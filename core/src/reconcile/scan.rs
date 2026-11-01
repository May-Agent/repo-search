//! The blocking half of reconcile: walking the worktree, hashing, reading and
//! chunking files. Runs on a dedicated thread, never on the caller's executor.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use ignore::WalkBuilder;

use super::{Error, PendingChunk, SkipReason, Skipped};
use crate::TokenCounter;
#[cfg(feature = "lang-c")]
use crate::chunk::C;
#[cfg(any(
    feature = "lang-c",
    feature = "lang-cpp",
    feature = "lang-go",
    feature = "lang-javascript",
    feature = "lang-python",
    feature = "lang-rust",
    feature = "lang-swift",
    feature = "lang-typescript"
))]
use crate::chunk::Chunker;
#[cfg(feature = "lang-cpp")]
use crate::chunk::Cpp;
#[cfg(feature = "lang-go")]
use crate::chunk::Go;
#[cfg(feature = "lang-javascript")]
use crate::chunk::JavaScript;
#[cfg(feature = "lang-python")]
use crate::chunk::Python;
#[cfg(feature = "lang-rust")]
use crate::chunk::Rust;
#[cfg(feature = "lang-swift")]
use crate::chunk::Swift;
use crate::chunk::{self, ConfigChunker, DocChunker, FileChunker, Language};
#[cfg(feature = "lang-typescript")]
use crate::chunk::{Tsx, TypeScript};
use crate::store::{
    ChunkRecord, ContentHash, FileChange, FileState, MemberRecord, StoredChunk, Unchunkable,
    UnchunkableReason, fts_body,
};

/// Directories that are never part of the indexed tree, even when ignore rules
/// are switched off.
const RESERVED_DIRS: [&str; 1] = [".git"];

/// The chunkers that carry state across files: the tree-sitter parsers
/// are reused, the documentation and configuration chunkers are stateless.
struct Chunkers {
    #[cfg(feature = "lang-c")]
    c: Chunker<C>,
    #[cfg(feature = "lang-cpp")]
    cpp: Chunker<Cpp>,
    #[cfg(feature = "lang-go")]
    go: Chunker<Go>,
    #[cfg(feature = "lang-javascript")]
    javascript: Chunker<JavaScript>,
    #[cfg(feature = "lang-python")]
    python: Chunker<Python>,
    #[cfg(feature = "lang-rust")]
    rust: Chunker<Rust>,
    #[cfg(feature = "lang-swift")]
    swift: Chunker<Swift>,
    #[cfg(feature = "lang-typescript")]
    typescript: Chunker<TypeScript>,
    #[cfg(feature = "lang-typescript")]
    tsx: Chunker<Tsx>,
}

impl Chunkers {
    fn new() -> Self {
        Self {
            #[cfg(feature = "lang-c")]
            c: Chunker::new(C),
            #[cfg(feature = "lang-cpp")]
            cpp: Chunker::new(Cpp),
            #[cfg(feature = "lang-go")]
            go: Chunker::new(Go),
            #[cfg(feature = "lang-javascript")]
            javascript: Chunker::new(JavaScript),
            #[cfg(feature = "lang-python")]
            python: Chunker::new(Python),
            #[cfg(feature = "lang-rust")]
            rust: Chunker::new(Rust),
            #[cfg(feature = "lang-swift")]
            swift: Chunker::new(Swift),
            #[cfg(feature = "lang-typescript")]
            typescript: Chunker::new(TypeScript),
            #[cfg(feature = "lang-typescript")]
            tsx: Chunker::new(Tsx),
        }
    }

    /// The records of `src` plus each chunk's text, in chunk order.
    fn chunk(
        &mut self,
        language: Language,
        path: &str,
        src: &str,
        tokens: &dyn TokenCounter,
    ) -> Result<Vec<(ChunkRecord, String)>, chunk::Error> {
        match language {
            #[cfg(feature = "lang-c")]
            Language::C => chunk_with(&mut self.c, path, src, tokens),
            #[cfg(feature = "lang-cpp")]
            Language::Cpp => chunk_with(&mut self.cpp, path, src, tokens),
            #[cfg(feature = "lang-go")]
            Language::Go => chunk_with(&mut self.go, path, src, tokens),
            #[cfg(feature = "lang-javascript")]
            Language::JavaScript => chunk_with(&mut self.javascript, path, src, tokens),
            #[cfg(feature = "lang-python")]
            Language::Python => chunk_with(&mut self.python, path, src, tokens),
            #[cfg(feature = "lang-rust")]
            Language::Rust => chunk_with(&mut self.rust, path, src, tokens),
            #[cfg(feature = "lang-swift")]
            Language::Swift => chunk_with(&mut self.swift, path, src, tokens),
            #[cfg(feature = "lang-typescript")]
            Language::TypeScript => chunk_with(&mut self.typescript, path, src, tokens),
            #[cfg(feature = "lang-typescript")]
            Language::Tsx => chunk_with(&mut self.tsx, path, src, tokens),
            Language::Doc(format) => chunk_with(&mut DocChunker::new(format), path, src, tokens),
            Language::Config(format) => {
                chunk_with(&mut ConfigChunker::new(format), path, src, tokens)
            }
        }
    }
}

/// The records of `src` as `chunker` cuts it, plus each chunk's text, in
/// chunk order.
pub(crate) fn chunk_with<C: FileChunker>(
    chunker: &mut C,
    path: &str,
    src: &str,
    tokens: &dyn TokenCounter,
) -> Result<Vec<(ChunkRecord, String)>, chunk::Error> {
    let chunks = chunker.chunk_file(path, src, tokens)?;
    Ok(chunks
        .iter()
        .map(|chunk| {
            let mut text = String::new();
            chunk.write_text(src, &mut text);
            let record = ChunkRecord {
                lines: chunk.lines.clone(),
                title: chunk.title.clone(),
                kind: chunker.chunk_kind_name(chunk.kind).into_owned(),
                members: chunk
                    .members
                    .iter()
                    .map(|m| MemberRecord {
                        bytes: m.bytes.start as u64..m.bytes.end as u64,
                        kind: chunker.member_kind_name(m.kind).into_owned(),
                        cut: match m.kind {
                            crate::chunk::MemberKind::Signature { cut } => cut as u64,
                            _ => 0,
                        },
                        symbol: m.symbol.clone(),
                    })
                    .collect(),
                content_hash: ContentHash::of_chunk(&chunk.title, &text),
                fts_body: fts_body(&text),
            };
            (record, text)
        })
        .collect())
}

/// Everything the walk decided, before anything is written.
pub(super) struct Plan {
    /// Changes for the one transaction; empty when nothing changed.
    pub changes: Vec<FileChange>,
    /// Text of every chunk of every re-chunked file, by content hash.
    pub texts: HashMap<ContentHash, PendingChunk>,
    /// Paths of the files written as `FileChange::Restamp` this pass: same
    /// bytes, moved stat or ignored flag. Their stored chunks stay.
    pub restamped: Vec<String>,
    pub files_scanned: u64,
    pub files_added: u64,
    pub files_modified: u64,
    pub files_removed: u64,
    pub files_restamped: u64,
    pub files_chunked: u64,
    pub skipped: Vec<Skipped>,
}

impl Plan {
    /// Records `record` as the file's new state. `was_chunked` is whether the
    /// index held chunks for it, which the change drops.
    fn unchunkable(&mut self, was_chunked: bool, record: Unchunkable) {
        if was_chunked {
            self.files_removed += 1;
        }
        self.skipped.push(Skipped {
            path: record.state.path.clone(),
            reason: record.reason.clone().into(),
        });
        self.changes.push(FileChange::Unchunkable(record));
    }
}

/// A file under the root that passed the filters: worktree-relative path,
/// language and fingerprint minus the content hash.
struct Candidate {
    rel: String,
    language: Language,
    size: u64,
    mtime_ns: i64,
    inode: u64,
}

impl Candidate {
    fn matches(&self, state: &FileState) -> bool {
        self.size == state.size && self.mtime_ns == state.mtime_ns && self.inode == state.inode
    }
}

/// One file's chunking outcome: its records and texts, or the failure.
type Chunked = Result<Vec<(ChunkRecord, String)>, chunk::Error>;

/// What the fingerprint pass decided for one candidate.
enum Work {
    /// Unchanged (or vanished mid-pass): stays stored as it is.
    Live,
    /// Same bytes, moved fingerprint: stored chunks stay.
    Restamp(FileState),
    /// Re-chunk: its new state and source, and the file's index in `jobs`.
    Chunk(FileState, String, usize),
    /// Recorded as unchunkable with this fingerprint: stays as it is.
    KeepUnchunkable(UnchunkableReason),
    /// Recorded as unchunkable, same bytes, moved fingerprint.
    RestampUnchunkable(FileState, UnchunkableReason),
    /// Newly found not to be UTF-8.
    NotUtf8(FileState),
}

/// Lists `root`'s files and diffs them against `stored`, the chunked files,
/// and `recorded`, the files that failed to chunk.
pub(super) fn plan(
    root: &Path,
    stored: HashMap<String, FileState>,
    recorded: HashMap<String, Unchunkable>,
    tokens: &dyn TokenCounter,
) -> Result<Plan, Error> {
    let Walked {
        candidates,
        skipped,
    } = list(root)?;
    let mut plan = Plan {
        changes: Vec::new(),
        texts: HashMap::new(),
        restamped: Vec::new(),
        files_scanned: candidates.len() as u64,
        files_added: 0,
        files_modified: 0,
        files_removed: 0,
        files_restamped: 0,
        files_chunked: 0,
        skipped,
    };
    // Paths whose stored rows this pass keeps, rewrites or replaces; any
    // other stored path is gone.
    let mut live: HashSet<String> = HashSet::with_capacity(candidates.len());
    // First the fingerprint diff decides every file's fate and reads the
    // ones that need chunking.
    let mut work = Vec::with_capacity(candidates.len());
    let mut jobs = Vec::new();
    for cand in candidates {
        let previous = stored.get(&cand.rel);
        let failed = recorded.get(&cand.rel);
        if previous.filter(|state| cand.matches(state)).is_some() {
            live.insert(cand.rel.clone());
            work.push((cand, Work::Live));
            continue;
        }
        if let Some(record) = failed.filter(|record| cand.matches(&record.state)) {
            live.insert(cand.rel.clone());
            let reason = record.reason.clone();
            work.push((cand, Work::KeepUnchunkable(reason)));
            continue;
        }
        let bytes = match fs::read(root.join(&cand.rel)) {
            Ok(bytes) => bytes,
            // Deleted between the stat and the read: a deletion like any
            // other, so it is left out of `live` and removed below.
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                work.push((cand, Work::Live));
                continue;
            }
            Err(source) => {
                return Err(Error::Io {
                    path: cand.rel,
                    source,
                });
            }
        };
        let content_hash = ContentHash::of_file(&bytes);
        let state = FileState {
            path: cand.rel.clone(),
            size: cand.size,
            mtime_ns: cand.mtime_ns,
            inode: cand.inode,
            content_hash,
        };
        if previous.is_some_and(|old| old.content_hash == content_hash) {
            live.insert(cand.rel.clone());
            plan.files_restamped += 1;
            work.push((cand, Work::Restamp(state)));
            continue;
        }
        if let Some(record) = failed.filter(|record| record.state.content_hash == content_hash) {
            live.insert(cand.rel.clone());
            plan.files_restamped += 1;
            let reason = record.reason.clone();
            work.push((cand, Work::RestampUnchunkable(state, reason)));
            continue;
        }
        let Ok(src) = String::from_utf8(bytes) else {
            tracing::warn!(path = %cand.rel, "recording a file that is not valid UTF-8");
            live.insert(cand.rel.clone());
            work.push((cand, Work::NotUtf8(state)));
            continue;
        };
        jobs.push(work.len());
        work.push((cand, Work::Chunk(state, src, jobs.len() - 1)));
    }
    // The files to chunk go to a worker per core, each with its own
    // chunkers; chunk order is preserved by job index.
    plan.files_chunked = jobs.len() as u64;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(jobs.len());
    let per_worker: Vec<Vec<(usize, Chunked)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut chunkers = Chunkers::new();
                    let mut out = Vec::new();
                    loop {
                        let job = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if job >= jobs.len() {
                            break;
                        }
                        let (cand, work) = &work[jobs[job]];
                        let Work::Chunk(_, src, _) = work else {
                            unreachable!("jobs holds only Chunk entries")
                        };
                        out.push((job, chunkers.chunk(cand.language, &cand.rel, src, tokens)));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut results: Vec<Option<Chunked>> = (0..jobs.len()).map(|_| None).collect();
    for done in per_worker {
        for (job, result) in done {
            results[job] = Some(result);
        }
    }
    for (cand, item) in work {
        match item {
            Work::Live => {}
            Work::Restamp(state) => {
                plan.restamped.push(state.path.clone());
                plan.changes.push(FileChange::Restamp(state));
            }
            Work::KeepUnchunkable(reason) => plan.skipped.push(Skipped {
                path: cand.rel,
                reason: reason.into(),
            }),
            Work::RestampUnchunkable(state, reason) => {
                plan.skipped.push(Skipped {
                    path: cand.rel,
                    reason: reason.into(),
                });
                plan.changes.push(FileChange::RestampUnchunkable(state));
            }
            Work::NotUtf8(state) => plan.unchunkable(
                stored.contains_key(&cand.rel),
                Unchunkable {
                    state,
                    reason: UnchunkableReason::NotUtf8,
                },
            ),
            Work::Chunk(state, _src, job) => {
                let chunked = match results[job].take().expect("every job ran") {
                    Ok(chunked) => chunked,
                    Err(error) => {
                        tracing::warn!(path = %cand.rel, %error, "recording a file that does not parse");
                        live.insert(cand.rel.clone());
                        plan.unchunkable(
                            stored.contains_key(&cand.rel),
                            Unchunkable {
                                state,
                                reason: UnchunkableReason::Malformed(error.to_string()),
                            },
                        );
                        continue;
                    }
                };
                let mut records = Vec::with_capacity(chunked.len());
                for (record, text) in chunked {
                    plan.texts
                        .entry(record.content_hash)
                        .or_insert_with(|| PendingChunk {
                            path: cand.rel.clone(),
                            title: record.title.clone(),
                            lines: record.lines.clone(),
                            content_hash: record.content_hash,
                            text,
                        });
                    records.push(record);
                }
                if stored.contains_key(&cand.rel) {
                    plan.files_modified += 1;
                } else {
                    plan.files_added += 1;
                }
                live.insert(cand.rel);
                plan.changes.push(FileChange::Replace(state, records));
            }
        }
    }
    // Whatever was stored but not seen alive is gone: deleted (including
    // vanished mid-pass) or no longer tracked. A chunked file that now
    // fails to chunk was counted as removed when it was recorded.
    let mut gone: Vec<String> = stored
        .values()
        .chain(recorded.values().map(|record| &record.state))
        .filter(|state| !live.contains(&state.path))
        .map(|state| state.path.clone())
        .collect();
    gone.sort_unstable();
    plan.files_removed += gone.iter().filter(|p| stored.contains_key(*p)).count() as u64;
    plan.changes
        .extend(gone.into_iter().map(FileChange::Remove));
    Ok(plan)
}

/// Rebuilds the text of stored chunks from their files' bytes and byte
/// ranges, without parsing. The pass reconciled the index moments ago, so the
/// stored rows are the truth; the hash check only guards against a file
/// rewritten in the meantime, whose chunks are then left out and handled by
/// the next pass (as are files deleted in the meantime). Each file is read
/// and validated once, however many of its chunks are rebuilt.
pub(super) fn read_chunks(
    root: &Path,
    mut stored: Vec<StoredChunk>,
) -> Result<Vec<PendingChunk>, Error> {
    stored.sort_by(|a, b| a.path.cmp(&b.path));
    let mut out = Vec::with_capacity(stored.len());
    let mut file: Option<(String, Option<String>)> = None;
    for chunk in stored {
        if file.as_ref().is_none_or(|(path, _)| *path != chunk.path) {
            match fs::read(root.join(&chunk.path)) {
                Ok(bytes) => {
                    file = Some((chunk.path.clone(), String::from_utf8(bytes).ok()));
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    file = None;
                    continue;
                }
                Err(source) => {
                    return Err(Error::Io {
                        path: chunk.path,
                        source,
                    });
                }
            }
        }
        let Some((_, Some(src))) = file.as_ref().filter(|(path, _)| *path == chunk.path) else {
            continue;
        };
        let Some(text) =
            chunk::rebuild_text(src, chunk.members.iter().map(|m| (m.bytes.clone(), m.kind)))
        else {
            continue;
        };
        if ContentHash::of_chunk(&chunk.title, &text) != chunk.content_hash {
            continue;
        }
        out.push(PendingChunk {
            path: chunk.path,
            title: chunk.title,
            lines: chunk.lines,
            content_hash: chunk.content_hash,
            text,
        });
    }
    Ok(out)
}

/// What the listing found.
struct Walked {
    /// Every supported file under the root, sorted by path.
    candidates: Vec<Candidate>,
    /// Files the listing could not index.
    skipped: Vec<Skipped>,
}

/// Whether `err` is an entry below the root that vanished between the
/// directory listing and the stat: a deletion, not a failure of the walk. An
/// error at the root itself (depth 0, or no depth) is never a deletion.
fn is_vanished(err: &ignore::Error) -> bool {
    err.depth().is_some_and(|depth| depth > 0)
        && err
            .io_error()
            .is_some_and(|e| e.kind() == io::ErrorKind::NotFound)
}

/// The files a pass looks at: `root`'s git-tracked files when it is inside
/// a git worktree, else the filtered walk's.
fn list(root: &Path) -> Result<Walked, Error> {
    match tracked(root) {
        Some(listed) => listed,
        None => walk_once(root),
    }
}

/// `root`'s git index entries: `None` when `root` is not inside a git
/// worktree (or the worktree's index cannot be read as files), else the
/// tracked regular files under it. Untracked and ignored files are never
/// listed; an entry that has no file on disk — deleted, or a path now held
/// by a directory or a symlink — is skipped.
fn tracked(root: &Path) -> Option<Result<Walked, Error>> {
    let repo = git2::Repository::discover(root).ok()?;
    let workdir = repo.workdir()?;
    // `root` is `workdir` or under it; the prefix in the index's
    // `/`-separated spelling selects the entries under `root`.
    let prefix = std::fs::canonicalize(root)
        .ok()
        .zip(std::fs::canonicalize(workdir).ok())
        .and_then(|(root, workdir)| root.strip_prefix(&workdir).ok().map(|p| p.to_owned()))
        .and_then(|prefix| rel_string(&prefix))?;
    Some((|| {
        // The index is sorted by path; a set dedups a conflicted entry's
        // stages and keeps the order.
        let mut paths = std::collections::BTreeSet::new();
        let mut skipped = Vec::new();
        for entry in repo.index()?.iter() {
            // A regular file only: a gitlink, symlink or tree entry has no
            // bytes of its own.
            let file = [
                git2::FileMode::Blob,
                git2::FileMode::BlobGroupWritable,
                git2::FileMode::BlobExecutable,
            ]
            .map(u32::from);
            if !file.contains(&entry.mode) {
                continue;
            }
            let Ok(path) = str::from_utf8(&entry.path) else {
                tracing::warn!(path = ?entry.path, "skipping a path that is not valid UTF-8");
                skipped.push(Skipped {
                    path: String::from_utf8_lossy(&entry.path).into_owned(),
                    reason: SkipReason::NotUtf8Path,
                });
                continue;
            };
            let rel = if prefix.is_empty() {
                path
            } else {
                match path
                    .strip_prefix(prefix.as_str())
                    .and_then(|rest| rest.strip_prefix('/'))
                {
                    Some(rel) => rel,
                    None => continue,
                }
            };
            paths.insert(rel.to_owned());
        }
        let mut candidates = Vec::with_capacity(paths.len());
        for rel in paths {
            match candidate(root, &rel) {
                Ok(Some(cand)) => candidates.push(cand),
                Ok(None) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(Walked {
            candidates,
            skipped,
        })
    })())
}

/// The candidate `rel` under `root` makes: `None` for a file that vanished,
/// is not regular, or speaks no language; an unstatable or unmeasurable
/// file is an `Err`.
fn candidate(root: &Path, rel: &str) -> Result<Option<Candidate>, Error> {
    let meta = match fs::symlink_metadata(root.join(rel)) {
        Ok(meta) => meta,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                path: rel.to_owned(),
                source,
            });
        }
    };
    if !meta.is_file() {
        return Ok(None);
    }
    let Some(language) = Language::of_path(Path::new(rel)) else {
        return Ok(None);
    };
    let mtime_ns = meta
        .mtime()
        .checked_mul(1_000_000_000)
        .and_then(|ns| ns.checked_add(meta.mtime_nsec()))
        .ok_or_else(|| Error::Mtime {
            path: rel.to_owned(),
            secs: meta.mtime(),
        })?;
    Ok(Some(Candidate {
        rel: rel.to_owned(),
        language,
        size: meta.len(),
        mtime_ns,
        inode: meta.ino(),
    }))
}

/// Walks every supported file under `root`: it is not inside a git
/// worktree, so no ignore rules apply — a collection's documents must not
/// be hidden by a rule a parent directory carries. Reserved directories are
/// skipped, dot-files are source too and symlinks are not followed.
fn walk_once(root: &Path) -> Result<Walked, Error> {
    let mut builder = WalkBuilder::new(root);
    builder
        .standard_filters(false)
        .hidden(false)
        .require_git(false)
        .follow_links(false)
        .filter_entry(|entry| {
            !(entry.depth() > 0
                && entry.file_type().is_some_and(|t| t.is_dir())
                && RESERVED_DIRS.iter().any(|d| entry.file_name() == *d))
        });
    let mut candidates = Vec::new();
    let mut skipped = Vec::new();
    for entry in builder.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) if is_vanished(&e) => continue,
            Err(e) => return Err(e.into()),
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Some(language) = Language::of_path(entry.path()) else {
            continue;
        };
        let rel = entry
            .path()
            .strip_prefix(root)
            .expect("the walker yields paths under its root");
        let Some(rel_utf8) = rel_string(rel) else {
            tracing::warn!(path = ?rel, "skipping a path that is not valid UTF-8");
            skipped.push(Skipped {
                path: rel.to_string_lossy().into_owned(),
                reason: SkipReason::NotUtf8Path,
            });
            continue;
        };
        let meta = match entry.metadata() {
            Ok(meta) => meta,
            Err(e) if is_vanished(&e) => continue,
            Err(e) => return Err(e.into()),
        };
        let mtime_ns = meta
            .mtime()
            .checked_mul(1_000_000_000)
            .and_then(|ns| ns.checked_add(meta.mtime_nsec()))
            .ok_or_else(|| Error::Mtime {
                path: rel_utf8.clone(),
                secs: meta.mtime(),
            })?;
        candidates.push(Candidate {
            rel: rel_utf8,
            language,
            size: meta.len(),
            mtime_ns,
            inode: meta.ino(),
        });
    }
    candidates.sort_unstable_by(|a, b| a.rel.cmp(&b.rel));
    Ok(Walked {
        candidates,
        skipped,
    })
}

/// `/`-separated form of a relative path; `None` when a component is not
/// UTF-8.
fn rel_string(rel: &Path) -> Option<String> {
    let mut out = String::new();
    for component in rel.components() {
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(component.as_os_str().to_str()?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::ops::Range;
    use std::os::unix::ffi::OsStrExt;

    use super::*;
    use crate::store::{StoredMember, StoredMemberKind};

    fn stored(path: &str, title: &str, text: &str, range: Range<u64>) -> StoredChunk {
        with_ranges(path, title, text, vec![(range, StoredMemberKind::Text)])
    }

    fn with_ranges(
        path: &str,
        title: &str,
        text: &str,
        members: Vec<(Range<u64>, StoredMemberKind)>,
    ) -> StoredChunk {
        StoredChunk {
            path: path.into(),
            title: title.into(),
            lines: 1..=1,
            members: members
                .into_iter()
                .map(|(bytes, kind)| StoredMember { bytes, kind })
                .collect(),
            content_hash: ContentHash::of_chunk(title, text),
        }
    }

    #[test]
    fn read_chunks_joins_member_ranges() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}\n\nfn b() {}\n").unwrap();
        let chunk = with_ranges(
            "a.rs",
            "t",
            "fn a() {}\nfn b() {}",
            vec![
                (0..9, StoredMemberKind::Text),
                (11..20, StoredMemberKind::Text),
            ],
        );
        let out = read_chunks(dir.path(), vec![chunk]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "fn a() {}\nfn b() {}");
    }

    #[test]
    fn a_signature_member_rebuilds_as_write_text_wrote_it() {
        // A line-split part after the first: the signature member — the
        // collapsed, cut signature plus ` …\n` — ahead of the part's text.
        let src = "fn f(\n    a: u8,\n    b: u8,\n) -> u8 {\n    a + b\n}\n";
        let chunk = chunk::Chunk {
            title: "t".into(),
            kind: chunk::ChunkKind::Packed,
            lines: 1..=7,
            members: smallvec::smallvec![
                chunk::Member {
                    bytes: 0..35,
                    kind: chunk::MemberKind::Signature { cut: 20 },
                    symbol: String::new(),
                },
                chunk::Member {
                    bytes: 35..50,
                    kind: chunk::MemberKind::Doc,
                    symbol: String::new(),
                },
            ],
        };
        let mut text = String::new();
        chunk.write_text(src, &mut text);
        let members = [
            (0..35, StoredMemberKind::Signature { cut: 20 }),
            (35..50, StoredMemberKind::Text),
        ];
        assert_eq!(
            chunk::rebuild_text(src, members.iter().cloned()),
            Some(text.clone())
        );
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), src).unwrap();
        let stored = with_ranges("a.rs", "t", &text, members.to_vec());
        let out = read_chunks(dir.path(), vec![stored]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, text);
    }

    #[test]
    fn read_chunks_rebuilds_every_chunk_of_one_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}\nfn b() {}\nfn c() {}\n").unwrap();
        let chunks = vec![
            stored("a.rs", "t", "fn a() {}", 0..9),
            stored("a.rs", "t", "fn b() {}", 10..19),
            stored("a.rs", "t", "fn c() {}", 20..29),
        ];
        let out = read_chunks(dir.path(), chunks).unwrap();
        assert_eq!(out.len(), 3);
        let texts: Vec<&str> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["fn a() {}", "fn b() {}", "fn c() {}"]);
    }

    #[test]
    fn read_chunks_skips_every_chunk_of_a_file_that_is_not_utf8() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bad.rs"), b"fn a() {}\xff\nfn b() {}\n").unwrap();
        fs::write(dir.path().join("good.rs"), "fn g() {}\n").unwrap();
        let chunks = vec![
            stored("bad.rs", "t", "fn a() {}", 0..9),
            stored("bad.rs", "t", "fn b() {}", 11..20),
            stored("good.rs", "t", "fn g() {}", 0..9),
        ];
        let out = read_chunks(dir.path(), chunks).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "good.rs");
        assert_eq!(out[0].text, "fn g() {}");
    }

    #[test]
    fn read_chunks_treats_a_vanished_file_as_a_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let chunk = stored("gone.rs", "t", "x", 0..1);
        assert!(read_chunks(dir.path(), vec![chunk]).unwrap().is_empty());
    }

    #[test]
    fn read_chunks_drops_chunks_of_a_rewritten_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn changed() {}\n").unwrap();
        let stale = stored("a.rs", "t", "fn a() {}", 0..9);
        let short = stored("a.rs", "t", "fn a() {}", 0..999);
        assert!(
            read_chunks(dir.path(), vec![stale, short])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_not_found_walk_error_is_a_vanished_entry() {
        let gone = ignore::Error::WithDepth {
            depth: 2,
            err: Box::new(ignore::Error::WithPath {
                path: "x".into(),
                err: Box::new(ignore::Error::Io(io::ErrorKind::NotFound.into())),
            }),
        };
        assert!(is_vanished(&gone));
        let root = ignore::Error::WithDepth {
            depth: 0,
            err: Box::new(ignore::Error::WithPath {
                path: "x".into(),
                err: Box::new(ignore::Error::Io(io::ErrorKind::NotFound.into())),
            }),
        };
        assert!(!is_vanished(&root));
        let denied = ignore::Error::Io(io::ErrorKind::PermissionDenied.into());
        assert!(!is_vanished(&denied));
    }

    #[test]
    fn rel_string_rejects_non_utf8_components() {
        assert_eq!(rel_string(Path::new("a/b.rs")).as_deref(), Some("a/b.rs"));
        assert_eq!(
            rel_string(Path::new(OsStr::from_bytes(b"a/b\xff.rs"))),
            None
        );
    }
}
