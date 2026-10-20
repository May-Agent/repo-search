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
use crate::chunk::{self, ConfigChunker, DocChunker, FileChunker, Language, MemberEmit};
#[cfg(feature = "lang-typescript")]
use crate::chunk::{Tsx, TypeScript};
use crate::store::{
    ChunkRecord, ContentHash, FileChange, FileState, MemberRecord, StoredChunk, StoredMember,
    StoredMemberKind, fts_body,
};

/// Directories that are never part of the indexed tree, even when ignore rules
/// are switched off.
const RESERVED_DIRS: [&str; 2] = [".git", ".repo-search"];

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

fn chunk_with<C: FileChunker>(
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
    pub files_scanned: u64,
    pub files_added: u64,
    pub files_modified: u64,
    pub files_removed: u64,
    pub files_restamped: u64,
    pub skipped: Vec<Skipped>,
}

/// A walked file that passed the filters: worktree-relative path, language,
/// and its fingerprint minus the content hash.
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
    /// Not indexable: the reason it is skipped.
    Skip(SkipReason),
}

/// Walks `root` and diffs it against `stored`.
pub(super) fn plan(
    root: &Path,
    include_ignored: bool,
    stored: HashMap<String, FileState>,
    tokens: &dyn TokenCounter,
) -> Result<Plan, Error> {
    let Walked {
        candidates,
        skipped,
    } = walk(root, include_ignored)?;
    let mut plan = Plan {
        changes: Vec::new(),
        texts: HashMap::new(),
        files_scanned: candidates.len() as u64,
        files_added: 0,
        files_modified: 0,
        files_removed: 0,
        files_restamped: 0,
        skipped,
    };
    let mut live: HashSet<String> = HashSet::with_capacity(candidates.len());
    // First the fingerprint diff decides every file's fate and reads the
    // ones that need chunking.
    let mut work = Vec::with_capacity(candidates.len());
    let mut jobs = Vec::new();
    for cand in candidates {
        let previous = stored.get(&cand.rel);
        if previous.is_some_and(|state| cand.matches(state)) {
            live.insert(cand.rel.clone());
            work.push((cand, Work::Live));
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
        let Ok(src) = String::from_utf8(bytes) else {
            tracing::warn!(path = %cand.rel, "skipping a file that is not valid UTF-8");
            work.push((cand, Work::Skip(SkipReason::NotUtf8)));
            continue;
        };
        jobs.push(work.len());
        work.push((cand, Work::Chunk(state, src, jobs.len() - 1)));
    }
    // The files to chunk go to a worker per core, each with its own
    // chunkers; chunk order is preserved by job index.
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
            Work::Restamp(state) => plan.changes.push(FileChange::Restamp(state)),
            Work::Skip(reason) => {
                plan.skipped.push(Skipped {
                    path: cand.rel,
                    reason,
                });
            }
            Work::Chunk(state, _src, job) => {
                let chunked = match results[job].take().expect("every job ran") {
                    Ok(chunked) => chunked,
                    Err(error) => {
                        tracing::warn!(path = %cand.rel, %error, "skipping a file that is not valid");
                        plan.skipped.push(Skipped {
                            path: cand.rel,
                            reason: SkipReason::Malformed,
                        });
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
    // vanished mid-pass), newly ignored, no longer valid UTF-8, or malformed.
    let mut gone: Vec<String> = stored
        .into_keys()
        .filter(|path| !live.contains(path))
        .collect();
    gone.sort_unstable();
    plan.files_removed = gone.len() as u64;
    plan.changes
        .extend(gone.into_iter().map(FileChange::Remove));
    Ok(plan)
}

/// Rebuilds the text of stored chunks from their files' bytes and byte
/// ranges, without parsing. The pass reconciled the index moments ago, so the
/// stored rows are the truth; the hash check only guards against a file
/// rewritten in the meantime, whose chunks are then left out and handled by
/// the next pass (as are files deleted in the meantime).
pub(super) fn read_chunks(
    root: &Path,
    mut stored: Vec<StoredChunk>,
) -> Result<Vec<PendingChunk>, Error> {
    stored.sort_by(|a, b| a.path.cmp(&b.path));
    let mut out = Vec::with_capacity(stored.len());
    let mut file: Option<(String, Vec<u8>)> = None;
    for chunk in stored {
        if file.as_ref().is_none_or(|(path, _)| *path != chunk.path) {
            match fs::read(root.join(&chunk.path)) {
                Ok(bytes) => file = Some((chunk.path.clone(), bytes)),
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
        let Some((_, bytes)) = file.as_ref().filter(|(path, _)| *path == chunk.path) else {
            continue;
        };
        let Some(text) = chunk_text(bytes, &chunk.members) else {
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

/// The member slices joined by `\n`, as `Chunk::write_text` builds them — a
/// signature member emits its slice whitespace-collapsed plus ` …\n`;
/// `None` when the file is not UTF-8 or a range no longer fits it.
fn chunk_text(bytes: &[u8], members: &[StoredMember]) -> Option<String> {
    let src = std::str::from_utf8(bytes).ok()?;
    let mut emit = Vec::with_capacity(members.len());
    for member in members {
        emit.push((
            usize::try_from(member.bytes.start).ok()?..usize::try_from(member.bytes.end).ok()?,
            match member.kind {
                StoredMemberKind::Signature { cut } => MemberEmit::Signature { cut },
                StoredMemberKind::Text => MemberEmit::Text,
            },
        ));
    }
    let mut text = String::new();
    chunk::write_members(src, emit, &mut text)?;
    Some(text)
}

/// What the walk found.
struct Walked {
    /// Every supported file under the root, sorted by path.
    candidates: Vec<Candidate>,
    /// Files the walk could not index.
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

fn walk(root: &Path, include_ignored: bool) -> Result<Walked, Error> {
    let mut builder = WalkBuilder::new(root);
    builder
        .standard_filters(!include_ignored)
        // Dot-files and dot-directories are source too.
        .hidden(false)
        // `.gitignore` applies in any tree, with or without a `.git` marker.
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

    fn stored(path: &str, title: &str, text: &str, range: Range<u64>) -> StoredChunk {
        with_ranges(path, title, text, vec![range])
    }

    fn with_ranges(path: &str, title: &str, text: &str, ranges: Vec<Range<u64>>) -> StoredChunk {
        StoredChunk {
            path: path.into(),
            title: title.into(),
            lines: 1..=1,
            members: ranges
                .into_iter()
                .map(|bytes| StoredMember {
                    bytes,
                    kind: StoredMemberKind::Text,
                })
                .collect(),
            content_hash: ContentHash::of_chunk(title, text),
        }
    }

    #[test]
    fn read_chunks_joins_member_ranges() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}\n\nfn b() {}\n").unwrap();
        let chunk = with_ranges("a.rs", "t", "fn a() {}\nfn b() {}", vec![0..9, 11..20]);
        let out = read_chunks(dir.path(), vec![chunk]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "fn a() {}\nfn b() {}");
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
