//! The blocking half of reconcile: walking the worktree, hashing, reading and
//! chunking files. Runs on a dedicated thread, never on the caller's executor.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use ignore::WalkBuilder;

use super::{Error, PendingChunk, SkipReason, Skipped};
use crate::chunk::{self, Chunker, ConfigChunker, DocChunker, FileChunker, Language, Rust};
use crate::store::{
    ChunkRecord, ContentHash, FileChange, FileState, MemberRecord, StoredChunk, fts_body,
};

/// Directories that are never part of the indexed tree, even when ignore rules
/// are switched off.
const RESERVED_DIRS: [&str; 2] = [".git", ".repo-search"];

/// The chunkers that carry state across files: the tree-sitter parser is
/// reused, the documentation and configuration chunkers are stateless.
struct Chunkers {
    rust: Chunker<Rust>,
}

impl Chunkers {
    fn new() -> Self {
        Self {
            rust: Chunker::new(Rust),
        }
    }

    /// The records of `src` plus each chunk's text, in chunk order.
    fn chunk(
        &mut self,
        language: Language,
        path: &str,
        src: &str,
    ) -> Result<Vec<(ChunkRecord, String)>, chunk::Error> {
        match language {
            Language::Rust => chunk_with(&mut self.rust, path, src),
            Language::Doc(format) => chunk_with(&mut DocChunker::new(format), path, src),
            Language::Config(format) => chunk_with(&mut ConfigChunker::new(format), path, src),
        }
    }
}

fn chunk_with<C: FileChunker>(
    chunker: &mut C,
    path: &str,
    src: &str,
) -> Result<Vec<(ChunkRecord, String)>, chunk::Error> {
    let chunks = chunker.chunk_file(path, src)?;
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

/// Walks `root` and diffs it against `stored`.
pub(super) fn plan(
    root: &Path,
    include_ignored: bool,
    stored: HashMap<String, FileState>,
) -> Result<Plan, Error> {
    let Walked {
        candidates,
        skipped,
    } = walk(root, include_ignored)?;
    let mut chunkers = Chunkers::new();
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
    for cand in candidates {
        let previous = stored.get(&cand.rel);
        if previous.is_some_and(|state| cand.matches(state)) {
            live.insert(cand.rel);
            continue;
        }
        let bytes = match fs::read(root.join(&cand.rel)) {
            Ok(bytes) => bytes,
            // Deleted between the stat and the read: a deletion like any
            // other, so it is left out of `live` and removed below.
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
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
            live.insert(cand.rel);
            plan.files_restamped += 1;
            plan.changes.push(FileChange::Restamp(state));
            continue;
        }
        let Ok(src) = String::from_utf8(bytes) else {
            tracing::warn!(path = %cand.rel, "skipping a file that is not valid UTF-8");
            plan.skipped.push(Skipped {
                path: cand.rel,
                reason: SkipReason::NotUtf8,
            });
            continue;
        };
        let chunked = match chunkers.chunk(cand.language, &cand.rel, &src) {
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
        if previous.is_some() {
            plan.files_modified += 1;
        } else {
            plan.files_added += 1;
        }
        live.insert(cand.rel);
        plan.changes.push(FileChange::Replace(state, records));
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
        let Some(text) = chunk_text(bytes, &chunk.ranges) else {
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

/// The member slices joined by `\n`, as `Chunk::write_text` builds them;
/// `None` when a range no longer fits the file or splits a character.
fn chunk_text(bytes: &[u8], ranges: &[Range<u64>]) -> Option<String> {
    let mut text = Vec::new();
    for (i, range) in ranges.iter().enumerate() {
        if i > 0 {
            text.push(b'\n');
        }
        let range = usize::try_from(range.start).ok()?..usize::try_from(range.end).ok()?;
        text.extend_from_slice(bytes.get(range)?);
    }
    String::from_utf8(text).ok()
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
            ranges,
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
