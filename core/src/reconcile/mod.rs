//! The reconcile pass: brings the index in line with the worktree and reports
//! how the set of chunks needing a vector changed; [`missing`] lists that set
//! whole.
//!
//! One pass walks the worktree, stats every supported file and diffs it
//! against the stored file table. Only files whose size, mtime or inode moved
//! are read; of those, files whose bytes hash the same as before just get a
//! fresh fingerprint, and the rest are re-chunked. A file that fails to chunk
//! is recorded with its fingerprint and error and then treated the same way:
//! left unread while its stat holds, restamped when only the stat moved, and
//! tried again when its content changed. All changes land in one transaction.
//! A first index is the same pass over an empty table.

mod scan;

use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use tokio::sync::oneshot;

use crate::TokenCounter;
use crate::store::{
    self, Applied, ContentHash, Index, StoredChunk, UnchunkableReason, VectorCache,
};

/// Failures of a reconcile pass.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The index or the vector cache failed; the pass wrote nothing it did not
    /// commit atomically.
    #[error(transparent)]
    Store(#[from] store::Error),

    /// Walking the worktree failed.
    #[error("walking the worktree: {0}")]
    Walk(#[from] ignore::Error),

    /// Reading a file failed for a reason other than it having vanished.
    #[error("reading {path}: {source}")]
    Io {
        /// Worktree-relative path.
        path: String,
        /// The underlying failure.
        source: std::io::Error,
    },

    /// The mtime of a file does not fit in signed 64-bit nanoseconds.
    #[error("mtime of {path} ({secs} s since the epoch) overflows nanoseconds in 64 bits")]
    Mtime {
        /// Worktree-relative path.
        path: String,
        /// Seconds since the Unix epoch.
        secs: i64,
    },

    /// The blocking worker thread could not be started.
    #[error("spawning the reconcile worker: {0}")]
    Spawn(std::io::Error),

    /// The blocking worker exited without answering; a panic on it surfaces
    /// here, never as a panic in the caller.
    #[error("reconcile worker exited without answering")]
    WorkerGone,
}

/// How a pass walks the worktree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Index files that `.gitignore`, `.git/info/exclude`, global excludes and
    /// `.ignore` files would hide (`--include-ignored`), flagged as ignored.
    /// A pass without it keeps the flagged files stored but does not look at
    /// them, and queues none of their chunks for embedding. Hidden files and
    /// dot-directories are indexed either way; `.git/` and `.repo-search/`
    /// never are.
    pub include_ignored: bool,
}

/// A chunk whose vector is missing from the cache, with the text to embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChunk {
    /// Worktree-relative path of the file.
    pub path: String,
    /// Embed under this title: `Embedder::embed_document(title, text)`.
    pub title: String,
    /// The chunk text.
    pub text: String,
    /// 1-based inclusive line range, for progress and diagnostics.
    pub lines: RangeInclusive<u32>,
    /// The vector cache key: store the vector at this hash.
    pub content_hash: ContentHash,
}

/// Why a file was left out of the index this pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The file's content is not valid UTF-8. Recorded in the index.
    NotUtf8,
    /// The file's path is not valid UTF-8; [`Skipped::path`] is lossy. Decided
    /// by the path, so not recorded.
    NotUtf8Path,
    /// A configuration file that does not parse in its format; carries the
    /// parser's error. Recorded in the index.
    Malformed(String),
}

impl From<UnchunkableReason> for SkipReason {
    fn from(reason: UnchunkableReason) -> Self {
        match reason {
            UnchunkableReason::NotUtf8 => Self::NotUtf8,
            UnchunkableReason::Malformed(error) => Self::Malformed(error),
        }
    }
}

/// A walked file that was not indexed; any chunks it had are dropped. A file
/// that vanished during the pass is a deletion, not a skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    /// Worktree-relative path, lossily converted when not UTF-8.
    pub path: String,
    /// What was wrong with it.
    pub reason: SkipReason,
}

/// What a pass did, for the freshness line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// Supported files found by the walk and stat'd.
    pub files_scanned: u64,
    /// Files new to the index.
    pub files_added: u64,
    /// Files re-chunked because their content changed.
    pub files_modified: u64,
    /// Files dropped: deleted, or newly unchunkable.
    pub files_removed: u64,
    /// Files whose stat or ignored flag moved but whose bytes did not; only
    /// the fingerprint and flag were rewritten.
    pub files_restamped: u64,
    /// Files read and handed to a chunker this pass, whether it succeeded or
    /// not.
    pub files_chunked: u64,
    /// Chunk rows inserted.
    pub chunks_added: u64,
    /// Chunk rows deleted, including those of re-chunked files.
    pub chunks_removed: u64,
    /// Chunk rows visible to a search with the pass's `include_ignored`.
    pub chunks_total: u64,
    /// Chunk rows whose vector is in the cache for the model.
    pub chunks_with_vector: u64,
    /// The index was built by another chunker (or token counter) and was
    /// emptied, so every file was re-chunked this pass.
    pub rechunked: bool,
    /// Every walked file that is not indexed: the unchunkable files known to
    /// the index, recorded this pass or before, and paths that are not UTF-8.
    pub skipped: Vec<Skipped>,
}

impl Report {
    /// Files whose indexed content changed: added, modified or removed.
    #[must_use]
    pub fn files_changed(&self) -> u64 {
        self.files_added + self.files_modified + self.files_removed
    }

    /// Fraction of chunk rows that have a vector; `1.0` for an empty index.
    #[must_use]
    pub fn vector_coverage(&self) -> f64 {
        if self.chunks_total == 0 {
            1.0
        } else {
            self.chunks_with_vector as f64 / self.chunks_total as f64
        }
    }
}

/// What a pass changed in the set of chunks its scope needs vectors for.
///
/// Applied in pass order on top of [`missing`] taken after some earlier
/// pass of the same scope, successive passes' changes keep that set
/// current: `added` holds the chunks the pass put in scope or touched that
/// have no cached vector, `dropped` the hashes it took out of scope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Changes {
    /// Chunks of files the pass re-chunked or restamped that have no cached
    /// vector, one per distinct hash, ordered by path then first line. A
    /// restamped file's chunks are offered again so a chunk [`missing`]
    /// could not rebuild while the file was mid-change is never lost.
    pub added: Vec<PendingChunk>,
    /// Hashes some row of which the pass deleted and of which no row
    /// visible in the scope remains; distinct, in deletion order.
    pub dropped: Vec<ContentHash>,
}

/// The outcome of a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconciled {
    /// What the pass did.
    pub report: Report,
    /// The pass's changes; `None` when it re-chunked an emptied index
    /// (`report.rechunked`), whose deleted hashes nothing recorded: a caller
    /// keeping a pending set rebuilds it with [`missing`].
    pub changes: Option<Changes>,
}

/// Brings `index` in line with the worktree at `worktree`, then reports how
/// the pass changed the chunks its scope needs vectors for; [`missing`]
/// lists that set whole.
///
/// `chunker` identifies the chunker, normally [`crate::CHUNKER_FINGERPRINT`].
/// An index stored under another `(chunker, model_id)` is emptied first and
/// the pass re-chunks every file; cached vectors are reused by content hash.
///
/// `tokens` is the model's token counter; the chunker holds every chunk's
/// embedded document to the cap it counts.
///
/// Every change to files, chunks and FTS commits in one transaction; a pass
/// that finds nothing changed performs no write.
///
/// # Errors
///
/// When the walk or a read fails, or the index or cache rejects a request. A
/// failed pass leaves the index as it was.
pub async fn reconcile(
    index: &Index,
    cache: &VectorCache,
    model_id: &str,
    chunker: &str,
    worktree: &Path,
    options: Options,
    tokens: Arc<dyn TokenCounter>,
) -> Result<Reconciled, Error> {
    let rechunked = index.align_chunker(chunker, model_id).await?;
    let stored: HashMap<_, _> = index
        .file_states()
        .await?
        .into_iter()
        .map(|state| (state.path.clone(), state))
        .collect();
    let recorded: HashMap<_, _> = index
        .unchunkables()
        .await?
        .into_iter()
        .map(|record| (record.state.path.clone(), record))
        .collect();
    let root = worktree.to_path_buf();
    let plan = run_blocking(move || {
        scan::plan(&root, options.include_ignored, stored, recorded, &*tokens)
    })
    .await??;

    let applied = if plan.changes.is_empty() {
        Applied::default()
    } else {
        index.apply(plan.changes).await?
    };
    tracing::debug!(
        scanned = plan.files_scanned,
        added = plan.files_added,
        modified = plan.files_modified,
        removed = plan.files_removed,
        restamped = plan.files_restamped,
        chunked = plan.files_chunked,
        "reconcile applied"
    );

    let coverage = coverage(index, cache, model_id, options.include_ignored).await?;
    let visible: HashSet<ContentHash> = coverage.distinct.iter().copied().collect();
    let chunks_with_vector = coverage
        .rows
        .iter()
        .filter(|h| coverage.cached.contains(h))
        .count() as u64;

    let changes = if rechunked {
        // The emptied index's deleted hashes were never recorded; a caller
        // keeping a pending set rebuilds it with `missing`.
        None
    } else {
        let mut added: Vec<PendingChunk> = plan
            .texts
            .into_values()
            .filter(|chunk| !coverage.cached.contains(&chunk.content_hash))
            .collect();
        let mut taken: HashSet<ContentHash> =
            added.iter().map(|chunk| chunk.content_hash).collect();
        if !plan.restamped.is_empty() {
            let stored: Vec<StoredChunk> = index
                .chunks_in(plan.restamped)
                .await?
                .into_iter()
                .filter(|row| {
                    visible.contains(&row.content_hash)
                        && !coverage.cached.contains(&row.content_hash)
                        && taken.insert(row.content_hash)
                })
                .collect();
            if !stored.is_empty() {
                let root = worktree.to_path_buf();
                let rebuilt = run_blocking(move || scan::read_chunks(&root, stored)).await??;
                added.extend(rebuilt);
            }
        }
        added.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then(a.lines.start().cmp(b.lines.start()))
        });
        let dropped = applied
            .removed_hashes
            .into_iter()
            .filter(|hash| !visible.contains(hash))
            .collect();
        Some(Changes { added, dropped })
    };

    let report = Report {
        files_scanned: plan.files_scanned,
        files_added: plan.files_added,
        files_modified: plan.files_modified,
        files_removed: plan.files_removed,
        files_restamped: plan.files_restamped,
        files_chunked: plan.files_chunked,
        chunks_added: applied.chunks_added,
        chunks_removed: applied.chunks_removed,
        chunks_total: coverage.rows.len() as u64,
        chunks_with_vector,
        rechunked,
        skipped: plan.skipped,
    };
    Ok(Reconciled { report, changes })
}

/// Every distinct chunk hash visible in `options`' scope that has no vector
/// cached for `model_id`, in index order, with its text: taken from `known`
/// when it holds the hash, otherwise rebuilt from the file's stored byte
/// ranges (each file read and validated once). A hash whose file changed or
/// vanished since the last pass is left out; the next pass offers it again.
///
/// # Errors
///
/// When the index or the cache rejects a read, or a file read fails for a
/// reason other than the file having vanished.
pub async fn missing(
    index: &Index,
    cache: &VectorCache,
    model_id: &str,
    worktree: &Path,
    options: Options,
    known: Vec<PendingChunk>,
) -> Result<Vec<PendingChunk>, Error> {
    let coverage = coverage(index, cache, model_id, options.include_ignored).await?;

    // Missing hashes, distinct, in index order.
    let missing: Vec<ContentHash> = coverage
        .distinct
        .iter()
        .copied()
        .filter(|hash| !coverage.cached.contains(hash))
        .collect();
    let mut texts: HashMap<ContentHash, PendingChunk> =
        known.into_iter().map(|c| (c.content_hash, c)).collect();
    // Chunks `known` does not hold: rebuilt from the stored byte ranges, no
    // parsing.
    let to_load: Vec<ContentHash> = missing
        .iter()
        .filter(|hash| !texts.contains_key(hash))
        .copied()
        .collect();
    if !to_load.is_empty() {
        let stored = index.chunks_by_hash(to_load).await?;
        let root = worktree.to_path_buf();
        let loaded = run_blocking(move || scan::read_chunks(&root, stored)).await??;
        texts.extend(loaded.into_iter().map(|c| (c.content_hash, c)));
    }
    let mut pending = Vec::with_capacity(missing.len());
    for hash in missing {
        // A hash with no text belongs to a file that changed or vanished
        // after the pass; the next pass handles it.
        if let Some(chunk) = texts.remove(&hash) {
            pending.push(chunk);
        }
    }
    Ok(pending)
}

/// A scope's chunk coverage: every stored row's hash, the same hashes
/// deduplicated, and which of them have a cached vector.
struct Coverage {
    /// Every stored chunk row's hash, in index order.
    rows: Vec<ContentHash>,
    /// The distinct hashes of `rows`, in first-seen index order.
    distinct: Vec<ContentHash>,
    /// The hashes of `distinct` with a vector cached under the model.
    cached: HashSet<ContentHash>,
}

/// Reads `index`'s coverage in `include_ignored`'s scope against `cache`.
async fn coverage(
    index: &Index,
    cache: &VectorCache,
    model_id: &str,
    include_ignored: bool,
) -> Result<Coverage, Error> {
    let rows = index.chunk_hashes(include_ignored).await?;
    let mut seen = HashSet::with_capacity(rows.len());
    let distinct: Vec<ContentHash> = rows
        .iter()
        .copied()
        .filter(|hash| seen.insert(*hash))
        .collect();
    let cached = cache.cached(model_id, &distinct).await?;
    Ok(Coverage {
        rows,
        distinct,
        cached,
    })
}

/// Runs blocking `work` on its own thread and awaits the result.
async fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Error> {
    let (tx, rx) = oneshot::channel();
    thread::Builder::new()
        .name("repo-search-reconcile".into())
        .spawn(move || {
            let _ = tx.send(work());
        })
        .map_err(Error::Spawn)?;
    rx.await.map_err(|_| Error::WorkerGone)
}
