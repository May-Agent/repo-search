//! The reconcile pass: brings the index in line with the worktree and reports
//! which chunks still need a vector.
//!
//! One pass walks the worktree, stats every supported file and diffs it
//! against the stored file table. Only files whose size, mtime or inode moved
//! are read; of those, files whose bytes hash the same as before just get a
//! fresh fingerprint, and the rest are re-chunked. All changes land in one
//! transaction. A first index is the same pass over an empty table.

mod scan;

use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use std::path::Path;
use std::thread;

use tokio::sync::oneshot;

use crate::store::{self, Applied, ContentHash, Index, VectorCache};

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
    /// `.ignore` files would hide (`--include-ignored`). Hidden files and
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The file's content is not valid UTF-8.
    NotUtf8,
    /// The file's path is not valid UTF-8; [`Skipped::path`] is lossy.
    NotUtf8Path,
    /// A configuration file that does not parse in its format.
    Malformed,
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
    /// Files dropped: deleted, newly ignored, or skipped.
    pub files_removed: u64,
    /// Files whose stat moved but whose bytes did not; only the fingerprint
    /// was rewritten.
    pub files_restamped: u64,
    /// Chunk rows inserted.
    pub chunks_added: u64,
    /// Chunk rows deleted, including those of re-chunked files.
    pub chunks_removed: u64,
    /// Chunk rows in the index after the pass.
    pub chunks_total: u64,
    /// Chunk rows whose vector is in the cache for the model.
    pub chunks_with_vector: u64,
    /// Walked files that could not be indexed.
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

/// The outcome of a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconciled {
    /// What the pass did.
    pub report: Report,
    /// One entry per distinct missing vector, in index order. The caller
    /// embeds them and stores the vectors in the cache.
    pub pending: Vec<PendingChunk>,
}

/// Brings `index` in line with the worktree at `worktree`, then lists the
/// chunks that have no vector cached for `model_id`.
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
    worktree: &Path,
    options: Options,
) -> Result<Reconciled, Error> {
    let stored: HashMap<_, _> = index
        .file_states()
        .await?
        .into_iter()
        .map(|state| (state.path.clone(), state))
        .collect();
    let root = worktree.to_path_buf();
    let plan = run_blocking(move || scan::plan(&root, options.include_ignored, stored)).await??;

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
        "reconcile applied"
    );

    let rows = index.chunk_hashes().await?;
    let distinct: Vec<ContentHash> = {
        let mut seen = HashSet::with_capacity(rows.len());
        rows.iter()
            .copied()
            .filter(|hash| seen.insert(*hash))
            .collect()
    };
    let cached = cache.cached(model_id, &distinct).await?;
    let chunks_with_vector = rows.iter().filter(|h| cached.contains(h)).count() as u64;

    // Missing hashes, distinct, in index order.
    let mut seen = HashSet::new();
    let missing: Vec<ContentHash> = rows
        .iter()
        .copied()
        .filter(|hash| !cached.contains(hash) && seen.insert(*hash))
        .collect();
    let mut texts = plan.texts;
    // Chunks of files this pass did not re-chunk: rebuilt from the stored
    // byte ranges, no parsing.
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

    let report = Report {
        files_scanned: plan.files_scanned,
        files_added: plan.files_added,
        files_modified: plan.files_modified,
        files_removed: plan.files_removed,
        files_restamped: plan.files_restamped,
        chunks_added: applied.chunks_added,
        chunks_removed: applied.chunks_removed,
        chunks_total: rows.len() as u64,
        chunks_with_vector,
        skipped: plan.skipped,
    };
    Ok(Reconciled { report, pending })
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
