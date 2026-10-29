//! The reconcile pass: brings the index in line with the worktree and reports
//! how the set of chunks needing a vector changed; [`Reconciler::missing`]
//! lists that set whole.
//!
//! One pass walks the worktree, stats every supported file and diffs it
//! against the stored file table. Only files whose size, mtime or inode moved
//! are read; of those, files whose bytes hash the same as before just get a
//! fresh fingerprint, and the rest are re-chunked. A file that fails to chunk
//! is recorded with its fingerprint and error and then treated the same way:
//! left unread while its stat holds, restamped when only the stat moved, and
//! tried again when its content changed. All changes land in one transaction.
//! A first index is the same pass over an empty table.
//!
//! A [`Reconciler`] keeps [`Books`] between passes: every stored row's hash
//! and side, and which hashes have a vector. A pass updates them from its
//! own changes and from the vectors the cache stored since the last pass, so
//! a pass that changed nothing reads no chunk row and looks up no hash. The
//! same books answer [`Reconciler::counts`] and [`Reconciler::missing`]
//! between passes, so neither reads every row or looks every hash up. Each
//! pass reports how the set of `(hash, side)` pairs changed ([`Presence`]),
//! which is what a retriever's vector indexes follow.

mod books;
mod scan;

pub use books::Books;
#[cfg(all(test, feature = "lang-rust"))]
pub(crate) use scan::chunk_with;

use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use tokio::sync::oneshot;

use crate::TokenCounter;
use crate::store::{
    self, Applied, ContentHash, Index, Side, StoredChunk, UnchunkableReason, VectorCache,
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
    /// dot-directories are indexed either way; `.git/` never is.
    pub include_ignored: bool,
}

/// A chunk whose vector is missing from the cache, with the text to embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChunk {
    /// Worktree-relative path of the file.
    pub path: String,
    /// Embed the document `crate::model::DOCUMENT` formats for this title
    /// and `text`.
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

/// What a pass read from the stores beyond the walk and the fingerprint
/// table: the measure of its cost that does not depend on the machine. A
/// pass whose reconciler holds books and that changed nothing reads no
/// chunk row and looks up no hash.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reads {
    /// Chunk rows read from the index: loading the books, and the stored
    /// chunks of restamped files.
    pub chunk_rows: u64,
    /// Hashes looked up one by one in the vector cache.
    pub cache_probes: u64,
    /// Entries read from the cache's slot log: the vectors stored since the
    /// books last read it.
    pub cache_slots: u64,
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
    /// Chunk rows on the docs side visible to a search with the pass's
    /// `include_ignored`.
    pub doc_chunks: u64,
    /// The index was built by another chunker (or token counter) and was
    /// emptied, so every file was re-chunked this pass.
    pub rechunked: bool,
    /// Every walked file that is not indexed: the unchunkable files known to
    /// the index, recorded this pass or before, and paths that are not UTF-8.
    pub skipped: Vec<Skipped>,
    /// What the pass read to settle coverage and its changes.
    pub reads: Reads,
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
/// Applied in pass order on top of [`Reconciler::missing`] taken after
/// some earlier pass of the same scope, successive passes' changes keep that
/// set current: `added` holds the chunks the pass put in scope or touched that
/// have no cached vector, `dropped` the hashes it took out of scope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Changes {
    /// Chunks of files the pass re-chunked or restamped that have no cached
    /// vector, one per distinct hash, ordered by path then first line. A
    /// restamped file's chunks are offered again so a chunk
    /// [`Missing::pending`] could not rebuild while the file was mid-change
    /// is never lost.
    pub added: Vec<PendingChunk>,
    /// Hashes some row of which the pass deleted and of which no row
    /// visible in the scope remains; distinct, in deletion order.
    pub dropped: Vec<ContentHash>,
}

/// How a pass changed the `(hash, side)` pairs the index holds: a content
/// hash with a side some stored row of that side holds, ignored files
/// included.
///
/// Applied in pass order on top of [`Books::pairs`] taken after some earlier
/// pass of the same reconciler, successive passes' presences keep that set
/// current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Presence {
    /// The reconciler read its books whole this pass (its first pass, the
    /// pass after a failed one, a re-chunk, or after [`Reconciler::counts`]
    /// loaded them): every pair the index holds.
    Whole(Vec<(ContentHash, Side)>),
    /// The pairs the pass added and removed; each pair is in at most one
    /// list, and both are empty when the pass changed no row.
    Changed {
        /// Pairs the index did not hold before the pass and holds now.
        appeared: Vec<(ContentHash, Side)>,
        /// Pairs the index held before the pass and holds no longer.
        vanished: Vec<(ContentHash, Side)>,
    },
}

/// The outcome of a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconciled {
    /// What the pass did.
    pub report: Report,
    /// The pass's changes; `None` when it re-chunked an emptied index
    /// (`report.rechunked`), whose deleted hashes nothing recorded: a caller
    /// keeping a pending set rebuilds it with [`Reconciler::missing`].
    pub changes: Option<Changes>,
    /// How the pass changed the pairs the index holds.
    pub presence: Presence,
}

/// A scope's coverage between passes, read from a reconciler's books.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// Chunk rows visible in the scope.
    pub chunks_total: u64,
    /// Chunk rows visible in the scope whose vector is in the cache.
    pub chunks_with_vector: u64,
    /// What answering read.
    pub reads: Reads,
}

/// Reconciles one index against its worktree, keeping the [`Books`] that
/// let a pass's reads follow what it changed.
///
/// The books are loaded by the first pass, by the pass after a failed one,
/// and by the pass that re-chunks an emptied index. Every change to the
/// index's rows must go through this reconciler: rows changed any other way
/// leave its books wrong, so whoever changes them builds a new one.
pub struct Reconciler {
    index: Index,
    cache: VectorCache,
    model_id: String,
    worktree: PathBuf,
    books: Option<Books>,
    /// The books were read whole since a pass last reported its presence,
    /// so the next pass reports [`Presence::Whole`].
    whole: bool,
}

impl Reconciler {
    /// A reconciler of `index` against the worktree at `worktree`, under
    /// `model_id` in `cache`; it holds no books until its first pass.
    pub fn new(
        index: Index,
        cache: VectorCache,
        model_id: impl Into<String>,
        worktree: impl Into<PathBuf>,
    ) -> Self {
        Self {
            index,
            cache,
            model_id: model_id.into(),
            worktree: worktree.into(),
            books: None,
            whole: true,
        }
    }

    /// The books after the last pass that succeeded; `None` before the
    /// first pass and after a failed one.
    #[must_use]
    pub fn books(&self) -> Option<&Books> {
        self.books.as_ref()
    }

    /// Brings the index in line with the worktree, then reports how the
    /// pass changed the chunks its scope needs vectors for;
    /// [`Reconciler::missing`] lists that set whole.
    ///
    /// `chunker` identifies the chunker, normally
    /// [`crate::CHUNKER_FINGERPRINT`]. An index stored under another
    /// `(chunker, model_id)` is emptied first and the pass re-chunks every
    /// file; cached vectors are reused by content hash.
    ///
    /// `tokens` is the model's token counter; the chunker holds every
    /// chunk's embedded document to the cap it counts.
    ///
    /// Every change to files, chunks and FTS commits in one transaction; a
    /// pass that finds nothing changed performs no write.
    ///
    /// # Errors
    ///
    /// When the walk or a read fails, or the index or cache rejects a
    /// request. A failed pass leaves the index as it was and drops the
    /// books; the next pass loads them again.
    pub async fn reconcile(
        &mut self,
        chunker: &str,
        options: Options,
        tokens: Arc<dyn TokenCounter>,
    ) -> Result<Reconciled, Error> {
        let books = self.books.take();
        let include_ignored = options.include_ignored;
        let rechunked = self.index.align_chunker(chunker, &self.model_id).await?;
        let (states, unchunkables) =
            tokio::try_join!(self.index.file_states(), self.index.unchunkables())?;
        let stored: HashMap<_, _> = states
            .into_iter()
            .map(|state| (state.path.clone(), state))
            .collect();
        let recorded: HashMap<_, _> = unchunkables
            .into_iter()
            .map(|record| (record.state.path.clone(), record))
            .collect();
        let root = self.worktree.clone();
        // The books held before the pass describe the index unless the
        // alignment just emptied it; then, and without books, they are read
        // whole while the worktree is walked.
        let books = async {
            match books {
                Some(books) if !rechunked => Ok((books, Reads::default(), false)),
                _ => Books::load(&self.index, &self.cache, &self.model_id)
                    .await
                    .map(|(books, reads)| (books, reads, true)),
            }
        };
        let (plan, books) = tokio::join!(
            run_blocking(move || {
                scan::plan(&root, include_ignored, stored, recorded, &*tokens)
            }),
            books,
        );
        let plan = plan??;
        let (mut books, mut reads, loaded) = books?;
        self.whole |= loaded;

        let mut fresh = Vec::new();
        for change in &plan.changes {
            books.record(change, &mut fresh);
        }
        let applied = if plan.changes.is_empty() {
            Applied::default()
        } else {
            self.index.apply(plan.changes).await?
        };

        // Vectors stored since the books last looked, and whether the
        // hashes new to the books have one; neither read waits on the other.
        let probe = async {
            if fresh.is_empty() {
                Ok(HashSet::new())
            } else {
                self.cache.cached(&self.model_id, &fresh).await
            }
        };
        let (slots, found) =
            tokio::try_join!(self.cache.since(&self.model_id, books.high_water()), probe)?;
        reads.cache_slots += slots.len() as u64;
        reads.cache_probes += fresh.len() as u64;
        books.advance(&slots);
        for hash in &found {
            books.mark_cached(hash);
        }

        let changes = if rechunked {
            // The emptied index's deleted hashes were never recorded; a
            // caller keeping a pending set rebuilds it with
            // `Reconciler::missing`.
            None
        } else {
            let mut added: Vec<PendingChunk> = plan
                .texts
                .into_values()
                .filter(|chunk| !books.is_cached(&chunk.content_hash))
                .collect();
            let mut taken: HashSet<ContentHash> =
                added.iter().map(|chunk| chunk.content_hash).collect();
            if !plan.restamped.is_empty() {
                let rows = self.index.chunks_in(plan.restamped).await?;
                reads.chunk_rows += rows.len() as u64;
                let stored: Vec<StoredChunk> = rows
                    .into_iter()
                    .filter(|row| {
                        books.is_visible(&row.content_hash, include_ignored)
                            && !books.is_cached(&row.content_hash)
                            && taken.insert(row.content_hash)
                    })
                    .collect();
                if !stored.is_empty() {
                    let root = self.worktree.clone();
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
                .filter(|hash| !books.is_visible(hash, include_ignored))
                .collect();
            Some(Changes { added, dropped })
        };

        tracing::debug!(
            scanned = plan.files_scanned,
            added = plan.files_added,
            modified = plan.files_modified,
            removed = plan.files_removed,
            restamped = plan.files_restamped,
            chunked = plan.files_chunked,
            chunk_rows = reads.chunk_rows,
            cache_probes = reads.cache_probes,
            cache_slots = reads.cache_slots,
            "reconcile applied"
        );
        let report = Report {
            files_scanned: plan.files_scanned,
            files_added: plan.files_added,
            files_modified: plan.files_modified,
            files_removed: plan.files_removed,
            files_restamped: plan.files_restamped,
            files_chunked: plan.files_chunked,
            chunks_added: applied.chunks_added,
            chunks_removed: applied.chunks_removed,
            chunks_total: books.chunks_total(include_ignored),
            chunks_with_vector: books.chunks_with_vector(include_ignored),
            doc_chunks: books.doc_chunks(include_ignored),
            rechunked,
            skipped: plan.skipped,
            reads,
        };
        let presence = books.take_presence();
        let presence = if std::mem::take(&mut self.whole) {
            Presence::Whole(books.pairs())
        } else {
            presence
        };
        self.books = Some(books);
        Ok(Reconciled {
            report,
            changes,
            presence,
        })
    }

    /// The coverage of `include_ignored`'s scope now, from the books: the
    /// rows as the last pass left them, and every vector the cache stored
    /// since the books last looked. Without books (before the first pass or
    /// after a failed one) they are read whole first and kept for the next
    /// pass.
    ///
    /// # Errors
    ///
    /// When the index or the cache rejects a read. The books are dropped
    /// then; the next pass loads them again.
    pub async fn counts(&mut self, include_ignored: bool) -> Result<Counts, Error> {
        let (books, reads) = self.current().await?;
        Ok(Counts {
            chunks_total: books.chunks_total(include_ignored),
            chunks_with_vector: books.chunks_with_vector(include_ignored),
            reads,
        })
    }

    /// Every distinct hash visible in `include_ignored`'s scope that has no
    /// vector now, from the books, as [`Self::counts`] reads them: only the
    /// vectors the cache stored since the books last looked, however many
    /// chunks already have one. Without books they are read whole first and
    /// kept for the next pass. [`Missing::pending`] later drops what was
    /// stored after this listing.
    ///
    /// # Errors
    ///
    /// When the index or the cache rejects a read. The books are dropped
    /// then; the next pass loads them again.
    pub async fn missing(&mut self, include_ignored: bool) -> Result<Missing, Error> {
        let (books, reads) = self.current().await?;
        let hashes = books.uncached(include_ignored);
        let high_water = books.high_water();
        Ok(Missing {
            hashes,
            high_water,
            index: self.index.clone(),
            cache: self.cache.clone(),
            model_id: self.model_id.clone(),
            worktree: self.worktree.clone(),
            reads,
        })
    }

    /// The books brought up to the cache's slot log, loaded whole first
    /// when the reconciler holds none, with what that read.
    async fn current(&mut self) -> Result<(&Books, Reads), Error> {
        let (mut books, mut reads) = match self.books.take() {
            Some(books) => (books, Reads::default()),
            None => {
                let loaded = Books::load(&self.index, &self.cache, &self.model_id).await?;
                self.whole = true;
                loaded
            }
        };
        let slots = self.cache.since(&self.model_id, books.high_water()).await?;
        reads.cache_slots += slots.len() as u64;
        books.advance(&slots);
        Ok((self.books.insert(books), reads))
    }
}

/// A scope's chunks with no vector, listed from a reconciler's books by
/// [`Reconciler::missing`]: their hashes, the first slot of the cache's log
/// the listing did not see, and the stores and worktree their texts are
/// rebuilt from.
pub struct Missing {
    hashes: Vec<ContentHash>,
    high_water: u64,
    index: Index,
    cache: VectorCache,
    model_id: String,
    worktree: PathBuf,
    /// What listing them read.
    pub reads: Reads,
}

impl std::fmt::Debug for Missing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Missing")
            .field("worktree", &self.worktree)
            .field("hashes", &self.hashes.len())
            .field("reads", &self.reads)
            .finish_non_exhaustive()
    }
}

/// The missing chunks with their texts, from [`Missing::pending`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    /// One chunk per missing hash whose text could be had, ordered by path
    /// then first line.
    pub chunks: Vec<PendingChunk>,
    /// What settling the list and rebuilding the texts read: the vectors
    /// stored since the listing, and the stored rows of the hashes still
    /// missing that `known` did not hold.
    pub reads: Reads,
}

impl Missing {
    /// The missing hashes, distinct, in no particular order.
    #[must_use]
    pub fn hashes(&self) -> &[ContentHash] {
        &self.hashes
    }

    /// Each hash still missing with its text. A hash whose vector the
    /// cache stored after the listing is dropped: read from the slot log,
    /// so a caller that flushes its writer first rebuilds no text the flush
    /// made moot. The text is taken from `known` when it holds the hash,
    /// otherwise rebuilt from one stored row's byte ranges (each file read
    /// and validated once). A hash whose file changed or vanished since the
    /// last pass is left out; the next pass offers it again.
    ///
    /// # Errors
    ///
    /// When the index or the cache rejects a read, or a file read fails for
    /// a reason other than the file having vanished.
    pub async fn pending(mut self, known: Vec<PendingChunk>) -> Result<Pending, Error> {
        let mut reads = Reads::default();
        let slots = self.cache.since(&self.model_id, self.high_water).await?;
        reads.cache_slots = slots.len() as u64;
        if !slots.is_empty() {
            let stored: HashSet<ContentHash> = slots.iter().map(|s| s.content_hash).collect();
            self.hashes.retain(|hash| !stored.contains(hash));
        }
        let mut texts: HashMap<ContentHash, PendingChunk> =
            known.into_iter().map(|c| (c.content_hash, c)).collect();
        // Chunks `known` does not hold: rebuilt from the stored byte ranges,
        // no parsing.
        let to_load: Vec<ContentHash> = self
            .hashes
            .iter()
            .filter(|hash| !texts.contains_key(hash))
            .copied()
            .collect();
        if !to_load.is_empty() {
            let stored = self.index.chunks_by_hash(to_load).await?;
            reads.chunk_rows = stored.len() as u64;
            let root = self.worktree;
            let loaded = run_blocking(move || scan::read_chunks(&root, stored)).await??;
            texts.extend(loaded.into_iter().map(|c| (c.content_hash, c)));
        }
        // A hash with no text belongs to a file that changed or vanished
        // after the pass; the next pass handles it.
        let mut chunks: Vec<PendingChunk> = self
            .hashes
            .iter()
            .filter_map(|hash| texts.remove(hash))
            .collect();
        chunks.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then(a.lines.start().cmp(b.lines.start()))
        });
        Ok(Pending { chunks, reads })
    }
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
