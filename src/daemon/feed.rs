//! The per-worktree feeder: turns each freshness barrier's outcome into the
//! worktree's next scheduler `Enqueue`, in barrier order, off the owner.
//! Token counting and a resync's text rebuild run here, so a query never
//! waits on them. A resync's set is listed from the owner's books: by the
//! barrier, or by asking the owner when a failed feed promotes a `Change`.

use std::path::PathBuf;
use std::sync::Arc;

use repo_search_core::TokenCounter;
use repo_search_core::qos::performance_cores;
use repo_search_core::reconcile::{self, Missing, PendingChunk};
use tokio::sync::{mpsc, oneshot};

use super::counter::Tokens;
use super::coverage::Asker;
use super::scheduler::{
    Enqueue, IndexClass, Interest, PendingItem, RepoId, SchedulerHandle, Stage1Done, WorktreeId,
};
use super::writer::WriterHandle;

/// Feeds the owner may queue ahead of the one in progress; the owner's
/// send waits only beyond that.
const INBOX: usize = 8;

/// What a barrier hands the feeder.
pub enum Feed {
    /// Flush the repository's writer, then settle the worktree's whole
    /// pending set in `options`' scope, rebuild its texts and replace the
    /// scheduler's interest with it.
    Resync {
        /// The scope.
        options: reconcile::Options,
        /// The set, listed from the owner's books inside the barrier.
        missing: Missing,
        /// Texts the pass already holds, so they need not be rebuilt.
        known: Vec<PendingChunk>,
    },
    /// One pass's changes, in the scope of the last `Resync`.
    Change(reconcile::Changes),
}

/// What the feeder does with one feed.
enum Work {
    /// Replace the scheduler's interest with the scope's whole set.
    Resync {
        options: reconcile::Options,
        /// The set as the barrier listed it; `None` for a `Change` a failed
        /// feed promoted, whose set the owner's books list on asking.
        listed: Option<Missing>,
        known: Vec<PendingChunk>,
    },
    /// Apply one pass's changes.
    Change(reconcile::Changes),
}

/// One barrier's feed.
pub struct FeedMsg {
    /// What to feed.
    pub feed: Feed,
    /// `Stage1` while the owner's phase is first-index.
    pub class: IndexClass,
    /// For `index --wait`: `Ok` once the `Enqueue` is in the scheduler's
    /// inbox, else why there is none.
    pub ack: Option<oneshot::Sender<Result<(), String>>>,
}

/// What a feeder is built from.
pub struct FeederConfig<T: Tokens> {
    /// The scheduler's worktree id.
    pub worktree: WorktreeId,
    /// The repository the worktree lives in.
    pub repo: RepoId,
    /// The worktree root, for diagnostics.
    pub root: PathBuf,
    /// Asks the owner's books for the set of a promoted resync.
    pub books: Asker,
    /// The repository's vector writer, handed to every `Enqueue`.
    pub writer: WriterHandle,
    /// The model's token counter.
    pub counter: Arc<T::Counter>,
    /// The embedding scheduler.
    pub sched: SchedulerHandle,
    /// Where stage-1 completion for `worktree` is announced.
    pub stage1_done: mpsc::Sender<Stage1Done>,
}

/// Spawns the feeder; it exits once the returned sender is dropped and its
/// inbox is drained.
pub fn spawn<T: Tokens>(config: FeederConfig<T>) -> mpsc::Sender<FeedMsg> {
    let FeederConfig {
        worktree,
        repo,
        root,
        books,
        writer,
        counter,
        sched,
        stage1_done,
    } = config;
    let (tx, mut rx) = mpsc::channel(INBOX);
    tokio::spawn(async move {
        // The last `Resync`'s scope; a `Change` applies in that scope.
        let mut scope: Option<reconcile::Options> = None;
        // A feed failed, so the scheduler's interest no longer matches what
        // the barriers reported: the next `Change` is promoted to a resync.
        let mut stale = false;
        while let Some(FeedMsg { feed, class, ack }) = rx.recv().await {
            let work = match feed {
                Feed::Change(changes) if stale => Work::Resync {
                    options: scope.expect("a Change follows a Resync"),
                    listed: None,
                    known: changes.added,
                },
                Feed::Resync {
                    options,
                    missing,
                    known,
                } => Work::Resync {
                    options,
                    listed: Some(missing),
                    known,
                },
                Feed::Change(changes) => Work::Change(changes),
            };
            let interest: Result<Interest, String> = match work {
                Work::Resync {
                    options,
                    listed,
                    known,
                } => {
                    scope = Some(options);
                    // Vectors the writer still buffers would read as
                    // missing: the scheduler would drop them from the
                    // `Replace`, but only after their text was rebuilt and
                    // counted for nothing. Settling the set reads the
                    // cache's slot log after the flush, inside this feed's
                    // barrier; a promoted `Change` has no listed set, so the
                    // owner's books list it now.
                    let pending = async {
                        writer
                            .flush()
                            .await
                            .map_err(|e| format!("vector writer flush: {e}"))?;
                        let missing = match listed {
                            Some(missing) => missing,
                            None => books.missing(options).await.map_err(|e| e.to_string())?,
                        };
                        let listed = missing.reads;
                        let pending = missing.pending(known).await.map_err(|e| e.to_string())?;
                        tracing::debug!(root = %root.display(),
                            listed_chunk_rows = listed.chunk_rows,
                            listed_cache_probes = listed.cache_probes,
                            listed_cache_slots = listed.cache_slots,
                            settled_cache_slots = pending.reads.cache_slots,
                            rebuilt_chunk_rows = pending.reads.chunk_rows,
                            pending = pending.chunks.len(), "resync settled");
                        Ok::<_, String>(pending.chunks)
                    }
                    .await;
                    match pending {
                        Ok(pending) => count_pending::<T>(pending, counter.clone())
                            .await
                            .map(Interest::Replace),
                        Err(message) => Err(message),
                    }
                }
                Work::Change(reconcile::Changes { added, dropped }) => {
                    count_pending::<T>(added, counter.clone())
                        .await
                        .map(|added| Interest::Change { added, dropped })
                }
            };
            let outcome = match interest {
                Ok(interest) => {
                    stale = false;
                    sched
                        .enqueue(Enqueue {
                            worktree,
                            repo,
                            class,
                            interest,
                            writer: writer.clone(),
                            stage1_done: stage1_done.clone(),
                        })
                        .await
                        .map_err(|_| "the embedding scheduler is gone".to_owned())
                }
                Err(message) => {
                    tracing::error!(%message, root = %root.display(),
                        "feeding the scheduler failed; the next barrier resyncs");
                    stale = true;
                    Err(message)
                }
            };
            if let Some(ack) = ack {
                let _ = ack.send(outcome);
            }
            // Only after this feed's `Enqueue`: until then the scheduler
            // keeps the hashes committed since the barrier began.
            sched.barrier(repo, false).await;
        }
    });
    tx
}

/// Counts each pending chunk's document tokens on a scoped worker pool of
/// `performance_cores()` threads, off the executor.
async fn count_pending<T: Tokens>(
    pending: Vec<PendingChunk>,
    counter: Arc<T::Counter>,
) -> Result<Vec<PendingItem>, String> {
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    let threads = performance_cores()
        .map_err(|e| e.to_string())?
        .min(pending.len());
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let size = pending.len().div_ceil(threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = pending
                .chunks(size)
                .map(|slice| {
                    let counter = &counter;
                    s.spawn(move || {
                        slice
                            .iter()
                            .map(|chunk| PendingItem {
                                tokens: u32::try_from(
                                    counter.document_tokens(&chunk.title, &chunk.text),
                                )
                                .expect("a chunk is capped at 512 tokens"),
                                chunk: chunk.clone(),
                            })
                            .collect::<Vec<PendingItem>>()
                    })
                })
                .collect();
            let items: Vec<PendingItem> = handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("counting thread"))
                .collect();
            let _ = tx.send(items);
        });
    });
    Ok(rx.await.expect("the counting pool answers"))
}
