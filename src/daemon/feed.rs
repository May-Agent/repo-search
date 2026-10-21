//! The per-worktree feeder: turns each freshness barrier's outcome into the
//! worktree's next scheduler `Enqueue`, in barrier order, off the owner.
//! Token counting and a full resync run here, so a query never waits on
//! them.

use std::path::PathBuf;
use std::sync::Arc;

use repo_search_core::TokenCounter;
use repo_search_core::qos::performance_cores;
use repo_search_core::reconcile::{self, PendingChunk};
use repo_search_core::store::{Index, VectorCache};
use tokio::sync::{mpsc, oneshot};

use super::Backend;
use super::scheduler::{
    Enqueue, IndexClass, Interest, PendingItem, RepoId, SchedulerHandle, Stage1Done, WorktreeId,
};
use super::writer::WriterHandle;

/// Feeds the owner may queue ahead of the one in progress; the owner's
/// send waits only beyond that.
const INBOX: usize = 8;

/// What a barrier hands the feeder.
pub enum Feed {
    /// Rebuild the worktree's whole pending set in `options`' scope with
    /// `reconcile::missing` and replace the scheduler's interest with it.
    Resync {
        /// The index the set is read from: the owner's current handle.
        index: Index,
        /// The scope.
        options: reconcile::Options,
        /// Texts the pass already holds, so `missing` need not re-read them.
        known: Vec<PendingChunk>,
    },
    /// One pass's changes, in the scope of the last `Resync`.
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
pub struct FeederConfig<B: Backend> {
    /// The scheduler's worktree id.
    pub worktree: WorktreeId,
    /// The repository the worktree lives in.
    pub repo: RepoId,
    /// The worktree root `missing` reads under.
    pub root: PathBuf,
    /// The model-id vectors are stored under.
    pub model_id: String,
    /// The repository's shared vector cache.
    pub cache: VectorCache,
    /// The repository's vector writer, handed to every `Enqueue`.
    pub writer: WriterHandle,
    /// The model's token counter.
    pub counter: Arc<B::Counter>,
    /// The embedding scheduler.
    pub sched: SchedulerHandle,
    /// Where stage-1 completion for `worktree` is announced.
    pub stage1_done: mpsc::Sender<Stage1Done>,
}

/// Spawns the feeder; it exits once the returned sender is dropped and its
/// inbox is drained.
pub fn spawn<B: Backend>(config: FeederConfig<B>) -> mpsc::Sender<FeedMsg> {
    let FeederConfig {
        worktree,
        repo,
        root,
        model_id,
        cache,
        writer,
        counter,
        sched,
        stage1_done,
    } = config;
    let (tx, mut rx) = mpsc::channel(INBOX);
    tokio::spawn(async move {
        // The last `Resync`'s index handle and scope; a `Change` applies in
        // that scope.
        let mut scope: Option<(Index, reconcile::Options)> = None;
        // A feed failed, so the scheduler's interest no longer matches what
        // the barriers reported: the next `Change` is promoted to a resync.
        let mut stale = false;
        while let Some(FeedMsg { feed, class, ack }) = rx.recv().await {
            let feed = match feed {
                Feed::Change(changes) if stale => {
                    let (index, options) = scope.clone().expect("a Change follows a Resync");
                    Feed::Resync {
                        index,
                        options,
                        known: changes.added,
                    }
                }
                feed => feed,
            };
            let interest: Result<Interest, String> = match feed {
                Feed::Resync {
                    index,
                    options,
                    known,
                } => {
                    scope = Some((index.clone(), options));
                    match reconcile::missing(&index, &cache, &model_id, &root, options, known).await
                    {
                        Ok(pending) => count_pending::<B>(pending, counter.clone())
                            .await
                            .map(Interest::Replace),
                        Err(e) => Err(e.to_string()),
                    }
                }
                Feed::Change(reconcile::Changes { added, dropped }) => {
                    count_pending::<B>(added, counter.clone())
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
            sched.barrier(worktree, false).await;
        }
    });
    tx
}

/// Counts each pending chunk's document tokens on a scoped worker pool of
/// `performance_cores()` threads, off the executor.
async fn count_pending<B: Backend>(
    pending: Vec<PendingChunk>,
    counter: Arc<B::Counter>,
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
