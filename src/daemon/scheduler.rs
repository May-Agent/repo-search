//! The scheduler actor: the sole owner of all embedding work.
//!
//! One async task holds the query queue (FIFO, always first), the two-ended
//! stage-1 queue (the ANE pops the shortest item, the GPU the longest in
//! same-bucket batches) and the background queue. Indexing items are
//! de-duplicated by `(repo, hash)` so worktrees of one repository share
//! vectors. The ANE and GPU threads pull their next job over [`SchedulerMsg`];
//! finished vectors go to the per-repository [`writer`]. A worktree's
//! interest is replaced whole on a fresh or rescoped pass and edited by each
//! later pass's changes, so a pass that changed nothing costs the scheduler
//! nothing. An `Enqueue` is applied inside the freshness barrier it came
//! from, and chunks that got a vector after that barrier began are dropped
//! from it, so a late update never queues committed work again.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use repo_search_core::qos::{self, QosClass};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::store::ContentHash;
use repo_search_core::{QueryTask, UnitVector};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{debug, error};

use super::writer::WriterHandle;
use super::{Backend, Error, Headroom, gpu};

/// Inbox capacity of the scheduler and every component channel.
const INBOX: usize = 64;

/// Stage 1 embeds only chunks up to this many tokens; larger chunks wait for
/// stage 2 at background QoS.
pub const STAGE1_MAX_TOKENS: usize = 256;

/// The largest batch the GPU thread is handed.
const GPU_BATCH: usize = 8;

/// A worktree the scheduler is tracking, as the registry numbered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorktreeId(pub u64);

/// A repository (a git common dir), as the registry numbered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepoId(pub u64);

/// One queued or in-flight indexing item.
pub type ItemId = u64;

/// Which queue an indexing item runs from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexClass {
    /// First-index work: stage-1 priority on the ANE and GPU.
    Stage1,
    /// Everything else: ANE only, at background QoS.
    Background,
}

/// A chunk handed to the scheduler with its token count.
#[derive(Debug)]
pub struct PendingItem {
    /// The chunk to embed.
    pub chunk: PendingChunk,
    /// `TokenCounter::document_tokens` of the chunk, counted by the caller.
    pub tokens: u32,
}

/// Sent to a worktree's `stage1_done` channel once, when every stage-1 item
/// it is interested in is embedded and flushed to the vector cache.
#[derive(Debug)]
pub struct Stage1Done;

/// A worktree's indexing work. A worktree's `Enqueue`s are applied in the
/// order sent, its first one is an `Interest::Replace`, and each is sent
/// between its barrier's `Barrier { running: true }` and
/// `Barrier { running: false }`.
#[derive(Debug)]
pub struct Enqueue {
    /// The worktree these items belong to.
    pub worktree: WorktreeId,
    /// The repository the worktree lives in.
    pub repo: RepoId,
    /// `Stage1` while the worktree's index is in its first pass.
    pub class: IndexClass,
    /// What the worktree now wants embedded.
    pub interest: Interest,
    /// This repo's vector writer; the embedding threads hand it every
    /// finished vector directly, and stage-1 completion is reported only
    /// after it has flushed.
    pub writer: WriterHandle,
    /// Where stage-1 completion for `worktree` is announced.
    pub stage1_done: mpsc::Sender<Stage1Done>,
}

/// How an `Enqueue` changes the worktree's interest.
///
/// Each `Change` is computed from the index right after its own pass:
/// `added` lists the hashes present in scope and uncached at that moment,
/// `dropped` the hashes absent from scope at that moment. Feeds are applied
/// in pass order, so for any hash the last feed that names it reflects the
/// latest pass that changed its presence, and a `Replace` computed later
/// than some queued `Change`s is corrected by them. A hash `missing` left
/// out because its file changed after the pass is offered again by the next
/// pass.
///
/// Both lists are read from the cache inside the barrier the `Enqueue`
/// belongs to. A hash that got a vector between that read and the
/// `Enqueue`, by any worktree of the repository or by a search, is dropped
/// from `Replace` and `added` when the `Enqueue` is applied (see
/// `OpenBarriers`).
#[derive(Debug)]
pub enum Interest {
    /// Every chunk of the worktree's scope that lacks a vector: replaces
    /// what the scheduler holds for it.
    Replace(Vec<PendingItem>),
    /// One reconcile pass's changes, on top of the interest the previous
    /// `Enqueue` left.
    Change {
        /// Chunks to want; already-wanted ones are a no-op.
        added: Vec<PendingItem>,
        /// Hashes to stop wanting; unknown or finished ones are a no-op.
        dropped: Vec<ContentHash>,
    },
}

/// A query's embedding request; query work preempts indexing on the ANE.
#[derive(Debug)]
pub enum QueryJob {
    /// One query embedding for a search.
    Query {
        /// The worktree asking; stats accumulate under it.
        worktree: WorktreeId,
        /// How the query is phrased for the model.
        task: QueryTask,
        /// The query text.
        text: String,
        /// When the job was handed to the scheduler.
        enqueued: Instant,
        /// `dispatched_indexing` at enqueue time; the scheduler assigns it.
        stamp: u64,
        /// The answer goes here.
        reply: oneshot::Sender<Result<UnitVector, EmbedError>>,
    },
    /// On-the-spot document embeddings for a search, in order.
    Documents {
        /// The worktree asking.
        worktree: WorktreeId,
        /// The chunks to embed.
        chunks: Vec<PendingChunk>,
        /// When the job was handed to the scheduler.
        enqueued: Instant,
        /// `dispatched_indexing` at enqueue time; the scheduler assigns it.
        stamp: u64,
        /// The vectors, one per chunk, go here.
        reply: oneshot::Sender<Result<Vec<UnitVector>, EmbedError>>,
    },
}

impl QueryJob {
    /// The job's worktree.
    pub(crate) fn worktree(&self) -> WorktreeId {
        match self {
            Self::Query { worktree, .. } | Self::Documents { worktree, .. } => *worktree,
        }
    }

    /// The job's enqueue instant.
    pub(crate) fn enqueued(&self) -> Instant {
        match self {
            Self::Query { enqueued, .. } | Self::Documents { enqueued, .. } => *enqueued,
        }
    }

    /// Whether the awaiting end hung up.
    fn closed(&self) -> bool {
        match self {
            Self::Query { reply, .. } => reply.is_closed(),
            Self::Documents { reply, .. } => reply.is_closed(),
        }
    }

    /// Answers the job with `error` if the reply channel is still open.
    fn fail(self, error: EmbedError) {
        match self {
            Self::Query { reply, .. } => {
                let _ = reply.send(Err(error));
            }
            Self::Documents { reply, .. } => {
                let _ = reply.send(Err(error));
            }
        }
    }
}

/// The query path's failure, as the owner sees it over `EmbedService`.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The model failed; the daemon is shutting down.
    #[error("model: {0}")]
    Model(String),
    /// The scheduler went away before answering.
    #[error("the embedding scheduler is gone")]
    SchedulerGone,
}

/// A job the scheduler hands the ANE thread.
#[derive(Debug)]
pub enum AneJob {
    /// A query job, preempting indexing; `overtaken` counts indexing items
    /// dispatched after the job was stamped and must be zero.
    Query {
        /// The job to run.
        job: QueryJob,
        /// Indexing items handed to the ANE after the job was enqueued.
        overtaken: u32,
    },
    /// One indexing chunk, at the QoS of the queue it came from.
    Index {
        /// The item, echoed back in `AneDone`.
        item: ItemId,
        /// The chunk to embed.
        chunk: PendingChunk,
        /// The thread's QoS for this job.
        qos: QosClass,
        /// The item's repo writer; the vector goes to it before `AneDone`.
        writer: WriterHandle,
    },
    /// Drop the embedder and exit.
    Stop,
}

/// A job the scheduler hands the GPU thread. There is no query variant: the
/// GPU never serves a query. Each batch item carries its repo writer; the
/// vectors go to it before `GpuDone`.
#[derive(Debug)]
pub enum GpuJob {
    /// One batch prediction, same bucket, up to [`GPU_BATCH`] documents.
    Batch(Vec<(ItemId, PendingChunk, WriterHandle)>),
    /// Drop the embedder and exit.
    Stop,
}

/// Query wait statistics a worktree accumulated since the last `TakeStats`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AneStats {
    /// The longest wait a query job spent behind an in-flight index chunk.
    pub wait: Duration,
    /// Sum of `overtaken` across the worktree's query jobs; must be zero.
    pub overtaken: u32,
}

/// Queue counts for a worktree's `Status`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// Items in the stage-1 queue.
    pub queued_stage1: u64,
    /// Items in the background queue.
    pub queued_background: u64,
    /// Items handed to an embedder, not yet done.
    pub in_flight: u64,
    /// Stage-1 completion already ran for this worktree; its `Stage1Done`
    /// is on its way to the owner once the writer flush lands.
    pub stage1_done: bool,
}

/// The GPU instance's state, as `Status` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuState {
    /// No GPU instance exists; `blocked` names why, when a check ran.
    Absent {
        /// Why the GPU cannot be opened, when known.
        blocked: Option<String>,
    },
    /// The GPU thread is opening its instance (a multi-second cold start).
    Loading,
    /// Embedding stage-1 index work.
    Running,
}

/// The scheduler's inbox. Every component the pipeline owns talks to the
/// scheduler through this one channel.
#[derive(Debug)]
pub enum SchedulerMsg {
    /// A worktree's indexing work: a `Replace` or a `Change` of its
    /// interest.
    Enqueue(Box<Enqueue>),
    /// A query job.
    Query(Box<QueryJob>),
    /// The ANE thread's pull for its next job.
    AnePull {
        /// The pulling thread, for QoS overrides.
        thread: qos::ThreadId,
        /// Where the next job is sent.
        reply: oneshot::Sender<AneJob>,
    },
    /// An indexing item the ANE finished; its vector is already with the
    /// item's repo writer.
    AneDone {
        /// The completed item.
        item: ItemId,
    },
    /// A query job the ANE finished.
    QueryDone {
        /// The job's worktree.
        worktree: WorktreeId,
        /// How long the job waited behind the index chunk in flight when it
        /// was enqueued.
        wait: Duration,
        /// The `overtaken` the job was dispatched with.
        overtaken: u32,
    },
    /// The GPU thread's pull for its next job.
    GpuPull {
        /// Where the next job is sent.
        reply: oneshot::Sender<GpuJob>,
    },
    /// The GPU thread opened its embedder.
    GpuStarted,
    /// The GPU thread exited: it pulled `Stop`, or it failed fatally.
    GpuGone,
    /// `open_gpu` returned `Blocked`; the thread exited. No respawn until
    /// headroom changes.
    GpuBlocked,
    /// A batch the GPU finished, in order; the vectors are already with each
    /// item's repo writer.
    GpuDone {
        /// The completed items.
        items: Vec<ItemId>,
    },
    /// A `Documents` job's vectors are committed to `repo`'s cache: queued
    /// items with these hashes are redundant and leave the queues; in-flight
    /// ones finish normally.
    DocumentsStored {
        /// The repository whose cache holds the vectors.
        repo: RepoId,
        /// The committed hashes.
        hashes: Vec<ContentHash>,
    },
    /// An embedder call or an `open_gpu` failed: fatal.
    EmbedderFailed {
        /// What failed, for the log and waiting queries.
        error: String,
    },
    /// The owner takes its accumulated query stats.
    TakeStats {
        /// The worktree asking.
        worktree: WorktreeId,
        /// Where the stats go.
        reply: oneshot::Sender<AneStats>,
    },
    /// Queue counts for a worktree's `Status`.
    CountsFor {
        /// The worktree asking.
        worktree: WorktreeId,
        /// Where the counts go.
        reply: oneshot::Sender<Counts>,
    },
    /// A freshness barrier of a worktree of `repo` is running (`running`)
    /// or ended. Barrier work keeps the daemon busy for the idle exit, and
    /// while any barrier of `repo` runs, the scheduler remembers the hashes
    /// of `repo` that get a vector (see [`OpenBarriers`]). A barrier's
    /// `Enqueue`, when it has one, is sent between its two reports.
    Barrier {
        /// The repository of the worktree reporting.
        repo: RepoId,
        /// Whether the barrier is running.
        running: bool,
    },
    /// An `index --wait` asks for the worktree's counts pushed on change;
    /// the scheduler builds the watch at handling — behind the worktree's
    /// `Enqueue` in this inbox — seeded with the live counts.
    WatchCounts {
        /// The worktree asking.
        worktree: WorktreeId,
        /// Where the watch receiver goes.
        reply: oneshot::Sender<watch::Receiver<Counts>>,
    },
    /// The GPU instance's state, for `Status`.
    GpuStatus {
        /// Where the state goes.
        reply: oneshot::Sender<GpuState>,
    },
    /// Stop dispatching; pulls are answered `Stop`.
    Stop,
}

/// The scheduler's handle; clones share the one inbox.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::Sender<SchedulerMsg>,
}

impl SchedulerHandle {
    /// The raw inbox sender, for the embedding threads and components that
    /// talk [`SchedulerMsg`].
    pub fn tx(&self) -> mpsc::Sender<SchedulerMsg> {
        self.tx.clone()
    }

    /// Enqueues a worktree's indexing work.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn enqueue(&self, enqueue: Enqueue) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::Enqueue(Box::new(enqueue)))
            .await
            .map_err(|_| Closed)
    }

    /// Sends a query job.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn query(&self, job: QueryJob) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::Query(Box::new(job)))
            .await
            .map_err(|_| Closed)
    }

    /// The worktree's accumulated query stats, taken (reset) by the call.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn take_stats(&self, worktree: WorktreeId) -> Result<AneStats, Closed> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SchedulerMsg::TakeStats { worktree, reply })
            .await
            .map_err(|_| Closed)?;
        rx.await.map_err(|_| Closed)
    }

    /// The worktree's queue counts, for `Status`.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn counts_for(&self, worktree: WorktreeId) -> Result<Counts, Closed> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SchedulerMsg::CountsFor { worktree, reply })
            .await
            .map_err(|_| Closed)?;
        rx.await.map_err(|_| Closed)
    }

    /// Reports a `Documents` job's vectors committed to `repo`'s cache.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn documents_stored(
        &self,
        repo: RepoId,
        hashes: Vec<ContentHash>,
    ) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::DocumentsStored { repo, hashes })
            .await
            .map_err(|_| Closed)
    }

    /// Reports a freshness barrier of a worktree of `repo` running or
    /// ended. `running: true` must be sent before the barrier's first cache
    /// read, and `running: false` after its `Enqueue`.
    pub async fn barrier(&self, repo: RepoId, running: bool) {
        let _ = self.tx.send(SchedulerMsg::Barrier { repo, running }).await;
    }

    /// The worktree's counts, pushed on change until the receiver is
    /// dropped; for `index --wait`.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn watch_counts(
        &self,
        worktree: WorktreeId,
    ) -> Result<watch::Receiver<Counts>, Closed> {
        // The scheduler creates the watch at `WatchCounts` handling —
        // FIFO-behind the worktree's `Enqueue` — seeded with the live
        // counts, and replies with the receiver.
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SchedulerMsg::WatchCounts { worktree, reply })
            .await
            .map_err(|_| Closed)?;
        rx.await.map_err(|_| Closed)
    }

    /// The GPU instance's state, for `Status`.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn gpu_status(&self) -> Result<GpuState, Closed> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SchedulerMsg::GpuStatus { reply })
            .await
            .map_err(|_| Closed)?;
        rx.await.map_err(|_| Closed)
    }

    /// Stops the scheduler; pulls are answered `Stop`.
    pub async fn stop(&self) {
        let _ = self.tx.send(SchedulerMsg::Stop).await;
    }
}

/// The scheduler is gone.
#[derive(Debug, thiserror::Error)]
#[error("the embedding scheduler is gone")]
pub struct Closed;

/// Where an item sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Queue {
    Stage1,
    Background,
    InFlight,
}

/// One indexing item: a `(repo, hash)` the worktrees sharing it wait for.
struct Item {
    repo: RepoId,
    hash: ContentHash,
    chunk: PendingChunk,
    tokens: u32,
    /// Derived: `Stage1` while any interested worktree is still in stage 1.
    class: IndexClass,
    queue: Queue,
    worktrees: Vec<WorktreeId>,
}

/// A repository's running freshness barriers, and the hashes that got a
/// vector since the first of them began.
///
/// Each barrier reads the vector cache after it began, and its `Enqueue`
/// lands before it ends. A hash that got a vector after that read may be
/// listed as missing, yet it has already left the queues. `committed`
/// holds every such hash, so `on_enqueue` drops it instead of queuing it a
/// second time. Once no barrier of the repository runs, no `Enqueue` built
/// from an earlier read can land, and the set goes with the entry.
#[derive(Default)]
struct OpenBarriers {
    /// Barriers begun and not yet ended.
    running: u64,
    /// Hashes whose vector went to the writer (`item_done`) or was
    /// committed by a search (`DocumentsStored`) while `running > 0`.
    committed: HashSet<ContentHash>,
}

/// How many of `interest` sit in each queue; kept in step with every
/// queue transition and interest change, so `counts_for` never scans.
#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    stage1: u64,
    background: u64,
    in_flight: u64,
}

impl Tally {
    /// The counter for `queue`.
    fn slot(&mut self, queue: Queue) -> &mut u64 {
        match queue {
            Queue::Stage1 => &mut self.stage1,
            Queue::Background => &mut self.background,
            Queue::InFlight => &mut self.in_flight,
        }
    }

    /// One more item of `interest` sits in `queue`.
    fn add(&mut self, queue: Queue) {
        *self.slot(queue) += 1;
    }

    /// One fewer item of `interest` sits in `queue`.
    fn sub(&mut self, queue: Queue) {
        let slot = self.slot(queue);
        *slot = slot.checked_sub(1).expect("queue tally underflow");
    }
}

/// What the scheduler tracks per worktree.
struct Worktree {
    /// The last applied `Enqueue` class.
    class: IndexClass,
    /// Stage-1 completion already ran.
    stage1_done: bool,
    /// Interested stage-1 items with `tokens <= STAGE1_MAX_TOKENS` still
    /// queued or in flight.
    stage1_left: usize,
    /// The items this worktree wants that the scheduler still holds: set by
    /// a `Replace`, edited by each `Change`, shrunk as items finish.
    interest: HashSet<ItemId>,
    /// `interest` counted by queue.
    tally: Tally,
    repo: RepoId,
    stage1_done_tx: mpsc::Sender<Stage1Done>,
}

impl Worktree {
    /// Whether stage-1 completion can still run for this worktree: it is in
    /// stage 1 and has not completed.
    fn stage1_active(&self) -> bool {
        self.class == IndexClass::Stage1 && !self.stage1_done
    }
}

/// Whether the GPU thread exists.
enum Gpu {
    Absent,
    Starting,
    Running,
}

/// The scheduler's state.
struct Scheduler<B: Backend> {
    /// Weak so the actor can end: every component exits once its last strong
    /// sender is dropped.
    tx: mpsc::WeakSender<SchedulerMsg>,
    backend: Arc<B>,
    dir: PathBuf,
    model_id: String,
    buckets: Box<[usize]>,
    fatal_tx: mpsc::Sender<Error>,
    /// The fatal error `run` was told about; queries arriving while stopping
    /// are answered with it.
    fatal: Option<String>,
    headroom: watch::Receiver<Headroom>,
    queries: VecDeque<QueryJob>,
    /// `(tokens, item)` ordered so `pop_first` is the shortest and `pop_last`
    /// the longest; `ItemId` is a sequence so the key is unique.
    stage1: BTreeMap<(u32, ItemId), ()>,
    background: BTreeMap<(u32, ItemId), ()>,
    items: HashMap<ItemId, Item>,
    by_hash: HashMap<(RepoId, ContentHash), ItemId>,
    worktrees: HashMap<WorktreeId, Worktree>,
    writers: HashMap<RepoId, WriterHandle>,
    next_item: ItemId,
    ane_waiting: Option<oneshot::Sender<AneJob>>,
    /// The ANE thread, from its last pull.
    ane_thread: Option<qos::ThreadId>,
    /// The index item currently on the ANE thread.
    ane_in_flight: Option<ItemId>,
    /// The override lifting the ANE thread while a query waits on its
    /// in-flight index job; dropped on the next `AnePull`.
    ane_override: Option<qos::Override>,
    /// `Stop` was sent to the ANE thread, or the thread is gone.
    ane_stopped: bool,
    /// Fires when the ANE thread exits: a thread that died before its first
    /// pull can never be answered `Stop`, and a panic unwinds past the send.
    ane_exited: watch::Receiver<bool>,
    gpu: Gpu,
    /// Exit receivers of every GPU thread ever spawned; handed to the
    /// caller at exit so it can join them without blocking the executor.
    gpu_exited: Vec<oneshot::Receiver<()>>,
    /// Where the GPU exit receivers go when the actor exits.
    exit_tx: Option<oneshot::Sender<Vec<oneshot::Receiver<()>>>>,
    /// `open_gpu` was `Blocked`; no respawn until headroom changes.
    gpu_blocked: bool,
    dispatched_indexing: u64,
    stats: HashMap<WorktreeId, AneStats>,
    /// Running freshness barriers per repository, counted: a barrier's end
    /// report can arrive after a later barrier already started. An entry
    /// exists only while its count is above zero.
    barriers: HashMap<RepoId, OpenBarriers>,
    /// `index --wait` watches, pushing each worktree's counts on change —
    /// one sender per waiting task, since waits no longer serialize on the
    /// owner.
    watchers: HashMap<WorktreeId, Vec<watch::Sender<Counts>>>,
    /// Query jobs on the ANE thread, not yet `QueryDone`.
    queries_in_flight: u64,
    /// The idle-exit tracker: every queued or in-flight unit of work reports
    /// busy through it.
    busy: watch::Sender<bool>,
    /// What the tracker was last told.
    last_busy: bool,
    stopping: bool,
}

/// What the scheduler is built from.
pub struct SchedulerConfig<B: Backend> {
    /// The embedding backend.
    pub backend: Arc<B>,
    /// The unpacked model directory.
    pub dir: PathBuf,
    /// The model-id vectors are stored under.
    pub model_id: String,
    /// The model's bucket lengths, ascending.
    pub buckets: Box<[usize]>,
    /// Host headroom for the GPU instance.
    pub headroom: watch::Receiver<Headroom>,
    /// Fatal errors go to `run` here.
    pub fatal: mpsc::Sender<Error>,
    /// The scheduler's busy state for the idle-exit tracker.
    pub busy: watch::Sender<bool>,
    /// Fires when the ANE thread exits.
    pub ane_exited: watch::Receiver<bool>,
}

/// Spawns the scheduler task. The ANE thread is spawned separately by the
/// caller; the GPU thread is spawned here whenever stage-1 work and headroom
/// allow it.
pub fn spawn<B: Backend>(
    config: SchedulerConfig<B>,
) -> (
    SchedulerHandle,
    oneshot::Receiver<Vec<oneshot::Receiver<()>>>,
) {
    let SchedulerConfig {
        backend,
        dir,
        model_id,
        buckets,
        headroom,
        fatal,
        busy,
        ane_exited,
    } = config;
    let (tx, rx) = mpsc::channel(INBOX);
    let (exit_tx, exited) = oneshot::channel();
    let scheduler = Scheduler {
        tx: tx.downgrade(),
        backend,
        dir,
        model_id,
        buckets,
        fatal_tx: fatal,
        fatal: None,
        headroom,
        queries: VecDeque::new(),
        stage1: BTreeMap::new(),
        background: BTreeMap::new(),
        items: HashMap::new(),
        by_hash: HashMap::new(),
        worktrees: HashMap::new(),
        writers: HashMap::new(),
        next_item: 0,
        ane_waiting: None,
        ane_thread: None,
        ane_in_flight: None,
        ane_override: None,
        ane_stopped: false,
        ane_exited,
        gpu: Gpu::Absent,
        gpu_exited: Vec::new(),
        exit_tx: Some(exit_tx),
        gpu_blocked: false,
        dispatched_indexing: 0,
        stats: HashMap::new(),
        barriers: HashMap::new(),
        watchers: HashMap::new(),
        queries_in_flight: 0,
        busy,
        last_busy: false,
        stopping: false,
    };
    tokio::spawn(async move {
        let mut scheduler = scheduler;
        scheduler.run(rx).await;
        // Every GPU thread's exit receiver goes to the owner of the
        // pipeline, which joins them without blocking the executor.
        if let Some(tx) = scheduler.exit_tx.take() {
            let _ = tx.send(scheduler.gpu_exited);
        }
    });
    (SchedulerHandle { tx }, exited)
}

impl<B: Backend> Scheduler<B> {
    async fn run(&mut self, mut rx: mpsc::Receiver<SchedulerMsg>) {
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    let Some(msg) = msg else { return };
                    self.handle(msg).await;
                }
                changed = self.headroom.changed() => {
                    if changed.is_err() {
                        debug!("the headroom monitor is gone; the scheduler exits");
                        return;
                    }
                    self.gpu_blocked = false;
                    self.eval_gpu();
                }
                gone = self.ane_exited.changed(), if !self.ane_stopped => {
                    self.ane_stopped = true;
                    if gone.is_err() {
                        // Unwound past the exit report: a panic on the ANE
                        // thread, fatal like any embedder failure.
                        self.on_fatal("the ANE thread died".to_owned()).await;
                    }
                }
            }
            self.dispatch_waiting();
            self.push_counts();
            self.update_busy();
            if self.stopping && self.ane_stopped && matches!(self.gpu, Gpu::Absent) {
                self.clear_busy();
                return;
            }
        }
    }

    async fn handle(&mut self, msg: SchedulerMsg) {
        match msg {
            SchedulerMsg::Enqueue(enqueue) => self.on_enqueue(*enqueue),
            SchedulerMsg::Query(job) => self.on_query(*job),
            SchedulerMsg::AnePull { thread, reply } => {
                self.ane_thread = Some(thread);
                // The in-flight index job is done; a held override ends.
                self.ane_override = None;
                self.on_ane_pull(reply);
            }
            SchedulerMsg::AneDone { item } => {
                assert_eq!(
                    self.ane_in_flight,
                    Some(item),
                    "AneDone for an item the ANE did not pull"
                );
                self.ane_in_flight = None;
                self.item_done(item);
            }
            SchedulerMsg::QueryDone {
                worktree,
                wait,
                overtaken,
            } => self.on_query_done(worktree, wait, overtaken),
            SchedulerMsg::GpuPull { reply } => self.on_gpu_pull(reply),
            SchedulerMsg::GpuStarted => self.gpu = Gpu::Running,
            SchedulerMsg::GpuGone => {
                self.gpu = Gpu::Absent;
                self.eval_gpu();
            }
            SchedulerMsg::GpuBlocked => {
                self.gpu = Gpu::Absent;
                self.gpu_blocked = true;
            }
            SchedulerMsg::GpuDone { items } => {
                for item in items {
                    self.item_done(item);
                }
            }
            SchedulerMsg::DocumentsStored { repo, hashes } => {
                for hash in hashes {
                    self.note_committed(repo, hash);
                    self.remove_queued(repo, hash);
                }
            }
            SchedulerMsg::EmbedderFailed { error } => self.on_fatal(error).await,
            SchedulerMsg::TakeStats { worktree, reply } => {
                let _ = reply.send(self.stats.remove(&worktree).unwrap_or_default());
            }
            SchedulerMsg::CountsFor { worktree, reply } => {
                let _ = reply.send(self.counts_for(worktree));
            }
            SchedulerMsg::Barrier { repo, running } => {
                if running {
                    self.barriers.entry(repo).or_default().running += 1;
                } else {
                    let open = self.barriers.get_mut(&repo).expect("unreported barrier");
                    open.running = open
                        .running
                        .checked_sub(1)
                        .expect("barrier count underflow");
                    if open.running == 0 {
                        // No `Enqueue` built from an earlier read can land.
                        self.barriers.remove(&repo);
                    }
                }
            }
            SchedulerMsg::WatchCounts { worktree, reply } => {
                let (watch, rx) = watch::channel(self.counts_for(worktree));
                self.watchers.entry(worktree).or_default().push(watch);
                let _ = reply.send(rx);
            }
            SchedulerMsg::GpuStatus { reply } => {
                let state = match self.gpu {
                    Gpu::Absent => GpuState::Absent {
                        blocked: self.headroom.borrow().clone().err().map(|b| b.to_string()),
                    },
                    Gpu::Starting => GpuState::Loading,
                    Gpu::Running => GpuState::Running,
                };
                let _ = reply.send(state);
            }
            SchedulerMsg::Stop => {
                self.stopping = true;
                for job in self.queries.drain(..) {
                    job.fail(EmbedError::SchedulerGone);
                }
            }
        }
    }

    /// Sends the ANE its next job if a pull is waiting and work exists, and
    /// keeps the GPU thread alive while it is wanted.
    fn dispatch_waiting(&mut self) {
        if self.stopping {
            if let Some(reply) = self.ane_waiting.take() {
                let _ = reply.send(AneJob::Stop);
                self.ane_stopped = true;
            }
            return;
        }
        self.eval_gpu();
        if self.ane_waiting.is_some() {
            let reply = self.ane_waiting.take().expect("checked above");
            self.dispatch_ane(reply);
        }
    }

    fn eval_gpu(&mut self) {
        if !self.gpu_wanted() || !matches!(self.gpu, Gpu::Absent) {
            return;
        }
        // A `None` means every strong sender is gone: shutdown is under way.
        let Some(tx) = self.tx.upgrade() else {
            return;
        };
        let exited = gpu::spawn(
            self.backend.clone(),
            self.dir.clone(),
            self.model_id.clone(),
            tx,
        );
        self.gpu = Gpu::Starting;
        self.gpu_exited.push(exited);
    }

    /// The GPU is wanted when stage-1 work is queued, the host has headroom
    /// and the daemon is not stopping or blocked from the GPU.
    fn gpu_wanted(&self) -> bool {
        !self.stopping
            && !self.gpu_blocked
            && !self.stage1.is_empty()
            && self.headroom.borrow().is_ok()
    }

    /// Applies a worktree's `Enqueue`: a `Replace` or a `Change` of its
    /// interest.
    fn on_enqueue(&mut self, enqueue: Enqueue) {
        let Enqueue {
            worktree,
            repo,
            class,
            mut interest,
            writer,
            stage1_done,
        } = enqueue;
        let open = self
            .barriers
            .get(&repo)
            .expect("an Enqueue lands inside one of its repository's barriers");
        // Chunks that got a vector after the read this interest was built
        // from: they need none, so they are neither queued nor counted.
        if !open.committed.is_empty() {
            let needs_vector =
                |item: &PendingItem| !open.committed.contains(&item.chunk.content_hash);
            match &mut interest {
                Interest::Replace(items) => items.retain(needs_vector),
                Interest::Change { added, .. } => added.retain(needs_vector),
            }
        }
        self.writers.entry(repo).or_insert(writer);
        match interest {
            Interest::Replace(items) => {
                self.replace_interest(worktree, repo, class, items, stage1_done)
            }
            Interest::Change { added, dropped } => {
                self.change_interest(worktree, repo, class, added, dropped)
            }
        }
        self.maybe_complete_stage1(worktree);
    }

    /// A `Replace` re-lists the worktree's whole interest: drop what it no
    /// longer names, then add what it does.
    fn replace_interest(
        &mut self,
        worktree: WorktreeId,
        repo: RepoId,
        class: IndexClass,
        items: Vec<PendingItem>,
        stage1_done: mpsc::Sender<Stage1Done>,
    ) {
        let keep: HashSet<ContentHash> = items.iter().map(|item| item.chunk.content_hash).collect();
        self.worktrees.entry(worktree).or_insert_with(|| Worktree {
            class,
            stage1_done: false,
            stage1_left: 0,
            interest: HashSet::new(),
            tally: Tally::default(),
            repo,
            stage1_done_tx: stage1_done.clone(),
        });
        let old: Vec<ItemId> = self.worktrees[&worktree].interest.iter().copied().collect();
        for item in old {
            if !keep.contains(&self.items[&item].hash) {
                self.worktrees
                    .get_mut(&worktree)
                    .expect("the worktree exists")
                    .interest
                    .remove(&item);
                self.lose_interest(worktree, item);
            }
        }
        let adds_small = items
            .iter()
            .any(|item| item.tokens as usize <= STAGE1_MAX_TOKENS);
        let state = self
            .worktrees
            .get_mut(&worktree)
            .expect("the worktree exists");
        state.class = class;
        // A `Stage1` enqueue that adds no small item keeps a past completion:
        // `Stage1Done` is sent once per stage-1 pass, not once per enqueue.
        if class == IndexClass::Stage1 && adds_small {
            state.stage1_done = false;
        }
        state.repo = repo;
        state.stage1_done_tx = stage1_done;
        for item in items {
            self.add_interest(worktree, repo, class, item);
        }
        let interest: Vec<ItemId> = self.worktrees[&worktree].interest.iter().copied().collect();
        for item in interest {
            self.refresh_class(item);
        }
        self.recompute_stage1_left(worktree);
    }

    /// A `Change` edits the interest the previous `Enqueue` left: drop under
    /// the old counting, then add under the new class.
    fn change_interest(
        &mut self,
        worktree: WorktreeId,
        repo: RepoId,
        class: IndexClass,
        added: Vec<PendingItem>,
        dropped: Vec<ContentHash>,
    ) {
        let was_active = self
            .worktrees
            .get_mut(&worktree)
            .expect("a worktree's first Enqueue is a Replace")
            .stage1_active();
        for hash in dropped {
            let Some(&item) = self.by_hash.get(&(repo, hash)) else {
                continue;
            };
            if !self
                .worktrees
                .get_mut(&worktree)
                .expect("the worktree exists")
                .interest
                .remove(&item)
            {
                continue;
            }
            self.lose_interest(worktree, item);
        }
        {
            let state = self
                .worktrees
                .get_mut(&worktree)
                .expect("the worktree exists");
            state.class = class;
            if class == IndexClass::Stage1
                && added
                    .iter()
                    .any(|item| item.tokens as usize <= STAGE1_MAX_TOKENS)
            {
                state.stage1_done = false;
            }
        }
        let flipped = self.worktrees[&worktree].stage1_active() != was_active;
        for item in added {
            self.add_interest(worktree, repo, class, item);
        }
        // A class refresh is needed only when the worktree's stage-1
        // activity flipped: a late `Stage1` `Change` after completion, or
        // the `Background` `Change` after `Stage1Done`.
        if flipped {
            let interest: Vec<ItemId> =
                self.worktrees[&worktree].interest.iter().copied().collect();
            for item in interest {
                self.refresh_class(item);
            }
            self.recompute_stage1_left(worktree);
        }
    }

    /// `worktree` wants `pending`: creates the `(repo, hash)` item when new,
    /// and when the worktree was not yet interested adds it to the item, the
    /// tally and `stage1_left`, then re-derives the item's class.
    fn add_interest(
        &mut self,
        worktree: WorktreeId,
        repo: RepoId,
        class: IndexClass,
        pending: PendingItem,
    ) {
        let PendingItem { chunk, tokens } = pending;
        let key = (repo, chunk.content_hash);
        let item = match self.by_hash.get(&key) {
            Some(&item) => item,
            None => {
                let item = self.next_item;
                self.next_item += 1;
                let queue = match class {
                    IndexClass::Stage1 => Queue::Stage1,
                    IndexClass::Background => Queue::Background,
                };
                self.queue_mut(queue).insert((tokens, item), ());
                self.items.insert(
                    item,
                    Item {
                        repo,
                        hash: chunk.content_hash,
                        chunk,
                        tokens,
                        class,
                        queue,
                        worktrees: Vec::new(),
                    },
                );
                self.by_hash.insert(key, item);
                item
            }
        };
        self.worktrees
            .get_mut(&worktree)
            .expect("the worktree exists")
            .interest
            .insert(item);
        let entry = self.items.get_mut(&item).expect("queued item exists");
        if entry.worktrees.contains(&worktree) {
            return;
        }
        entry.worktrees.push(worktree);
        let (queue, item_class, item_tokens) = (entry.queue, entry.class, entry.tokens);
        // This item was not yet counted: add it under its queue as it sits
        // now, before the refresh below may flip it.
        let state = self
            .worktrees
            .get_mut(&worktree)
            .expect("the worktree exists");
        state.tally.add(queue);
        if item_class == IndexClass::Stage1
            && item_tokens as usize <= STAGE1_MAX_TOKENS
            && state.stage1_active()
        {
            state.stage1_left += 1;
        }
        // A Background→Stage1 flip increments every interested active
        // worktree, this one included; it was not counted above.
        self.refresh_class(item);
    }

    /// Recomputes `stage1_left` for `worktree` from its interest list; an
    /// inactive worktree counts nothing.
    fn recompute_stage1_left(&mut self, worktree: WorktreeId) {
        let left = if self.worktrees[&worktree].stage1_active() {
            self.worktrees[&worktree]
                .interest
                .iter()
                .filter(|item| {
                    let item = &self.items[item];
                    item.tokens as usize <= STAGE1_MAX_TOKENS && item.class == IndexClass::Stage1
                })
                .count()
        } else {
            0
        };
        self.worktrees
            .get_mut(&worktree)
            .expect("the worktree exists")
            .stage1_left = left;
    }

    /// A worktree lost interest in `item`: drop it from the item's list and
    /// the item entirely when nobody is left.
    fn lose_interest(&mut self, worktree: WorktreeId, item: ItemId) {
        let entry = self.items.get_mut(&item).expect("interested item exists");
        entry.worktrees.retain(|w| *w != worktree);
        let queue = entry.queue;
        self.worktrees
            .get_mut(&worktree)
            .expect("interested worktree exists")
            .tally
            .sub(queue);
        if entry.class == IndexClass::Stage1
            && entry.tokens as usize <= STAGE1_MAX_TOKENS
            && let Some(state) = self.worktrees.get_mut(&worktree)
            && state.stage1_active()
        {
            state.stage1_left = state
                .stage1_left
                .checked_sub(1)
                .expect("stage1_left underflow");
        }
        self.refresh_class(item);
        let entry = &self.items[&item];
        if entry.worktrees.is_empty() && entry.queue != Queue::InFlight {
            self.remove_item(item);
        }
    }

    /// Re-derives an item's class from its worktrees, moves it between the
    /// queues on a change, and keeps the `stage1_left` counts right.
    fn refresh_class(&mut self, item: ItemId) {
        let entry = &self.items[&item];
        let class = if entry
            .worktrees
            .iter()
            .any(|w| self.worktrees[w].stage1_active())
        {
            IndexClass::Stage1
        } else {
            IndexClass::Background
        };
        if class == entry.class {
            return;
        }
        let small = entry.tokens as usize <= STAGE1_MAX_TOKENS;
        let interested: Vec<WorktreeId> = entry.worktrees.clone();
        let (queue, tokens) = {
            let entry = self.items.get_mut(&item).expect("exists");
            entry.class = class;
            (entry.queue, entry.tokens)
        };
        match (queue, class) {
            (Queue::Stage1, IndexClass::Background) => {
                self.stage1.remove(&(tokens, item));
                self.background.insert((tokens, item), ());
                self.set_queue(item, Queue::Background);
            }
            (Queue::Background, IndexClass::Stage1) => {
                self.background.remove(&(tokens, item));
                self.stage1.insert((tokens, item), ());
                self.set_queue(item, Queue::Stage1);
            }
            _ => {}
        }
        if !small {
            return;
        }
        for w in interested {
            let state = self
                .worktrees
                .get_mut(&w)
                .expect("interested worktree exists");
            if !state.stage1_active() {
                continue;
            }
            match class {
                IndexClass::Stage1 => state.stage1_left += 1,
                IndexClass::Background => {
                    state.stage1_left = state
                        .stage1_left
                        .checked_sub(1)
                        .expect("stage1_left underflow");
                }
            }
        }
    }

    /// Removes an item from its queue, the maps and the hash index, and
    /// drops it from every interested worktree's `interest`.
    fn remove_item(&mut self, item: ItemId) {
        let entry = self.items.remove(&item).expect("item exists");
        self.by_hash.remove(&(entry.repo, entry.hash));
        match entry.queue {
            Queue::Stage1 => {
                self.stage1.remove(&(entry.tokens, item));
            }
            Queue::Background => {
                self.background.remove(&(entry.tokens, item));
            }
            Queue::InFlight => {}
        }
        for worktree in &entry.worktrees {
            if let Some(state) = self.worktrees.get_mut(worktree) {
                state.interest.remove(&item);
                state.tally.sub(entry.queue);
            }
        }
    }

    /// Moves `item` to `to`, updating every interested worktree's tally.
    /// Every `entry.queue` assignment goes through here.
    fn set_queue(&mut self, item: ItemId, to: Queue) {
        let entry = self.items.get(&item).expect("queued item exists");
        let old = entry.queue;
        for &worktree in &entry.worktrees {
            let tally = &mut self
                .worktrees
                .get_mut(&worktree)
                .expect("interested worktree exists")
                .tally;
            tally.sub(old);
            tally.add(to);
        }
        self.items.get_mut(&item).expect("queued item exists").queue = to;
    }

    fn queue_mut(&mut self, queue: Queue) -> &mut BTreeMap<(u32, ItemId), ()> {
        match queue {
            Queue::Stage1 => &mut self.stage1,
            Queue::Background => &mut self.background,
            Queue::InFlight => unreachable!("in-flight items live in no queue"),
        }
    }

    /// A query job arrived: fail it while stopping, else stamp it and queue
    /// it behind earlier queries.
    fn on_query(&mut self, mut job: QueryJob) {
        if self.stopping {
            let error = match &self.fatal {
                Some(error) => EmbedError::Model(error.clone()),
                None => EmbedError::SchedulerGone,
            };
            job.fail(error);
            return;
        }
        let stamp = self.dispatched_indexing;
        match &mut job {
            QueryJob::Query { stamp: s, .. } | QueryJob::Documents { stamp: s, .. } => *s = stamp,
        }
        // The query waits behind the index job on the ANE; lift that thread
        // out of its job's class so a background chunk cannot starve it.
        if self.ane_in_flight.is_some()
            && self.ane_override.is_none()
            && let Some(thread) = self.ane_thread
        {
            self.ane_override = Some(qos::Override::start(thread, QosClass::UserInteractive));
        }
        self.queries.push_back(job);
    }

    /// The ANE asked for work: queries first, then stage 1 shortest-first,
    /// then background. Anything else keeps the pull waiting.
    fn on_ane_pull(&mut self, reply: oneshot::Sender<AneJob>) {
        assert!(
            self.ane_waiting.is_none(),
            "the ANE pulled twice without a reply"
        );
        if self.stopping {
            let _ = reply.send(AneJob::Stop);
            self.ane_stopped = true;
            return;
        }
        self.dispatch_ane(reply);
    }

    fn dispatch_ane(&mut self, reply: oneshot::Sender<AneJob>) {
        while let Some(job) = self.queries.pop_front() {
            if job.closed() {
                continue;
            }
            let stamp = match &job {
                QueryJob::Query { stamp, .. } | QueryJob::Documents { stamp, .. } => *stamp,
            };
            let overtaken = u32::try_from(self.dispatched_indexing - stamp)
                .expect("overtaken cannot exceed dispatched_indexing");
            self.queries_in_flight += 1;
            let _ = reply.send(AneJob::Query { job, overtaken });
            return;
        }
        for (queue, qos) in [
            (Queue::Stage1, QosClass::UserInitiated),
            (Queue::Background, QosClass::Background),
        ] {
            if let Some((&key, &())) = self.queue_mut(queue).first_key_value() {
                self.queue_mut(queue).remove(&key);
                let item = key.1;
                self.set_queue(item, Queue::InFlight);
                let entry = &self.items[&item];
                let repo = entry.repo;
                let chunk = entry.chunk.clone();
                self.dispatched_indexing += 1;
                self.ane_in_flight = Some(item);
                let _ = reply.send(AneJob::Index {
                    item,
                    chunk,
                    qos,
                    writer: self.writers[&repo].clone(),
                });
                return;
            }
        }
        self.ane_waiting = Some(reply);
    }

    /// The GPU pulled: answer with a same-bucket batch off the long end, or
    /// `Stop` when the GPU is no longer wanted.
    fn on_gpu_pull(&mut self, reply: oneshot::Sender<GpuJob>) {
        assert!(
            matches!(self.gpu, Gpu::Running),
            "a GPU pull arrived before GpuStarted"
        );
        if !self.gpu_wanted() {
            let _ = reply.send(GpuJob::Stop);
            return;
        }
        // The batch's bucket is the longest queued item's; up to `GPU_BATCH`
        // items off the back of `stage1` that share it go in one batch.
        let mut batch = Vec::with_capacity(GPU_BATCH);
        let mut bucket = None;
        while batch.len() < GPU_BATCH {
            let Some((&(_, item), &())) = self.stage1.last_key_value() else {
                break;
            };
            let item_bucket = self.bucket(self.items[&item].tokens);
            match bucket {
                None => bucket = Some(item_bucket),
                Some(b) if b == item_bucket => {}
                Some(_) => break,
            }
            self.stage1.pop_last();
            self.set_queue(item, Queue::InFlight);
            let entry = &self.items[&item];
            let repo = entry.repo;
            let chunk = entry.chunk.clone();
            batch.push((item, chunk, self.writers[&repo].clone()));
        }
        if batch.is_empty() {
            let _ = reply.send(GpuJob::Stop);
        } else {
            let _ = reply.send(GpuJob::Batch(batch));
        }
    }

    /// The index into `buckets` of the smallest bucket that holds `tokens`,
    /// or the largest there is.
    fn bucket(&self, tokens: u32) -> usize {
        self.buckets
            .iter()
            .position(|&b| tokens as usize <= b)
            .unwrap_or(self.buckets.len() - 1)
    }

    /// An embedder finished `item`: the vector already went to the repo
    /// writer, so drop the item and close stage 1 for any worktree it
    /// completed.
    fn item_done(&mut self, item: ItemId) {
        let entry = self.items.remove(&item).expect("done item exists");
        self.by_hash.remove(&(entry.repo, entry.hash));
        self.note_committed(entry.repo, entry.hash);
        for worktree in &entry.worktrees {
            if let Some(state) = self.worktrees.get_mut(worktree) {
                state.interest.remove(&item);
                state.tally.sub(entry.queue);
            }
        }
        for worktree in entry.worktrees {
            if entry.tokens as usize <= STAGE1_MAX_TOKENS && entry.class == IndexClass::Stage1 {
                let state = self
                    .worktrees
                    .get_mut(&worktree)
                    .expect("interested worktree");
                if state.stage1_active() {
                    state.stage1_left = state
                        .stage1_left
                        .checked_sub(1)
                        .expect("stage1_left underflow");
                }
            }
            self.maybe_complete_stage1(worktree);
        }
    }

    /// A query job completed: accumulate the worktree's stats.
    fn on_query_done(&mut self, worktree: WorktreeId, wait: Duration, overtaken: u32) {
        self.queries_in_flight = self
            .queries_in_flight
            .checked_sub(1)
            .expect("a QueryDone without an in-flight query");
        let stats = self.stats.entry(worktree).or_default();
        stats.wait = stats.wait.max(wait);
        stats.overtaken = stats.overtaken.saturating_add(overtaken);
    }

    /// Removes a queued item whose vector was committed another way;
    /// in-flight items are untouched.
    fn remove_queued(&mut self, repo: RepoId, hash: ContentHash) {
        let Some(&item) = self.by_hash.get(&(repo, hash)) else {
            return;
        };
        if self.items[&item].queue == Queue::InFlight {
            return;
        }
        let worktrees = self.items[&item].worktrees.clone();
        let small_stage1 = {
            let entry = &self.items[&item];
            entry.tokens as usize <= STAGE1_MAX_TOKENS && entry.class == IndexClass::Stage1
        };
        self.remove_item(item);
        for worktree in worktrees {
            if small_stage1 {
                let state = self
                    .worktrees
                    .get_mut(&worktree)
                    .expect("interested worktree");
                if state.stage1_active() {
                    state.stage1_left = state
                        .stage1_left
                        .checked_sub(1)
                        .expect("stage1_left underflow");
                }
            }
            self.maybe_complete_stage1(worktree);
        }
    }

    /// `hash` of `repo` has a vector, committed or on its way to the
    /// writer. While a barrier of `repo` runs it is remembered, so an
    /// `Enqueue` built from a read inside that barrier never queues it.
    fn note_committed(&mut self, repo: RepoId, hash: ContentHash) {
        if let Some(open) = self.barriers.get_mut(&repo) {
            open.committed.insert(hash);
        }
    }

    /// Runs stage-1 completion for `worktree` when its stage-1 items are all
    /// embedded.
    fn maybe_complete_stage1(&mut self, worktree: WorktreeId) {
        let Some(state) = self.worktrees.get(&worktree) else {
            return;
        };
        if !state.stage1_active() || state.stage1_left > 0 {
            return;
        }
        self.complete_stage1(worktree);
    }

    /// Stage 1 for `worktree` is done: its remaining stage-1 items move to
    /// background, then `Stage1Done` goes out once the repo writer flushed.
    fn complete_stage1(&mut self, worktree: WorktreeId) {
        let state = self
            .worktrees
            .get_mut(&worktree)
            .expect("the worktree exists");
        state.stage1_done = true;
        let interest: Vec<ItemId> = state.interest.iter().copied().collect();
        for item in interest {
            self.refresh_class(item);
        }
        let state = &self.worktrees[&worktree];
        let writer = self.writers[&state.repo].clone();
        let stage1_done = state.stage1_done_tx.clone();
        tokio::spawn(async move {
            if writer.flush().await.is_ok() {
                let _ = stage1_done.send(Stage1Done).await;
            }
        });
    }

    /// An embedder failure: answer every waiting query with it, log it, tell
    /// `run` and stop dispatching. There is no retry and no fallback.
    async fn on_fatal(&mut self, error: String) {
        error!(%error, "embedder failed");
        for job in self.queries.drain(..) {
            job.fail(EmbedError::Model(error.clone()));
        }
        self.fatal = Some(error.clone());
        self.stopping = true;
        self.ane_override = None;
        if let Some(reply) = self.ane_waiting.take() {
            let _ = reply.send(AneJob::Stop);
            self.ane_stopped = true;
        }
        let _ = self.fatal_tx.send(Error::Embedder(error)).await;
    }

    fn counts_for(&self, worktree: WorktreeId) -> Counts {
        let Some(state) = self.worktrees.get(&worktree) else {
            return Counts::default();
        };
        Counts {
            queued_stage1: state.tally.stage1,
            queued_background: state.tally.background,
            in_flight: state.tally.in_flight,
            stage1_done: state.stage1_done,
        }
    }

    /// Pushes each watched worktree's counts when they changed, dropping
    /// senders whose `index --wait` went away.
    fn push_counts(&mut self) {
        let mut watchers = std::mem::take(&mut self.watchers);
        watchers.retain(|worktree, watches| {
            let counts = self.counts_for(*worktree);
            watches.retain(|watch| {
                let _ = watch.send_if_modified(|current| {
                    if *current != counts {
                        *current = counts;
                        true
                    } else {
                        false
                    }
                });
                !watch.is_closed()
            });
            !watches.is_empty()
        });
        self.watchers = watchers;
    }

    /// Queued or in-flight work of any kind keeps the daemon busy: query
    /// jobs, indexing items and running barriers.
    fn update_busy(&mut self) {
        let busy = !self.queries.is_empty()
            || self.queries_in_flight > 0
            || !self.items.is_empty()
            || !self.barriers.is_empty();
        if busy != self.last_busy {
            self.last_busy = busy;
            self.busy.send_replace(busy);
        }
    }

    /// A stopped scheduler holds no work; clear the report so the idle-exit
    /// tracker never keeps a dead source.
    fn clear_busy(&mut self) {
        if self.last_busy {
            self.last_busy = false;
            self.busy.send_replace(false);
        }
    }
}
