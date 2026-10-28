//! The scheduler actor: the sole owner of all embedding work, against the
//! embedding server's HTTP API.
//!
//! The actor keeps the index semantics the pipeline is built on — stage-1
//! and background classes, `(repo, hash)` de-duplication, worktree
//! interest, freshness barriers — and hands the server a bounded window
//! of in-flight requests instead of embedding on its own threads. The
//! window holds the 8 shortest and the 24 longest remaining items, filled
//! from the stage-1 queue before background, one input per request:
//! the server's ANE then sees the true global shortest and its GPU the
//! true global longest. Stage-1 items go at `u=3`, background at `u=7`;
//! queries and a search's `Documents` job preempt indexing at `u=0`.
//! HTTP/1.1 cannot change a request's priority, so a `Background`→`Stage1`
//! class flip of an in-flight item aborts its request and requeues it;
//! the reverse is left alone — at most 32 items finish slightly early. A
//! `Documents` job embeds every chunk it needs in one `u=0` request,
//! holding each away from indexing first — a chunk already in flight has
//! its request aborted, since the daemon cannot tell whether the server
//! is running it or it is still queued. Finished vectors go to the
//! per-repository [`writer`] inside each request task, before its
//! completion lands here.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use repo_search_core::embed_api::{ServerTiming, Urgency};
use repo_search_core::model::{BUCKETS, DOCUMENT};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::store::ContentHash;
use repo_search_core::{QueryTask, UnitVector};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::AbortHandle;
use tracing::{debug, error};

use super::Error;
use super::counter::PromptCounter;
use super::server::Embeddings;
use super::writer::WriterHandle;

/// Inbox capacity of the scheduler and every component channel.
const INBOX: usize = 64;

/// Stage 1 embeds only chunks up to this many tokens; larger chunks wait
/// for stage 2 at background urgency.
pub const STAGE1_MAX_TOKENS: usize = BUCKETS[1];

/// Short-end flights: the server's ANE embeds one item at a time, so a
/// few queued items cover daemon stalls and forwarded-socket round trips.
const SHORT_SLOTS: usize = 8;

/// Long-end flights: one running GPU batch of 8, a full same-bucket batch
/// of 8 queued, and slack for a batch to form near bucket boundaries.
const LONG_SLOTS: usize = 24;

/// The daemon's bound of embedding requests in flight.
const IN_FLIGHT: usize = SHORT_SLOTS + LONG_SLOTS;

/// The `Priority` urgency stage-1 items embed at.
const URGENCY_STAGE1: u8 = 3;

/// The `Priority` urgency background items embed at.
const URGENCY_BACKGROUND: u8 = 7;

/// The `Priority` urgency queries and `Documents` jobs embed at.
const URGENCY_INTERACTIVE: u8 = 0;

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
    /// First-index work: stage-1 priority, sent at `u=3`.
    Stage1,
    /// Everything else: sent at `u=7`.
    Background,
}

/// A chunk handed to the scheduler with its token count.
#[derive(Debug, Clone)]
pub struct PendingItem {
    /// The chunk to embed.
    pub chunk: PendingChunk,
    /// `TokenCounter::document_tokens` of the chunk, counted by the caller.
    pub tokens: u32,
}

/// Sent to a worktree's `stage1_done` channel once, when every stage-1
/// item it is interested in is embedded and flushed to the vector cache.
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
    /// This repo's vector writer; the request tasks hand it every
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
/// `dropped` the hashes absent from scope at that moment. Feeds are
/// applied in pass order, so for any hash the last feed that names it
/// reflects the latest pass that changed its presence, and a `Replace`
/// computed later than some queued `Change`s is corrected by them. A hash
/// `missing` left out because its file changed after the pass is offered
/// again by the next pass.
///
/// Both lists are read from the cache inside the barrier the `Enqueue`
/// belongs to. A hash that got a vector between that read and the
/// `Enqueue`, by any worktree of the repository or by a search, is
/// dropped from `Replace` and `added` when the `Enqueue` is applied (see
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

/// A query embedding for a search; query work preempts indexing at
/// `u=0`. Its request runs in its own task, outside the in-flight window.
#[derive(Debug)]
pub struct QueryJob {
    /// The worktree asking; stats accumulate under it.
    pub worktree: WorktreeId,
    /// How the query is phrased for the model.
    pub task: QueryTask,
    /// The query text.
    pub text: String,
    /// When the job was handed to the scheduler.
    pub enqueued: Instant,
    /// Unused: a query's overtaken count comes back in `Server-Timing`.
    pub stamp: u64,
    /// The answer goes here.
    pub reply: oneshot::Sender<Result<UnitVector, EmbedError>>,
}

/// A search's on-the-spot document embeddings. Each chunk's vector goes
/// to `writer`: from the job's own `u=0` request, or it got there before
/// the job landed. A chunk in flight on an indexing request has that
/// request aborted and is embedded by the job instead; the cache keeps
/// whichever of its two vectors is committed first. The job lands inside
/// the search's window (`Barrier { running: true }` sent before the
/// search's first cache read, `running: false` after the job is
/// answered).
#[derive(Debug)]
pub struct DocumentsJob {
    /// The worktree asking; stats accumulate under it.
    pub worktree: WorktreeId,
    /// The repository the chunks belong to.
    pub repo: RepoId,
    /// The chunks; their content hashes are distinct.
    pub chunks: Vec<PendingChunk>,
    /// `repo`'s vector writer.
    pub writer: WriterHandle,
    /// When the job was handed to the scheduler.
    pub enqueued: Instant,
    /// Answered `Ok` once every chunk's vector is with `writer`; the
    /// caller flushes it to commit them.
    pub reply: oneshot::Sender<Result<(), EmbedError>>,
}

/// A `Documents` job the scheduler is tracking, as it numbered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpotId(u64);

/// The query path's failure, as the owner sees it over `EmbedService`.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The server failed; the daemon is shutting down.
    #[error("model: {0}")]
    Model(String),
    /// The scheduler went away before answering.
    #[error("the embedding scheduler is gone")]
    SchedulerGone,
    /// The repository's vector writer failed to commit the vectors.
    #[error("vector writer: {0}")]
    Writer(String),
}

/// Query wait statistics a worktree accumulated since the last
/// `TakeStats`, fed by the `Server-Timing` header of each request.
#[derive(Debug, Default, Clone, Copy)]
pub struct AneStats {
    /// The longest a request of this worktree waited behind lower-urgency
    /// work on the server.
    pub wait: Duration,
    /// Sum of `overtaken` across the worktree's requests; must be zero
    /// for `u=0` work.
    pub overtaken: u32,
}

/// Queue counts for a worktree's `Status`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// Items in the stage-1 queue.
    pub queued_stage1: u64,
    /// Items in the background queue.
    pub queued_background: u64,
    /// Items in a request to the server, or held by a search's
    /// `Documents` job, not yet done.
    pub in_flight: u64,
    /// Stage-1 completion already ran for this worktree; its `Stage1Done`
    /// is on its way to the owner once the writer flush lands.
    pub stage1_done: bool,
}

/// The scheduler's inbox. Every component the pipeline owns — and every
/// request task it spawned — talks to the scheduler through this one
/// channel.
#[derive(Debug)]
pub enum SchedulerMsg {
    /// A worktree's indexing work: a `Replace` or a `Change` of its
    /// interest.
    Enqueue(Box<Enqueue>),
    /// A query embedding job.
    Query(Box<QueryJob>),
    /// A search's on-the-spot `Documents` job.
    Documents(Box<DocumentsJob>),
    /// An indexing item's request landed; its vector is already with the
    /// item's repo writer.
    IndexDone {
        /// The completed item.
        item: ItemId,
        /// The flight's generation, checked against the current one: a
        /// completion of a cancelled and requeued flight is ignored.
        generation: u64,
    },
    /// A request's embed call failed, past its retry: fatal.
    ServerFailed {
        /// What failed, for the log and waiting queries.
        error: String,
    },
    /// A `Documents` job's request landed; its vectors are already with
    /// the job's writer.
    DocumentsDone {
        /// The job.
        spot: SpotId,
        /// The request's `Server-Timing`.
        timing: ServerTiming,
    },
    /// A query job's request ended; its reply is already sent.
    QueryDone {
        /// The job's worktree.
        worktree: WorktreeId,
        /// The request's `Server-Timing`, or its failure past its retry,
        /// which is fatal.
        outcome: Result<ServerTiming, String>,
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
    /// A freshness barrier of a worktree of `repo`, or a search's window,
    /// is running (`running`) or ended. Both keep the daemon busy for the
    /// idle exit, and while any of them runs, the scheduler remembers the
    /// hashes of `repo` that get a vector (see [`OpenBarriers`]). A
    /// barrier's `Enqueue`, when it has one, is sent between its two
    /// reports, and so is a search's `Documents` job.
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
    /// Stop dispatching; in-flight requests finish, none are sent anew,
    /// and the scheduler exits once the last of them has reported.
    Stop,
}

/// The scheduler's handle; clones share the one inbox.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::Sender<SchedulerMsg>,
}

impl SchedulerHandle {
    /// The raw inbox sender, for components that talk [`SchedulerMsg`].
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

    /// Sends a query embedding job.
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

    /// Sends a search's `Documents` job.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn documents(&self, job: DocumentsJob) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::Documents(Box::new(job)))
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

    /// Reports a freshness barrier of a worktree of `repo`, or a search's
    /// window, running or ended. `running: true` must be sent before the
    /// first cache read the barrier or the search makes, and
    /// `running: false` after its `Enqueue` or its `Documents` job's
    /// answer.
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

    /// Stops the scheduler; nothing is dispatched anew.
    pub async fn stop(&self) {
        let _ = self.tx.send(SchedulerMsg::Stop).await;
    }
}

/// The scheduler is gone.
#[derive(Debug, thiserror::Error)]
#[error("the embedding scheduler is gone")]
pub struct Closed;

/// What a query's request `input` is: `task`'s prompt for `query`, cut to
/// the model's cap. Production is [`PromptCounter::query_input`].
pub trait QueryInput: Send + Sync + 'static {
    /// `task`'s prompt for `query`, truncated to the model's token cap.
    fn input(&self, task: QueryTask, query: &str) -> String;
}

impl QueryInput for PromptCounter {
    fn input(&self, task: QueryTask, query: &str) -> String {
        self.query_input(task, query)
    }
}

/// Where an item sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Queue {
    Stage1,
    Background,
    InFlight,
    /// Held by a search's `Documents` job: in no queue, and no indexing
    /// request takes it.
    Held,
}

/// One indexing item: a `(repo, hash)` the worktrees sharing it wait for.
struct Item {
    repo: RepoId,
    hash: ContentHash,
    chunk: PendingChunk,
    tokens: u32,
    /// Derived: `Stage1` while any interested worktree is still in
    /// stage 1.
    class: IndexClass,
    queue: Queue,
    worktrees: Vec<WorktreeId>,
}

/// A repository's running freshness barriers and search windows, and the
/// hashes that got a vector since the first of them began.
///
/// Each barrier reads the vector cache after it began, and its `Enqueue`
/// lands before it ends. A search's window reads the cache at its vector
/// sync, and its `Documents` job lands before it ends; `on_documents`
/// skips the job's chunks found here the same way. A hash that got a
/// vector after that read may be listed as missing, yet it has already
/// left the queues. `committed` holds every such hash, so `on_enqueue`
/// drops it instead of queuing it a second time. Once no barrier of the
/// repository runs, no `Enqueue` built from an earlier read can land,
/// and the set goes with the entry.
#[derive(Default)]
struct OpenBarriers {
    /// Barriers and search windows begun and not yet ended.
    running: u64,
    /// Hashes whose vector went to the writer (`item_done`,
    /// `on_documents_done`) while `running > 0`. A chunk embedded both by
    /// an indexing request and by a search's job is inserted twice; the
    /// set keeps it once.
    committed: HashSet<ContentHash>,
}

/// Which end of the queues a flight came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// One of the `SHORT_SLOTS` shortest remaining items.
    Short,
    /// One of the `LONG_SLOTS` longest remaining items.
    Long,
}

/// An indexing request in flight on the server.
struct Flight {
    /// The window slot the flight fills.
    end: End,
    /// The flight's generation: an aborted and redispatched item leaves
    /// its old flight's completion stale.
    generation: u64,
    /// Aborts the request task; the server drops the request's queued
    /// item once its connection closes.
    task: AbortHandle,
}

/// A `Documents` job from its arrival until it is answered or dropped.
struct Spot {
    worktree: WorktreeId,
    repo: RepoId,
    /// The held chunks' hashes; finished at `DocumentsDone`, or released
    /// when the job is dropped or stopped unanswered.
    held: Vec<ContentHash>,
    /// The job's `u=0` request task.
    task: AbortHandle,
    reply: oneshot::Sender<Result<(), EmbedError>>,
}

/// The queue an item of `class` sits in.
fn queue_of(class: IndexClass) -> Queue {
    match class {
        IndexClass::Stage1 => Queue::Stage1,
        IndexClass::Background => Queue::Background,
    }
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
            Queue::InFlight | Queue::Held => &mut self.in_flight,
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
    /// The items this worktree wants that the scheduler still holds: set
    /// by a `Replace`, edited by each `Change`, shrunk as items finish.
    interest: HashSet<ItemId>,
    /// `interest` counted by queue.
    tally: Tally,
    repo: RepoId,
    stage1_done_tx: mpsc::Sender<Stage1Done>,
}

impl Worktree {
    /// Whether stage-1 completion can still run for this worktree: it is
    /// in stage 1 and has not completed.
    fn stage1_active(&self) -> bool {
        self.class == IndexClass::Stage1 && !self.stage1_done
    }
}

/// The scheduler's state.
struct Scheduler<E: Embeddings, Q: QueryInput> {
    /// Weak so the actor can end: every component exits once its last
    /// strong sender is dropped.
    tx: mpsc::WeakSender<SchedulerMsg>,
    /// The embedding server's client.
    embeddings: Arc<E>,
    /// The query input formatter: the prompt the server tokenizes, capped.
    query_input: Arc<Q>,
    fatal_tx: mpsc::Sender<Error>,
    /// The fatal error `run` was told about; jobs arriving while stopping
    /// are answered with it.
    fatal: Option<String>,
    /// `(tokens, item)` ordered so `pop_first` is the shortest and
    /// `pop_last` the longest; `ItemId` is a sequence so the key is
    /// unique.
    stage1: BTreeMap<(u32, ItemId), ()>,
    background: BTreeMap<(u32, ItemId), ()>,
    items: HashMap<ItemId, Item>,
    by_hash: HashMap<(RepoId, ContentHash), ItemId>,
    worktrees: HashMap<WorktreeId, Worktree>,
    writers: HashMap<RepoId, WriterHandle>,
    next_item: ItemId,
    /// The indexing requests in flight: at most `SHORT_SLOTS` short-end
    /// and `LONG_SLOTS` long-end items.
    flights: HashMap<ItemId, Flight>,
    /// The flight generation counter: a redispatched item gets a new one.
    next_generation: u64,
    /// `Documents` jobs not yet answered.
    spots: HashMap<SpotId, Spot>,
    /// The next `SpotId`.
    next_spot: u64,
    /// Per `(repo, hash)`, how many unanswered `Documents` jobs hold it
    /// for their request. An item of a key here is `Held`.
    held: HashMap<(RepoId, ContentHash), u32>,
    stats: HashMap<WorktreeId, AneStats>,
    /// Running freshness barriers per repository, counted: a barrier's
    /// end report can arrive after a later barrier already started. An
    /// entry exists only while its count is above zero.
    barriers: HashMap<RepoId, OpenBarriers>,
    /// `index --wait` watches, pushing each worktree's counts on change —
    /// one sender per waiting task, since waits no longer serialize on
    /// the owner.
    watchers: HashMap<WorktreeId, Vec<watch::Sender<Counts>>>,
    /// Query jobs whose request tasks have not reported yet.
    queries_in_flight: u64,
    /// The idle-exit tracker: every queued or in-flight unit of work
    /// reports busy through it.
    busy: watch::Sender<bool>,
    /// What the tracker was last told.
    last_busy: bool,
    stopping: bool,
}

/// What the scheduler is built from.
pub struct SchedulerConfig<E: Embeddings, Q: QueryInput> {
    /// The embedding server's client.
    pub embeddings: Arc<E>,
    /// The query input formatter.
    pub query_input: Arc<Q>,
    /// Fatal errors go to `run` here.
    pub fatal: mpsc::Sender<Error>,
    /// The scheduler's busy state for the idle-exit tracker.
    pub busy: watch::Sender<bool>,
}

/// Spawns the scheduler task; nothing leaves it until `Enqueue`s arrive.
pub fn spawn<E: Embeddings, Q: QueryInput>(config: SchedulerConfig<E, Q>) -> SchedulerHandle {
    let SchedulerConfig {
        embeddings,
        query_input,
        fatal,
        busy,
    } = config;
    let (tx, rx) = mpsc::channel(INBOX);
    let scheduler = Scheduler {
        tx: tx.downgrade(),
        embeddings,
        query_input,
        fatal_tx: fatal,
        fatal: None,
        stage1: BTreeMap::new(),
        background: BTreeMap::new(),
        items: HashMap::new(),
        by_hash: HashMap::new(),
        worktrees: HashMap::new(),
        writers: HashMap::new(),
        next_item: 0,
        flights: HashMap::new(),
        next_generation: 0,
        spots: HashMap::new(),
        next_spot: 0,
        held: HashMap::new(),
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
    });
    SchedulerHandle { tx }
}

impl<E: Embeddings, Q: QueryInput> Scheduler<E, Q> {
    async fn run(&mut self, mut rx: mpsc::Receiver<SchedulerMsg>) {
        while let Some(msg) = rx.recv().await {
            self.handle(msg).await;
            self.reap_spots();
            self.fill();
            self.push_counts();
            self.update_busy();
            if self.stopping && self.flights.is_empty() && self.queries_in_flight == 0 {
                // Stopped, and every request that was out has reported:
                // nothing can change any more.
                break;
            }
        }
        // Stopped and drained, or every sender is gone: the owner dropped
        // its handle and every request task ended.
        self.clear_busy();
    }

    async fn handle(&mut self, msg: SchedulerMsg) {
        match msg {
            SchedulerMsg::Enqueue(enqueue) => self.on_enqueue(*enqueue),
            SchedulerMsg::Query(job) => self.on_query(*job),
            SchedulerMsg::Documents(job) => self.on_documents(*job),
            SchedulerMsg::IndexDone { item, generation } => self.on_index_done(item, generation),
            SchedulerMsg::ServerFailed { error } => self.on_fatal(error).await,
            SchedulerMsg::DocumentsDone { spot, timing } => self.on_documents_done(spot, timing),
            SchedulerMsg::QueryDone { worktree, outcome } => {
                self.on_query_done(worktree, outcome).await
            }
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
                        // No `Enqueue` built from an earlier read can
                        // land.
                        self.barriers.remove(&repo);
                    }
                }
            }
            SchedulerMsg::WatchCounts { worktree, reply } => {
                let (watch, rx) = watch::channel(self.counts_for(worktree));
                self.watchers.entry(worktree).or_default().push(watch);
                let _ = reply.send(rx);
            }
            SchedulerMsg::Stop => {
                self.stopping = true;
                // Queued `Documents` work is failed and its holds go back
                // to indexing, as a dropped search's would.
                let spots: Vec<Spot> = self.spots.drain().map(|(_, spot)| spot).collect();
                for spot in spots {
                    spot.task.abort();
                    self.release_holds(spot.repo, spot.held);
                    let _ = spot.reply.send(Err(EmbedError::SchedulerGone));
                }
            }
        }
    }

    /// A `Documents` job whose reply went away — a dropped search — has
    /// its request aborted and its held chunks handed back to indexing.
    /// Runs after every handled message; a drop noticed only at
    /// `DocumentsDone` is still correct, since its vectors were written.
    fn reap_spots(&mut self) {
        let dropped: Vec<SpotId> = self
            .spots
            .iter()
            .filter(|(_, spot)| spot.reply.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in dropped {
            let spot = self.spots.remove(&id).expect("checked above");
            spot.task.abort();
            self.release_holds(spot.repo, spot.held);
        }
    }

    /// Refills the in-flight window after every message: up to
    /// `SHORT_SLOTS` flights off the short end and `LONG_SLOTS` off the
    /// long end, each queue taken from stage 1 before background.
    fn fill(&mut self) {
        if self.stopping {
            return;
        }
        let Some(tx) = self.tx.upgrade() else {
            // Every strong sender is gone: shutdown is under way.
            return;
        };
        for end in [End::Short, End::Long] {
            let slots = match end {
                End::Short => SHORT_SLOTS,
                End::Long => LONG_SLOTS,
            };
            while self.flights.values().filter(|f| f.end == end).count() < slots {
                let Some(item) = self.pop_queued(end) else {
                    break;
                };
                self.dispatch(item, end, &tx);
            }
        }
        debug_assert!(
            self.flights.len() <= IN_FLIGHT,
            "the in-flight window overran its bound"
        );
    }

    /// The next item for `end`: stage 1 first, else background — the
    /// shortest item for `Short`, the longest for `Long`.
    fn pop_queued(&mut self, end: End) -> Option<ItemId> {
        for queue in [Queue::Stage1, Queue::Background] {
            let map = self.queue_mut(queue);
            let key = match end {
                End::Short => map.first_key_value(),
                End::Long => map.last_key_value(),
            }
            .map(|(&key, &())| key);
            if let Some(key) = key {
                map.remove(&key);
                return Some(key.1);
            }
        }
        None
    }

    /// Sends `item`'s chunk to the server in its own request at its
    /// class's urgency and registers the flight.
    fn dispatch(&mut self, item: ItemId, end: End, tx: &mpsc::Sender<SchedulerMsg>) {
        self.set_queue(item, Queue::InFlight);
        let generation = self.next_generation;
        self.next_generation += 1;
        let entry = &self.items[&item];
        let input = DOCUMENT.format(&entry.chunk.title, &entry.chunk.text);
        let urgency = Urgency::new(match entry.class {
            IndexClass::Stage1 => URGENCY_STAGE1,
            IndexClass::Background => URGENCY_BACKGROUND,
        });
        let hash = entry.chunk.content_hash;
        let writer = self.writers[&entry.repo].clone();
        let embeddings = self.embeddings.clone();
        let tx = tx.clone();
        let task = tokio::spawn(async move {
            match embeddings.embed(vec![input], urgency).await {
                Ok(answered) => {
                    let vector = answered
                        .vectors
                        .into_iter()
                        .next()
                        .expect("one input embeds to one vector");
                    let _ = writer.vector(hash, vector).await;
                    let _ = tx.send(SchedulerMsg::IndexDone { item, generation }).await;
                }
                Err(error) => {
                    let _ = tx
                        .send(SchedulerMsg::ServerFailed {
                            error: error.to_string(),
                        })
                        .await;
                }
            }
        });
        self.flights.insert(
            item,
            Flight {
                end,
                generation,
                task: task.abort_handle(),
            },
        );
    }

    /// A flight's request landed: the vector is with the writer. Stale
    /// completions — of a flight aborted and requeued, or aborted for a
    /// search's `Documents` job — are ignored: the item may already be
    /// held or done by the newer flight.
    fn on_index_done(&mut self, item: ItemId, generation: u64) {
        let current = self
            .items
            .get(&item)
            .is_some_and(|i| i.queue == Queue::InFlight)
            && self
                .flights
                .get(&item)
                .is_some_and(|flight| flight.generation == generation);
        if !current {
            return;
        }
        self.flights.remove(&item);
        self.item_done(item);
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

    /// A `Replace` re-lists the worktree's whole interest: drop what it
    /// no longer names, then add what it does.
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
        // A `Stage1` enqueue that adds no small item keeps a past
        // completion: `Stage1Done` is sent once per stage-1 pass, not
        // once per enqueue.
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

    /// A `Change` edits the interest the previous `Enqueue` left: drop
    /// under the old counting, then add under the new class.
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

    /// `worktree` wants `pending`: creates the `(repo, hash)` item when
    /// new, and when the worktree was not yet interested adds it to the
    /// item, the tally and `stage1_left`, then re-derives the item's
    /// class.
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
                let queue = if self.held.contains_key(&key) {
                    // A search's `Documents` job is embedding this hash.
                    Queue::Held
                } else {
                    let queue = queue_of(class);
                    self.queue_mut(queue).insert((tokens, item), ());
                    queue
                };
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
        // This item was not yet counted: add it under its queue as it
        // sits now, before the refresh below may flip it.
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

    /// A worktree lost interest in `item`: drop it from the item's list
    /// and the item entirely when nobody is left.
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
        if !entry.worktrees.is_empty() {
            return;
        }
        if entry.queue == Queue::InFlight {
            // Nobody wants the vector: closing the request cancels it on
            // the server, where most of the window still sits queued.
            let flight = self.flights.remove(&item).expect("an in-flight item flies");
            flight.task.abort();
        }
        self.remove_item(item);
    }

    /// Re-derives an item's class from its worktrees, moves it between
    /// the queues on a change, and keeps the `stage1_left` counts right.
    /// A `Background`→`Stage1` flip of an in-flight item aborts its
    /// request — HTTP/1.1 cannot re-prioritize one — and requeues it so
    /// `fill` sends it at stage-1 urgency. The reverse flip leaves the
    /// flight alone: it finishes slightly early, which is harmless.
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
            (Queue::InFlight, IndexClass::Stage1) => {
                let flight = self.flights.remove(&item).expect("an in-flight item flies");
                flight.task.abort();
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
            Queue::InFlight | Queue::Held => {}
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
            Queue::InFlight | Queue::Held => {
                unreachable!("in-flight and held items live in no queue")
            }
        }
    }

    /// A query embedding job arrived: fail it while stopping, else send
    /// its request at `u=0` in a task of its own, outside the window.
    fn on_query(&mut self, job: QueryJob) {
        if self.stopping {
            let _ = job.reply.send(Err(self.stopping_error()));
            return;
        }
        let Some(tx) = self.tx.upgrade() else {
            let _ = job.reply.send(Err(EmbedError::SchedulerGone));
            return;
        };
        let input = self.query_input.input(job.task, &job.text);
        let embeddings = self.embeddings.clone();
        let worktree = job.worktree;
        self.queries_in_flight += 1;
        tokio::spawn(async move {
            match embeddings
                .embed(vec![input], Urgency::new(URGENCY_INTERACTIVE))
                .await
            {
                Ok(answered) => {
                    let vector = answered
                        .vectors
                        .into_iter()
                        .next()
                        .expect("one input embeds to one vector");
                    let _ = job.reply.send(Ok(vector));
                    let _ = tx
                        .send(SchedulerMsg::QueryDone {
                            worktree,
                            outcome: Ok(answered.timing),
                        })
                        .await;
                }
                Err(error) => {
                    let message = error.to_string();
                    let _ = job.reply.send(Err(EmbedError::Model(message.clone())));
                    let _ = tx
                        .send(SchedulerMsg::QueryDone {
                            worktree,
                            outcome: Err(message),
                        })
                        .await;
                }
            }
        });
    }

    /// What a job arriving while stopping is answered with.
    fn stopping_error(&self) -> EmbedError {
        match &self.fatal {
            Some(error) => EmbedError::Model(error.clone()),
            None => EmbedError::SchedulerGone,
        }
    }

    /// A search's `Documents` job arrived. Each chunk already has a
    /// vector (it got one inside the search's window), or is held away
    /// from indexing and embedded by the job's `u=0` request — an
    /// in-flight indexing request for it is aborted, since the daemon
    /// cannot tell whether the server is running it or it is still
    /// queued. The job is answered once its request lands.
    fn on_documents(&mut self, job: DocumentsJob) {
        let DocumentsJob {
            worktree,
            repo,
            chunks,
            writer,
            enqueued: _,
            reply,
        } = job;
        if self.stopping {
            let _ = reply.send(Err(self.stopping_error()));
            return;
        }
        let id = SpotId(self.next_spot);
        self.next_spot += 1;
        let open = self
            .barriers
            .get(&repo)
            .expect("a Documents job lands inside its search's window");
        let mut embed = Vec::new();
        let mut held = Vec::new();
        for chunk in chunks {
            let hash = chunk.content_hash;
            if open.committed.contains(&hash) {
                continue;
            }
            held.push(hash);
            embed.push(chunk);
        }
        for &hash in &held {
            self.hold(repo, hash);
        }
        if embed.is_empty() {
            let _ = reply.send(Ok(()));
            return;
        }
        let Some(tx) = self.tx.upgrade() else {
            self.release_holds(repo, held);
            let _ = reply.send(Err(EmbedError::SchedulerGone));
            return;
        };
        let inputs: Vec<(ContentHash, String)> = embed
            .iter()
            .map(|chunk| {
                (
                    chunk.content_hash,
                    DOCUMENT.format(&chunk.title, &chunk.text),
                )
            })
            .collect();
        let embeddings = self.embeddings.clone();
        let task = tokio::spawn(async move {
            let texts: Vec<String> = inputs.iter().map(|(_, text)| text.clone()).collect();
            match embeddings
                .embed(texts, Urgency::new(URGENCY_INTERACTIVE))
                .await
            {
                Ok(answered) => {
                    for ((hash, _), vector) in inputs.into_iter().zip(answered.vectors) {
                        let _ = writer.vector(hash, vector).await;
                    }
                    let _ = tx
                        .send(SchedulerMsg::DocumentsDone {
                            spot: id,
                            timing: answered.timing,
                        })
                        .await;
                }
                Err(error) => {
                    let _ = tx
                        .send(SchedulerMsg::ServerFailed {
                            error: error.to_string(),
                        })
                        .await;
                }
            }
        });
        self.spots.insert(
            id,
            Spot {
                worktree,
                repo,
                held,
                task: task.abort_handle(),
                reply,
            },
        );
    }

    /// A `Documents` job embeds `hash` of `repo`: no indexing request
    /// takes it until the job's request landed or the job is gone. An
    /// in-flight indexing request for it is aborted.
    fn hold(&mut self, repo: RepoId, hash: ContentHash) {
        *self.held.entry((repo, hash)).or_default() += 1;
        let Some(&item) = self.by_hash.get(&(repo, hash)) else {
            return;
        };
        let (queue, tokens) = {
            let entry = &self.items[&item];
            (entry.queue, entry.tokens)
        };
        match queue {
            Queue::Stage1 | Queue::Background => {
                self.queue_mut(queue).remove(&(tokens, item));
                self.set_queue(item, Queue::Held);
            }
            Queue::Held => {}
            Queue::InFlight => {
                let flight = self.flights.remove(&item).expect("an in-flight item flies");
                flight.task.abort();
                if self.items[&item].worktrees.is_empty() {
                    // Kept only for its flight: no worktree wants it, so
                    // the job's vector is all it needs, and a released
                    // hold must not requeue it.
                    self.remove_item(item);
                } else {
                    self.set_queue(item, Queue::Held);
                }
            }
        }
    }

    /// One job fewer holds `hash` of `repo`; `true` when none is left.
    fn unhold(&mut self, repo: RepoId, hash: ContentHash) -> bool {
        let key = (repo, hash);
        let count = self.held.get_mut(&key).expect("a held hash is counted");
        *count -= 1;
        if *count > 0 {
            return false;
        }
        self.held.remove(&key);
        true
    }

    /// A `Documents` job is gone unanswered: each of `held` goes back to
    /// its queue unless another job still holds it.
    fn release_holds(&mut self, repo: RepoId, held: Vec<ContentHash>) {
        for hash in held {
            if !self.unhold(repo, hash) {
                continue;
            }
            let Some(&item) = self.by_hash.get(&(repo, hash)) else {
                continue;
            };
            let (queue, tokens) = {
                let entry = &self.items[&item];
                assert_eq!(entry.queue, Queue::Held, "an item of a held hash is Held");
                (queue_of(entry.class), entry.tokens)
            };
            self.queue_mut(queue).insert((tokens, item), ());
            self.set_queue(item, queue);
        }
    }

    /// The job's request landed and its vectors went to the writer. Each
    /// held item is finished as an indexing item would be.
    fn on_documents_done(&mut self, id: SpotId, timing: ServerTiming) {
        let Some(spot) = self.spots.remove(&id) else {
            // The job was dropped or stopped before its request landed.
            return;
        };
        self.note_timing(spot.worktree, timing);
        let repo = spot.repo;
        for hash in spot.held {
            // The item finishes whether or not another job still holds
            // the hash: its vector is with the writer either way.
            self.unhold(repo, hash);
            self.note_committed(repo, hash);
            if let Some(&item) = self.by_hash.get(&(repo, hash)) {
                assert_eq!(
                    self.items[&item].queue,
                    Queue::Held,
                    "an item of a held hash is Held"
                );
                self.item_done(item);
            }
        }
        let _ = spot.reply.send(Ok(()));
    }

    /// A request finished `item`: the vector already went to the repo
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

    /// A query job's request ended: accumulate the worktree's stats, or
    /// go fatal on its failure.
    async fn on_query_done(&mut self, worktree: WorktreeId, outcome: Result<ServerTiming, String>) {
        self.queries_in_flight = self
            .queries_in_flight
            .checked_sub(1)
            .expect("a QueryDone without an in-flight query");
        match outcome {
            Ok(timing) => self.note_timing(worktree, timing),
            Err(error) => self.on_fatal(error).await,
        }
    }

    /// A request's `Server-Timing` into the worktree's stats.
    fn note_timing(&mut self, worktree: WorktreeId, timing: ServerTiming) {
        let stats = self.stats.entry(worktree).or_default();
        stats.wait = stats.wait.max(timing.wait);
        stats.overtaken = stats
            .overtaken
            .saturating_add(u32::try_from(timing.overtaken).unwrap_or(u32::MAX));
    }

    /// `hash` of `repo` has a vector, committed or on its way to the
    /// writer. While a barrier or search window of `repo` runs it is
    /// remembered, so neither an `Enqueue` built from a read inside that
    /// barrier nor a search's `Documents` job queues or embeds it again.
    /// Noting a hash twice is a no-op.
    fn note_committed(&mut self, repo: RepoId, hash: ContentHash) {
        if let Some(open) = self.barriers.get_mut(&repo) {
            open.committed.insert(hash);
        }
    }

    /// Runs stage-1 completion for `worktree` when its stage-1 items are
    /// all embedded.
    fn maybe_complete_stage1(&mut self, worktree: WorktreeId) {
        let Some(state) = self.worktrees.get(&worktree) else {
            return;
        };
        if !state.stage1_active() || state.stage1_left > 0 {
            return;
        }
        self.complete_stage1(worktree);
    }

    /// Stage 1 for `worktree` is done: its remaining stage-1 items move
    /// to background, then `Stage1Done` goes out once the repo writer
    /// flushed.
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

    /// A request failed past its retry, or the server answered non-2xx:
    /// answer every unanswered `Documents` job with the error, abort the
    /// indexing window — its items go back to their queues — log it, tell
    /// `run` and stop dispatching. There is no fallback. Requests already
    /// out can fail the same way after the first; only the first failure
    /// is reported.
    async fn on_fatal(&mut self, error: String) {
        if self.fatal.is_some() {
            debug!(%error, "a later request failed after the fatal error");
            return;
        }
        error!(%error, "the embedding server failed");
        self.fatal = Some(error.clone());
        self.stopping = true;
        for (_, spot) in self.spots.drain() {
            spot.task.abort();
            let _ = spot.reply.send(Err(EmbedError::Model(error.clone())));
        }
        let flights: Vec<(ItemId, Flight)> = self.flights.drain().collect();
        for (item, flight) in flights {
            flight.task.abort();
            let (queue, tokens) = {
                let entry = &self.items[&item];
                (queue_of(entry.class), entry.tokens)
            };
            self.queue_mut(queue).insert((tokens, item), ());
            self.set_queue(item, queue);
        }
        let _ = self.fatal_tx.send(Error::Server(error)).await;
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
    /// jobs, indexing items, unanswered `Documents` jobs and running
    /// barriers.
    fn update_busy(&mut self) {
        let busy = self.queries_in_flight > 0
            || !self.items.is_empty()
            || !self.spots.is_empty()
            || !self.barriers.is_empty();
        if busy != self.last_busy {
            self.last_busy = busy;
            self.busy.send_replace(busy);
        }
    }

    /// A stopped scheduler holds no work; clear the report so the
    /// idle-exit tracker never keeps a dead source.
    fn clear_busy(&mut self) {
        if self.last_busy {
            self.last_busy = false;
            self.busy.send_replace(false);
        }
    }
}
