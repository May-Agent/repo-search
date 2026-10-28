//! The hardware scheduler: the sole owner of all embedding work.
//!
//! One async task holds three urgency-keyed queues. `Interactive` (u=0) is
//! a FIFO served by the ANE only; `Normal` (u=1..=3) and `Background`
//! (u=4..=7) are token-length-ordered, the ANE drawing the shortest and the
//! GPU the longest in same-bucket batches — off `Normal` on a wide-class
//! chip, off `Background` on a base one, where it joins only once `Normal`
//! is empty. Every input of a request is one item; the request is answered
//! when its last item lands, vectors in input order, with the request's
//! `wait`/`overtaken` for its `Server-Timing`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use repo_search_core::UnitVector;
use repo_search_core::embed_api::{ServerTiming, Urgency};
use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{debug, error};

use crate::backend::{Backend, Headroom};
use crate::chip::ChipClass;
use crate::gpu;

/// Inbox capacity of the scheduler and every component channel.
const INBOX: usize = 64;

/// The largest batch the GPU thread is handed.
const GPU_BATCH: usize = 8;

/// An embedding request, as the requester numbered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

/// One input of a request, as the scheduler numbered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ItemId {
    /// The request the input belongs to.
    pub request: RequestId,
    /// The input's 0-based index inside the request.
    pub index: usize,
}

/// A request's urgency class: which queue its items run from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// u=0: FIFO on the ANE, never on the GPU.
    Interactive,
    /// u=1..=3: the ANE at user-initiated QoS, and the GPU on a wide chip.
    Normal,
    /// u=4..=7: the ANE at background QoS, and the GPU on a base chip.
    Background,
}

impl Class {
    fn of(urgency: Urgency) -> Self {
        match urgency.get() {
            0 => Self::Interactive,
            1..=3 => Self::Normal,
            _ => Self::Background,
        }
    }

    /// The ANE thread's QoS while embedding the class.
    fn qos(self) -> QosClass {
        match self {
            Self::Interactive => QosClass::UserInteractive,
            Self::Normal => QosClass::UserInitiated,
            Self::Background => QosClass::Background,
        }
    }
}

/// A request's answer: its vectors in input order and its timing.
#[derive(Debug)]
pub struct Answered {
    /// The inputs' vectors, in input order.
    pub vectors: Vec<UnitVector>,
    /// `wait` is the longest an item of the request spent behind
    /// lower-urgency work, `overtaken` the lower-urgency items it overtook.
    pub timing: ServerTiming,
}

/// A request handed to the scheduler: every input becomes one item.
#[derive(Debug)]
pub struct Submit {
    /// The request's id, assigned by the caller.
    pub request: RequestId,
    /// Its RFC 9218 urgency: 0 is most urgent.
    pub urgency: Urgency,
    /// One already-encoded input per vector wanted.
    pub inputs: Vec<Vec<u32>>,
    /// The answer goes here.
    pub reply: oneshot::Sender<Result<Answered, String>>,
}

/// One request the scheduler is tracking.
struct Pending {
    /// Unanswered items left.
    left: usize,
    /// The vectors, by input index; `Some` once the item landed.
    slots: Vec<Option<UnitVector>>,
    /// The longest wait any item reported at dispatch.
    wait: Duration,
    /// The overtaken count the items reported, summed.
    overtaken: u64,
    /// The answer goes here.
    reply: oneshot::Sender<Result<Answered, String>>,
}

/// One queued item.
struct Item {
    /// The input's token ids.
    ids: Vec<u32>,
    /// `ids.len()`, the queue key's length.
    tokens: u32,
    /// Which queue the item sits in.
    class: Class,
    /// When the request was handed to the scheduler.
    enqueued: Instant,
    /// `dispatched_low` at enqueue time, for a u=0 item's overtaken.
    stamp: u64,
}

/// A job the scheduler hands the ANE thread.
#[derive(Debug)]
pub enum AneJob {
    /// One input, at the QoS of the queue it came from.
    Item {
        /// The item, echoed back in `AneDone`.
        item: ItemId,
        /// Its token ids.
        ids: Vec<u32>,
        /// The thread's QoS for this job.
        qos: QosClass,
    },
    /// Drop the embedder and exit.
    Stop,
}

/// A job the scheduler hands the GPU thread: a same-bucket batch, up to
/// [`GPU_BATCH`] inputs.
#[derive(Debug)]
pub enum GpuJob {
    /// One batch prediction.
    Batch(Vec<(ItemId, Vec<u32>)>),
    /// Drop the embedder and exit.
    Stop,
}

/// The GPU instance's state, for `/status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuState {
    /// No GPU instance exists; `blocked` names why, when a check ran.
    Absent {
        /// Why the GPU cannot be opened, when known.
        blocked: Option<String>,
    },
    /// The GPU thread is opening its instance (a multi-second cold start).
    Loading,
    /// Embedding.
    Running,
}

/// The scheduler's inbox. Every component the pipeline owns talks to the
/// scheduler through this one channel.
#[derive(Debug)]
pub enum SchedulerMsg {
    /// A request to embed.
    Submit(Box<Submit>),
    /// Cancel a request: its queued items leave their queues, in-flight
    /// items' vectors are dropped.
    Cancel(RequestId),
    /// The ANE thread's pull for its next job.
    AnePull {
        /// The pulling thread, for QoS overrides.
        thread: qos::ThreadId,
        /// Where the next job is sent.
        reply: oneshot::Sender<AneJob>,
    },
    /// An item the ANE finished.
    AneDone {
        /// The completed item.
        item: ItemId,
        /// Its vector.
        vector: UnitVector,
        /// When the embed ended on the thread; a non-interactive item's
        /// `ended` starts a u=0 item's wait window.
        ended: Instant,
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
    /// A batch the GPU finished, in order.
    GpuDone {
        /// The completed items with their vectors.
        items: Vec<(ItemId, UnitVector)>,
    },
    /// An embedder call or an `open_gpu` failed: fatal.
    EmbedderFailed {
        /// What failed, for the log and waiting replies.
        error: String,
    },
    /// Stop dispatching; pulls are answered `Stop`.
    Stop,
}

/// The scheduler's handle; clones share the one inbox.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::Sender<SchedulerMsg>,
    /// A3: the GPU instance's state for `GET /status`, pushed on every
    /// transition.
    gpu: watch::Receiver<GpuState>,
}

impl SchedulerHandle {
    /// The raw inbox sender, for the embedding threads and components that
    /// talk [`SchedulerMsg`].
    pub fn tx(&self) -> mpsc::Sender<SchedulerMsg> {
        self.tx.clone()
    }

    /// The GPU instance's state, pushed on every transition; `/status`
    /// reads it through `borrow`.
    pub fn gpu_state(&self) -> watch::Receiver<GpuState> {
        self.gpu.clone()
    }

    /// Sends a request to embed.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn submit(&self, submit: Submit) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::Submit(Box::new(submit)))
            .await
            .map_err(|_| Closed)
    }

    /// Cancels a request.
    ///
    /// # Errors
    ///
    /// When the scheduler is gone.
    pub async fn cancel(&self, request: RequestId) -> Result<(), Closed> {
        self.tx
            .send(SchedulerMsg::Cancel(request))
            .await
            .map_err(|_| Closed)
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

/// Where the GPU instance is in its lifecycle.
#[derive(Debug)]
enum Gpu {
    Absent,
    /// `gpu::spawn` ran; `GpuStarted` has not arrived.
    Starting,
    Running,
}

struct Scheduler<B: Backend> {
    /// Weak so the actor can end: every component exits once its last
    /// strong sender is dropped.
    tx: mpsc::WeakSender<SchedulerMsg>,
    backend: Arc<B>,
    dir: PathBuf,
    buckets: Box<[usize]>,
    fatal_tx: mpsc::Sender<String>,
    /// The fatal error `run` was told about; submits arriving while
    /// stopping are answered with it.
    fatal: Option<String>,
    headroom: watch::Receiver<Headroom>,
    /// The host's chip class; it decides which queue the GPU draws from.
    chip: ChipClass,
    /// u=0 items queued for the ANE, FIFO.
    interactive: VecDeque<ItemId>,
    /// `(tokens, item)` ordered so `first` is the shortest and `last` the
    /// longest; `ItemId` is a sequence so the key is unique.
    normal: BTreeMap<(u32, ItemId), ()>,
    background: BTreeMap<(u32, ItemId), ()>,
    /// Every live item, queued or on hardware.
    items: HashMap<ItemId, Item>,
    /// Requests not yet answered; an item whose request is gone drops its
    /// vector.
    requests: HashMap<RequestId, Pending>,
    ane_waiting: Option<oneshot::Sender<AneJob>>,
    /// The ANE thread, from its last pull.
    ane_thread: Option<qos::ThreadId>,
    /// The item currently on the ANE thread, and its class.
    ane_in_flight: Option<(ItemId, Class)>,
    /// The override lifting the ANE thread while a u=0 item waits on its
    /// in-flight lower-class job; dropped on the next `AnePull`.
    ane_override: Option<qos::Override>,
    /// `Stop` was sent to the ANE thread, or the thread is gone.
    ane_stopped: bool,
    /// Fires when the ANE thread exits: a thread that died before its
    /// first pull can never be answered `Stop`, and a panic unwinds past
    /// the send.
    ane_exited: watch::Receiver<bool>,
    gpu: Gpu,
    /// Exit receivers of every GPU thread ever spawned; handed to the
    /// caller at exit so it can join them without blocking the executor.
    gpu_exited: Vec<oneshot::Receiver<()>>,
    /// Where the GPU exit receivers go when the actor exits.
    exit_tx: Option<oneshot::Sender<Vec<oneshot::Receiver<()>>>>,
    /// `open_gpu` was `Blocked`; no respawn until headroom changes.
    gpu_blocked: bool,
    /// Inputs in the GPU's current batch, for the busy report.
    gpu_in_flight: usize,
    /// Lower-class items handed to the ANE, ever; a u=0 item's `stamp`
    /// measures how many overtook it (must be zero).
    dispatched_low: u64,
    /// When the last lower-class ANE item ended; a u=0 item's wait is
    /// measured from it to its enqueue.
    last_low_end: Option<Instant>,
    /// The GPU state `/status` reads, pushed on every transition.
    gpu_state: watch::Sender<GpuState>,
    /// The idle-exit tracker: every queued or in-flight unit of work
    /// reports busy through it.
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
    /// The model's bucket lengths, ascending.
    pub buckets: Box<[usize]>,
    /// Host headroom for the GPU instance.
    pub headroom: watch::Receiver<Headroom>,
    /// The host's chip class: a base chip keeps the GPU off normal work.
    pub chip: ChipClass,
    /// Fatal errors go to `run` here.
    pub fatal: mpsc::Sender<String>,
    /// The scheduler's busy state for the idle-exit tracker.
    pub busy: watch::Sender<bool>,
    /// Fires when the ANE thread exits.
    pub ane_exited: watch::Receiver<bool>,
    /// A3: the GPU state `/status` reports; the scheduler pushes every
    /// transition.
    pub gpu_state: watch::Sender<GpuState>,
}

/// Spawns the scheduler task. The ANE thread is spawned separately by the
/// caller; the GPU thread is spawned here whenever work in its queue and
/// headroom allow it.
pub fn spawn<B: Backend>(
    config: SchedulerConfig<B>,
) -> (
    SchedulerHandle,
    oneshot::Receiver<Vec<oneshot::Receiver<()>>>,
) {
    let SchedulerConfig {
        backend,
        dir,
        buckets,
        headroom,
        chip,
        fatal,
        busy,
        ane_exited,
        gpu_state,
    } = config;
    let (tx, rx) = mpsc::channel(INBOX);
    let (exit_tx, exited) = oneshot::channel();
    let handle = SchedulerHandle {
        tx: tx.clone(),
        gpu: gpu_state.subscribe(),
    };
    let scheduler = Scheduler {
        tx: tx.downgrade(),
        backend,
        dir,
        buckets,
        fatal_tx: fatal,
        fatal: None,
        headroom,
        chip,
        interactive: VecDeque::new(),
        normal: BTreeMap::new(),
        background: BTreeMap::new(),
        items: HashMap::new(),
        requests: HashMap::new(),
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
        gpu_in_flight: 0,
        dispatched_low: 0,
        last_low_end: None,
        gpu_state,
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
    (handle, exited)
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
            self.push_gpu_state();
            self.update_busy();
            if self.stopping && self.ane_stopped && matches!(self.gpu, Gpu::Absent) {
                self.clear_busy();
                return;
            }
        }
    }

    async fn handle(&mut self, msg: SchedulerMsg) {
        match msg {
            SchedulerMsg::Submit(submit) => self.on_submit(*submit),
            SchedulerMsg::Cancel(request) => self.on_cancel(request),
            SchedulerMsg::AnePull { thread, reply } => {
                self.ane_thread = Some(thread);
                // The in-flight job is done; a held override ends.
                self.ane_override = None;
                self.on_ane_pull(reply);
            }
            SchedulerMsg::AneDone {
                item,
                vector,
                ended,
            } => {
                let Some((in_flight, class)) = self.ane_in_flight else {
                    panic!("AneDone for an item the ANE did not pull");
                };
                assert_eq!(in_flight, item, "AneDone for an item the ANE did not pull");
                self.ane_in_flight = None;
                if class != Class::Interactive {
                    self.last_low_end = Some(ended);
                }
                self.item_done(item, vector);
            }
            SchedulerMsg::GpuPull { reply } => self.on_gpu_pull(reply),
            SchedulerMsg::GpuStarted => {
                self.gpu = Gpu::Running;
                self.push_gpu_state();
            }
            SchedulerMsg::GpuGone => {
                self.gpu = Gpu::Absent;
                self.gpu_in_flight = 0;
                self.push_gpu_state();
                self.eval_gpu();
            }
            SchedulerMsg::GpuBlocked => {
                self.gpu = Gpu::Absent;
                self.gpu_blocked = true;
                self.push_gpu_state();
            }
            SchedulerMsg::GpuDone { items } => {
                self.gpu_in_flight = self
                    .gpu_in_flight
                    .checked_sub(items.len())
                    .expect("a GpuDone without an in-flight batch");
                for (item, vector) in items {
                    self.item_done(item, vector);
                }
            }
            SchedulerMsg::EmbedderFailed { error } => self.on_fatal(error).await,
            SchedulerMsg::Stop => {
                self.stopping = true;
                for (_, request) in self.requests.drain() {
                    let _ = request.reply.send(Err(error_message(&self.fatal)));
                }
                self.interactive.clear();
                self.normal.clear();
                self.background.clear();
                self.items.clear();
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
        let exited = gpu::spawn(self.backend.clone(), self.dir.clone(), tx);
        self.gpu = Gpu::Starting;
        self.push_gpu_state();
        self.gpu_exited.push(exited);
    }

    /// The GPU is wanted when its queue holds work, the host has headroom
    /// and the server is not stopping or blocked from the GPU. A wide-class
    /// chip serves it from the normal queue; a base-class one from the
    /// background queue, and only once the normal queue is empty — a GPU
    /// executing alongside the ANE there makes the normal queue finish
    /// later.
    fn gpu_wanted(&self) -> bool {
        let queued = match self.chip {
            ChipClass::Wide => self.live_work(&self.normal),
            ChipClass::Base => !self.live_work(&self.normal) && self.live_work(&self.background),
        };
        !self.stopping && !self.gpu_blocked && queued && self.headroom.borrow().is_ok()
    }

    /// Whether `queue` holds an item whose request is still waiting on it:
    /// an item whose request was cancelled or closed counts as nothing.
    fn live_work(&self, queue: &BTreeMap<(u32, ItemId), ()>) -> bool {
        queue.keys().any(|&(_, id)| {
            self.requests
                .get(&id.request)
                .is_some_and(|request| !request.reply.is_closed())
        })
    }

    /// The queue the GPU draws its batches from: normal on a wide-class
    /// chip, background on a base one.
    fn gpu_queue_mut(&mut self) -> &mut BTreeMap<(u32, ItemId), ()> {
        match self.chip {
            ChipClass::Wide => &mut self.normal,
            ChipClass::Base => &mut self.background,
        }
    }

    /// Registers a request: one item per input, in the queue of its
    /// urgency's class. A u=0 request also lifts the ANE thread while a
    /// lower-class job is in flight, so the thread's priority matches what
    /// it is about to run.
    fn on_submit(&mut self, submit: Submit) {
        let Submit {
            request,
            urgency,
            inputs,
            reply,
        } = submit;
        assert!(!inputs.is_empty(), "a request submits at least one input");
        if let Some(error) = &self.fatal {
            let _ = reply.send(Err(error.clone()));
            return;
        }
        if self.stopping {
            let _ = reply.send(Err("the embedding scheduler is gone".to_owned()));
            return;
        }
        let class = Class::of(urgency);
        let left = inputs.len();
        let mut slots = Vec::with_capacity(left);
        slots.resize_with(left, || None);
        self.requests.insert(
            request,
            Pending {
                left,
                slots,
                wait: Duration::ZERO,
                overtaken: 0,
                reply,
            },
        );
        let enqueued = Instant::now();
        for (index, ids) in inputs.into_iter().enumerate() {
            let id = ItemId { request, index };
            let tokens = u32::try_from(ids.len()).expect("an input fits u32");
            self.items.insert(
                id,
                Item {
                    ids,
                    tokens,
                    class,
                    enqueued,
                    stamp: self.dispatched_low,
                },
            );
            match class {
                Class::Interactive => self.interactive.push_back(id),
                Class::Normal => {
                    self.normal.insert((tokens, id), ());
                }
                Class::Background => {
                    self.background.insert((tokens, id), ());
                }
            }
        }
        self.lift_ane_for_interactive();
    }

    /// A u=0 item is queued while the ANE runs a lower class: lift the
    /// thread to user-interactive until it pulls next.
    fn lift_ane_for_interactive(&mut self) {
        if self.interactive.is_empty() {
            return;
        }
        let Some((_, class)) = self.ane_in_flight else {
            return;
        };
        if class == Class::Interactive || self.ane_override.is_some() {
            return;
        }
        if let Some(thread) = self.ane_thread {
            self.ane_override = Some(qos::Override::start(thread, QosClass::UserInteractive));
        }
    }

    /// Drops a request: its queued items leave their queues and its
    /// in-flight items' vectors are dropped when they land. The request's
    /// reply gets nothing — the caller cancelled it.
    fn on_cancel(&mut self, request: RequestId) {
        self.requests.remove(&request);
        self.interactive.retain(|id| id.request != request);
        self.normal.retain(|&(_, id), _| id.request != request);
        self.background.retain(|&(_, id), _| id.request != request);
        self.items.retain(|id, _| id.request != request);
    }

    /// The ANE asked for work: u=0 first, then normal shortest-first, then
    /// background. Anything else keeps the pull waiting.
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
        while let Some(item) = self.interactive.pop_front() {
            if self.purge_if_dead(item) {
                continue;
            }
            let entry = self.items.get(&item).expect("a queued item exists");
            let request = self
                .requests
                .get_mut(&item.request)
                .expect("a live item's request exists");
            let wait = self
                .last_low_end
                .map(|end| end.saturating_duration_since(entry.enqueued))
                .unwrap_or_default();
            request.wait = request.wait.max(wait);
            request.overtaken += self.dispatched_low - entry.stamp;
            let entry = self.items.remove(&item).expect("a queued item exists");
            self.ane_in_flight = Some((item, entry.class));
            let _ = reply.send(AneJob::Item {
                item,
                ids: entry.ids,
                qos: entry.class.qos(),
            });
            return;
        }
        for class in [Class::Normal, Class::Background] {
            loop {
                let item = {
                    let queue = match class {
                        Class::Normal => &mut self.normal,
                        Class::Background => &mut self.background,
                        Class::Interactive => unreachable!(),
                    };
                    let Some((&(_, item), &())) = queue.first_key_value() else {
                        break;
                    };
                    queue.pop_first();
                    item
                };
                if self.purge_if_dead(item) {
                    continue;
                }
                let entry = self.items.remove(&item).expect("a queued item exists");
                self.dispatched_low += 1;
                self.ane_in_flight = Some((item, entry.class));
                let _ = reply.send(AneJob::Item {
                    item,
                    ids: entry.ids,
                    qos: entry.class.qos(),
                });
                return;
            }
        }
        self.ane_waiting = Some(reply);
    }

    /// Whether `item`'s request is gone or its reply closed: the item is
    /// then purged from `items` and the caller skips it. A closed reply is
    /// the backstop for cancellation — a request whose connection dropped
    /// never embeds again.
    fn purge_if_dead(&mut self, item: ItemId) -> bool {
        let closed = match self.requests.get(&item.request) {
            None => true,
            Some(request) => request.reply.is_closed(),
        };
        if closed {
            self.items.remove(&item);
        }
        closed
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
        // The batch's bucket is the longest queued item's; up to
        // `GPU_BATCH` items off the back of the GPU's queue that share it go
        // in one batch.
        let mut batch = Vec::with_capacity(GPU_BATCH);
        let mut bucket = None;
        while batch.len() < GPU_BATCH {
            let item = {
                let queue = self.gpu_queue_mut();
                let Some((&(_, item), &())) = queue.last_key_value() else {
                    break;
                };
                queue.pop_last();
                item
            };
            if self.purge_if_dead(item) {
                continue;
            }
            let item_bucket = self.bucket(self.items[&item].tokens);
            match bucket {
                None => bucket = Some(item_bucket),
                Some(b) if b == item_bucket => {}
                Some(_) => {
                    // The next batch's bucket: the item goes back.
                    let tokens = self.items[&item].tokens;
                    self.gpu_queue_mut().insert((tokens, item), ());
                    break;
                }
            }
            let entry = self.items.remove(&item).expect("a queued item exists");
            batch.push((item, entry.ids));
        }
        if batch.is_empty() {
            let _ = reply.send(GpuJob::Stop);
        } else {
            self.gpu_in_flight = batch.len();
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

    /// An embedder returned `item`'s vector: land it in the request's slot
    /// and, when the last item lands, answer the request.
    fn item_done(&mut self, item: ItemId, vector: UnitVector) {
        let Some(request) = self.requests.get_mut(&item.request) else {
            return;
        };
        request.slots[item.index] = Some(vector);
        request.left = request
            .left
            .checked_sub(1)
            .expect("a request's items underflow");
        if request.left > 0 {
            return;
        }
        let request = self
            .requests
            .remove(&item.request)
            .expect("the request is unanswered");
        let vectors = request
            .slots
            .into_iter()
            .map(|slot| slot.expect("an answered item has a vector"))
            .collect();
        let _ = request.reply.send(Ok(Answered {
            vectors,
            timing: ServerTiming {
                wait: request.wait,
                overtaken: request.overtaken,
            },
        }));
    }

    /// An embedder failure: answer every waiting request with it, log it,
    /// tell `run` and stop dispatching. There is no retry and no fallback.
    async fn on_fatal(&mut self, error: String) {
        error!(%error, "embedder failed");
        for (_, request) in self.requests.drain() {
            let _ = request.reply.send(Err(error.clone()));
        }
        self.interactive.clear();
        self.normal.clear();
        self.background.clear();
        self.items.clear();
        self.fatal = Some(error.clone());
        self.stopping = true;
        self.ane_override = None;
        if let Some(reply) = self.ane_waiting.take() {
            let _ = reply.send(AneJob::Stop);
            self.ane_stopped = true;
        }
        let _ = self.fatal_tx.send(error).await;
    }

    /// Pushes the GPU state `/status` reports: `Loading` while the instance
    /// opens, `Running` when it embeds, `Absent` with the headroom's reason
    /// otherwise.
    fn push_gpu_state(&mut self) {
        let state = match self.gpu {
            Gpu::Absent => GpuState::Absent {
                blocked: self.headroom.borrow().clone().err().map(|b| b.to_string()),
            },
            Gpu::Starting => GpuState::Loading,
            Gpu::Running => GpuState::Running,
        };
        self.gpu_state.send_if_modified(|current| {
            if *current == state {
                return false;
            }
            *current = state;
            true
        });
    }

    /// A pending request or hardware work in flight keeps the server busy:
    /// queued items live in their requests, and the ANE/GPU each track the
    /// items already handed out.
    fn update_busy(&mut self) {
        let busy =
            !self.requests.is_empty() || self.ane_in_flight.is_some() || self.gpu_in_flight > 0;
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

/// The message a drained or rejected request's reply carries: the fatal
/// error when there is one, else the scheduler's gone.
fn error_message(fatal: &Option<String>) -> String {
    fatal
        .clone()
        .unwrap_or_else(|| "the embedding scheduler is gone".to_owned())
}
