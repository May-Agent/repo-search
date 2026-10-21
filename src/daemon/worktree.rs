//! The per-worktree owner task.
//!
//! One task per worktree handles its inbox one message at a time, which the
//! `Retriever`'s `&mut` search requires. It owns the index, the retriever
//! built on [`AneHandle`], the last barrier's report, the FSEvents watcher,
//! the HEAD commit and the index phase, and runs the freshness barrier,
//! whose changes its feeder orders into the scheduler. Its spawned
//! `index --wait` tasks and its feeder read off its books by asking it.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use repo_search_core::qos::performance_cores;
use repo_search_core::reconcile::{self, Counts, PendingChunk, Reconciler, Report};
use repo_search_core::search::{self, EmbedService, Reconciliation, Retriever};
use repo_search_core::store::{Index, Phase, VectorCache};
use repo_search_core::{CHUNKER_FINGERPRINT, QueryTask, TokenCounter, UnitVector};
use tokio::sync::{mpsc, oneshot, watch};

use crate::protocol::{
    Answer, ErrorKind, ErrorReport, Event, Final, Format, IndexStatus, ModelStatus, Progress,
    QueryStats, SearchRequest, StatusReport, WirePhase,
};

use super::Backend;
use super::coverage::{self, Ask, AskError, Asker};
use super::feed::{self, Feed, FeedMsg, FeederConfig};
use super::model::{Model, ModelState};
use super::registry::RegistryHandle;
use super::scheduler::{
    DocumentsJob, EmbedError, GpuState, IndexClass, QueryJob, RepoId, SchedulerHandle,
    SchedulerMsg, Stage1Done, WorktreeId,
};
use super::watch::{WatchMsg, Watcher};
use super::writer::WriterHandle;

/// `index --wait` coverage frames are sent at most this often.
const COVERAGE_INTERVAL: Duration = Duration::from_millis(250);

/// The owner's inbox: one message per connection's request.
#[derive(Debug)]
pub enum OwnerMsg {
    /// A `search` request.
    Search {
        /// The wire request.
        request: SearchRequest,
        /// Progress and the closing `Final` go here.
        events: mpsc::Sender<Event>,
    },
    /// An `index` request.
    Index {
        /// Whether to wait for full vector coverage.
        wait: bool,
        /// Events sink.
        events: mpsc::Sender<Event>,
    },
    /// A `status` request; never creates `.repo-search/`.
    Status {
        /// Events sink.
        events: mpsc::Sender<Event>,
    },
}

impl OwnerMsg {
    fn events(&self) -> mpsc::Sender<Event> {
        match self {
            Self::Search { events, .. } | Self::Index { events, .. } | Self::Status { events } => {
                events.clone()
            }
        }
    }
}

/// Everything the registry hands a new owner.
pub struct OwnerConfig<B: Backend> {
    /// The scheduler's worktree id.
    pub id: WorktreeId,
    /// Canonical worktree root.
    pub root: PathBuf,
    /// The repository's canonical git common dir.
    pub common_dir: PathBuf,
    /// The worktree's git dir (`--absolute-git-dir`).
    pub git_dir: PathBuf,
    /// The scheduler's repository id.
    pub repo: RepoId,
    /// The model-id vectors are stored under; known before the model loads.
    pub model_id: String,
    /// The model readiness watch.
    pub model: watch::Receiver<ModelState<B>>,
    /// The registry, for the shared cache and writer, `DaemonStatus`'s
    /// worktree count, and `Forget`.
    pub registry: RegistryHandle,
    /// Daemon start, for `DaemonStatus::uptime_s`.
    pub started: Instant,
    /// Set when the daemon begins shutdown.
    pub stopping: watch::Receiver<bool>,
}

/// Spawns the owner task; the returned sender is its inbox.
pub fn spawn<B: Backend>(config: OwnerConfig<B>) -> mpsc::Sender<OwnerMsg> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(Owner::new(config, rx).run());
    tx
}

struct Owner<B: Backend> {
    id: WorktreeId,
    root: PathBuf,
    common_dir: PathBuf,
    git_dir: PathBuf,
    repo: RepoId,
    cache: Option<VectorCache>,
    writer: Option<WriterHandle>,
    model_id: String,
    model: watch::Receiver<ModelState<B>>,
    registry: RegistryHandle,
    started: Instant,
    stopping: watch::Receiver<bool>,
    rx: mpsc::Receiver<OwnerMsg>,
    index: Option<Index>,
    /// `(dev, ino)` of `index.db` as opened; a recreated file means the
    /// handle points at an unlinked inode.
    index_id: Option<(u64, u64)>,
    phase: Phase,
    retriever: Option<Retriever<AneHandle>>,
    report: Option<Report>,
    commit: String,
    watcher: Option<Watcher>,
    /// The scope whose whole pending set the feeder was last told to hold
    /// (a `Resync`), every later pass's changes following it; `None` until
    /// the first barrier and after anything that may have broken that —
    /// a failed barrier or a reopened index.
    sync: Option<reconcile::Options>,
    /// Reconciles `index`, keeping its books between barriers; built at
    /// the first barrier or `Status` and dropped with the index handle it
    /// holds. Its books answer the coverage of `Status` and of
    /// `index --wait`'s frames and every resync's set, and every barrier's
    /// presence is passed on to the retriever, which never reads the
    /// index's hashes itself.
    reconciler: Option<Reconciler>,
    /// The feeder, spawned at the first barrier.
    feeder: Option<mpsc::Sender<feed::FeedMsg>>,
    stage1_tx: mpsc::Sender<Stage1Done>,
    stage1_rx: mpsc::Receiver<Stage1Done>,
    /// The applied phase, published to `index --wait` tasks spawned off
    /// the owner.
    phase_tx: watch::Sender<Phase>,
    /// The spawned `index --wait` tasks the owner has not yet seen finish.
    waits: tokio::task::JoinSet<()>,
    /// Handed to each `index --wait` task; kept so `asks` never closes.
    asker: Asker,
    /// The waits' and the feeder's asks, answered from the reconciler's
    /// books.
    asks: mpsc::Receiver<Ask>,
}

impl<B: Backend> Owner<B> {
    fn new(config: OwnerConfig<B>, rx: mpsc::Receiver<OwnerMsg>) -> Self {
        let (stage1_tx, stage1_rx) = mpsc::channel(8);
        let (phase_tx, _) = watch::channel(Phase::FirstIndex);
        let (asker, asks) = coverage::channel();
        Self {
            id: config.id,
            root: config.root,
            common_dir: config.common_dir,
            git_dir: config.git_dir,
            repo: config.repo,
            cache: None,
            writer: None,
            model_id: config.model_id,
            model: config.model,
            registry: config.registry,
            started: config.started,
            stopping: config.stopping,
            rx,
            index: None,
            index_id: None,
            phase: Phase::FirstIndex,
            retriever: None,
            report: None,
            commit: String::new(),
            watcher: None,
            sync: None,
            reconciler: None,
            feeder: None,
            stage1_tx,
            stage1_rx,
            phase_tx,
            waits: tokio::task::JoinSet::new(),
            asker,
            asks,
        }
    }

    async fn run(mut self) {
        'run: loop {
            let mut stopping = self.stopping.clone();
            tokio::select! {
                biased;
                done = self.stage1_rx.recv() => {
                    if done.is_none() {
                        break 'run;
                    }
                    self.apply_stage1_done().await;
                }
                // Before the inbox: a frame's ask is cheap, and a stream of
                // requests must not hold a wait's frames back.
                Some(ask) = self.asks.recv() => self.answer(ask).await,
                msg = self.rx.recv() => {
                    let Some(msg) = msg else { break 'run };
                    let events = msg.events();
                    // The request runs to completion; a client hang-up cancels
                    // only the parts that select on `events.closed()` — never
                    // a barrier, whose busy marker must be cleared by its own
                    // completion message.
                    let quit = tokio::select! {
                        _ = self.handle(msg) => false,
                        _ = stopping.changed() => {
                            send(&events, shutting_down()).await;
                            true
                        }
                    };
                    if quit {
                        break 'run;
                    }
                }
                msg = async {
                    match &mut self.watcher {
                        Some(watcher) => watcher.changed.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match msg {
                        Some(WatchMsg::Changed) => self.changed().await,
                        Some(WatchMsg::Gone) => {
                            // The worktree root is gone: answer the queued
                            // requests, tell the registry to forget the owner
                            // and exit — the next request spawns a fresh one.
                            self.rx.close();
                            while let Ok(msg) = self.rx.try_recv() {
                                send(&msg.events(), shutting_down()).await;
                            }
                            self.registry.forget(self.root.clone(), self.id).await;
                            break 'run;
                        }
                        // The debounce task died with the watcher; it cannot
                        // happen while the `FsEventWatcher` lives.
                        None => {
                            tracing::error!(
                                root = %self.root.display(),
                                "the watcher ended; edits stop being embedded in the background"
                            );
                            self.watcher = None;
                        }
                    }
                }
                _ = self.stopping.changed() => break 'run,
            }
        }
        // The owner does not leave while a spawned `index --wait` is still
        // answering. A wait parked on the phase watch completes only once
        // its `Stage1Done` is applied, a wait's frames and `Indexed` are
        // counted off the books, and its ack may wait on a feeder whose
        // promoted resync asks the books, so the drain keeps serving both.
        let mut stage1_open = true;
        while !self.waits.is_empty() {
            tokio::select! {
                done = self.stage1_rx.recv(), if stage1_open => {
                    match done {
                        Some(_) => self.apply_stage1_done().await,
                        None => stage1_open = false,
                    }
                }
                Some(ask) = self.asks.recv() => self.answer(ask).await,
                joined = self.waits.join_next() => {
                    if joined.is_none() {
                        break;
                    }
                }
            }
        }
    }

    /// The `Ready` model, or `None` while it is still on its way.
    fn ready(&self) -> Option<Arc<Model<B>>> {
        match &*self.model.borrow() {
            ModelState::Ready(model) => Some(model.clone()),
            _ => None,
        }
    }

    /// The repository's shared cache and writer, opened on first need so a
    /// `Status` on a never-indexed repository creates nothing.
    async fn shared(&mut self) -> Result<(VectorCache, WriterHandle), String> {
        if self.cache.is_none() {
            let shared = self
                .registry
                .shared(self.common_dir.clone())
                .await
                .ok_or_else(|| "the daemon is shutting down".to_owned())?
                .map_err(|e| e.to_string())?;
            self.cache = Some(shared.cache);
            self.writer = Some(shared.writer);
        }
        Ok((
            self.cache.clone().expect("opened above"),
            self.writer.clone().expect("opened above"),
        ))
    }

    /// Opens the index on first use, reading the stored phase and the file's
    /// identity; a `Status` on a worktree with no index file never reaches
    /// here.
    async fn ensure_index(&mut self) -> Result<(), String> {
        if self.index.is_none() {
            let index = Index::open(&self.root).await.map_err(|e| e.to_string())?;
            let meta = tokio::fs::metadata(self.root.join(".repo-search/index.db"))
                .await
                .map_err(|e| e.to_string())?;
            let phase = index.phase().await.map_err(|e| e.to_string())?;
            self.index_id = Some(file_id(&meta));
            self.index = Some(index);
            self.set_phase(phase);
        }
        Ok(())
    }

    /// Before every barrier: the index file must still be the inode it was
    /// opened with — a recreated `.repo-search/` leaves the handle writing
    /// into an unlinked file. On a mismatch the index, its reconciler and the
    /// retriever are dropped and reopened.
    async fn verify_index(&mut self) -> Result<(), String> {
        let Some(id) = self.index_id else {
            return Ok(());
        };
        let meta = tokio::fs::metadata(self.root.join(".repo-search/index.db")).await;
        if matches!(&meta, Ok(meta) if file_id(meta) == id) {
            return Ok(());
        }
        self.index = None;
        self.retriever = None;
        self.index_id = None;
        self.sync = None;
        self.reconciler = None;
        self.set_phase(Phase::FirstIndex);
        self.ensure_index().await
    }

    /// The watcher starts on the first `Search` or `Index`, never on a
    /// `Status`. A watcher that fails to start fails the request: without
    /// it edits are never embedded in the background.
    fn ensure_watcher(&mut self) -> Result<(), String> {
        if self.watcher.is_none() {
            self.watcher =
                Some(super::watch::start(&self.root, &self.git_dir).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    async fn handle(&mut self, msg: OwnerMsg) {
        match msg {
            OwnerMsg::Search { request, events } => self.search(request, &events).await,
            OwnerMsg::Index { wait, events } => self.index_request(wait, &events).await,
            OwnerMsg::Status { events } => self.status(&events).await,
        }
    }

    async fn search(&mut self, request: SearchRequest, events: &mpsc::Sender<Event>) {
        let total_start = Instant::now();
        let Some(model) = self.ready() else {
            send(
                events,
                error(ErrorKind::Model, "the model is not ready".into()),
            )
            .await;
            return;
        };
        if let Err(message) = self.ensure_index().await {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        if let Err(message) = self.ensure_watcher() {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        if self.shared().await.is_err() {
            send(
                events,
                error(
                    ErrorKind::ShuttingDown,
                    "the daemon is shutting down".into(),
                ),
            )
            .await;
            return;
        }
        if !send(events, Event::Progress(Progress::Reconciling)).await {
            return;
        }
        let core_request = search::Request {
            query: request.query,
            mode: request.mode.into(),
            filters: search::Filters {
                paths: request.paths,
                langs: request.langs,
                kinds: request.kinds,
                symbol: request.symbol,
            },
            budget: request.budget as usize,
            files_only: request.files_only,
            include_ignored: request.include_ignored,
        };
        let sched = model.scheduler.clone();
        // The search's window: it opens before the freshness barrier's
        // writer flush, so a vector the retrieval's cache reads miss is
        // remembered until the retrieval's `Documents` job lands, and it
        // closes after that job is answered (see `DocumentsJob`).
        sched.barrier(self.repo, true).await;
        let retrieved = self
            .barrier_and_retrieve(&model, &core_request, events)
            .await;
        sched.barrier(self.repo, false).await;
        let Some(Retrieved { barrier_us, result }) = retrieved else {
            return;
        };
        // The stats are taken on every path so a failed search cannot leak
        // its wait or overtaken into the next one.
        let stats = match self.take_stats(&model).await {
            Ok(stats) => stats,
            Err(()) => {
                send(
                    events,
                    error(ErrorKind::ShuttingDown, "scheduler gone".into()),
                )
                .await;
                return;
            }
        };
        let Some(result) = result else {
            return;
        };
        let results = match result {
            Ok(results) => results,
            Err(e) => {
                let kind = match &e {
                    search::Error::Embed(EmbedError::Model(_)) => ErrorKind::Model,
                    search::Error::Embed(EmbedError::SchedulerGone) => ErrorKind::ShuttingDown,
                    search::Error::Embed(EmbedError::Writer(_)) => ErrorKind::Internal,
                    _ => ErrorKind::Search,
                };
                send(events, error(kind, e.to_string())).await;
                return;
            }
        };
        let total_us = total_start.elapsed().as_micros() as u64;
        tracing::info!(root = %self.root.display(), barrier_us, total_us, "search answered");
        let answer = Answer {
            output: match request.format {
                Format::Text => results.to_text(),
                Format::Json => results.to_json(),
            },
            has_results: !results.is_empty(),
            stats: QueryStats {
                barrier_us,
                indexing_wait_us: stats.wait.as_micros() as u64,
                indexing_overtaken: stats.overtaken,
                embedded_on_the_spot: results.embedded_on_the_spot,
                total_us,
            },
        };
        let _ = send(events, Event::Final(Final::Answer(answer))).await;
    }

    /// The freshness barrier, then the retrieval, inside the search's
    /// window the caller holds open. `None` when an error event was
    /// already sent.
    async fn barrier_and_retrieve(
        &mut self,
        model: &Arc<Model<B>>,
        request: &search::Request,
        events: &mpsc::Sender<Event>,
    ) -> Option<Retrieved> {
        let barrier_start = Instant::now();
        let (report, commit) = match self.barrier(model, request.include_ignored, None).await {
            Ok(done) => done,
            Err(message) => {
                send(events, error(ErrorKind::Search, message)).await;
                return None;
            }
        };
        let barrier_us = barrier_start.elapsed().as_micros() as u64;

        if self.retriever.is_none() {
            let knn_threads = match performance_cores() {
                Ok(cores) => NonZeroUsize::new(cores),
                Err(e) => {
                    send(events, error(ErrorKind::Internal, e.to_string())).await;
                    return None;
                }
            };
            let Some(knn_threads) = knn_threads else {
                send(
                    events,
                    error(ErrorKind::Internal, "no performance cores".into()),
                )
                .await;
                return None;
            };
            let handle = AneHandle {
                worktree: self.id,
                repo: self.repo,
                model_id: model.id.clone(),
                sched: model.scheduler.tx(),
                writer: self.writer.clone().expect("opened by shared"),
            };
            // Seeded from the books the barrier just left; every later
            // barrier's presence follows (see `barrier_steps`).
            let pairs = self
                .reconciler
                .as_ref()
                .and_then(Reconciler::books)
                .expect("a barrier that succeeded leaves books")
                .pairs();
            match Retriever::new(
                self.index.as_ref().expect("opened").clone(),
                self.cache.clone().expect("opened above"),
                handle,
                self.root.clone(),
                knn_threads,
                pairs,
            ) {
                Ok(retriever) => self.retriever = Some(retriever),
                Err(e) => {
                    send(events, error(ErrorKind::Internal, e.to_string())).await;
                    return None;
                }
            }
        }
        // Reset the accumulated query stats so a search reports only its own.
        if let Err(()) = self.reset_stats(model).await {
            send(
                events,
                error(ErrorKind::ShuttingDown, "scheduler gone".into()),
            )
            .await;
            return None;
        }
        let counter = BudgetCounter::<B>(model.counter.clone());
        // Only the retrieval cancels on a hang-up: the barrier above always
        // runs to completion.
        let result = tokio::select! {
            result = self
                .retriever
                .as_mut()
                .expect("built above")
                .search(
                    request,
                    Reconciliation::new(commit, &report),
                    &counter,
                ) => Some(result),
            _ = events.closed() => None,
        };
        Some(Retrieved { barrier_us, result })
    }

    /// Drops this worktree's accumulated query stats; `Err` when the
    /// scheduler is gone.
    async fn reset_stats(&self, model: &Arc<Model<B>>) -> Result<(), ()> {
        model
            .scheduler
            .take_stats(self.id)
            .await
            .map(|_| ())
            .map_err(|_| ())
    }

    /// Takes this worktree's accumulated query stats; `Err` when the
    /// scheduler is gone.
    async fn take_stats(&self, model: &Arc<Model<B>>) -> Result<super::scheduler::AneStats, ()> {
        model.scheduler.take_stats(self.id).await.map_err(|_| ())
    }

    async fn index_request(&mut self, wait: bool, events: &mpsc::Sender<Event>) {
        let Some(model) = self.ready() else {
            send(
                events,
                error(ErrorKind::Model, "the model is not ready".into()),
            )
            .await;
            return;
        };
        if let Err(message) = self.ensure_index().await {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        if let Err(message) = self.ensure_watcher() {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        if self.shared().await.is_err() {
            send(
                events,
                error(
                    ErrorKind::ShuttingDown,
                    "the daemon is shutting down".into(),
                ),
            )
            .await;
            return;
        }
        if !send(events, Event::Progress(Progress::Reconciling)).await {
            return;
        }
        let (ack_tx, ack_rx) = oneshot::channel::<Result<(), String>>();
        let (report, _) = match self.barrier(&model, false, Some(ack_tx)).await {
            Ok(done) => done,
            Err(message) => {
                send(events, error(ErrorKind::Internal, message)).await;
                return;
            }
        };
        if !wait {
            let _ = send(
                events,
                Event::Final(Final::Indexed {
                    with_vector: report.chunks_with_vector,
                    total: report.chunks_total,
                }),
            )
            .await;
            return;
        }
        // The wait must not occupy the owner — a `Search` or `Status` for
        // this worktree answers while indexing runs. It moves into a
        // spawned task the owner joins before leaving.
        self.waits.spawn(index_wait(IndexWait {
            events: events.clone(),
            sched: model.scheduler.clone(),
            worktree: self.id,
            ack: ack_rx,
            coverage: self.asker.clone(),
            writer: self.writer.as_ref().expect("opened").clone(),
            phase: self.phase_tx.subscribe(),
            stopping: self.stopping.clone(),
        }));
    }

    async fn status(&mut self, events: &mpsc::Sender<Event>) {
        let db = self.root.join(".repo-search/index.db");
        let index_exists = match tokio::fs::try_exists(&db).await {
            Ok(exists) => exists,
            Err(e) => {
                send(events, error(ErrorKind::Internal, e.to_string())).await;
                return;
            }
        };
        // A `Status` never creates an index: only an existing file is opened,
        // and an opened handle is checked against the file on disk.
        if index_exists && let Err(message) = self.verify_index().await {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        if index_exists && let Err(message) = self.ensure_index().await {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        let sched = self.ready().map(|m| m.scheduler.clone());
        let id = self.id;
        let counts = async {
            match &sched {
                Some(sched) => sched.counts_for(id).await,
                None => Ok(Default::default()),
            }
        };
        let gpu = async {
            match &sched {
                Some(sched) => sched.gpu_status().await,
                None => Ok(GpuState::Absent { blocked: None }),
            }
        };
        let index = if index_exists {
            self.index.clone()
        } else {
            None
        };
        let phase = async {
            match &index {
                Some(index) => Some(index.phase().await),
                None => None,
            }
        };
        // Independent reads run together; the coverage counts wait on the
        // stage-1 completion below.
        let (phase, counts, gpu, worktrees) =
            tokio::join!(phase, counts, gpu, self.registry.count());
        let (Ok(counts), Ok(gpu), Some(worktrees)) = (counts, gpu, worktrees) else {
            send(events, shutting_down()).await;
            return;
        };
        let index_status = match phase {
            None => None,
            Some(phase) => {
                let phase = match phase {
                    Ok(phase) => phase,
                    Err(e) => {
                        send(events, error(ErrorKind::Internal, e.to_string())).await;
                        return;
                    }
                };
                // The scheduler already completed stage 1 but its
                // `Stage1Done` has not been applied: wait for it so `Status`
                // never reports a phase older than the scheduler's own.
                let mut phase = phase;
                if counts.stage1_done && phase == Phase::FirstIndex {
                    if self.stage1_rx.recv().await.is_none() {
                        send(events, shutting_down()).await;
                        return;
                    }
                    self.apply_stage1_done().await;
                    phase = self.phase;
                }
                if let Err(message) = self.shared().await {
                    send(events, error(ErrorKind::Internal, message)).await;
                    return;
                }
                // From the books the barriers keep: no chunk row is read and
                // no hash looked up, only the vectors stored since.
                let coverage = match self.reconciler().counts(false).await {
                    Ok(coverage) => coverage,
                    Err(e) => {
                        send(events, error(ErrorKind::Internal, e.to_string())).await;
                        return;
                    }
                };
                tracing::debug!(root = %self.root.display(),
                                chunk_rows = coverage.reads.chunk_rows,
                                cache_probes = coverage.reads.cache_probes,
                                cache_slots = coverage.reads.cache_slots, "status counted");
                Some(IndexStatus {
                    phase: WirePhase::from(phase),
                    chunks_total: coverage.chunks_total,
                    chunks_with_vector: coverage.chunks_with_vector,
                    queued_stage1: counts.queued_stage1,
                    queued_background: counts.queued_background,
                    in_flight: counts.in_flight,
                })
            }
        };
        let model = match &*self.model.borrow() {
            ModelState::Absent => ModelStatus::Downloading {
                received: 0,
                total: None,
            },
            ModelState::Downloading { received, total } => ModelStatus::Downloading {
                received: *received,
                total: *total,
            },
            ModelState::Loading => ModelStatus::Loading,
            ModelState::Ready(model) => ModelStatus::Ready {
                model_id: model.id.clone(),
            },
            ModelState::Failed(message) | ModelState::Fatal(message) => ModelStatus::Failed {
                message: message.clone(),
            },
        };
        let gpu = match gpu {
            GpuState::Absent { blocked } => crate::protocol::GpuStatus::Absent { blocked },
            GpuState::Loading => crate::protocol::GpuStatus::Loading,
            GpuState::Running => crate::protocol::GpuStatus::Running,
        };
        let report = StatusReport {
            worktree: self.root.clone(),
            index: index_status,
            model,
            daemon: crate::protocol::DaemonStatus {
                version: env!("CARGO_PKG_VERSION").to_owned(),
                pid: std::process::id(),
                uptime_s: self.started.elapsed().as_secs(),
                worktrees,
                gpu,
            },
        };
        let _ = send(events, Event::Final(Final::Status(report))).await;
    }

    /// The freshness barrier: flush the writer, then `reconcile` and
    /// `git rev-parse HEAD` concurrently, then hand the pass's outcome to
    /// the feeder, which orders it into the scheduler. It never cancels:
    /// its busy marker is cleared by the feeder, or here on failure.
    /// `Barrier { running: true }` goes out before the writer flush, so a
    /// vector the flush leaves uncommitted is reported to the scheduler
    /// after the barrier began, and the scheduler drops it from this
    /// barrier's `Enqueue` if the read missed it.
    async fn barrier(
        &mut self,
        model: &Arc<Model<B>>,
        include_ignored: bool,
        ack: Option<oneshot::Sender<Result<(), String>>>,
    ) -> Result<(Report, String), String> {
        self.verify_index().await?;
        let sched = model.scheduler.clone();
        sched.barrier(self.repo, true).await;
        match self.barrier_steps(model, include_ignored).await {
            Ok((report, commit, feed)) => {
                self.report = Some(report.clone());
                self.commit = commit.clone();
                self.sync = Some(reconcile::Options { include_ignored });
                let resync = matches!(feed, Feed::Resync { .. });
                let class = match self.phase {
                    Phase::FirstIndex => IndexClass::Stage1,
                    Phase::Maintained => IndexClass::Background,
                };
                self.feed(model, FeedMsg { feed, class, ack }).await;
                tracing::info!(root = %self.root.display(), files_changed = report.files_changed(),
                                  resync, chunk_rows = report.reads.chunk_rows,
                                  cache_probes = report.reads.cache_probes,
                                  cache_slots = report.reads.cache_slots, "barrier done");

                Ok((report, commit))
            }
            Err(message) => {
                self.sync = None;
                drop(ack);
                sched.barrier(self.repo, false).await;
                Err(message)
            }
        }
    }

    /// Hands `msg` to the feeder. A resync the feeder promotes after a
    /// failed feed asks the books before it takes the next feed, so asks
    /// are answered while the feeder's inbox is full: the owner never waits
    /// on a feeder that waits on it.
    async fn feed(&mut self, model: &Arc<Model<B>>, msg: FeedMsg) {
        let feeder = self.feeder(model);
        let permit = loop {
            tokio::select! {
                biased;
                permit = feeder.reserve() => {
                    break permit.expect("the feeder runs while the owner holds its sender");
                }
                Some(ask) = self.asks.recv() => self.answer(ask).await,
            }
        };
        permit.send(msg);
    }

    /// The feeder, spawned on first use; the writer is open by then (every
    /// barrier caller ran `shared` first).
    fn feeder(&mut self, model: &Arc<Model<B>>) -> mpsc::Sender<feed::FeedMsg> {
        if self.feeder.is_none() {
            self.feeder = Some(feed::spawn(FeederConfig::<B> {
                worktree: self.id,
                repo: self.repo,
                root: self.root.clone(),
                books: self.asker.clone(),
                writer: self.writer.clone().expect("opened"),
                counter: model.counter.clone(),
                sched: model.scheduler.clone(),
                stage1_done: self.stage1_tx.clone(),
            }));
        }
        self.feeder.as_ref().expect("spawned above").clone()
    }

    /// A spawned `index --wait`'s or the feeder's ask, answered from the
    /// books; an owner whose index failed to reopen has none to answer
    /// from. The cache is open: only a barrier or an `Index` request, both
    /// after `shared`, spawn the feeder or a wait.
    async fn answer(&mut self, ask: Ask) {
        let reconciler = self.index.is_some().then(|| self.reconciler());
        ask.answer(reconciler).await;
    }

    /// The reconciler of the open index, built on first use; the index and
    /// cache are open by then.
    fn reconciler(&mut self) -> &mut Reconciler {
        self.reconciler.get_or_insert_with(|| {
            Reconciler::new(
                self.index.clone().expect("opened"),
                self.cache.clone().expect("opened"),
                self.model_id.clone(),
                self.root.clone(),
            )
        })
    }

    /// The barrier's writer flush, pass and `git rev-parse`, and the feed
    /// the pass leaves for the feeder.
    async fn barrier_steps(
        &mut self,
        model: &Arc<Model<B>>,
        include_ignored: bool,
    ) -> Result<(Report, String, Feed), String> {
        self.writer
            .as_ref()
            .expect("opened")
            .flush()
            .await
            .map_err(|e| format!("vector writer flush: {e}"))?;
        let counter = model.counter.clone();
        let root = self.root.clone();
        let reconciler = self.reconciler();
        let (reconciled, head) = tokio::join!(
            reconciler.reconcile(
                CHUNKER_FINGERPRINT,
                reconcile::Options { include_ignored },
                counter,
            ),
            tokio::process::Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["rev-parse", "HEAD"])
                .output(),
        );
        let reconciled = reconciled.map_err(|e| e.to_string())?;
        // The pass is committed: the retriever follows it whatever happens
        // next, or its vector indexes would part from the books.
        if let Some(retriever) = &self.retriever {
            retriever
                .follow(reconciled.presence)
                .await
                .map_err(|e| e.to_string())?;
        }
        let head = head.map_err(|e| e.to_string())?;
        if !head.status.success() {
            return Err(format!(
                "git rev-parse HEAD failed: {}",
                String::from_utf8_lossy(&head.stderr)
            ));
        }
        let commit = String::from_utf8_lossy(&head.stdout).trim().to_owned();
        let options = reconcile::Options { include_ignored };
        // A `Change` is enough when the feeder provably holds this scope's
        // set: anything else is a resync — the first barrier, a scope flip,
        // a rechunk or a reopened index. Its set is listed from the books
        // the pass just brought up to date, here inside the barrier, so it
        // reads no chunk row and looks up no hash; the feeder rebuilds the
        // texts.
        let feed = match reconciled.changes {
            Some(changes) if self.sync == Some(options) => Feed::Change(changes),
            changes => Feed::Resync {
                options,
                missing: self
                    .reconciler()
                    .missing(include_ignored)
                    .await
                    .map_err(|e| e.to_string())?,
                known: changes.map(|changes| changes.added).unwrap_or_default(),
            },
        };
        Ok((reconciled.report, commit, feed))
    }

    /// A watcher burst: the same barrier as a query, answered nowhere and
    /// always without ignored files.
    async fn changed(&mut self) {
        let Some(model) = self.ready() else {
            return;
        };
        if let Err(error) = self.ensure_index().await {
            tracing::error!(%error, root = %self.root.display(), "watcher barrier: index failed");
            return;
        }
        if let Err(error) = self.shared().await {
            tracing::error!(%error, root = %self.root.display(), "watcher barrier: shared state failed");
            return;
        }
        if let Err(error) = self.barrier(&model, false, None).await {
            tracing::error!(%error, root = %self.root.display(), "watcher barrier failed");
        }
    }

    /// `Stage1Done`: the scheduler flushed every stage-1 vector. The
    /// transition is idempotent — a `Stage1` change adding a small chunk
    /// after completion can deliver it twice.
    async fn apply_stage1_done(&mut self) {
        if self.phase == Phase::Maintained {
            return;
        }
        let Some(index) = self.index.clone() else {
            return;
        };
        match index.set_phase(Phase::Maintained).await {
            Ok(()) => self.set_phase(Phase::Maintained),
            Err(error) => {
                tracing::error!(%error, "set_phase(maintained) failed; the worktree stays first-index")
            }
        }
    }

    /// Sets the phase and publishes it: `index --wait` tasks read it off
    /// the watch rather than the owner's `stage1_rx`.
    fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
        self.phase_tx.send_replace(phase);
    }
}

/// Everything an `index --wait` needs once its barrier ran; owned by the
/// spawned wait task, never by the owner.
struct IndexWait {
    /// The request's events sink.
    events: mpsc::Sender<Event>,
    /// The scheduler, for the counts watch.
    sched: SchedulerHandle,
    /// The worktree the wait covers.
    worktree: WorktreeId,
    /// Resolves once the barrier's `Enqueue` is in the scheduler's inbox,
    /// or with why there is none.
    ack: oneshot::Receiver<Result<(), String>>,
    /// The owner's books, asked for each frame's and `Indexed`'s coverage;
    /// the wait reads neither the index nor the cache itself.
    coverage: Asker,
    /// The repository's writer, flushed before `Indexed` claims coverage.
    writer: WriterHandle,
    /// The owner's phase watch; `Maintained` releases a first-index wait.
    phase: watch::Receiver<Phase>,
    /// The daemon's shutdown flag.
    stopping: watch::Receiver<bool>,
}

/// The tail of an `index --wait`, run off the owner: coverage frames while
/// the scheduler drains the enqueue, the stage-1 completion off the phase
/// watch, then the counted `Indexed`. Cancels on a client hang-up and
/// answers `ShuttingDown` once the daemon begins stopping.
async fn index_wait(wait: IndexWait) {
    let IndexWait {
        events,
        sched,
        worktree,
        ack,
        coverage,
        writer,
        mut phase,
        mut stopping,
    } = wait;
    // The barrier's `Enqueue` must be in the scheduler's inbox before the
    // `WatchCounts` the ack orders it behind; then the watch's seed
    // already counts the enqueued items.
    match ack.await {
        Ok(Ok(())) => {}
        Ok(Err(message)) => {
            send(&events, error(ErrorKind::Internal, message)).await;
            return;
        }
        Err(_) => {
            send(
                &events,
                error(ErrorKind::ShuttingDown, "scheduler gone".into()),
            )
            .await;
            return;
        }
    }
    let mut counts = match sched.watch_counts(worktree).await {
        Ok(counts) => counts,
        Err(_) => {
            send(
                &events,
                error(ErrorKind::ShuttingDown, "scheduler gone".into()),
            )
            .await;
            return;
        }
    };
    let mut sent: Option<(u64, u64)> = None;
    let mut last_send_at: Option<Instant> = None;
    let mut dirty = true;
    loop {
        let remaining = {
            let counts = counts.borrow_and_update();
            counts.queued_stage1 + counts.queued_background + counts.in_flight
        };
        if remaining == 0 {
            break;
        }
        let throttled = last_send_at.is_some_and(|at| at.elapsed() < COVERAGE_INTERVAL);
        if dirty && !throttled {
            let Some(counts) = ask(&coverage, &events).await else {
                return;
            };
            let frame = (counts.chunks_with_vector, counts.chunks_total);
            if sent != Some(frame) {
                let (with_vector, total) = frame;
                if !send(
                    &events,
                    Event::Progress(Progress::Coverage { with_vector, total }),
                )
                .await
                {
                    return;
                }
                sent = Some(frame);
                last_send_at = Some(Instant::now());
            }
            dirty = false;
            continue;
        }
        // Wait for a counts change or, while a report is pending, the
        // throttle deadline.
        let deadline = last_send_at
            .map(|at| at + COVERAGE_INTERVAL)
            .unwrap_or_else(Instant::now);
        tokio::select! {
            changed = counts.changed() => {
                if changed.is_err() {
                    return;
                }
                dirty = true;
            }
            _ = tokio::time::sleep_until(deadline.into()), if dirty => {}
            _ = events.closed() => return,
            _ = stopped(&mut stopping) => {
                send(&events, shutting_down()).await;
                return;
            }
        }
    }
    // Every pending chunk is embedded; commit the last ones before
    // answering so `Indexed` is a durable claim.
    if writer.flush().await.is_err() {
        send(
            &events,
            error(ErrorKind::Internal, "vector writer failed".into()),
        )
        .await;
        return;
    }
    if *phase.borrow() == Phase::FirstIndex {
        // Stage-1 completion is already in flight: the owner applies the
        // `Stage1Done` and publishes `Maintained` on this watch.
        tokio::select! {
            owner_gone = async {
                phase.wait_for(|phase| *phase == Phase::Maintained).await.is_err()
            } => {
                if owner_gone {
                    // The owner is gone; the wait cannot outlive it.
                    send(&events, shutting_down()).await;
                    return;
                }
            }
            _ = events.closed() => return,
            _ = stopped(&mut stopping) => {
                send(&events, shutting_down()).await;
                return;
            }
        }
    }
    // `Indexed` is counted, not asserted: the books after the flush above,
    // which see every vector it committed.
    let Some(counts) = ask(&coverage, &events).await else {
        return;
    };
    let _ = send(
        &events,
        Event::Final(Final::Indexed {
            with_vector: counts.chunks_with_vector,
            total: counts.chunks_total,
        }),
    )
    .await;
}

/// The owner's coverage count now; `None` once the reply's final event is
/// sent or the client hung up. The owner answers between its requests, and
/// in its shutdown drain until every wait is done, so only a hang-up cuts
/// an ask short.
async fn ask(coverage: &Asker, events: &mpsc::Sender<Event>) -> Option<Counts> {
    tokio::select! {
        counts = coverage.now() => match counts {
            Ok(counts) => Some(counts),
            Err(AskError::OwnerGone) => {
                send(events, shutting_down()).await;
                None
            }
            Err(AskError::Books(message)) => {
                send(events, error(ErrorKind::Internal, message)).await;
                None
            }
        },
        _ = events.closed() => None,
    }
}

/// Completes once `stopping` shows `true` or its sender is gone.
async fn stopped(stopping: &mut watch::Receiver<bool>) {
    let _ = stopping.wait_for(|stopping| *stopping).await;
}

/// `(dev, ino)` of a file's metadata.
fn file_id(meta: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

/// Sends `event` unless the request's client hung up.
async fn send(events: &mpsc::Sender<Event>, event: Event) -> bool {
    events.send(event).await.is_ok()
}

fn error(kind: ErrorKind, message: String) -> Event {
    Event::Final(Final::Error(ErrorReport { kind, message }))
}

fn shutting_down() -> Event {
    error(
        ErrorKind::ShuttingDown,
        "the daemon is shutting down".to_owned(),
    )
}

/// What a search's barrier and retrieval hand back.
struct Retrieved {
    /// The freshness barrier's duration, in µs.
    barrier_us: u64,
    /// The retrieval's outcome; `None` when the client hung up.
    result: Option<Result<search::Results, search::Error<EmbedError>>>,
}

/// The output-budget counter over the model's tokenizer.
struct BudgetCounter<B: Backend>(Arc<B::Counter>);

impl<B: Backend> search::TokenCounter for BudgetCounter<B> {
    fn count(&self, text: &str) -> usize {
        self.0.count(text)
    }
}

/// The `Retriever`'s embedder: query work goes to the scheduler, so the
/// ANE thread shares queries and indexing and the GPU never runs query
/// work. A `Documents` job's vectors reach the cache through the
/// repository's writer.
#[derive(Clone)]
struct AneHandle {
    worktree: WorktreeId,
    repo: RepoId,
    model_id: String,
    sched: mpsc::Sender<SchedulerMsg>,
    /// The repository's writer: on-the-spot vectors go there, and
    /// `documents` flushes it before it returns.
    writer: WriterHandle,
}

impl EmbedService for AneHandle {
    type Error = EmbedError;

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn query(&self, task: QueryTask, text: String) -> Result<UnitVector, Self::Error> {
        let (reply, rx) = oneshot::channel();
        self.sched
            .send(SchedulerMsg::Query(Box::new(QueryJob {
                worktree: self.worktree,
                task,
                text,
                enqueued: Instant::now(),
                stamp: 0,
                reply,
            })))
            .await
            .map_err(|_| EmbedError::SchedulerGone)?;
        rx.await.map_err(|_| EmbedError::SchedulerGone)?
    }

    async fn documents(&self, chunks: Vec<PendingChunk>) -> Result<(), Self::Error> {
        let (reply, rx) = oneshot::channel();
        self.sched
            .send(SchedulerMsg::Documents(Box::new(DocumentsJob {
                worktree: self.worktree,
                repo: self.repo,
                chunks,
                writer: self.writer.clone(),
                enqueued: Instant::now(),
                reply,
            })))
            .await
            .map_err(|_| EmbedError::SchedulerGone)?;
        rx.await.map_err(|_| EmbedError::SchedulerGone)??;
        // Every vector is with the writer: commit them for the
        // retriever's read.
        self.writer
            .flush()
            .await
            .map_err(|error| EmbedError::Writer(error.to_string()))
    }
}
