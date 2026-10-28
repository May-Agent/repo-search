//! The model loader: download, load, and the readiness watch.
//!
//! One task owns `watch::Sender<ModelState>` and runs the load sequence:
//! [`ModelSource::ensure`], then the ANE thread — which loads the encoder
//! and opens the model off the executor — and the scheduler, publishing
//! [`ModelState::Ready`] once the thread reports its embedder open. A
//! failure publishes [`ModelState::Failed`]; a request that finds it sends
//! [`LoaderMsg::Retry`], which starts the sequence over only while the
//! state is still `Failed`. An `open_ane` failure on a verified archive is
//! fatal instead: it publishes [`ModelState::Fatal`] — which no `Retry`
//! restarts — and reports to `run` over the fatal channel. The task also
//! owns the embedding threads' lifetime: on [`LoaderMsg::Shutdown`] it
//! cancels a download in flight — the only phase that has no thread to
//! stop — awaits a running attempt rather than aborting it, stops the
//! scheduler (whose `Stop` answers the threads' next pull) and waits for
//! the ANE and GPU threads' exit signals.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};

use repo_search_core::model::Fetch;

use crate::BusySource;
use crate::ane;
use crate::backend::{Backend, Headroom, LoadStage};
use crate::chip::ChipClass;
use crate::scheduler::{self, SchedulerHandle};

/// A model cache the loader downloads from. `ModelStore` in the main
/// crate implements it; tests hand a directory back directly.
pub trait ModelSource: Send + Sync + 'static {
    /// The model build's identity: the archive sha256, `MODEL_ID` on the
    /// wire.
    fn model_id(&self) -> String;

    /// The unpacked model directory, downloading it on first use.
    /// `progress` receives a [`Fetch`] after each downloaded chunk.
    fn ensure(
        &self,
        progress: &watch::Sender<Fetch>,
    ) -> impl Future<Output = Result<PathBuf, String>> + Send;
}

/// Where the model is on its way to readiness.
pub enum ModelState<B: Backend> {
    /// `ensure` has not produced a directory yet.
    Absent,
    /// The archive is downloading.
    Downloading {
        /// Bytes received so far.
        received: u64,
        /// The server's declared length, when known.
        total: Option<u64>,
    },
    /// The archive is verified; the encoder and the ANE instance are
    /// loading. The stage `/status` reports rides the loader's `stage`
    /// watch.
    Loading,
    /// Ready to embed.
    Ready(Arc<Model<B>>),
    /// Download or load failed; a request finding it sends
    /// [`LoaderMsg::Retry`].
    Failed(String),
    /// The load failed fatally — `open_ane` on a verified archive — and
    /// the server is shutting down. No `Retry` is honored after this.
    Fatal(String),
}

/// The loaded model and the pipeline built on it.
pub struct Model<B: Backend> {
    /// The archive sha256; `MODEL_ID` on the wire.
    pub id: String,
    /// The unpacked model directory.
    pub dir: PathBuf,
    /// The model's text encoder, shared by every request task.
    pub encoder: Arc<B::Encoder>,
    /// The model's bucket lengths, ascending.
    pub buckets: Box<[usize]>,
    /// The embedding scheduler the model's pipeline runs on.
    pub scheduler: SchedulerHandle,
}

/// The loader's inbox.
#[derive(Debug)]
pub enum LoaderMsg {
    /// Restart the load sequence after a [`ModelState::Failed`].
    Retry,
    /// Stop the pipeline and join the embedding threads, then reply.
    Shutdown {
        /// Signalled once the threads have exited.
        reply: oneshot::Sender<()>,
    },
}

/// The loader's handle: the readiness watch, the load-stage watch and the
/// inbox.
pub struct Loader<B: Backend> {
    /// The model's current state.
    pub state: watch::Receiver<ModelState<B>>,
    /// The ANE pipeline's `open_ane` stage while `state` is `Loading` —
    /// `/status` reports it verbatim.
    pub stage: watch::Receiver<LoadStage>,
    /// The loader's inbox sender.
    pub tx: mpsc::Sender<LoaderMsg>,
}

/// The embedding threads an attempt produced, which the loader must stop
/// and join at shutdown — whether the attempt succeeded or failed.
struct Pipeline {
    /// The scheduler and the receiver handing over its GPU threads' exit
    /// signals when the actor exits; absent when the attempt died before
    /// the scheduler existed.
    sched: Option<SchedulerParts>,
    /// The ANE thread's exit signal; the scheduler watches a clone of it.
    ane_exited: watch::Receiver<bool>,
}

struct SchedulerParts {
    handle: SchedulerHandle,
    exited: oneshot::Receiver<Vec<oneshot::Receiver<()>>>,
}

/// One load attempt's outcome.
type Attempt<B> = Result<(Arc<Model<B>>, Pipeline), LoadFail>;

/// A load failure; `Retryable` publishes `Failed`, `Fatal` — an `open_ane`
/// failure on a verified, loaded archive — also reports to `run`.
enum LoadFail {
    /// The download was cancelled by a shutdown; no thread ever existed.
    Cancelled,
    /// A download, verification, encoder or bucket failure: `Failed`, and
    /// a request may retry.
    Retryable {
        /// The error text.
        message: String,
        /// The threads the attempt already spawned, to be stopped.
        pipeline: Option<Pipeline>,
    },
    /// `open_ane` on a verified archive, or the load outliving the bound:
    /// fatal per the fatal-errors rule.
    Fatal {
        /// The error text.
        message: String,
        /// The threads the attempt already spawned, to be stopped.
        pipeline: Option<Pipeline>,
    },
}

impl LoadFail {
    /// The threads the attempt spawned, which the caller stops and joins.
    fn pipeline(self) -> Option<Pipeline> {
        match self {
            Self::Cancelled => None,
            Self::Retryable { pipeline, .. } | Self::Fatal { pipeline, .. } => pipeline,
        }
    }
}

/// What the loader is built from.
pub struct LoaderConfig<B: Backend, M: ModelSource> {
    /// The embedding backend.
    pub backend: Arc<B>,
    /// The model source; `ensure` starts at once.
    pub model: Arc<M>,
    /// The cap on the wait for the ANE thread's `encoded` and `loaded`
    /// reports: a load past it fails fatally with the stage it stopped in.
    pub bound: Duration,
    /// The host's chip class, handed to the scheduler: a base chip keeps
    /// the GPU off normal work.
    pub chip: ChipClass,
    /// Host headroom for the GPU instance.
    pub headroom: watch::Receiver<Headroom>,
    /// The loader's busy state for the idle-exit tracker.
    pub busy: watch::Sender<bool>,
    /// Where busy sources register with the idle-exit tracker.
    pub busy_reg: mpsc::Sender<BusySource>,
    /// Fatal errors go to `run` here.
    pub fatal: mpsc::Sender<String>,
    /// The GPU state `/status` reports, handed to the scheduler.
    pub gpu_state: watch::Sender<crate::scheduler::GpuState>,
}

/// Spawns the loader task; it starts the load sequence at once.
pub fn spawn<B: Backend, M: ModelSource>(config: LoaderConfig<B, M>) -> Loader<B> {
    let LoaderConfig {
        backend,
        model,
        bound,
        chip,
        headroom,
        busy,
        busy_reg,
        fatal,
        gpu_state,
    } = config;
    let (state_tx, state) = watch::channel(ModelState::Absent);
    let (stage_tx, stage) = watch::channel(LoadStage::Encoder);
    let (tx, mut rx) = mpsc::channel::<LoaderMsg>(64);
    tokio::spawn(async move {
        let mut join: Option<tokio::task::JoinHandle<Attempt<B>>>;
        // The running attempt's download-cancel sender; always `Some`
        // exactly while `join` is `Some`.
        let mut cancel: Option<oneshot::Sender<()>>;
        let mut pipeline: Option<Pipeline> = None;
        let (fetch_tx, mut fetch_rx) = watch::channel(Fetch {
            received: 0,
            total: None,
        });
        macro_rules! attempt {
            () => {{
                let (cancel_tx, cancel_rx) = oneshot::channel();
                cancel = Some(cancel_tx);
                join = Some(tokio::spawn(load(
                    LoadParts {
                        backend: backend.clone(),
                        model: model.clone(),
                        bound,
                        chip,
                        fetch: fetch_tx.clone(),
                        headroom: headroom.clone(),
                        busy_reg: busy_reg.clone(),
                        fatal: fatal.clone(),
                        gpu_state: gpu_state.clone(),
                        state: state_tx.clone(),
                        stage: stage_tx.clone(),
                    },
                    cancel_rx,
                )));
                busy.send_replace(true);
            }};
        }
        attempt!();
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    None | Some(LoaderMsg::Shutdown { .. }) => {
                        let reply = match msg {
                            Some(LoaderMsg::Shutdown { reply }) => Some(reply),
                            _ => None,
                        };
                        // Never abort: an attempt that already spawned the
                        // ANE thread owns it, so the thread must come back
                        // through the attempt's pipeline to be stopped.
                        // Cancelling reaches only the download phase; every
                        // later phase runs to completion and hands its
                        // threads over.
                        if let Some(cancel) = cancel.take() {
                            let _ = cancel.send(());
                        }
                        if let Some(task) = join.take() {
                            match task.await {
                                Ok(Ok((_model, pipe))) => stop_pipeline(pipe).await,
                                Ok(Err(fail)) => {
                                    if let Some(parts) = fail.pipeline() {
                                        stop_pipeline(parts).await;
                                    }
                                }
                                Err(_) => {}
                            }
                        }
                        if let Some(pipeline) = pipeline.take() {
                            stop_pipeline(pipeline).await;
                        }
                        busy.send_replace(false);
                        if let Some(reply) = reply {
                            let _ = reply.send(());
                        }
                        return;
                    }
                    // A Retry is honored only while the state is `Failed`
                    // and no attempt runs: a stale Retry arriving after
                    // `Ready` must not build a second pipeline.
                    Some(LoaderMsg::Retry) => {
                        if matches!(*state_tx.borrow(), ModelState::Failed(_))
                            && join.is_none()
                        {
                            attempt!();
                        }
                    }
                },
                changed = fetch_rx.changed(), if join.is_some() => {
                    // The state moves forward only: a `Fetch` landing
                    // after the attempt sent `Loading` must not push it
                    // back to `Downloading`.
                    if changed.is_ok() {
                        let fetch = *fetch_rx.borrow();
                        state_tx.send_if_modified(|state| {
                            let ModelState::Downloading { received, total } = state else {
                                return false;
                            };
                            if *received == fetch.received && *total == fetch.total {
                                return false;
                            }
                            *received = fetch.received;
                            *total = fetch.total;
                            true
                        });
                    }
                }
                result = async { join.as_mut().expect("set").await }, if join.is_some() => {
                    join = None;
                    cancel = None;
                    let mut fatal_message = None;
                    let state = match result {
                        Ok(Ok((model, pipe))) => {
                            pipeline = Some(pipe);
                            ModelState::Ready(model)
                        }
                        Ok(Err(LoadFail::Cancelled)) => unreachable!(
                            "the cancel signal fires only on shutdown, which awaits the attempt itself"
                        ),
                        Ok(Err(LoadFail::Retryable { message, pipeline: parts })) => {
                            if let Some(parts) = parts {
                                stop_pipeline(parts).await;
                            }
                            ModelState::Failed(message)
                        }
                        Ok(Err(LoadFail::Fatal { message, pipeline: parts })) => {
                            if let Some(parts) = parts {
                                stop_pipeline(parts).await;
                            }
                            fatal_message = Some(message.clone());
                            ModelState::Fatal(message)
                        }
                        Err(_) => ModelState::Failed("model loader task ended".to_owned()),
                    };
                    busy.send_replace(false);
                    let _ = state_tx.send(state);
                    if let Some(message) = fatal_message {
                        let _ = fatal.send(message).await;
                    }
                }
            }
        }
    });
    Loader { state, stage, tx }
}

/// Stops an attempt's scheduler — whose `Stop` answers the embedding
/// threads' pulls — and waits on every thread's exit signal.
async fn stop_pipeline(mut pipeline: Pipeline) {
    if let Some(SchedulerParts { handle, exited }) = pipeline.sched {
        handle.stop().await;
        if let Ok(gpu) = exited.await {
            for thread in gpu {
                let _ = thread.await;
            }
        }
    }
    // `true` on a clean exit, `Err` if the thread unwound past the report.
    let _ = pipeline.ane_exited.changed().await;
}

fn retryable(message: String, pipeline: Option<Pipeline>) -> LoadFail {
    LoadFail::Retryable { message, pipeline }
}

/// What one load attempt is built from: the pieces shared across retries.
struct LoadParts<B: Backend, M: ModelSource> {
    backend: Arc<B>,
    model: Arc<M>,
    /// The cap on the wait for the ANE thread's `encoded` and `loaded`
    /// reports.
    bound: Duration,
    /// The host's chip class, handed to the scheduler.
    chip: ChipClass,
    fetch: watch::Sender<Fetch>,
    headroom: watch::Receiver<Headroom>,
    busy_reg: mpsc::Sender<BusySource>,
    fatal: mpsc::Sender<String>,
    gpu_state: watch::Sender<crate::scheduler::GpuState>,
    state: watch::Sender<ModelState<B>>,
    /// The `open_ane` stage `/status` reports; the attempt forwards the
    /// ANE thread's own watch into it.
    stage: watch::Sender<LoadStage>,
}

/// One load attempt: `ensure`, then the ANE thread — which loads the
/// encoder and opens the model — then the scheduler, whose inbox the
/// thread is handed. Returns the model with the threads' exit signals, or
/// the failure with the half-built pipeline for the caller to stop.
async fn load<B: Backend, M: ModelSource>(
    parts: LoadParts<B, M>,
    mut cancel: oneshot::Receiver<()>,
) -> Attempt<B> {
    let LoadParts {
        backend,
        model,
        bound,
        chip,
        fetch,
        headroom,
        busy_reg,
        fatal,
        gpu_state,
        state,
        stage,
    } = parts;
    // Every attempt starts the state machine over: a retry's `Failed`
    // becomes a fresh `Downloading` — absent progress — and the fetch
    // arm only ever advances a `Downloading` state, never a later one
    // backwards. The progress channel is reset first so a stale `Fetch`
    // of the last attempt cannot arrive after it.
    fetch.send_replace(Fetch {
        received: 0,
        total: None,
    });
    let _ = state.send(ModelState::Downloading {
        received: 0,
        total: None,
    });
    // Only the download observes the cancel signal: it is the one phase
    // that has spawned no thread, so a cancelled attempt returns with no
    // pipeline to stop. Everything past `ensure` runs to completion.
    let dir = tokio::select! {
        dir = model.ensure(&fetch) => dir.map_err(|e| retryable(e, None))?,
        _ = &mut cancel => return Err(LoadFail::Cancelled),
    };
    let _ = state.send(ModelState::Loading);
    let model_id = model.model_id();
    let ane::AneThread {
        encoded,
        sched: ane_sched,
        stage: ane_stage,
        loaded,
        exited: ane_exited,
    } = ane::spawn(backend.clone(), dir.clone());
    // Forward the thread's stage into the loader's watch so `/status`
    // reports where an `open_ane` call is; a second receiver serves the
    // attempt's own timeout reads.
    let stage_read = ane_stage.clone();
    let mut stage_fwd = ane_stage;
    tokio::spawn(async move {
        loop {
            let _ = stage.send_replace(*stage_fwd.borrow_and_update());
            if stage_fwd.changed().await.is_err() {
                return;
            }
        }
    });
    // One deadline covers every wait on the thread's reports: a load past
    // `bound` fails with the stage `stage` last reported.
    let deadline = tokio::time::Instant::now() + bound;
    let (encoder, buckets) = match tokio::time::timeout_at(deadline, encoded).await {
        Err(_) => return Err(timed_out(bound, *stage_read.borrow())),
        Ok(Ok(Ok(out))) => out,
        Ok(Ok(Err(message))) => {
            return Err(retryable(message, Some(Pipeline::just_ane(ane_exited))));
        }
        Ok(Err(_)) => {
            return Err(retryable(
                "the ANE thread went away".to_owned(),
                Some(Pipeline::just_ane(ane_exited)),
            ));
        }
    };
    let (sched_busy_tx, sched_busy_rx) = watch::channel(false);
    if busy_reg
        .send(BusySource {
            state: sched_busy_rx,
        })
        .await
        .is_err()
    {
        return Err(retryable(
            "the busy aggregator is gone".to_owned(),
            Some(Pipeline::just_ane(ane_exited)),
        ));
    }
    let (sched, sched_exited) = scheduler::spawn(scheduler::SchedulerConfig {
        backend,
        dir: dir.clone(),
        buckets: buckets.clone(),
        headroom,
        chip,
        fatal: fatal.clone(),
        busy: sched_busy_tx,
        ane_exited: ane_exited.clone(),
        gpu_state,
    });
    let make = |handle: SchedulerHandle,
                exited: oneshot::Receiver<Vec<oneshot::Receiver<()>>>,
                ane_exited: watch::Receiver<bool>| {
        Pipeline {
            sched: Some(SchedulerParts { handle, exited }),
            ane_exited,
        }
    };
    match tokio::time::timeout_at(deadline, loaded).await {
        // The thread is still inside the call that timed out, so the
        // attempt reports `pipeline: None` rather than a pipeline it
        // cannot stop: `sched` and the unsent `ane_sched` drop here, which
        // ends the scheduler and makes a late return exit the thread; a
        // permanently wedged call dies with the server's fatal exit.
        Err(_) => Err(timed_out(bound, *stage_read.borrow())),
        Ok(Err(_)) => Err(retryable(
            "the ANE thread went away".to_owned(),
            Some(make(sched, sched_exited, ane_exited)),
        )),
        Ok(Ok(Err(message))) => Err(LoadFail::Fatal {
            message,
            pipeline: Some(make(sched, sched_exited, ane_exited)),
        }),
        Ok(Ok(Ok(_thread))) => {
            // The thread waits on its scheduler handoff after reporting
            // `loaded`; it is handed over only once the load is confirmed.
            let _ = ane_sched.send(sched.tx());
            Ok((
                Arc::new(Model {
                    id: model_id,
                    dir,
                    encoder: Arc::new(encoder),
                    buckets,
                    scheduler: sched.clone(),
                }),
                make(sched, sched_exited, ane_exited),
            ))
        }
    }
}

/// The load outlived `bound` while in `stage`: fatal — the client gets the
/// error and no `Retry` restarts the attempt.
fn timed_out(bound: Duration, stage: LoadStage) -> LoadFail {
    tracing::error!(
        ?stage,
        bound_s = bound.as_secs(),
        "the model load timed out"
    );
    LoadFail::Fatal {
        message: format!(
            "the model load exceeded the {}s bound during {stage:?}",
            bound.as_secs()
        ),
        pipeline: None,
    }
}

impl Pipeline {
    /// A pipeline whose only thread is the ANE one — a failure before the
    /// scheduler existed.
    fn just_ane(ane_exited: watch::Receiver<bool>) -> Self {
        Self {
            sched: None,
            ane_exited,
        }
    }
}
