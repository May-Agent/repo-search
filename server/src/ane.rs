//! The ANE thread: one embedder, one input at a time.
//!
//! The thread owns the backend's ANE instance for its whole life. One input
//! at a time is what bounds an urgent request's wait to the item already in
//! flight when it was enqueued. Every blocking Core ML call happens here,
//! off the async executor.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot, watch};

use crate::backend::{Backend, EncoderLoad, LoadStage, TokenEmbedder};
use crate::scheduler::{AneJob, SchedulerMsg};

/// The running ANE thread.
pub struct AneThread<E> {
    /// Resolves once `load_encoder` returned on the thread: the model's
    /// text encoder and bucket lengths, or the load error's message.
    pub encoded: oneshot::Receiver<EncoderLoad<E, String>>,
    /// The scheduler's inbox goes here once it is spawned; dropping it
    /// exits the thread once `loaded` was sent.
    pub sched: oneshot::Sender<mpsc::Sender<SchedulerMsg>>,
    /// Where the load is: `Encoder` first, `Open` once `open_ane` starts,
    /// and `WarmUp` once the model is open and the buckets warm — the
    /// stage a timed-out load names.
    pub stage: watch::Receiver<LoadStage>,
    /// Resolves once `open_ane` returned: the thread's [`qos::ThreadId`], or
    /// the load error's message.
    pub loaded: oneshot::Receiver<Result<qos::ThreadId, String>>,
    /// Fires `true` when the thread exits and errs when it dies without
    /// reporting — a panic unwinds past the send. Shared between the
    /// scheduler, which ends its run when the thread is gone, and the
    /// loader, which joins the thread at shutdown.
    pub exited: watch::Receiver<bool>,
}

/// Spawns the `repo-search-ane` thread. It loads the encoder and opens the
/// ANE instance on its own thread — the blocking calls stay off the
/// executor — reports each over its oneshot, then pulls jobs off the
/// scheduler it is handed until `Stop` or a dropped channel.
pub fn spawn<B: Backend>(backend: Arc<B>, dir: PathBuf) -> AneThread<B::Encoder> {
    let (encoded_tx, encoded) = oneshot::channel();
    let (sched_tx, sched_rx) = oneshot::channel();
    let (stage_tx, stage) = watch::channel(LoadStage::Encoder);
    let (loaded_tx, loaded) = oneshot::channel();
    let (exited_tx, exited) = watch::channel(false);
    std::thread::Builder::new()
        .name("repo-search-ane".to_owned())
        .spawn(move || {
            run(
                backend.as_ref(),
                &dir,
                encoded_tx,
                sched_rx,
                stage_tx,
                loaded_tx,
            );
            let _ = exited_tx.send(true);
        })
        .expect("failed to spawn repo-search-ane");
    AneThread {
        encoded,
        sched: sched_tx,
        stage,
        loaded,
        exited,
    }
}

fn run<B: Backend>(
    backend: &B,
    dir: &std::path::Path,
    encoded: oneshot::Sender<EncoderLoad<B::Encoder, String>>,
    sched_rx: oneshot::Receiver<mpsc::Sender<SchedulerMsg>>,
    stage: watch::Sender<LoadStage>,
    loaded: oneshot::Sender<Result<qos::ThreadId, String>>,
) {
    let loaded_out = match backend.load_encoder(dir) {
        Ok(out) => out,
        Err(error) => {
            let _ = encoded.send(Err(error.to_string()));
            return;
        }
    };
    if encoded.send(Ok(loaded_out)).is_err() {
        return;
    }
    stage.send_replace(LoadStage::Open);
    tracing::info!(model = MODEL_ID, "loading the ANE model instance");
    let started = Instant::now();
    let mut embedder = match backend.open_ane(dir, &stage) {
        Ok(embedder) => embedder,
        Err(error) => {
            let _ = loaded.send(Err(error.to_string()));
            return;
        }
    };
    tracing::info!(
        model = MODEL_ID,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "the ANE model instance loaded"
    );
    let me = qos::ThreadId::current();
    let _ = loaded.send(Ok(me));
    // The job's QoS applies only around its embed call; the pull and the
    // wait for the next job run at user-interactive, so an urgent item never
    // waits on a thread the scheduler dropped to background priority.
    let mut qos = None;
    set_qos(&mut qos, QosClass::UserInteractive);
    // The loader hands over the scheduler once it exists; a dropped channel
    // means the pipeline was never wanted.
    let sched = match sched_rx.blocking_recv() {
        Ok(sched) => sched,
        Err(_) => return,
    };
    let sched = &sched;

    loop {
        let (reply, rx) = oneshot::channel();
        if sched
            .blocking_send(SchedulerMsg::AnePull { thread: me, reply })
            .is_err()
        {
            return;
        }
        let job = match rx.blocking_recv() {
            Ok(job) => job,
            Err(_) => return,
        };
        match job {
            AneJob::Stop => return,
            AneJob::Item {
                item,
                ids,
                qos: class,
            } => {
                set_qos(&mut qos, class);
                let result = embedder.embed(&ids);
                set_qos(&mut qos, QosClass::UserInteractive);
                match result {
                    Ok(vector) => {
                        let ended = Instant::now();
                        if sched
                            .blocking_send(SchedulerMsg::AneDone {
                                item,
                                vector,
                                ended,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sched.blocking_send(SchedulerMsg::EmbedderFailed {
                            error: error.to_string(),
                        });
                    }
                }
            }
        }
    }
}

fn set_qos(qos: &mut Option<QosClass>, class: QosClass) {
    if *qos != Some(class) {
        qos::set_current_thread(class);
        *qos = Some(class);
    }
}

use repo_search_core::model::MODEL_ID;
