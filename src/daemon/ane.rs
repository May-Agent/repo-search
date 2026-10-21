//! The ANE thread: one `Embedder`, one chunk at a time.
//!
//! The thread owns the backend's ANE instance for its whole life. One chunk
//! at a time is what bounds a query's wait to the index chunk already in
//! flight when it was enqueued. Every blocking Core ML call happens here, off
//! the async executor.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use repo_search_core::Embedder;
use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot, watch};

use super::scheduler::{AneJob, EmbedError, QueryJob, SchedulerMsg};
use super::{Backend, CounterLoad, LoadStage};

/// The running ANE thread.
pub struct AneThread<C> {
    /// Resolves once `load_counter` returned on the thread: the model's
    /// counter and bucket lengths, or the load error's message.
    pub counted: oneshot::Receiver<CounterLoad<C, String>>,
    /// The scheduler's inbox goes here once it is spawned; dropping it
    /// exits the thread once `loaded` was sent.
    pub sched: oneshot::Sender<mpsc::Sender<SchedulerMsg>>,
    /// Where the load is: `Counter` first, `Open` once `open_ane` starts,
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

/// Spawns the `repo-search-ane` thread. It loads the counter and opens the
/// ANE instance on its own thread — the blocking calls stay off the
/// executor — reports each over its oneshot, then pulls jobs off the
/// scheduler it is handed until `Stop` or a dropped channel.
pub fn spawn<B: Backend>(backend: Arc<B>, dir: PathBuf, model_id: String) -> AneThread<B::Counter> {
    let (counted_tx, counted) = oneshot::channel();
    let (sched_tx, sched_rx) = oneshot::channel();
    let (stage_tx, stage) = watch::channel(LoadStage::Counter);
    let (loaded_tx, loaded) = oneshot::channel();
    let (exited_tx, exited) = watch::channel(false);
    std::thread::Builder::new()
        .name("repo-search-ane".to_owned())
        .spawn(move || {
            run(
                backend.as_ref(),
                &dir,
                &model_id,
                counted_tx,
                sched_rx,
                stage_tx,
                loaded_tx,
            );
            let _ = exited_tx.send(true);
        })
        .expect("failed to spawn repo-search-ane");
    AneThread {
        counted,
        sched: sched_tx,
        stage,
        loaded,
        exited,
    }
}

fn run<B: Backend>(
    backend: &B,
    dir: &Path,
    model_id: &str,
    counted: oneshot::Sender<CounterLoad<B::Counter, String>>,
    sched_rx: oneshot::Receiver<mpsc::Sender<SchedulerMsg>>,
    stage: watch::Sender<LoadStage>,
    loaded: oneshot::Sender<Result<qos::ThreadId, String>>,
) {
    let loaded_out = match backend.load_counter(dir) {
        Ok(out) => out,
        Err(error) => {
            let _ = counted.send(Err(error.to_string()));
            return;
        }
    };
    if counted.send(Ok(loaded_out)).is_err() {
        return;
    }
    stage.send_replace(LoadStage::Open);
    tracing::info!(model = %model_id, "loading the ANE model instance");
    let started = Instant::now();
    let mut embedder = match backend.open_ane(dir, model_id, &stage) {
        Ok(embedder) => embedder,
        Err(error) => {
            let _ = loaded.send(Err(error.to_string()));
            return;
        }
    };
    tracing::info!(
        model = %model_id,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "the ANE model instance loaded"
    );
    let me = qos::ThreadId::current();
    let _ = loaded.send(Ok(me));
    // The job's QoS applies only around its embed call; the pull and the
    // wait for the next job run at user-interactive, so a queued query never
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

    // End of the last index job; the wait of a query job is measured from it.
    let mut last_index_end = None;
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
            AneJob::Index {
                item,
                chunk,
                qos: class,
                writer,
            } => {
                set_qos(&mut qos, class);
                let result = embedder.embed_document(&chunk.title, &chunk.text);
                set_qos(&mut qos, QosClass::UserInteractive);
                match result {
                    Ok(vector) => {
                        last_index_end = Some(Instant::now());
                        let _ = writer.blocking_vector(chunk.content_hash, vector);
                        if sched.blocking_send(SchedulerMsg::AneDone { item }).is_err() {
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
            AneJob::Query { job, overtaken } => {
                set_qos(&mut qos, QosClass::UserInteractive);
                let worktree = job.worktree;
                let wait = last_index_end
                    .map(|end| end.saturating_duration_since(job.enqueued))
                    .unwrap_or_default();
                if let Some(error) = run_query(&mut embedder, job) {
                    let _ = sched.blocking_send(SchedulerMsg::EmbedderFailed { error });
                }
                if sched
                    .blocking_send(SchedulerMsg::QueryDone {
                        worktree,
                        wait,
                        overtaken,
                    })
                    .is_err()
                {
                    return;
                }
            }
            AneJob::Documents {
                spot,
                worktree,
                chunks,
                writer,
                enqueued,
                overtaken,
            } => {
                set_qos(&mut qos, QosClass::UserInteractive);
                let wait = last_index_end
                    .map(|end| end.saturating_duration_since(enqueued))
                    .unwrap_or_default();
                for chunk in &chunks {
                    match embedder.embed_document(&chunk.title, &chunk.text) {
                        Ok(vector) => {
                            let _ = writer.blocking_vector(chunk.content_hash, vector);
                        }
                        Err(error) => {
                            // Fatal: the scheduler answers the job with it.
                            let _ = sched.blocking_send(SchedulerMsg::EmbedderFailed {
                                error: error.to_string(),
                            });
                            break;
                        }
                    }
                }
                if sched
                    .blocking_send(SchedulerMsg::DocumentsDone {
                        spot,
                        worktree,
                        wait,
                        overtaken,
                    })
                    .is_err()
                {
                    return;
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

/// Embeds a query job and answers its reply; `Some` carries the failure's
/// error text when the model failed.
fn run_query<E>(embedder: &mut E, job: QueryJob) -> Option<String>
where
    E: repo_search_core::Embedder,
{
    let QueryJob {
        task, text, reply, ..
    } = job;
    match embedder.embed_query(task, &text) {
        Ok(vector) => {
            let _ = reply.send(Ok(vector));
            None
        }
        Err(error) => {
            let error = error.to_string();
            let _ = reply.send(Err(EmbedError::Model(error.clone())));
            Some(error)
        }
    }
}
