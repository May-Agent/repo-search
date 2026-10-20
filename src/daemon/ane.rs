//! The ANE thread: one `Embedder`, one chunk at a time.
//!
//! The thread owns the backend's ANE instance for its whole life. One chunk
//! at a time is what bounds a query's wait to the index chunk already in
//! flight when it was enqueued. Every blocking Core ML call happens here, off
//! the async executor.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use repo_search_core::Embedder;
use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot};

use super::Backend;
use super::scheduler::{AneJob, EmbedError, QueryJob, SchedulerMsg};

/// The running ANE thread.
pub struct AneThread {
    /// Resolves once `open_ane` returned: the thread's [`qos::ThreadId`], or
    /// the load error's message so the model loader can publish `Failed`.
    pub loaded: oneshot::Receiver<Result<qos::ThreadId, String>>,
    /// The thread's join handle.
    pub join: JoinHandle<()>,
}

/// Spawns the `repo-search-ane` thread. It opens the ANE instance on its own
/// thread, reports `loaded`, then pulls jobs off the scheduler until `Stop`
/// or a dropped channel.
pub fn spawn<B: Backend>(
    backend: Arc<B>,
    dir: PathBuf,
    model_id: String,
    sched: mpsc::Sender<SchedulerMsg>,
) -> AneThread {
    let (loaded_tx, loaded) = oneshot::channel::<Result<qos::ThreadId, String>>();
    let join = std::thread::Builder::new()
        .name("repo-search-ane".to_owned())
        .spawn(move || run(backend.as_ref(), &dir, &model_id, &sched, loaded_tx))
        .expect("failed to spawn repo-search-ane");
    AneThread { loaded, join }
}

fn run<B: Backend>(
    backend: &B,
    dir: &Path,
    model_id: &str,
    sched: &mpsc::Sender<SchedulerMsg>,
    loaded: oneshot::Sender<Result<qos::ThreadId, String>>,
) {
    let mut embedder = match backend.open_ane(dir, model_id) {
        Ok(embedder) => embedder,
        Err(error) => {
            let _ = loaded.send(Err(error.to_string()));
            return;
        }
    };
    let me = qos::ThreadId::current();
    let _ = loaded.send(Ok(me));

    // The job's QoS applies only around its embed call; the pull and the wait
    // for the next job run at user-interactive, so a queued query never waits
    // on a thread the scheduler dropped to background priority.
    let mut qos = None;
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
                let worktree = job.worktree();
                let wait = last_index_end
                    .map(|end| end.saturating_duration_since(job.enqueued()))
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
    match job {
        QueryJob::Query {
            task, text, reply, ..
        } => match embedder.embed_query(task, &text) {
            Ok(vector) => {
                let _ = reply.send(Ok(vector));
                None
            }
            Err(error) => {
                let error = error.to_string();
                let _ = reply.send(Err(EmbedError::Model(error.clone())));
                Some(error)
            }
        },
        QueryJob::Documents { chunks, reply, .. } => {
            let mut vectors = Vec::with_capacity(chunks.len());
            let mut failure = None;
            for chunk in &chunks {
                match embedder.embed_document(&chunk.title, &chunk.text) {
                    Ok(vector) => vectors.push(vector),
                    Err(error) => {
                        failure = Some(error.to_string());
                        break;
                    }
                }
            }
            let result = match &failure {
                None => Ok(vectors),
                Some(error) => Err(EmbedError::Model(error.clone())),
            };
            let _ = reply.send(result);
            failure
        }
    }
}
