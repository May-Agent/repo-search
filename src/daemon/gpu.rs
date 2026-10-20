//! The GPU thread: batch stage-1 indexing on `CPU_AND_GPU`.
//!
//! The scheduler spawns this thread when stage-1 work and headroom allow.
//! Opening the instance — a multi-second cold start — happens on this thread,
//! off every other path. `Stop` drops the embedder, releasing the Core ML GPU
//! instance, and the thread exits. There is no query path here by
//! construction: [`GpuJob`] has no query variant.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use repo_search_core::BatchEmbedder;
use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot};

use super::scheduler::{GpuJob, SchedulerMsg};
use super::{Backend, GpuOpen};

/// Spawns the `repo-search-gpu` thread at user-initiated QoS. The returned
/// receiver resolves when the thread exits — after the embedder's `Drop`
/// released the Core ML GPU instance — and is the shutdown join.
pub fn spawn<B: Backend>(
    backend: Arc<B>,
    dir: PathBuf,
    model_id: String,
    sched: mpsc::Sender<SchedulerMsg>,
) -> oneshot::Receiver<()> {
    let (exited_tx, exited) = oneshot::channel();
    std::thread::Builder::new()
        .name("repo-search-gpu".to_owned())
        .spawn(move || {
            run(backend.as_ref(), &dir, &model_id, &sched);
            let _ = exited_tx.send(());
        })
        .expect("failed to spawn repo-search-gpu");
    exited
}

fn run<B: Backend>(backend: &B, dir: &Path, model_id: &str, sched: &mpsc::Sender<SchedulerMsg>) {
    qos::set_current_thread(QosClass::UserInitiated);
    let mut embedder = match backend.open_gpu(dir, model_id) {
        Ok(embedder) => embedder,
        // Headroom changed between the scheduler's check and the open; not
        // fatal, the scheduler holds off the GPU until headroom changes.
        Err(GpuOpen::Blocked(_)) => {
            let _ = sched.blocking_send(SchedulerMsg::GpuBlocked);
            return;
        }
        Err(GpuOpen::Failed(error)) => {
            let _ = sched.blocking_send(SchedulerMsg::EmbedderFailed {
                error: error.to_string(),
            });
            let _ = sched.blocking_send(SchedulerMsg::GpuGone);
            return;
        }
    };
    if sched.blocking_send(SchedulerMsg::GpuStarted).is_err() {
        return;
    }
    loop {
        let (reply, rx) = oneshot::channel();
        if sched
            .blocking_send(SchedulerMsg::GpuPull { reply })
            .is_err()
        {
            break;
        }
        match rx.blocking_recv() {
            Err(_) | Ok(GpuJob::Stop) => break,
            Ok(GpuJob::Batch(batch)) => {
                let docs: Vec<(&str, &str)> = batch
                    .iter()
                    .map(|(_, chunk, _)| (chunk.title.as_str(), chunk.text.as_str()))
                    .collect();
                match embedder.embed_documents(&docs) {
                    Ok(vectors) => {
                        let items = batch.iter().map(|(item, ..)| *item).collect();
                        for ((_, chunk, writer), vector) in batch.into_iter().zip(vectors) {
                            let _ = writer.blocking_vector(chunk.content_hash, vector);
                        }
                        if sched
                            .blocking_send(SchedulerMsg::GpuDone { items })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sched.blocking_send(SchedulerMsg::EmbedderFailed {
                            error: error.to_string(),
                        });
                        break;
                    }
                }
            }
        }
    }
    drop(embedder);
    let _ = sched.blocking_send(SchedulerMsg::GpuGone);
}
