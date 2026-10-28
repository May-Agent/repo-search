//! The GPU thread: batch embedding on `CPU_AND_GPU`.
//!
//! The scheduler spawns this thread when work in its queue — the normal
//! queue on a wide-class chip, background on a base one — and headroom
//! allow. Opening the instance — a multi-second cold start — happens on
//! this thread, off every other path. `Stop` drops the embedder, releasing
//! the Core ML GPU instance, and the thread exits. There is no urgent path
//! here by construction: [`GpuJob`] carries no u=0 items.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use repo_search_core::model::MODEL_ID;
use repo_search_core::qos::{self, QosClass};
use tokio::sync::{mpsc, oneshot};

use crate::backend::{Backend, GpuOpen, TokenBatchEmbedder};
use crate::scheduler::{GpuJob, ItemId, SchedulerMsg};

/// Spawns the `repo-search-gpu` thread at user-initiated QoS. The returned
/// receiver resolves when the thread exits — after the embedder's `Drop`
/// released the Core ML GPU instance — and is the shutdown join.
pub fn spawn<B: Backend>(
    backend: Arc<B>,
    dir: PathBuf,
    sched: mpsc::Sender<SchedulerMsg>,
) -> oneshot::Receiver<()> {
    let (exited_tx, exited) = oneshot::channel();
    std::thread::Builder::new()
        .name("repo-search-gpu".to_owned())
        .spawn(move || {
            run(backend.as_ref(), &dir, &sched);
            let _ = exited_tx.send(());
        })
        .expect("failed to spawn repo-search-gpu");
    exited
}

fn run<B: Backend>(backend: &B, dir: &Path, sched: &mpsc::Sender<SchedulerMsg>) {
    qos::set_current_thread(QosClass::UserInitiated);
    tracing::info!(model = MODEL_ID, "loading the GPU model instance");
    let started = Instant::now();
    let mut embedder = match backend.open_gpu(dir) {
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
    tracing::info!(
        model = MODEL_ID,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "the GPU model instance loaded"
    );
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
                let (items, inputs): (Vec<ItemId>, Vec<Vec<u32>>) = batch.into_iter().unzip();
                match embedder.embed_batch(&inputs) {
                    Ok(vectors) => {
                        let items = items.into_iter().zip(vectors).collect();
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
