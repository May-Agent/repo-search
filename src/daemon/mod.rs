//! The `search` daemon: the embedding pipeline and, behind it, the model
//! loader, the worktree registry and the socket server.

pub mod ane;
pub mod coreml;
pub mod gpu;
pub mod scheduler;
pub mod writer;

use std::path::Path;

use repo_search_core::{BatchEmbedder, Embedder, TokenCounter};
use repo_search_coreml::GpuBlocked;

/// Host headroom for the GPU embedder: `Ok(())` when the GPU instance may run.
pub type Headroom = Result<(), GpuBlocked>;

/// The counter for chunking and queue ordering, with the model's bucket
/// lengths in ascending order.
pub type CounterLoad<C, E> = Result<(C, Box<[usize]>), E>;

/// The model backend behind the daemon's embedding pipeline.
///
/// The seam that lets the scheduler-level tests run every component against a
/// stub embedder. Production is [`coreml::CoreMlBackend`].
pub trait Backend: Send + Sync + 'static {
    /// The single-chunk instance on the ANE thread, shared by queries and
    /// indexing.
    type Ane: Embedder + Send + 'static;
    /// The batch instance on the GPU thread; serves stage-1 indexing only.
    type Gpu: BatchEmbedder + Send + 'static;
    /// Counts the model's tokens for chunking and queue ordering.
    type Counter: TokenCounter + 'static;
    /// The backend's model-load failure.
    type LoadError: std::error::Error + Send + Sync + 'static;
    /// The headroom monitor's guard; dropping it stops the monitor.
    type Monitor: Send + 'static;

    /// The counter for chunking and queue ordering, and the model's bucket
    /// lengths, ascending.
    ///
    /// # Errors
    ///
    /// When the model directory cannot be read.
    fn load_counter(&self, dir: &Path) -> CounterLoad<Self::Counter, Self::LoadError>;

    /// Opens the ANE instance; called on the ANE thread, which owns the
    /// returned embedder. Implementations warm every bucket before returning
    /// so the first query does not pay a compile.
    ///
    /// # Errors
    ///
    /// When the model cannot be loaded.
    fn open_ane(&self, dir: &Path, model_id: &str) -> Result<Self::Ane, Self::LoadError>;

    /// Opens the GPU instance; called on the GPU thread, which owns the
    /// returned embedder.
    ///
    /// # Errors
    ///
    /// [`GpuOpen::Blocked`] when the host has no GPU headroom — the headroom
    /// may have changed between the check and the open — and
    /// [`GpuOpen::Failed`] for every other load failure, which is fatal.
    fn open_gpu(&self, dir: &Path, model_id: &str) -> Result<Self::Gpu, GpuOpen<Self::LoadError>>;

    /// Calls `on_change` once with the current headroom, then on every
    /// power-source or thermal change.
    fn watch_headroom(&self, on_change: Box<dyn Fn(Headroom) + Send + Sync>) -> Self::Monitor;
}

/// The outcome of opening the GPU instance.
#[derive(Debug)]
pub enum GpuOpen<E> {
    /// The host refuses the GPU instance right now: not fatal, retry later.
    Blocked(GpuBlocked),
    /// The model could not be loaded: fatal.
    Failed(E),
}

/// A failure that shuts the daemon down. Delivered to `run` over the fatal
/// channel every component shares.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An [`Embedder`] or [`BatchEmbedder`] call, or an `open_gpu`, failed.
    /// There is no retry and no fallback to the other instance.
    #[error("embedder failed: {0}")]
    Embedder(String),
    /// A writer's `put_many` failed.
    #[error("vector writer failed: {0}")]
    Writer(String),
}
