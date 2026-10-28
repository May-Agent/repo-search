//! The model backend behind the server's embedding pipeline.
//!
//! The seam that lets the scheduler-level tests run every component
//! against a stub embedder. Production is [`crate::coreml::CoreMlBackend`].

use std::path::Path;

use repo_search_core::UnitVector;
use repo_search_coreml::GpuBlocked;
use tokio::sync::watch;

/// Host headroom for the GPU embedder: `Ok(())` when it may run.
pub type Headroom = Result<(), GpuBlocked>;

/// The text encoder and the model's bucket lengths, ascending.
pub type EncoderLoad<E, Er> = Result<(E, Box<[usize]>), Er>;

/// Embeds already-tokenized inputs, one at a time.
pub trait TokenEmbedder: Send {
    /// The embedder's failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Embeds `ids`.
    ///
    /// # Errors
    ///
    /// When the embed call fails.
    fn embed(&mut self, ids: &[u32]) -> Result<UnitVector, Self::Error>;
}

/// Embeds already-tokenized inputs in batches.
pub trait TokenBatchEmbedder: TokenEmbedder {
    /// Embeds every input of `ids` in one batch prediction. Callers batch
    /// per bucket: the inputs must share a bucket length.
    ///
    /// # Errors
    ///
    /// When the batch call fails.
    fn embed_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Self::Error>;
}

/// Turns request text into token ids.
pub trait TextEncoder: Send + Sync {
    /// Encodes `text` with special tokens; the error is the message a
    /// `context_length_exceeded` or `invalid_request` reply carries.
    ///
    /// # Errors
    ///
    /// When `text` encodes past the model's largest bucket, or encoding
    /// fails.
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
}

/// The model backend: the text encoder plus the ANE and GPU instances.
pub trait Backend: Send + Sync + 'static {
    /// The single-input instance on the ANE thread.
    type Ane: TokenEmbedder + Send + 'static;
    /// The batch instance on the GPU thread; never serves u=0 work.
    type Gpu: TokenBatchEmbedder + Send + 'static;
    /// The text encoder request inputs are tokenized with.
    type Encoder: TextEncoder + 'static;
    /// The backend's model-load failure.
    type LoadError: std::error::Error + Send + Sync + 'static;
    /// The headroom monitor's guard; dropping it stops the monitor.
    type Monitor: Send + 'static;

    /// The text encoder and the model's bucket lengths, ascending.
    ///
    /// # Errors
    ///
    /// When the model directory cannot be read.
    fn load_encoder(&self, dir: &Path) -> EncoderLoad<Self::Encoder, Self::LoadError>;

    /// Opens the ANE instance; called on the ANE thread, which owns the
    /// returned embedder. Implementations warm every bucket before
    /// returning so the first request does not pay a compile. `stage` is
    /// already `Open` on entry; an implementation sends
    /// [`LoadStage::WarmUp`] once the model is open and the warm-up embeds
    /// begin, so a load that outlives the bound names the stage it stopped
    /// in.
    ///
    /// # Errors
    ///
    /// When the model cannot be loaded.
    fn open_ane(
        &self,
        dir: &Path,
        stage: &watch::Sender<LoadStage>,
    ) -> Result<Self::Ane, Self::LoadError>;

    /// Opens the GPU instance; called on the GPU thread, which owns the
    /// returned embedder.
    ///
    /// # Errors
    ///
    /// [`GpuOpen::Blocked`] when the host has no GPU headroom — the headroom
    /// may have changed between the check and the open — and
    /// [`GpuOpen::Failed`] for every other load failure, which is fatal.
    fn open_gpu(&self, dir: &Path) -> Result<Self::Gpu, GpuOpen<Self::LoadError>>;

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

/// How far the ANE pipeline's load got, reported on `/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStage {
    /// `load_encoder` is reading the tokenizer and the manifest.
    Encoder,
    /// `open_ane` is inside the compiled-model open — the ANE compile when
    /// the system's compiled-model cache misses.
    Open,
    /// The model is open; the per-bucket warm-up embeds are running.
    WarmUp,
}
