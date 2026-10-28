//! The production [`Backend`] over `repo-search-coreml`.

use std::path::Path;

use repo_search_core::UnitVector;
use repo_search_coreml::{ComputeUnits, CoreMlEmbedder, Encoder, Error, HeadroomMonitor};
use tokio::sync::watch;

use crate::backend::{
    Backend, EncoderLoad, GpuOpen, Headroom, LoadStage, TextEncoder, TokenBatchEmbedder,
    TokenEmbedder,
};

impl TokenEmbedder for CoreMlEmbedder {
    type Error = Error;

    fn embed(&mut self, ids: &[u32]) -> Result<UnitVector, Self::Error> {
        self.embed_tokens(ids)
    }
}

impl TokenBatchEmbedder for CoreMlEmbedder {
    fn embed_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Self::Error> {
        self.embed_tokens_batch(ids)
    }
}

impl TextEncoder for Encoder {
    fn encode(&self, text: &str) -> Result<Vec<u32>, crate::backend::EncodeError> {
        use crate::backend::EncodeError;
        Encoder::encode(self, text).map_err(|error| match error {
            repo_search_coreml::Error::TooLong { .. } => EncodeError::TooLong(error.to_string()),
            error => EncodeError::Failed(error.to_string()),
        })
    }
}

/// The server's Core ML backend.
pub struct CoreMlBackend;

impl Backend for CoreMlBackend {
    type Ane = CoreMlEmbedder;
    type Gpu = CoreMlEmbedder;
    type Encoder = Encoder;
    type LoadError = Error;
    type Monitor = HeadroomMonitor;

    fn load_encoder(&self, dir: &Path) -> EncoderLoad<Self::Encoder, Self::LoadError> {
        let encoder = Encoder::open(dir)?;
        let buckets = encoder.buckets().into();
        Ok((encoder, buckets))
    }

    fn open_ane(
        &self,
        dir: &Path,
        stage: &watch::Sender<LoadStage>,
    ) -> Result<Self::Ane, Self::LoadError> {
        let mut embedder = CoreMlEmbedder::open(dir, ComputeUnits::CpuAndNeuralEngine)?;
        stage.send_replace(LoadStage::WarmUp);
        // Warm every bucket once so the first request does not pay the ANE
        // compile.
        let buckets: Vec<usize> = embedder.buckets().collect();
        for len in buckets {
            embedder.embed_tokens(&vec![0; len])?;
        }
        Ok(embedder)
    }

    fn open_gpu(&self, dir: &Path) -> Result<Self::Gpu, GpuOpen<Self::LoadError>> {
        CoreMlEmbedder::open(dir, ComputeUnits::CpuAndGpu).map_err(|error| match error {
            Error::GpuBlocked(blocked) => GpuOpen::Blocked(blocked),
            other => GpuOpen::Failed(other),
        })
    }

    fn watch_headroom(&self, on_change: Box<dyn Fn(Headroom) + Send + Sync>) -> Self::Monitor {
        HeadroomMonitor::start(on_change).expect("failed to start the headroom monitor")
    }
}
