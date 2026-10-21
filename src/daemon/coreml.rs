//! The production [`Backend`] over `repo-search-coreml`.

use std::path::Path;

use repo_search_coreml::{ComputeUnits, CoreMlEmbedder, DocumentCounter, Error, HeadroomMonitor};

use super::{Backend, GpuOpen, Headroom, LoadStage};

/// The daemon's Core ML backend.
pub struct CoreMlBackend;

impl Backend for CoreMlBackend {
    type Ane = CoreMlEmbedder;
    type Gpu = CoreMlEmbedder;
    type Counter = DocumentCounter;
    type LoadError = Error;
    type Monitor = HeadroomMonitor;

    fn load_counter(&self, dir: &Path) -> Result<(Self::Counter, Box<[usize]>), Self::LoadError> {
        let counter = DocumentCounter::open(dir)?;
        let buckets = counter.buckets().into();
        Ok((counter, buckets))
    }

    fn open_ane(
        &self,
        dir: &Path,
        model_id: &str,
        stage: &tokio::sync::watch::Sender<LoadStage>,
    ) -> Result<Self::Ane, Self::LoadError> {
        let mut embedder = CoreMlEmbedder::open(dir, model_id, ComputeUnits::CpuAndNeuralEngine)?;
        stage.send_replace(LoadStage::WarmUp);
        // Warm every bucket once so the first query does not pay the ANE
        // compile.
        let buckets: Vec<usize> = embedder.counter().buckets().to_vec();
        for len in buckets {
            embedder.embed_tokens(&vec![0; len])?;
        }
        Ok(embedder)
    }

    fn open_gpu(&self, dir: &Path, model_id: &str) -> Result<Self::Gpu, GpuOpen<Self::LoadError>> {
        CoreMlEmbedder::open(dir, model_id, ComputeUnits::CpuAndGpu).map_err(|error| match error {
            Error::GpuBlocked(blocked) => GpuOpen::Blocked(blocked),
            other => GpuOpen::Failed(other),
        })
    }

    fn watch_headroom(&self, on_change: Box<dyn Fn(Headroom) + Send + Sync>) -> Self::Monitor {
        HeadroomMonitor::start(on_change).expect("failed to start the headroom monitor")
    }
}
