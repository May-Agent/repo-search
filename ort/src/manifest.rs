//! The `manifest.json` a model directory ships beside the ONNX model.

use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// A model directory's manifest: the ONNX file, its identity and the
/// bucket lengths the server pads inputs to.
#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    /// The model build's identity: what `/v1/models` lists and the vector
    /// cache keys embeddings under.
    pub model_id: String,
    /// The ONNX file, relative to the model directory.
    pub model: String,
    /// Width of the `sentence_embedding` output.
    pub embedding_dim: usize,
    /// The sequence lengths the server pads inputs to, ascending.
    pub buckets: Vec<usize>,
}

impl Manifest {
    pub(crate) fn load(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("manifest.json");
        let bytes = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|source| Error::Manifest {
            path: path.clone(),
            source,
        })?;
        if manifest.model_id.is_empty() {
            return Err(Error::ManifestContent {
                path,
                reason: "empty model_id",
            });
        }
        if manifest.buckets.is_empty() {
            return Err(Error::ManifestContent {
                path,
                reason: "no buckets",
            });
        }
        if manifest.buckets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::ManifestContent {
                path,
                reason: "buckets are not strictly ascending",
            });
        }
        if manifest.embedding_dim == 0 {
            return Err(Error::ManifestContent {
                path,
                reason: "zero embedding_dim",
            });
        }
        Ok(manifest)
    }
}
