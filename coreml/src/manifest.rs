//! The `manifest.json` the conversion pipeline ships beside the compiled model.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// A model directory's manifest: one Core ML function per token bucket.
#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    /// Core ML function name to the sequence length it accepts (`len_128`: 128).
    pub functions: BTreeMap<String, usize>,
    /// Width of the `embedding` output.
    pub embedding_dim: usize,
    /// Declared inputs; `input_ids` must be int32, `attention_mask` float16.
    pub inputs: BTreeMap<String, Io>,
    /// Declared outputs; exactly one, the embedding.
    pub outputs: BTreeMap<String, Io>,
}

/// A declared input or output tensor.
#[derive(Debug, Deserialize)]
pub(crate) struct Io {
    /// `int32`, `float16`, ...
    pub dtype: String,
    /// Tensor shape; the sequence dimension is the string `length`.
    pub shape: Vec<serde_json::Value>,
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
        if manifest.functions.is_empty() {
            return Err(Error::ManifestContent {
                path,
                reason: "no functions",
            });
        }
        if manifest.outputs.len() != 1 {
            return Err(Error::ManifestContent {
                path,
                reason: "expected exactly one output",
            });
        }
        let output = manifest.outputs.values().next().expect("one output");
        let dim = serde_json::Value::from(manifest.embedding_dim as u64);
        if output.dtype != "float16" || output.shape.last() != Some(&dim) {
            return Err(Error::ManifestContent {
                path,
                reason: "output is not float16 [1, embedding_dim]",
            });
        }
        for (name, dtype) in [("input_ids", "int32"), ("attention_mask", "float16")] {
            match manifest.inputs.get(name) {
                Some(io) if io.dtype == dtype => {}
                _ => {
                    return Err(Error::ManifestContent {
                        path,
                        reason: "missing or mistyped input",
                    });
                }
            }
        }
        Ok(manifest)
    }
}
