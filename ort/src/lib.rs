//! ONNX Runtime token embedding for repo-search: the Linux and Windows
//! counterpart of `repo-search-coreml`. This crate is compiled out on
//! macOS — the library is empty there and its dependencies are all
//! target-gated, so ONNX never enters an Apple build.
#![cfg(not(target_os = "macos"))]

use std::io;
use std::path::{Path, PathBuf};

use manifest::Manifest;
use ort::session::Session;
use ort::value::Tensor;
use repo_search_core::UnitVector;
use repo_search_core::embed_api::LoadStage;
use tokenizers::Tokenizer;
use tokio::sync::watch;

mod manifest;

pub mod server;

/// A [`OrtEmbedder`] or [`Encoder`] failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file in the model directory could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The I/O failure.
        source: io::Error,
    },
    /// `manifest.json` is not valid JSON.
    #[error("{path}: {source}")]
    Manifest {
        /// The manifest path.
        path: PathBuf,
        /// The parse failure.
        source: serde_json::Error,
    },
    /// `manifest.json` parsed but violates the model contract.
    #[error("{path}: {reason}")]
    ManifestContent {
        /// The manifest path.
        path: PathBuf,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// `tokenizer.json` failed to load, or encoding failed.
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// The tokenizer has no `<pad>` token.
    #[error("tokenizer has no <pad> token")]
    NoPadToken,
    /// The vocabulary does not fit the inputs' element type.
    #[error("tokenizer vocabulary exceeds the token-id type")]
    VocabTooLarge,
    /// The model contract names an input the ONNX graph does not take.
    #[error("model has no {0} input")]
    MissingInput(String),
    /// An input beyond `input_ids`/`attention_mask` cannot be fed: the
    /// model contract feeds a zero-length first dimension, which needs a
    /// static feature width.
    #[error("model input {0} has no static feature width")]
    ExtraInput(String),
    /// The model has no `sentence_embedding` output.
    #[error("model has no sentence_embedding output")]
    NoEmbedding,
    /// The input exceeds the largest bucket.
    #[error("{tokens} tokens exceeds the {max}-token maximum")]
    TooLong {
        /// The encoding's length.
        tokens: usize,
        /// The largest bucket.
        max: usize,
    },
    /// A batched embed's inputs do not share a bucket.
    #[error("inputs of one batch must share a bucket")]
    MixedBucket,
    /// The model returned the wrong embedding width or count.
    #[error("expected {want} embedding values, got {got}")]
    EmbeddingDim {
        /// What the model returned.
        got: usize,
        /// The manifest's `embedding_dim`.
        want: usize,
    },
    /// The ONNX Runtime failure.
    #[error("onnx runtime: {0}")]
    Ort(#[from] ort::Error),
}

/// The text encoder request inputs are tokenized with: a clone of the
/// model's tokenizer plus its largest bucket. Cloned handles share the
/// tokenizer's tables and are cheap.
#[derive(Clone)]
pub struct Encoder {
    tokenizer: Tokenizer,
    max_len: usize,
}

impl Encoder {
    /// Encodes `text` with special tokens.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when `text` encodes past the largest bucket, and
    /// [`Error::Tokenizer`] when encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(Error::Tokenizer)?
            .get_ids()
            .to_vec();
        if ids.len() > self.max_len {
            return Err(Error::TooLong {
                tokens: ids.len(),
                max: self.max_len,
            });
        }
        Ok(ids)
    }
}

/// An EmbeddingGemma ONNX session: one `input_ids`/`attention_mask` pair
/// plus any further graph inputs fed an empty tensor, and the
/// `sentence_embedding` output read back as a [`UnitVector`].
pub struct OrtEmbedder {
    session: Session,
    /// Every input of the session beyond `input_ids`/`attention_mask`,
    /// with the shape it is fed: a zero-length token dimension.
    extras: Vec<(String, Vec<i64>)>,
    encoder: Encoder,
    pad: i64,
    buckets: Box<[usize]>,
    dim: usize,
    /// The manifest's declared model id, what `/v1/models` lists.
    model_id: String,
}

impl OrtEmbedder {
    /// Loads the model directory (`manifest.json`, `tokenizer.json`, the
    /// named ONNX file and its sidecar data) on the CPU execution provider,
    /// then runs one warm embed so the first request does not pay the
    /// initial-run cost. `stage` is already `Encoder` on entry and tracks
    /// the pipeline's progress for `/status`.
    ///
    /// # Errors
    ///
    /// The manifest, tokenizer and model failures in [`Error`].
    pub fn open(dir: impl AsRef<Path>, stage: &watch::Sender<LoadStage>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.get_vocab_size(true) > i64::MAX as usize {
            return Err(Error::VocabTooLarge);
        }
        let pad = tokenizer
            .token_to_id("<pad>")
            .ok_or(Error::NoPadToken)? as i64;
        let max_len = *manifest.buckets.last().expect("manifest requires buckets");
        let encoder = Encoder { tokenizer, max_len };

        let _ = stage.send(LoadStage::Open);
        let environment = ort::init().with_name("repo-search-ort").build()?;
        let session = Session::builder(&environment)?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|error| Error::Ort(ort::Error::from(error)))?
            .commit_from_file(dir.join(&manifest.model))?;

        let mut seen = session
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<std::collections::HashSet<String>>();
        for name in ["input_ids", "attention_mask"] {
            if !seen.remove(name) {
                return Err(Error::MissingInput(name.to_owned()));
            }
        }
        // EmbeddingGemma's multimodal graph takes already-encoded feature
        // tensors for other modalities; the text-only server feeds each a
        // tensor with a zero-length token dimension. The feature width is
        // the input's static last dimension.
        let mut extras = Vec::new();
        for input in session.inputs() {
            if !seen.contains(input.name()) {
                continue;
            }
            let width = match input.dtype() {
                ort::value::ValueType::Tensor {
                    ty: ort::value::TensorElementType::Float32,
                    shape,
                    ..
                } if shape.last().is_some_and(|width| *width > 0) => {
                    *shape.last().expect("checked")
                }
                _ => return Err(Error::ExtraInput(input.name().to_owned())),
            };
            extras.push((input.name().to_owned(), vec![0, width]));
        }

        let _ = stage.send(LoadStage::WarmUp);
        let mut embedder = Self {
            session,
            extras,
            encoder,
            pad,
            buckets: manifest.buckets.into_boxed_slice(),
            dim: manifest.embedding_dim,
            model_id: manifest.model_id,
        };
        // One run at the largest bucket: the graph's initial compile and
        // the allocator's warm-up are paid here, not on the first request.
        let largest = *embedder.buckets.last().expect("manifest requires buckets");
        let warm = vec![pad as u32; largest];
        let _ = embedder.embed_tokens(&warm)?;
        Ok(embedder)
    }

    /// The manifest's `model_id`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The tokenizer request inputs encode with; cloneable.
    pub fn encoder(&self) -> Encoder {
        self.encoder.clone()
    }

    /// The bucket sequence lengths, ascending.
    pub fn buckets(&self) -> impl Iterator<Item = usize> + '_ {
        self.buckets.iter().copied()
    }

    /// The smallest bucket `len` fits.
    pub(crate) fn bucket_of(&self, len: usize) -> Result<usize, Error> {
        self.buckets
            .iter()
            .copied()
            .find(|&bucket| len <= bucket)
            .ok_or(Error::TooLong {
                tokens: len,
                max: *self.buckets.last().expect("manifest requires buckets"),
            })
    }

    /// Embeds already-tokenized prompt text, routed to the smallest bucket
    /// that holds it.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when `ids` exceeds the largest bucket, and the
    /// ONNX Runtime failures in [`Error`].
    pub fn embed_tokens(&mut self, ids: &[u32]) -> Result<UnitVector, Error> {
        let [vector] = self
            .run(&[ids.to_vec()])?
            .try_into()
            .map_err(|_| Error::EmbeddingDim {
                got: 0,
                want: 1,
            })?;
        Ok(vector)
    }

    /// One batch prediction over already-tokenized inputs sharing a bucket.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when an input exceeds the largest bucket,
    /// [`Error::MixedBucket`] when the inputs do not share a bucket, and
    /// the ONNX Runtime failures in [`Error`].
    pub fn embed_tokens_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Error> {
        self.run(ids)
    }

    /// The shared run: buckets the inputs by their first element's bucket,
    /// feeds every graph input, reads `sentence_embedding` back.
    fn run(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Error> {
        let Some(first) = ids.first() else {
            return Ok(Vec::new());
        };
        let bucket = self.bucket_of(first.len())?;
        let mut input_ids = Vec::with_capacity(ids.len() * bucket);
        let mut mask = Vec::with_capacity(ids.len() * bucket);
        for ids in ids {
            if self.bucket_of(ids.len())? != bucket {
                return Err(Error::MixedBucket);
            }
            input_ids.extend(ids.iter().map(|&id| id as i64));
            input_ids.extend(std::iter::repeat_n(self.pad, bucket - ids.len()));
            mask.extend(std::iter::repeat_n(1i64, ids.len()));
            mask.extend(std::iter::repeat_n(0i64, bucket - ids.len()));
        }
        let shape = vec![ids.len() as i64, bucket as i64];
        let mut inputs: Vec<(String, ort::session::SessionInputValue)> = vec![
            (
                "input_ids".to_owned(),
                Tensor::from_array((shape.clone(), input_ids))?.into(),
            ),
            (
                "attention_mask".to_owned(),
                Tensor::from_array((shape, mask))?.into(),
            ),
        ];
        for (name, shape) in &self.extras {
            inputs.push((
                name.clone(),
                Tensor::<f32>::from_array((shape.clone(), Vec::new()))?.into(),
            ));
        }
        let outputs = self.session.run(inputs)?;
        let embedding = &outputs["sentence_embedding"];
        let (shape, floats) = embedding.try_extract_tensor::<f32>()?;
        let per = shape
            .last()
            .copied()
            .map(|d| usize::try_from(d).unwrap_or(0))
            .unwrap_or(0);
        if per != self.dim {
            return Err(Error::EmbeddingDim {
                got: per,
                want: self.dim,
            });
        }
        let mut vectors = Vec::with_capacity(ids.len());
        for row in floats.chunks_exact(self.dim) {
            vectors.push(UnitVector::normalize(row.to_vec()));
        }
        Ok(vectors)
    }
}
