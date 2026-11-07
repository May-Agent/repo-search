//! ONNX Runtime token embedding for repo-search: the Linux and Windows
//! counterpart of `repo-search-coreml`. This crate is compiled out on
//! macOS — the library is empty there and its dependencies are all
//! target-gated, so ONNX never enters an Apple build.
#![cfg(not(target_os = "macos"))]

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// `REPO_SEARCH_ORT_EP` named a provider this build does not know.
    #[error("unknown execution provider: {0}")]
    UnknownProvider(String),
    /// `REPO_SEARCH_ORT_EP` mixed `cpu` with accelerator providers; CPU
    /// runs only when no accelerator is available, never alongside one.
    #[error("cpu cannot share a provider chain with accelerators")]
    CpuMixed,
    /// `REPO_SEARCH_ORT_DEVICES` is not a positive integer.
    #[error("REPO_SEARCH_ORT_DEVICES must be a positive integer, got {0:?}")]
    BadDeviceCap(String),
    /// The ONNX Runtime failure.
    #[error("onnx runtime: {0}")]
    Ort(#[from] ort::Error),
}

/// The text encoder request inputs are tokenized with: a clone of the
/// model's tokenizer plus its largest bucket. Cloned handles share the one
/// tokenizer: `Tokenizer` itself deep-copies its vocabulary and merge
/// tables, so a handle is an `Arc`.
#[derive(Clone)]
pub struct Encoder {
    tokenizer: Arc<Tokenizer>,
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

/// The candidate execution-provider chains, in preference order; the
/// first that commits a session wins.
///
/// CPU is only ever a chain of its own: the `cpu` provider never shares a
/// chain with an accelerator, so it is the explicit last resort rather than
/// part of an accelerator's registration. ONNX Runtime's implicit CPU
/// provider still takes the shape operators no accelerator claims; that is
/// expected and not a fallback. `REPO_SEARCH_ORT_EP` names one explicit
/// comma-separated chain — mixing `cpu` with an accelerator there is a
/// startup error — while the unset default tries each accelerator alone,
/// ending on CPU only when none of them registered.
fn ep_chains() -> Result<Vec<Vec<String>>, Error> {
    let spec = match std::env::var("REPO_SEARCH_ORT_EP") {
        Ok(spec) => spec,
        Err(_) => {
            return Ok(vec![
                vec!["qnn".to_owned()],
                vec!["openvino-npu".to_owned()],
                vec!["directml".to_owned()],
                vec!["cuda".to_owned()],
                vec!["cpu".to_owned()],
            ]);
        }
    };
    let chain: Vec<String> = spec
        .split(',')
        .map(|name| name.trim().to_lowercase())
        .filter(|name| !name.is_empty())
        .collect();
    if chain.len() > 1 && chain.iter().any(|name| name == "cpu") {
        return Err(Error::CpuMixed);
    }
    if chain.is_empty() {
        return Err(Error::UnknownProvider(String::new()));
    }
    Ok(vec![chain])
}

/// Whether the provider `name` selects among several devices by id.
fn takes_device_id(name: &str) -> bool {
    matches!(name, "cuda" | "tensorrt" | "nvrtx")
}

/// Whether every provider of `chain` runs a graph at any sequence length
/// without a per-shape cost. The NPU providers compile for fixed shapes and
/// TensorRT builds an engine per shape, so their batches keep the bucket
/// length.
fn takes_any_width(chain: &[String]) -> bool {
    chain
        .iter()
        .all(|name| matches!(name.as_str(), "cuda" | "cpu"))
}

/// Tensor Cores take fp16 operands in multiples of this many elements, so a
/// batch narrowed below its bucket rounds its width up to it.
const WIDTH_ALIGN: usize = 8;

/// The saturation probe doubles a batch while doing so raises throughput by
/// at least this fraction: past that, a larger run mostly lengthens the
/// wait of a more urgent request queued behind it.
const MIN_DOUBLING_GAIN: f64 = 0.1;

/// Timed runs per batch size in the saturation probe; their median counts.
const PROBE_RUNS: usize = 3;

/// The padded sequence length of a batch whose longest input has `longest`
/// tokens and whose bucket is `bucket`.
fn batch_width(narrow: bool, longest: usize, bucket: usize) -> usize {
    if narrow {
        longest.next_multiple_of(WIDTH_ALIGN).min(bucket)
    } else {
        bucket
    }
}

/// How an embedder pads a batch: to its bucket, or to its longest input
/// when the provider chain runs any width.
#[derive(Debug, Clone)]
pub struct Widths {
    pub(crate) narrow: bool,
    /// The bucket lengths, ascending.
    pub(crate) buckets: Box<[usize]>,
}

impl Widths {
    /// The padded sequence length of a batch whose longest input has
    /// `longest` tokens; the encoder caps every input at the largest
    /// bucket.
    pub fn of(&self, longest: usize) -> usize {
        let bucket = self
            .buckets
            .iter()
            .copied()
            .find(|&bucket| longest <= bucket)
            .expect("the encoder capped every input");
        batch_width(self.narrow, longest, bucket)
    }

    /// The smallest bucket.
    pub fn smallest(&self) -> usize {
        self.buckets[0]
    }
}

/// The `REPO_SEARCH_ORT_DEVICES` cap on how many devices to serve from;
/// `None` when unset.
fn device_cap() -> Result<Option<usize>, Error> {
    match std::env::var("REPO_SEARCH_ORT_DEVICES") {
        Err(_) => Ok(None),
        Ok(spec) => match spec.trim().parse::<usize>() {
            Ok(cap) if cap > 0 => Ok(Some(cap)),
            _ => Err(Error::BadDeviceCap(spec)),
        },
    }
}

/// Commits a session over `model`, registering `chain`'s providers in
/// order — ONNX Runtime runs on the first provider that accepts each
/// node, so an NPU-capable name placed first prefers the NPU. Providers
/// that select a device are bound to `device`. Every registration is
/// `error_on_failure`: ONNX Runtime otherwise logs a failed registration
/// and runs on the CPU, so a failure here is an `Err`.
fn commit_session(
    environment: &ort::environment::Environment,
    chain: &[String],
    device: usize,
    model: PathBuf,
) -> Result<Session, Error> {
    let id = i32::try_from(device).expect("device index fits a device id");
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::with_capacity(chain.len());
    for name in chain {
        let provider = match name.as_str() {
            "cpu" => ort::ep::CPU::default().build(),
            "openvino" => ort::ep::OpenVINO::default().build(),
            "openvino-npu" => ort::ep::OpenVINO::default().with_device_type("NPU").build(),
            "qnn" => ort::ep::QNN::default().build(),
            "directml" => ort::ep::DirectML::default().build(),
            "cuda" => ort::ep::CUDA::default().with_device_id(id).build(),
            "tensorrt" => ort::ep::TensorRT::default().with_device_id(id).build(),
            "nvrtx" => ort::ep::NVRTX::default()
                .with_device_id(device as u32)
                .build(),
            "xnnpack" => ort::ep::XNNPACK::default().build(),
            "onednn" => ort::ep::OneDNN::default().build(),
            other => return Err(Error::UnknownProvider(other.to_owned())),
        };
        providers.push(provider.error_on_failure());
    }
    Ok(Session::builder(environment)?
        .with_execution_providers(providers)
        .map_err(ort::Error::from)?
        .commit_from_file(model)?)
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
    /// Whether a batch is padded to its longest input rather than to its
    /// bucket: the chain's providers run any sequence length at no
    /// per-shape cost.
    narrow: bool,
    dim: usize,
    /// The manifest's declared model id, what `/v1/models` lists.
    model_id: String,
    /// The padded tokens per run that saturate this device.
    run_tokens: usize,
}

impl OrtEmbedder {
    /// Loads the model directory (`manifest.json`, `tokenizer.json`, the
    /// named ONNX file and its sidecar data) on the first execution-provider
    /// chain that works, then runs one warm embed per session so the first
    /// request does not pay the initial-run cost, and measures the batch
    /// that saturates each device. When that chain's
    /// accelerator selects among devices (CUDA, TensorRT, NVRTX), the same
    /// chain is opened on device 1, 2, ... until one fails or
    /// `REPO_SEARCH_ORT_DEVICES` devices are open; each opened device yields
    /// one embedder. `stage` is already `Encoder` on entry and tracks the
    /// pipeline's progress for `/status`.
    ///
    /// # Errors
    ///
    /// The manifest, tokenizer and model failures in [`Error`], and
    /// [`Error::BadDeviceCap`]. A failure on a device beyond the first only
    /// ends the search.
    pub fn open_devices(
        dir: impl AsRef<Path>,
        stage: &watch::Sender<LoadStage>,
    ) -> Result<Vec<Self>, Error> {
        let dir = dir.as_ref();
        let cap = device_cap()?;
        let manifest = Manifest::load(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.get_vocab_size(true) > i64::MAX as usize {
            return Err(Error::VocabTooLarge);
        }
        let pad = tokenizer.token_to_id("<pad>").ok_or(Error::NoPadToken)? as i64;
        let max_len = *manifest.buckets.last().expect("manifest requires buckets");
        let encoder = Encoder {
            tokenizer: Arc::new(tokenizer),
            max_len,
        };

        let _ = stage.send(LoadStage::Open);
        let environment = ort::init().with_name("repo-search-ort").build()?;
        // Provider preference: candidate chains in order; the first whose
        // session commits AND passes the warm run wins — a provider that
        // registers but cannot execute falls through to the next
        // candidate. An explicit `REPO_SEARCH_ORT_EP` is a single chain,
        // so its failure is fatal and never a silent drop to CPU.
        let candidates = ep_chains()?;
        let mut selected = None;
        let mut last_error = None;
        for chain in &candidates {
            match Self::open_on(
                &environment,
                chain,
                0,
                dir,
                &manifest,
                encoder.clone(),
                pad,
                stage,
            ) {
                Ok(opened) => {
                    tracing::info!(providers = ?chain, "execution provider selected");
                    selected = Some((chain, opened));
                    break;
                }
                Err(error) => {
                    tracing::warn!(providers = ?chain, %error, "provider chain unavailable");
                    last_error = Some(error);
                }
            }
        }
        let (chain, first) =
            selected.ok_or_else(|| last_error.expect("candidates are nonempty"))?;
        let mut embedders = vec![first];
        if chain.iter().any(|name| takes_device_id(name)) {
            while cap.is_none_or(|cap| embedders.len() < cap) {
                let device = embedders.len();
                match Self::open_on(
                    &environment,
                    chain,
                    device,
                    dir,
                    &manifest,
                    encoder.clone(),
                    pad,
                    stage,
                ) {
                    Ok(opened) => embedders.push(opened),
                    Err(error) => {
                        tracing::debug!(device, %error, "no further device");
                        break;
                    }
                }
            }
        }
        Ok(embedders)
    }

    /// One candidate chain's try: commits the session, checks the model
    /// contract, then proves the provider executes with one warm run at
    /// the largest bucket — paying the graph's initial compile and the
    /// allocator's warm-up here, not on the first request.
    #[allow(clippy::too_many_arguments)]
    fn open_on(
        environment: &ort::environment::Environment,
        chain: &[String],
        device: usize,
        dir: &Path,
        manifest: &Manifest,
        encoder: Encoder,
        pad: i64,
        stage: &watch::Sender<LoadStage>,
    ) -> Result<Self, Error> {
        let session = commit_session(environment, chain, device, dir.join(&manifest.model))?;
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
            buckets: manifest.buckets.clone().into_boxed_slice(),
            narrow: takes_any_width(chain),
            dim: manifest.embedding_dim,
            model_id: manifest.model_id.clone(),
            run_tokens: 0,
        };
        let largest = *embedder.buckets.last().expect("manifest requires buckets");
        let warm = vec![pad as u32; largest];
        embedder.embed_tokens(&warm)?;
        embedder.run_tokens = embedder.saturating_batch(&warm)? * largest;
        tracing::info!(
            device,
            run_tokens = embedder.run_tokens,
            "device saturation measured"
        );
        Ok(embedder)
    }

    /// The smallest batch of `ids` past which doubling the batch raises
    /// throughput by less than [`MIN_DOUBLING_GAIN`]. Saturation depends
    /// on the device, the provider and the graph, so it is measured here
    /// rather than known beforehand.
    fn saturating_batch(&mut self, ids: &[u32]) -> Result<usize, Error> {
        let mut batch = 1;
        let mut rate = self.throughput(batch, ids)?;
        loop {
            let doubled = self.throughput(batch * 2, ids)?;
            if doubled < rate * (1.0 + MIN_DOUBLING_GAIN) {
                return Ok(batch);
            }
            batch *= 2;
            rate = doubled;
        }
    }

    /// Inputs per second over `batch` copies of `ids`: the median of
    /// [`PROBE_RUNS`] runs after one that settles the shape.
    fn throughput(&mut self, batch: usize, ids: &[u32]) -> Result<f64, Error> {
        let inputs = vec![ids.to_vec(); batch];
        self.run(&inputs)?;
        let mut times = Vec::with_capacity(PROBE_RUNS);
        for _ in 0..PROBE_RUNS {
            let started = std::time::Instant::now();
            self.run(&inputs)?;
            times.push(started.elapsed());
        }
        times.sort_unstable();
        Ok(batch as f64 / times[PROBE_RUNS / 2].as_secs_f64())
    }

    /// The padded tokens a run should carry to keep this device saturated:
    /// its measured saturating batch at the largest bucket.
    pub fn run_tokens(&self) -> usize {
        self.run_tokens
    }

    /// How this embedder pads a batch.
    pub fn widths(&self) -> Widths {
        Widths {
            narrow: self.narrow,
            buckets: self.buckets.clone(),
        }
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
            .map_err(|_| Error::EmbeddingDim { got: 0, want: 1 })?;
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
        for ids in ids {
            if self.bucket_of(ids.len())? != bucket {
                return Err(Error::MixedBucket);
            }
        }
        let longest = ids.iter().map(Vec::len).max().expect("ids is nonempty");
        let width = batch_width(self.narrow, longest, bucket);
        let mut input_ids = Vec::with_capacity(ids.len() * width);
        let mut mask = Vec::with_capacity(ids.len() * width);
        for ids in ids {
            input_ids.extend(ids.iter().map(|&id| id as i64));
            input_ids.extend(std::iter::repeat_n(self.pad, width - ids.len()));
            mask.extend(std::iter::repeat_n(1i64, ids.len()));
            mask.extend(std::iter::repeat_n(0i64, width - ids.len()));
        }
        let shape = vec![ids.len() as i64, width as i64];
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn only_shape_free_providers_narrow_a_batch() {
        assert!(takes_any_width(&chain(&["cuda"])));
        assert!(takes_any_width(&chain(&["cpu"])));
        assert!(!takes_any_width(&chain(&["tensorrt"])));
        assert!(!takes_any_width(&chain(&["qnn"])));
        assert!(!takes_any_width(&chain(&["openvino-npu"])));
        assert!(!takes_any_width(&chain(&["cuda", "tensorrt"])));
    }

    #[test]
    fn a_narrow_batch_pads_to_its_longest_input_in_tensor_core_steps() {
        assert_eq!(batch_width(true, 129, 256), 136);
        assert_eq!(batch_width(true, 136, 256), 136);
        assert_eq!(batch_width(true, 1, 128), 8);
        // Never past the bucket.
        assert_eq!(batch_width(true, 256, 256), 256);
        // A fixed-shape provider keeps the bucket.
        assert_eq!(batch_width(false, 129, 256), 256);
    }
}
