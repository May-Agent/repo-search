//! Core ML token embedding for `repo-search` (Apple Silicon).
//!
//! A [`CoreMlEmbedder`] is the EmbeddingGemma 2 conversion's
//! [`ondevice_coreml::Model`] — one function per bucket (`len_128`,
//! `len_256`, `len_512`) sharing the weights of one `.mlmodelc` — plus what
//! embedding means for it: token ids padded with `<pad>` and an attention
//! mask of ones in, the `float16 [1, dim]` embedding normalized to a
//! [`UnitVector`] out. Embedding is blocking; the caller owns the thread and
//! its QoS.

use std::path::Path;

use ondevice_coreml::{DataType, Dim, Input, Manifest, Model, Prediction, Sequence, f16};
use repo_search_core::UnitVector;
use tokenizers::Tokenizer;

pub use ondevice_coreml::{ComputeUnits, GpuBlocked, HeadroomMonitor, gpu_headroom};

/// The token-id input: `int32 [1, length]`.
const INPUT_IDS: &str = "input_ids";

/// The attention-mask input: `float16 [1, length]`, one over the tokens.
const ATTENTION_MASK: &str = "attention_mask";

/// A [`CoreMlEmbedder`] or [`Encoder`] failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model directory or Core ML failed.
    #[error(transparent)]
    Model(ondevice_coreml::Error),
    /// The manifest describes a model the embedder cannot drive.
    #[error("the model is not an embedding model: {0}")]
    Contract(&'static str),
    /// `tokenizer.json` failed to load, or encoding failed.
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// The tokenizer has no `<pad>` token.
    #[error("tokenizer has no <pad> token")]
    NoPadToken,
    /// The tokenizer's vocabulary does not fit the `int32` model input.
    #[error("tokenizer vocabulary exceeds i32::MAX")]
    VocabTooLarge,
    /// A token id does not fit the `int32` model input.
    #[error("token id {0} exceeds i32::MAX")]
    TokenIdTooLarge(u32),
    /// The embedding's element count differs from the manifest.
    #[error("embedding has {got} elements, expected {want}")]
    EmbeddingDim {
        /// What the model produced.
        got: usize,
        /// What the manifest declared.
        want: usize,
    },
    /// The input exceeded the largest bucket; for documents that is a chunker
    /// bug (the chunker guarantees the cap), so this fails loudly.
    #[error("input has {tokens} tokens, over the {max}-token cap")]
    TooLong {
        /// Tokens in the input.
        tokens: usize,
        /// Largest bucket length.
        max: usize,
    },
    /// GPU compute units were requested without [`gpu_headroom`].
    #[error("GPU compute units unavailable: {0}")]
    GpuBlocked(#[from] GpuBlocked),
}

impl From<ondevice_coreml::Error> for Error {
    fn from(error: ondevice_coreml::Error) -> Self {
        match error {
            ondevice_coreml::Error::TooLong { len, max } => Self::TooLong { tokens: len, max },
            error => Self::Model(error),
        }
    }
}

/// The embedding output's name and width, once `manifest` is checked
/// against what the embedder feeds and reads.
fn contract(manifest: &Manifest) -> Result<(String, usize), Error> {
    let typed = |name: &str, data_type: DataType| {
        manifest
            .inputs()
            .get(name)
            .is_some_and(|spec| spec.data_type == data_type)
    };
    if !typed(INPUT_IDS, DataType::Int32) || !typed(ATTENTION_MASK, DataType::Float16) {
        return Err(Error::Contract(
            "the inputs are not int32 input_ids and float16 attention_mask",
        ));
    }
    let mut outputs = manifest.outputs().iter();
    let (Some((name, spec)), None) = (outputs.next(), outputs.next()) else {
        return Err(Error::Contract("expected exactly one output"));
    };
    match (spec.data_type, spec.shape.as_slice()) {
        (DataType::Float16, [Dim::Fixed(1), Dim::Fixed(dim)]) => Ok((name.clone(), *dim)),
        _ => Err(Error::Contract("the output is not float16 [1, dim]")),
    }
}

/// Embeds token ids through the compiled multifunction Core ML model.
pub struct CoreMlEmbedder {
    model: Model,
    pad: i32,
    /// The single declared output's feature name.
    embedding: String,
    dim: usize,
}

/// Turns request text into token ids: the model's tokenizer loaded without
/// the compiled model, capped at the largest bucket.
pub struct Encoder {
    tokenizer: Tokenizer,
    /// Ascending.
    buckets: Box<[usize]>,
}

impl Encoder {
    /// Loads `tokenizer.json` and `manifest.json` from the model directory;
    /// the compiled model is not needed for encoding.
    ///
    /// # Errors
    ///
    /// The manifest and tokenizer failures of [`CoreMlEmbedder::open`].
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        contract(&manifest)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        Ok(Self {
            tokenizer,
            buckets: manifest.functions().iter().map(|f| f.len).collect(),
        })
    }

    /// The model's bucket lengths in tokens, ascending.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// Encodes `text` with special tokens on.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when the encoding exceeds the largest bucket, and
    /// [`Error::Tokenizer`] when encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let ids = self.tokenizer.encode(text, true)?.get_ids().to_vec();
        let max = self.buckets[self.buckets.len() - 1];
        if ids.len() > max {
            return Err(Error::TooLong {
                tokens: ids.len(),
                max,
            });
        }
        Ok(ids)
    }
}

impl CoreMlEmbedder {
    /// Loads the model directory (the extracted archive: `*.mlmodelc`,
    /// `tokenizer.json`, `manifest.json`) for `units`.
    ///
    /// # Errors
    ///
    /// `GpuBlocked` when `units` is [`ComputeUnits::CpuAndGpu`] without
    /// [`gpu_headroom`]; the model, manifest-contract and tokenizer failures
    /// above.
    pub fn open(dir: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, Error> {
        if let ComputeUnits::CpuAndGpu = units {
            gpu_headroom()?;
        }
        let dir = dir.as_ref();
        let model = Model::open(dir, units)?;
        let (embedding, dim) = contract(model.manifest())?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.get_vocab_size(true) > i32::MAX as usize {
            return Err(Error::VocabTooLarge);
        }
        let pad = i32::try_from(tokenizer.token_to_id("<pad>").ok_or(Error::NoPadToken)?)
            .expect("vocab fits i32");
        Ok(Self {
            model,
            pad,
            embedding,
            dim,
        })
    }

    /// The bucket sequence lengths, ascending.
    pub fn buckets(&self) -> impl Iterator<Item = usize> + '_ {
        self.model.manifest().functions().iter().map(|f| f.len)
    }

    /// Embeds already-tokenized prompt text: `ids` are token ids of a string
    /// the prompt format already produced (document or query prompt), routed
    /// to the smallest bucket that holds them.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when `ids` exceeds the largest bucket, and the Core
    /// ML failures above.
    pub fn embed_tokens(&mut self, ids: &[u32]) -> Result<UnitVector, Error> {
        let ids = token_ids(ids)?;
        let mask = vec![f16::ONE; ids.len()];
        let prediction = self.model.predict(&[&self.ids(&ids), &Self::mask(&mask)])?;
        self.read(&prediction)
    }

    /// One Core ML batch prediction over already-tokenized inputs, padded to
    /// the smallest bucket that holds the longest. The scheduler batches per
    /// bucket, so that is every input's own bucket.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when an input exceeds the largest bucket, and the
    /// Core ML failures above.
    pub fn embed_tokens_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Error> {
        let ids = ids
            .iter()
            .map(|ids| token_ids(ids))
            .collect::<Result<Vec<_>, _>>()?;
        let masks: Vec<Vec<f16>> = ids.iter().map(|ids| vec![f16::ONE; ids.len()]).collect();
        let ids_inputs: Vec<Sequence<'_, i32>> = ids.iter().map(|ids| self.ids(ids)).collect();
        let mask_inputs: Vec<Sequence<'_, f16>> =
            masks.iter().map(|mask| Self::mask(mask)).collect();
        let items: Vec<[&dyn Input; 2]> = ids_inputs
            .iter()
            .zip(&mask_inputs)
            .map(|(ids, mask)| [ids as &dyn Input, mask as &dyn Input])
            .collect();
        let batch: Vec<&[&dyn Input]> = items.iter().map(|item| item.as_slice()).collect();
        self.model
            .predict_batch(&batch)?
            .iter()
            .map(|prediction| self.read(prediction))
            .collect()
    }

    /// The token-id input for `ids`, padded with `<pad>`.
    fn ids<'a>(&self, ids: &'a [i32]) -> Sequence<'a, i32> {
        Sequence {
            name: INPUT_IDS,
            values: ids,
            pad: self.pad,
        }
    }

    /// The attention-mask input over `mask`'s ones, padded with zeros.
    fn mask(mask: &[f16]) -> Sequence<'_, f16> {
        Sequence {
            name: ATTENTION_MASK,
            values: mask,
            pad: f16::ZERO,
        }
    }

    /// The prediction's embedding, normalized.
    fn read(&self, prediction: &Prediction) -> Result<UnitVector, Error> {
        let embedding = prediction.output::<f16>(&self.embedding)?;
        if embedding.data().len() != self.dim {
            return Err(Error::EmbeddingDim {
                got: embedding.data().len(),
                want: self.dim,
            });
        }
        Ok(UnitVector::normalize(
            embedding.data().iter().map(|v| v.to_f32()).collect(),
        ))
    }
}

/// `ids` as the `int32` the model takes.
fn token_ids(ids: &[u32]) -> Result<Vec<i32>, Error> {
    ids.iter()
        .map(|&id| i32::try_from(id).map_err(|_| Error::TokenIdTooLarge(id)))
        .collect()
}
