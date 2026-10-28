//! Core ML token embedding for `repo-search` (Apple Silicon).
//!
//! A [`CoreMlEmbedder`] wraps one compiled multifunction model: one `MLModel`
//! per bucket function (`len_128`, `len_256`, `len_512`), all loaded from the
//! same `.mlmodelc` so the weights are shared. Embedding is blocking; the
//! caller owns the thread and its QoS.

use std::io;
use std::path::{Path, PathBuf};

use std::ffi::c_void;
use std::ptr::NonNull;

use block2::StackBlock;
use numkong::f16;
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLComputeUnits, MLDictionaryFeatureProvider,
    MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration, MLMultiArray,
    MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSInteger, NSNumber, NSString, NSURL};
use repo_search_core::UnitVector;
use tokenizers::Tokenizer;

use manifest::Manifest;

mod manifest;
mod power;

pub use power::{GpuBlocked, HeadroomMonitor, gpu_headroom};

/// The hardware a [`CoreMlEmbedder`]'s functions run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnits {
    /// Neural Engine + CPU: the query and background-indexing instance.
    CpuAndNeuralEngine,
    /// GPU + CPU: the daemon's stage-1 indexing instance, which never serves
    /// queries. Construction is refused without [`gpu_headroom`].
    CpuAndGpu,
}

/// A [`CoreMlEmbedder`] failure.
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
    /// The model directory holds no `*.mlmodelc`, or several.
    #[error("{0}: expected exactly one .mlmodelc bundle")]
    ModelDir(PathBuf),
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
    /// Core ML returned an `NSError`.
    #[error("{domain} error {code}: {message}")]
    CoreMl {
        /// The error's domain.
        domain: String,
        /// The error's code.
        code: NSInteger,
        /// `localizedDescription`.
        message: String,
    },
    /// The prediction produced no embedding multi-array.
    #[error("prediction returned no embedding multi-array")]
    NoEmbedding,
    /// A batch prediction returned a different number of results than inputs.
    #[error("batch returned {got} predictions for {want} inputs")]
    BatchSize {
        /// What the model produced.
        got: usize,
        /// The number of inputs.
        want: usize,
    },
    /// The embedding multi-array is not `float16` with unit last stride.
    #[error("embedding is {dtype:?} with last stride {stride}, expected contiguous float16")]
    EmbeddingLayout {
        /// The array's data type.
        dtype: MLMultiArrayDataType,
        /// The array's last stride, in elements.
        stride: NSInteger,
    },
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

/// One bucket function's `MLModel`.
struct Bucket {
    /// Sequence length the function accepts.
    len: usize,
    model: Retained<MLModel>,
}

/// Embeds token ids through a compiled multifunction Core ML model.
pub struct CoreMlEmbedder {
    pad: i32,
    /// Ascending by `len`.
    buckets: Box<[Bucket]>,
    /// `buckets.last().len`: the token cap.
    max_len: usize,
    dim: usize,
    /// The single declared output's feature name.
    embedding: String,
}

// SAFETY: `MLModel` and `MLMultiArray` may be used from any thread provided
// calls on the same object do not overlap. `CoreMlEmbedder` is `!Sync` and
// every method that touches them takes `&mut self`, so calls on its objects
// never overlap; moving the embedder between threads preserves that.
unsafe impl Send for CoreMlEmbedder {}

/// Turns request text into token ids: the model's tokenizer loaded without
/// the compiled model, capped at the largest bucket.
pub struct Encoder {
    tokenizer: Tokenizer,
    max_len: usize,
    buckets: Box<[usize]>,
}

impl Encoder {
    /// Loads `tokenizer.json` and `manifest.json` from the model directory;
    /// the compiled model is not needed for encoding.
    ///
    /// # Errors
    ///
    /// The manifest, tokenizer or model load failures of
    /// [`CoreMlEmbedder::open`], minus the model's own.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let buckets = bucket_lengths(&manifest);
        let max_len = *buckets
            .last()
            .expect("manifest validation requires a function");
        Ok(Self {
            tokenizer,
            max_len,
            buckets,
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
        let ids = encode(&self.tokenizer, text)?;
        if ids.len() > self.max_len {
            return Err(Error::TooLong {
                tokens: ids.len(),
                max: self.max_len,
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
    /// [`gpu_headroom`]; the manifest, tokenizer or model load failures above.
    pub fn open(dir: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, Error> {
        if let ComputeUnits::CpuAndGpu = units {
            gpu_headroom()?;
        }
        let dir = dir.as_ref();
        let manifest = Manifest::load(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.get_vocab_size(true) > i32::MAX as usize {
            return Err(Error::VocabTooLarge);
        }
        let pad = i32::try_from(tokenizer.token_to_id("<pad>").ok_or(Error::NoPadToken)?)
            .expect("vocab fits i32");
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&compiled_bundle(dir)?.to_string_lossy()),
            true,
        );
        let units = match units {
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
        };
        let mut buckets: Vec<Bucket> = manifest
            .functions
            .iter()
            .map(|(function, len)| {
                // SAFETY: plain Objective-C allocation and init; there are no
                // invariants the caller must uphold.
                let config = unsafe { MLModelConfiguration::new() };
                // SAFETY: `config` is a live configuration, `units` a valid
                // `MLComputeUnits`, and `function` names a function the
                // manifest declares.
                unsafe {
                    config.setComputeUnits(units);
                    config.setFunctionName(Some(&NSString::from_str(function)));
                }
                // SAFETY: `url` is the compiled bundle and `config` a valid
                // configuration; a load failure is reported through `error`.
                let model =
                    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                        .map_err(core_ml)?;
                Ok(Bucket { len: *len, model })
            })
            .collect::<Result<_, Error>>()?;
        buckets.sort_unstable_by_key(|b| b.len);
        let max_len = buckets.last().expect("manifest requires functions").len;

        let embedding = manifest
            .outputs
            .into_keys()
            .next()
            .expect("manifest validation checked the output count");
        Ok(Self {
            pad,
            buckets: buckets.into_boxed_slice(),
            max_len,
            dim: manifest.embedding_dim,
            embedding,
        })
    }

    /// The bucket sequence lengths, ascending.
    pub fn buckets(&self) -> impl Iterator<Item = usize> + '_ {
        self.buckets.iter().map(|b| b.len)
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
        let bucket = self.bucket_for(ids.len())?;
        let provider = input_features(ids, bucket.len, self.pad)?;
        // SAFETY: `provider` is a valid MLFeatureProvider.
        let out = unsafe {
            bucket
                .model
                .predictionFromFeatures_error(&ProtocolObject::from_retained(provider))
        }
        .map_err(core_ml)?;
        read_embedding(&out, &self.embedding, self.dim)
    }

    /// One Core ML batch prediction over already-tokenized inputs, all of
    /// which must land in the first one's bucket.
    ///
    /// # Panics
    ///
    /// When the inputs do not all fall in the first one's bucket: the
    /// scheduler batches per bucket, so a mix is a caller bug.
    ///
    /// # Errors
    ///
    /// [`Error::TooLong`] when the first input exceeds the largest bucket,
    /// [`Error::BatchSize`] when the model returns a different count and the
    /// Core ML failures above.
    pub fn embed_tokens_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Error> {
        let Some(first) = ids.first() else {
            return Ok(Vec::new());
        };
        let bucket = self.bucket_for(first.len())?;
        let providers = ids
            .iter()
            .map(|ids| {
                assert_eq!(
                    self.bucket_for(ids.len())?.len,
                    bucket.len,
                    "inputs of one batch must share a bucket"
                );
                let provider = input_features(ids, bucket.len, self.pad)?;
                Ok(ProtocolObject::from_retained(provider))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let providers = NSArray::from_retained_slice(&providers);
        // SAFETY: `providers` holds valid MLFeatureProviders.
        let batch = unsafe {
            MLArrayBatchProvider::initWithFeatureProviderArray(
                MLArrayBatchProvider::alloc(),
                &providers,
            )
        };
        // SAFETY: `batch` is a valid MLBatchProvider.
        let out = unsafe {
            bucket
                .model
                .predictionsFromBatch_error(&ProtocolObject::from_retained(batch))
        }
        .map_err(core_ml)?;
        // SAFETY: `out` is a batch provider the model returned.
        let count = unsafe { out.count() } as usize;
        if count != ids.len() {
            return Err(Error::BatchSize {
                got: count,
                want: ids.len(),
            });
        }
        (0..ids.len())
            .map(|i| {
                // SAFETY: `i` is below `out.count()`.
                let features = unsafe { out.featuresAtIndex(i as NSInteger) };
                read_embedding(&features, &self.embedding, self.dim)
            })
            .collect()
    }

    /// The smallest bucket that holds `tokens`.
    fn bucket_for(&self, tokens: usize) -> Result<&Bucket, Error> {
        self.buckets
            .iter()
            .find(|b| tokens <= b.len)
            .ok_or(Error::TooLong {
                tokens,
                max: self.max_len,
            })
    }
}

/// Tokenizes `text` with `tokenizer` (special tokens on).
fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    Ok(tokenizer.encode(text, true)?.get_ids().to_vec())
}

/// The manifest's function lengths, ascending.
fn bucket_lengths(manifest: &Manifest) -> Box<[usize]> {
    let mut lens: Vec<usize> = manifest.functions.values().copied().collect();
    lens.sort_unstable();
    lens.into_boxed_slice()
}

/// The one `*.mlmodelc` bundle in `dir`, canonicalized.
fn compiled_bundle(dir: &Path) -> Result<PathBuf, Error> {
    let mut bundles = std::fs::read_dir(dir)
        .map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .map(|entry| {
            entry.map(|entry| entry.path()).map_err(|source| Error::Io {
                path: dir.to_path_buf(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|e| e == "mlmodelc"));
    let bundle = bundles
        .next()
        .ok_or_else(|| Error::ModelDir(dir.to_path_buf()))?;
    if bundles.next().is_some() {
        return Err(Error::ModelDir(dir.to_path_buf()));
    }
    bundle.canonicalize().map_err(|source| Error::Io {
        path: bundle,
        source,
    })
}

/// The model inputs for `ids` padded to the bucket length `len`.
fn input_features(
    ids: &[u32],
    len: usize,
    pad: i32,
) -> Result<Retained<MLDictionaryFeatureProvider>, Error> {
    // ObjC handler blocks must not panic, so every conversion happens
    // before they run and the blocks are plain copies.
    let ids: Vec<i32> = ids
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| Error::TokenIdTooLarge(id)))
        .collect::<Result<_, Error>>()?;
    let shape = NSArray::from_retained_slice(&[NSNumber::new_usize(1), NSNumber::new_usize(len)]);
    // SAFETY: `alloc` and `initWithShape:dataType:` are sound here; the
    // returned array owns `len` elements and stays alive through the
    // feature provider, so Core ML's deferred release frees memory it owns.
    let input_ids = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Int32,
        )
    }
    .map_err(core_ml)?;
    let fill_ids = StackBlock::new(
        |ptr: NonNull<c_void>, _size: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
            // SAFETY: `strides` is a live `NSArray` holding one entry per
            // dimension of the rank-2 array, so index 1 is in bounds.
            let stride = unsafe { strides.as_ref() }.objectAtIndex(1).as_isize() as usize;
            // SAFETY: `ptr` is the backing store for `len` i32s `stride`
            // elements apart, valid for the block's duration; every
            // element is written exactly once.
            unsafe {
                let p = ptr.as_ptr().cast::<i32>();
                for (i, &id) in ids.iter().enumerate() {
                    p.add(i * stride).write(id);
                }
                for i in ids.len()..len {
                    p.add(i * stride).write(pad);
                }
            }
        },
    );
    // SAFETY: `fill_ids` writes only within the `len` elements `ptr` and
    // `strides` describe, and keeps no pointer past the block.
    unsafe { input_ids.getMutableBytesWithHandler(&fill_ids) };
    // SAFETY: as for `input_ids` above, with the float16 element type.
    let mask = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(core_ml)?;
    let one = f16::from_f32(1.0).0;
    let fill_mask = StackBlock::new(
        |ptr: NonNull<c_void>, _size: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
            // SAFETY: `strides` is a live `NSArray` holding one entry per
            // dimension of the rank-2 array, so index 1 is in bounds.
            let stride = unsafe { strides.as_ref() }.objectAtIndex(1).as_isize() as usize;
            // SAFETY: `ptr` is the backing store for `len` fp16s `stride`
            // elements apart, valid for the block's duration; every
            // element is written exactly once.
            unsafe {
                let p = ptr.as_ptr().cast::<u16>();
                for i in 0..len {
                    p.add(i * stride).write(if i < ids.len() { one } else { 0 });
                }
            }
        },
    );
    // SAFETY: `fill_mask` writes only within the `len` elements `ptr` and
    // `strides` describe, and keeps no pointer past the block.
    unsafe { mask.getMutableBytesWithHandler(&fill_mask) };
    let ids_v = feature_value(&input_ids);
    let mask_v = feature_value(&mask);
    let features: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
        &[&*ns("input_ids"), &*ns("attention_mask")],
        &[&ids_v, &mask_v],
    );
    // SAFETY: `features` pairs NSString keys with feature values.
    unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &features,
        )
    }
    .map_err(core_ml)
}

/// The embedding `provider` holds under the feature name `embedding`.
fn read_embedding(
    provider: &ProtocolObject<dyn MLFeatureProvider>,
    embedding: &str,
    dim: usize,
) -> Result<UnitVector, Error> {
    // SAFETY: `provider` is a feature provider the model returned.
    let value =
        unsafe { provider.featureValueForName(&ns(embedding)) }.ok_or(Error::NoEmbedding)?;
    // SAFETY: `value` is a live feature value; `multiArrayValue` returns
    // nil when the feature is not a multi-array.
    let array = unsafe { value.multiArrayValue() }.ok_or(Error::NoEmbedding)?;
    // SAFETY: `array` is a live multi-array.
    let got = unsafe { array.count() } as usize;
    if got != dim {
        return Err(Error::EmbeddingDim { got, want: dim });
    }
    // SAFETY: `array` is a live multi-array.
    let dtype = unsafe { array.dataType() };
    // SAFETY: `array` is a live multi-array.
    let strides = unsafe { array.strides() };
    let stride = strides.lastObject().map_or(-1, |s| s.as_isize());
    if dtype != MLMultiArrayDataType::Float16 || stride != 1 {
        return Err(Error::EmbeddingLayout { dtype, stride });
    }
    let out = std::cell::Cell::new(Vec::new());
    let read = StackBlock::new(|ptr: NonNull<c_void>, _size: NSInteger| {
        // SAFETY: `ptr` is `size` bytes — `dim` contiguous fp16s given the
        // dtype and stride checks above — valid for the block's duration;
        // every element is read once.
        let p = ptr.as_ptr().cast::<u16>();
        out.set(
            (0..dim)
                .map(|i| f16(unsafe { p.add(i).read() }).to_f32())
                .collect(),
        );
    });
    // SAFETY: `read` reads only within `ptr`..`ptr + size` while the block
    // runs and keeps no pointer past it.
    unsafe { array.getBytesWithHandler(&read) };
    Ok(UnitVector::normalize(out.into_inner()))
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

fn feature_value(array: &MLMultiArray) -> Retained<MLFeatureValue> {
    // SAFETY: `array` is a live multi-array.
    unsafe { MLFeatureValue::featureValueWithMultiArray(array) }
}

fn core_ml(error: Retained<NSError>) -> Error {
    Error::CoreMl {
        domain: error.domain().to_string(),
        code: error.code(),
        message: error.localizedDescription().to_string(),
    }
}
