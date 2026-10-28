//! The wire types of the embedding server's OpenAI-compatible API:
//! `POST /v1/embeddings`, `GET /v1/models`, the RFC 9218 `Priority`
//! request header and the `Server-Timing` response header.

use std::fmt;
use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};
use sfv::{
    BareItem, Decimal, Dictionary, FieldType, Item, List, ListEntry, Parameters, Parser, integer,
    key_ref, string_ref, token_ref,
};

/// `POST /v1/embeddings`. Non-standard body fields are rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    /// The model id to embed with; anything but [`crate::model::MODEL_ID`]
    /// is a `model_not_found` 404.
    pub model: String,
    /// One string or an array of strings; token arrays fail to parse.
    pub input: Input,
    /// Matryoshka-truncated output width; only the model's widths are
    /// valid. `None` is the full width.
    #[serde(default)]
    pub dimensions: Option<u32>,
    /// `float` (the default) or `base64`.
    #[serde(default)]
    pub encoding_format: Option<EncodingFormat>,
    /// Standard field; accepted and ignored.
    #[serde(default)]
    pub user: Option<String>,
}

/// The `input` field: a string or an array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    /// `input: "text"`.
    One(String),
    /// `input: ["text", ...]`.
    Many(Vec<String>),
}

/// The `encoding_format` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// A JSON array of f32.
    Float,
    /// A base64 string of little-endian f32 bytes.
    Base64,
}

/// `object: "list"` of `data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// `"list"`.
    pub object: ListTag,
    /// One entry per input, in input order.
    pub data: Vec<Embedding>,
    /// The model id the vectors came from.
    pub model: String,
    /// Token usage of the request.
    pub usage: Usage,
}

/// `object: "list"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListTag {
    /// `"list"`.
    #[serde(rename = "list")]
    List,
}

/// `GET /v1/models`: `object: "list"` of `data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// `"list"`.
    pub object: ListTag,
    /// The models this server embeds.
    pub data: Vec<ModelEntry>,
}

/// `object: "model"` in the `GET /v1/models` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// The model's id.
    pub id: String,
    /// `"model"`.
    pub object: ModelTag,
    /// Creation time; always 0, the card has no epoch.
    pub created: u64,
    /// The owner, always `repo-search`.
    pub owned_by: String,
}

/// `object: "model"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTag {
    /// `"model"`.
    #[serde(rename = "model")]
    Model,
}

/// One input's vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// `"embedding"`.
    pub object: EmbeddingTag,
    /// The input's 0-based position.
    pub index: u32,
    /// The vector, in the request's `encoding_format`.
    pub embedding: EmbeddingData,
}

/// `object: "embedding"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingTag {
    /// `"embedding"`.
    #[serde(rename = "embedding")]
    Embedding,
}

/// The `embedding` field, in the request's `encoding_format`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    /// `float`: a JSON array of f32.
    Float(Vec<f32>),
    /// `base64`: little-endian f32 bytes.
    Base64(String),
}

/// `usage` on an embeddings response: `prompt_tokens` always equals
/// `total_tokens` — there is no completion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens the request's inputs encoded to, special tokens included.
    pub prompt_tokens: u32,
    /// The same count: the request produces no completion tokens.
    pub total_tokens: u32,
}

/// The standard error envelope: `{"error": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    /// The error itself.
    pub error: ErrorDetail,
}

/// The `error` member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable failure, also the server's log line.
    pub message: String,
    /// `invalid_request_error` or `server_error`.
    pub r#type: String,
    /// The offending parameter, when one applies.
    pub param: Option<String>,
    /// A machine-readable code such as `context_length_exceeded`, when one
    /// applies.
    pub code: Option<String>,
}

/// `GET /status`: the server's own report — model state (with archive
/// download progress and load stage) and GPU state, in the shapes the
/// daemon's `status` already knows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusBody {
    /// The model's state.
    pub model: ModelStatus,
    /// The GPU embedding instance's state.
    pub gpu: GpuStatus,
}

/// The model's state on `GET /status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelStatus {
    /// `ensure` has not produced a directory yet.
    Absent,
    /// The archive is downloading.
    Downloading {
        /// Bytes received so far.
        received: u64,
        /// The server's declared length, when known.
        total: Option<u64>,
    },
    /// The archive is verified; the encoder and the ANE instance load.
    Loading {
        /// The `open_ane` pipeline stage.
        stage: LoadStage,
    },
    /// Ready to embed.
    Ready {
        /// The model id.
        id: String,
    },
    /// Download or load failed; a request retries.
    Failed {
        /// The failure's message.
        message: String,
    },
    /// The load failed fatally; the server is shutting down.
    Fatal {
        /// The failure's message.
        message: String,
    },
}

/// The model load's stage while `model.type` is `loading`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStage {
    /// Reading the tokenizer and the manifest.
    Encoder,
    /// Inside the compiled-model open.
    Open,
    /// The per-bucket warm-up embeds.
    WarmUp,
}

/// The GPU embedding instance's state — the daemon's `GpuStatus` shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GpuStatus {
    /// No GPU instance exists.
    Absent {
        /// Why it cannot be created, when the check has run.
        blocked: Option<String>,
    },
    /// Being created (a multi-second cold start).
    Loading,
    /// Embedding normal-urgency work.
    Running,
}

/// An RFC 9218 `Priority` urgency: 0 is most urgent, 7 least.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Urgency(u8);

impl Urgency {
    /// The urgency a missing or invalid `Priority` header means.
    pub const DEFAULT: Self = Self(3);

    /// An urgency of `value`.
    ///
    /// # Panics
    ///
    /// When `value` exceeds 7.
    #[must_use]
    pub const fn new(value: u8) -> Self {
        assert!(value <= 7, "urgency is 0..=7");
        Self(value)
    }

    /// The urgency value, 0 to 7.
    #[must_use]
    pub fn get(self) -> u8 {
        self.0
    }

    /// Parses an RFC 9218 `Priority` dictionary header: the integer member
    /// `u` in `0..=7` is taken; `i`, unknown keys, non-integer values and
    /// out-of-range values are ignored. A missing or unparseable header
    /// means [`Urgency::DEFAULT`].
    #[must_use]
    pub fn from_header(header: Option<&str>) -> Self {
        let Some(header) = header else {
            return Self::DEFAULT;
        };
        let Ok(dictionary) = Parser::new(header).parse::<Dictionary>() else {
            return Self::DEFAULT;
        };
        let Some(ListEntry::Item(item)) = dictionary.get("u") else {
            return Self::DEFAULT;
        };
        match item.bare_item {
            BareItem::Integer(value) if (0..=7).contains(&i64::from(value)) => {
                Self(i64::from(value) as u8)
            }
            _ => Self::DEFAULT,
        }
    }

    /// `u=N`, for the daemon's requests.
    #[must_use]
    pub fn to_header(self) -> String {
        let mut dictionary = Dictionary::new();
        dictionary.insert(
            key_ref("u").to_owned(),
            Item::new(integer(i64::from(self.0))).into(),
        );
        dictionary
            .serialize()
            .expect("a nonempty dictionary serializes")
    }
}

/// The `Server-Timing` metrics an embeddings response carries: how long its
/// items waited behind lower-urgency work and how many such items they
/// overtook. `wait` is a request's max over its items, `overtaken` the sum.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ServerTiming {
    /// The longest an item of the request spent queued behind
    /// lower-urgency work.
    pub wait: Duration,
    /// The lower-urgency items the request's items overtook, summed.
    pub overtaken: u64,
}

impl ServerTiming {
    /// `wait;dur=<ms>, overtaken;desc="<n>"` — `dur` in milliseconds, as
    /// Server-Timing requires. `wait` is quantized to whole microseconds:
    /// `dur` carries three decimal places.
    #[must_use]
    pub fn to_header(self) -> String {
        let micros =
            i64::try_from(self.wait.as_micros()).expect("a queue wait fits i64 microseconds");
        let mut dur = Parameters::new();
        dur.insert(
            key_ref("dur").to_owned(),
            Decimal::from_integer_scaled_1000(integer(micros)).into(),
        );
        let mut desc = Parameters::new();
        desc.insert(
            key_ref("desc").to_owned(),
            BareItem::String(string_ref(&self.overtaken.to_string()).to_owned()),
        );
        let list: List = vec![
            Item::with_params(token_ref("wait"), dur).into(),
            Item::with_params(token_ref("overtaken"), desc).into(),
        ];
        list.serialize().expect("a nonempty list serializes")
    }

    /// Parses a `Server-Timing` header: `wait`'s `dur` decimal in
    /// milliseconds and `overtaken`'s `desc` string integer. Unrecognized
    /// metrics, malformed parameters and an unparseable header are ignored;
    /// absent fields stay zero.
    #[must_use]
    pub fn from_header(header: &str) -> Self {
        let mut timing = Self::default();
        let Ok(list) = Parser::new(header).parse::<List>() else {
            return timing;
        };
        for entry in list {
            let ListEntry::Item(item) = entry else {
                continue;
            };
            let BareItem::Token(metric) = &item.bare_item else {
                continue;
            };
            match metric.as_str() {
                "wait" => {
                    if let Some(BareItem::Decimal(ms)) = item.params.get("dur") {
                        timing.wait = Duration::from_secs_f64(f64::from(*ms) / 1e3);
                    }
                }
                "overtaken" => match item.params.get("desc") {
                    Some(BareItem::String(desc)) => {
                        timing.overtaken = desc.as_str().parse().unwrap_or(0);
                    }
                    Some(BareItem::Integer(desc)) => {
                        timing.overtaken = u64::try_from(i64::from(*desc)).unwrap_or(0);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        timing
    }
}

impl fmt::Display for ServerTiming {
    /// [`ServerTiming::to_header`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_header())
    }
}

/// A [`decode_vector`] failure.
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    /// The string is not valid base64.
    #[error("invalid base64: {0}")]
    Base64(#[from] base64::DecodeError),
    /// The decoded bytes do not hold whole f32s.
    #[error("base64 vector has {0} bytes, not a multiple of 4")]
    Length(usize),
}

/// The `base64` wire form of `vector`: its f32s as little-endian bytes.
#[must_use]
pub fn encode_vector(vector: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for f in vector {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decodes a `base64` vector.
///
/// # Errors
///
/// [`VectorError::Base64`] for malformed base64 and
/// [`VectorError::Length`] when the decoded bytes do not hold whole f32s.
pub fn decode_vector(text: &str) -> Result<Vec<f32>, VectorError> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(text)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(VectorError::Length(bytes.len()));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips() {
        let request = EmbeddingRequest {
            model: "m".to_owned(),
            input: Input::Many(vec!["a".to_owned(), "b".to_owned()]),
            dimensions: Some(256),
            encoding_format: Some(EncodingFormat::Base64),
            user: None,
        };
        let json = serde_json::to_string(&request).expect("serializes");
        let back: EmbeddingRequest = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.model, "m");
        assert_eq!(back.dimensions, Some(256));
        assert_eq!(back.encoding_format, Some(EncodingFormat::Base64));
        assert!(matches!(back.input, Input::Many(v) if v == ["a", "b"]));

        let one: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":"x"}"#).expect("deserializes");
        assert!(matches!(one.input, Input::One(s) if s == "x"));
        assert_eq!(one.dimensions, None);
        assert_eq!(one.encoding_format, None);
        assert_eq!(one.user, None);
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        assert!(
            serde_json::from_str::<EmbeddingRequest>(r#"{"model":"m","input":"x","nope":1}"#)
                .is_err()
        );
    }

    #[test]
    fn a_token_array_input_is_rejected() {
        assert!(
            serde_json::from_str::<EmbeddingRequest>(r#"{"model":"m","input":[1,2,3]}"#).is_err()
        );
    }

    #[test]
    fn response_round_trips() {
        let response = EmbeddingResponse {
            object: ListTag::List,
            data: vec![
                Embedding {
                    object: EmbeddingTag::Embedding,
                    index: 0,
                    embedding: EmbeddingData::Float(vec![0.5, -0.5]),
                },
                Embedding {
                    object: EmbeddingTag::Embedding,
                    index: 1,
                    embedding: EmbeddingData::Base64(encode_vector(&[1.0, 2.0])),
                },
            ],
            model: "m".to_owned(),
            usage: Usage {
                prompt_tokens: 7,
                total_tokens: 7,
            },
        };
        let json = serde_json::to_value(&response).expect("serializes");
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        let back: EmbeddingResponse = serde_json::from_value(json).expect("deserializes");
        assert_eq!(back.data.len(), 2);
        assert_eq!(
            back.data[0].embedding,
            EmbeddingData::Float(vec![0.5, -0.5])
        );
        assert!(matches!(back.data[1].embedding, EmbeddingData::Base64(_)));
    }

    #[test]
    fn error_body_round_trips() {
        let body = ErrorBody {
            error: ErrorDetail {
                message: "bad".to_owned(),
                r#type: "invalid_request_error".to_owned(),
                param: None,
                code: Some("context_length_exceeded".to_owned()),
            },
        };
        let json = serde_json::to_string(&body).expect("serializes");
        let back: ErrorBody = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.error.r#type, "invalid_request_error");
        assert_eq!(back.error.code.as_deref(), Some("context_length_exceeded"));
    }

    #[test]
    fn float_vectors_round_trip() {
        let vector = [0.1, -0.2, 3.5, f32::MIN_POSITIVE];
        let encoded = EmbeddingData::Float(vector.to_vec());
        let json = serde_json::to_string(&encoded).expect("serializes");
        let back: EmbeddingData = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, encoded);
    }

    #[test]
    fn base64_vectors_round_trip() {
        let vector = [0.1, -0.2, 3.5, f32::MIN_POSITIVE];
        let text = encode_vector(&vector);
        assert_eq!(decode_vector(&text).expect("decodes"), vector);
        let encoded = EmbeddingData::Base64(text);
        let json = serde_json::to_string(&encoded).expect("serializes");
        let back: EmbeddingData = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, encoded);
    }

    #[test]
    fn bad_base64_and_odd_lengths_fail() {
        assert!(matches!(decode_vector("!!!"), Err(VectorError::Base64(_))));
        // Three bytes: whole f32s never come out of it.
        assert!(matches!(decode_vector("AAAA"), Err(VectorError::Length(3))));
    }

    #[test]
    fn urgency_defaults_to_three() {
        assert_eq!(Urgency::from_header(None), Urgency::DEFAULT);
        for header in ["", "u=9", "u=x", "garbage", "i", "u=-1"] {
            assert_eq!(
                Urgency::from_header(Some(header)),
                Urgency::DEFAULT,
                "{header:?}"
            );
        }
    }

    #[test]
    fn urgency_reads_the_u_member() {
        for (header, want) in [
            ("u=0", 0),
            ("u=7, i", 7),
            ("i, u=2", 2),
            ("u=3", 3),
            ("sf=1, u=5", 5),
        ] {
            assert_eq!(
                Urgency::from_header(Some(header)),
                Urgency::new(want),
                "{header:?}"
            );
        }
        for urgency in 0..=7 {
            assert_eq!(
                Urgency::from_header(Some(&Urgency::new(urgency).to_header())),
                Urgency::new(urgency)
            );
        }
    }

    #[test]
    fn server_timing_formats_and_parses() {
        let timing = ServerTiming {
            wait: Duration::from_micros(1234),
            overtaken: 5,
        };
        assert_eq!(timing.to_header(), "wait;dur=1.234, overtaken;desc=\"5\"");
        assert_eq!(ServerTiming::from_header(&timing.to_header()), timing);
        assert_eq!(ServerTiming::from_header(""), ServerTiming::default());
        assert_eq!(
            ServerTiming::from_header("wait;dur=0.500, overtaken;desc=\"0\", other"),
            ServerTiming {
                wait: Duration::from_micros(500),
                overtaken: 0,
            }
        );
    }
}
