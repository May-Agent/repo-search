//! The model card: everything `repo-search` knows about the published
//! EmbeddingGemma 2 Core ML build, as data.
//!
//! `release-gate.yml` reads [`ARCHIVE`]'s pin back out of this file with
//! sed — the first `url:` line and the first line holding only a 64-hex
//! literal — so [`ARCHIVE`] stays above [`TOKENIZER`] and keeps its
//! sha256 literal on a line of its own.

use ondevice::{Archive, File, Pin, Sha256};

/// The published EmbeddingGemma 2 Core ML archive.
#[rustfmt::skip]
pub const ARCHIVE: Pin<'static, Archive> = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-coreml/resolve/5bb9f08270c0e2968d048a0a35e120863f32962c/eg2-coreml-914f7f89-57287aaa.tar.zst",
    sha256: Sha256::from_hex(
        "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c",
    ),
    size: 417_260_546,
    kind: Archive,
};

/// The published EmbeddingGemma 2 ONNX archive served off macOS:
/// `manifest.json`, `tokenizer.json`, the fp16 graph and its weights,
/// packed as `tar.zst` like the Core ML build, as `repo-search-convert-onnx`
/// builds it. fp16 is the default on every execution provider: CUDA's
/// tensor cores run it about three times faster than the int8 graph, and on
/// a CPU it is no slower.
#[cfg(not(target_os = "macos"))]
#[rustfmt::skip]
pub const ONNX_ARCHIVE: Pin<'static, Archive> = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-onnx/resolve/2984e5b2329ad4d0b9580821095c440090c88faf/eg2-onnx-fp16-914f7f89-715a4677.tar.zst",
    sha256: Sha256::from_hex(
        "715a46779e5895affb7c71672778ff336d395c06c720db1b8c3f43b52c035a9c",
    ),
    size: 444_377_768,
    kind: Archive,
};

/// The int8 (weight-quantized) variant of the ONNX archive, opted into with
/// `REPO_SEARCH_ORT_ARCHIVE=int8`.
#[cfg(not(target_os = "macos"))]
#[rustfmt::skip]
pub const ONNX_INT8_ARCHIVE: Pin<'static, Archive> = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-onnx/resolve/090c8ce7415de7af5bcb3f8ba106ee6a1cfc2563/eg2-onnx-914f7f89-a4acdbe8.tar.zst",
    sha256: Sha256::from_hex(
        "a4acdbe8cbcdc2d2a8e95b81317a694c966f2de64a534dda5189b34faad14acf",
    ),
    size: 291_242_960,
    kind: Archive,
};

/// The tokenizer the daemon counts and prefixes chunks with, pinned at the
/// model's upstream revision; the archive ships the same bytes.
#[rustfmt::skip]
pub const TOKENIZER: Pin<'static, File> = Pin {
    url: "https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json",
    sha256: Sha256::from_hex(
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
    size: 32_170_510,
    kind: File,
};

/// [`ARCHIVE`]'s digest in hex, the storage [`MODEL_ID`] borrows.
#[cfg(target_os = "macos")]
const ARCHIVE_HEX: &ondevice::Hex = &ARCHIVE.sha256.hex();

/// The model build's identity, on `/v1/models` and in the vector cache:
/// the archive's sha256, so vectors from different builds never mix.
#[cfg(target_os = "macos")]
pub const MODEL_ID: &str = ARCHIVE_HEX.as_str();

/// The model build's identity off macOS: `onnx-` plus the sha256 of the
/// pinned ONNX graph file the server's manifest names, so ONNX and Core
/// ML vectors never mix. The manifest a model directory ships declares
/// this same id.
#[cfg(not(target_os = "macos"))]
pub const MODEL_ID: &str = "onnx-730f8bdb5a3f42b2a02d3a2d23852e70c82c184d4bebcfa236c383069e25628b";

/// The model build's identity of [`ONNX_INT8_ARCHIVE`], which its manifest
/// declares: the int8 graph is a different build from [`MODEL_ID`]'s, so
/// their vectors never mix.
#[cfg(not(target_os = "macos"))]
pub const ONNX_INT8_MODEL_ID: &str =
    "onnx-d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953";

/// Human-readable name of the embedded model, shown by `search --version`.
#[cfg(target_os = "macos")]
pub const MODEL_NAME: &str = "google/embeddinggemma-2 (Core ML conversion)";

/// Human-readable name of the embedded model, shown by `search --version`.
#[cfg(not(target_os = "macos"))]
pub const MODEL_NAME: &str = "google/embeddinggemma-2 (ONNX conversion)";

/// SPDX identifier of the model's license, shown by `search --version`.
pub const MODEL_LICENSE: &str = "Apache-2.0";

/// The sequence lengths the model's functions accept, ascending; the
/// scheduler batches GPU work inside one bucket.
pub const BUCKETS: [usize; 3] = [128, 256, 512];

/// The largest document the model embeds: the largest bucket.
pub const MAX_TOKENS: usize = BUCKETS[BUCKETS.len() - 1];

/// The prompt EmbeddingGemma 2 embeds document chunks under:
/// `title: {title} | text: {text}`.
pub struct DocumentPrompt {
    /// Before the title.
    pub before: &'static str,
    /// Between title and text.
    pub between: &'static str,
    /// After the text.
    pub after: &'static str,
}

/// The document prompt [`crate::TokenCounter`] counts and the embedding
/// server sees.
pub const DOCUMENT: DocumentPrompt = DocumentPrompt {
    before: "title: ",
    between: " | text: ",
    after: "",
};

impl DocumentPrompt {
    /// The document `title` and `text` format to.
    #[must_use]
    pub fn format(&self, title: &str, text: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.before, title, self.between, text, self.after
        )
    }

    /// The byte length of the document `title` and the `text` parts format
    /// to, without building it.
    #[must_use]
    pub fn bytes(&self, title: &str, text: &[&str]) -> usize {
        self.before.len()
            + title.len()
            + self.between.len()
            + text.iter().map(|part| part.len()).sum::<usize>()
            + self.after.len()
    }

    /// The bytes of a `title` document's head: everything before the text.
    /// [`crate::TokenCounter::fit_end`] subtracts it to map a document
    /// offset back into `text`.
    #[must_use]
    pub fn head_len(&self, title: &str) -> usize {
        self.before.len() + title.len() + self.between.len()
    }
}

/// How a query is phrased for the embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryTask {
    /// The code list: `task: code retrieval | query: <query>`.
    CodeRetrieval,
    /// The docs list: `task: question answering | query: <query>`.
    QuestionAnswering,
}

impl QueryTask {
    /// The prompt text the model is given for `query`.
    #[must_use]
    pub fn prompt(self, query: &str) -> String {
        let task = match self {
            Self::CodeRetrieval => "code retrieval",
            Self::QuestionAnswering => "question answering",
        };
        format!("task: {task} | query: {query}")
    }
}

/// Progress of a model-file download, reported on a `watch` channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fetch {
    /// Bytes received so far.
    pub received: u64,
    /// The file's pinned length; `None` before the download starts.
    pub total: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_formats_title_and_text() {
        assert_eq!(
            DOCUMENT.format("main.rs", "fn main() {}"),
            "title: main.rs | text: fn main() {}"
        );
        assert_eq!(DOCUMENT.format("t", ""), "title: t | text: ");
    }

    #[test]
    fn document_bytes_matches_the_formatted_length() {
        let text = ["fn main() {\n", "    run();\n", "}"];
        assert_eq!(
            DOCUMENT.bytes("main.rs", &text),
            DOCUMENT.format("main.rs", &text.concat()).len()
        );
    }

    #[test]
    fn document_head_len_covers_the_prefix() {
        let doc = DOCUMENT.format("main.rs", "body");
        assert_eq!(
            &doc[..DOCUMENT.head_len("main.rs")],
            "title: main.rs | text: "
        );
    }

    #[test]
    fn prompts_name_the_task() {
        assert_eq!(
            QueryTask::CodeRetrieval.prompt("find x"),
            "task: code retrieval | query: find x"
        );
        assert_eq!(
            QueryTask::QuestionAnswering.prompt("what is x"),
            "task: question answering | query: what is x"
        );
    }
}
