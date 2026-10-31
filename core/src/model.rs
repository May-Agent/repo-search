//! The model card: everything `repo-search` knows about the published
//! EmbeddingGemma 2 Core ML build, as data.
//!
//! `release-gate.yml` reads [`ARCHIVE`]'s pin back out of this file with
//! sed — the first `url:` line and the first line holding only a 64-hex
//! literal — so [`ARCHIVE`] stays above [`TOKENIZER`] and keeps its
//! sha256 literal on a line of its own.

/// A pinned download: the URL it comes from and the sha256 its bytes
/// verify against.
pub struct Pin {
    /// Download URL, pinned to an immutable upstream revision.
    pub url: &'static str,
    /// Expected sha256 of the downloaded bytes, lowercase hex.
    pub sha256: &'static str,
}

/// The published EmbeddingGemma 2 Core ML archive.
#[rustfmt::skip]
pub const ARCHIVE: Pin = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-coreml/resolve/5bb9f08270c0e2968d048a0a35e120863f32962c/eg2-coreml-914f7f89-57287aaa.tar.zst",
    sha256:
        "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c",
};

/// The published EmbeddingGemma 2 ONNX archive served off macOS:
/// `manifest.json`, `tokenizer.json`, the quantized graph and its
/// weights, packed as `tar.zst` like the Core ML build.
#[cfg(not(target_os = "macos"))]
#[rustfmt::skip]
pub const ONNX_ARCHIVE: Pin = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-onnx/resolve/090c8ce7415de7af5bcb3f8ba106ee6a1cfc2563/eg2-onnx-914f7f89-a4acdbe8.tar.zst",
    sha256:
        "a4acdbe8cbcdc2d2a8e95b81317a694c966f2de64a534dda5189b34faad14acf",
};

/// The fp16 variant of the ONNX archive, served on NVIDIA GPUs: CUDA's
/// tensor cores run fp16 natively, where the int8 quantization gains
/// nothing.
#[cfg(not(target_os = "macos"))]
#[rustfmt::skip]
pub const ONNX_FP16_ARCHIVE: Pin = Pin {
    url: "https://huggingface.co/lexoliu/repo-search-embeddinggemma2-onnx/resolve/d4e1b374f97451cf89262d03f4f8e387822511d4/eg2-onnx-fp16-914f7f89-ac32bf70.tar.zst",
    sha256:
        "ac32bf7023a81cf0d58e5f487ecb8e46979898d966f606bc564c5dba8585b414",
};

/// The tokenizer the daemon counts and prefixes chunks with, pinned at the
/// model's upstream revision; the archive ships the same bytes.
pub const TOKENIZER: Pin = Pin {
    url: "https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json",
    sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
};

/// The model build's identity, on `/v1/models` and in the vector cache:
/// the archive's sha256, so vectors from different builds never mix.
#[cfg(target_os = "macos")]
pub const MODEL_ID: &str = ARCHIVE.sha256;

/// The model build's identity off macOS: `onnx-` plus the sha256 of the
/// pinned ONNX graph file the server's manifest names, so ONNX and Core
/// ML vectors never mix. The manifest a model directory ships declares
/// this same id.
#[cfg(not(target_os = "macos"))]
pub const MODEL_ID: &str = "onnx-d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953";

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
    /// The server's declared length, when known.
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
