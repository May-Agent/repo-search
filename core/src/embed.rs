//! The embedding interface every model backend implements.

/// A unit-length embedding; only constructible by normalizing.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitVector(Box<[f32]>);

impl UnitVector {
    /// Scales `v` to unit L2 length.
    ///
    /// # Panics
    ///
    /// When `v` is empty, has zero length, or holds a non-finite component: a model
    /// that produces such an output is broken, and a vector built from it would
    /// poison every similarity it takes part in.
    #[must_use]
    pub fn normalize(mut v: Vec<f32>) -> Self {
        assert!(!v.is_empty(), "cannot normalize an empty vector");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm.is_finite() && norm > 0.0,
            "cannot normalize a vector of length {norm}"
        );
        let inv = norm.recip();
        for x in &mut v {
            *x *= inv;
        }
        Self(v.into_boxed_slice())
    }

    /// The components, of unit L2 length.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
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

/// Counts the model's tokens. The chunker enforces its cap on every chunk's
/// embedded document through this interface; the tokenizer and the document
/// prompt the count applies to live in the embedding crate, so core never
/// formats a prompt.
pub trait TokenCounter: Send + Sync {
    /// The number of tokens `text` encodes to, special tokens included — the
    /// count the model consumes.
    fn count(&self, text: &str) -> usize;

    /// Whether the document the model embeds for `title` — its text is
    /// `text` joined — encodes to at most `max` tokens. Implementations may
    /// answer without encoding when a byte bound already decides it.
    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool;

    /// An estimate of where the document the model embeds for `title` —
    /// `text` joined — runs out of tokens: the byte offset into the joined
    /// `text` the document's first `max` tokens end at, special tokens
    /// included. `None` when the document fits.
    ///
    /// This is an estimate, not a boundary: merges spanning the cut can
    /// move where the `max`-th token truly ends, so callers confirm it with
    /// [`TokenCounter::fits`].
    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize>;

    /// Tokens of the document the model embeds for `title` and `text`,
    /// special tokens included.
    fn document_tokens(&self, title: &str, text: &str) -> usize;
}

/// A text encoder. Calls block: callers run it on a thread they own and set its QoS.
pub trait Embedder {
    /// The backend's failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Identifies the model build; vectors with different ids never mix.
    fn model_id(&self) -> &str;

    /// Embeds a document chunk under its title.
    ///
    /// # Errors
    ///
    /// When the backend fails to run the model.
    fn embed_document(&mut self, title: &str, text: &str) -> Result<UnitVector, Self::Error>;

    /// Embeds a search query for `task`; the implementation applies
    /// [`QueryTask::prompt`].
    ///
    /// # Errors
    ///
    /// When the backend fails to run the model.
    fn embed_query(&mut self, task: QueryTask, query: &str) -> Result<UnitVector, Self::Error>;
}

/// An encoder that embeds several documents in one model call.
pub trait BatchEmbedder: Embedder {
    /// One vector per `(title, text)`, in order. All documents fall in the same bucket.
    ///
    /// # Errors
    ///
    /// When the backend fails to run the model.
    fn embed_documents(&mut self, docs: &[(&str, &str)]) -> Result<Vec<UnitVector>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::{QueryTask, UnitVector};

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

    #[test]
    fn normalize_scales_to_unit_length() {
        let v = UnitVector::normalize(vec![3.0, 4.0]);
        assert_eq!(v.as_slice(), &[0.6, 0.8]);
    }

    #[test]
    #[should_panic(expected = "length 0")]
    fn normalize_rejects_zero() {
        let _ = UnitVector::normalize(vec![0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn normalize_rejects_empty() {
        let _ = UnitVector::normalize(Vec::new());
    }
}
