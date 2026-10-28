//! The daemon's token counter: the model's own tokenizer, downloaded
//! through [`crate::model_store::TokenizerStore`], applying the document
//! and query prompts the embedding server sees.

use std::path::Path;

use repo_search_core::TokenCounter;
use repo_search_core::model::{BUCKETS, DOCUMENT, MAX_TOKENS, QueryTask};
use tokenizers::Tokenizer;

/// Counts the model's tokens for the chunker's cap and for query
/// truncation. The tokenizer file is the pinned `tokenizer.json` the
/// archive ships, and the prompts it formats are [`DOCUMENT`] and
/// [`QueryTask::prompt`] — the same the server tokenizes, so counts
/// here and encodes there can never skew.
pub struct PromptCounter {
    tokenizer: Tokenizer,
    /// Tokens the post-processor adds to every encoding (`<s>`, `</s>`).
    specials: usize,
}

impl PromptCounter {
    /// The counter over an already-loaded `tokenizer`.
    ///
    /// # Errors
    ///
    /// When the tokenizer cannot encode the empty string used to measure
    /// its added special tokens.
    pub fn of(tokenizer: Tokenizer) -> Result<Self, tokenizers::Error> {
        let specials = tokenizer.encode("", true)?.len();
        Ok(Self {
            tokenizer,
            specials,
        })
    }

    /// Loads the tokenizer from `path` — the file
    /// [`crate::model_store::TokenizerStore`] caches.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or parsed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, tokenizers::Error> {
        Self::of(Tokenizer::from_file(path)?)
    }

    /// The model's bucket lengths in tokens, ascending.
    pub fn buckets(&self) -> &[usize] {
        &BUCKETS
    }

    /// The request `input` for a query: `task`'s prompt for `query`, cut
    /// to the model's cap. The cut starts at the byte offset where the
    /// `MAX_TOKENS - specials`-th token begins, then shrinks a char at a
    /// time while `count` still disagrees — merges spanning the cut can
    /// push the encoding back over.
    #[must_use]
    pub fn query_input(&self, task: QueryTask, query: &str) -> String {
        let mut prompt = task.prompt(query);
        if self.count(&prompt) <= MAX_TOKENS {
            return prompt;
        }
        let encoding = self
            .tokenizer
            .encode(&*prompt, true)
            .expect("a loaded tokenizer encodes every input");
        let mut cut = encoding
            .get_offsets()
            .get(MAX_TOKENS - self.specials)
            .map(|(start, _)| *start)
            .unwrap_or(prompt.len())
            .min(prompt.len());
        while !prompt.is_char_boundary(cut) {
            cut -= 1;
        }
        prompt.truncate(cut);
        while self.count(&prompt) > MAX_TOKENS && prompt.pop().is_some() {}
        prompt
    }

    /// The document `title` and `text` encode to, when the byte bound
    /// does not decide it first: `None` when it fits `max` tokens, else
    /// the formatted prompt.
    fn document_over(&self, title: &str, text: &[&str], max: usize) -> Option<String> {
        let bytes = DOCUMENT.bytes(title, text);
        // Every token covers at least one byte, so a document whose bytes
        // plus the special tokens stay under `max` fits without an encode.
        if bytes + self.specials <= max {
            return None;
        }
        let mut doc = String::with_capacity(bytes);
        doc.push_str(DOCUMENT.before);
        doc.push_str(title);
        doc.push_str(DOCUMENT.between);
        for part in text {
            doc.push_str(part);
        }
        doc.push_str(DOCUMENT.after);
        Some(doc)
    }
}

impl TokenCounter for PromptCounter {
    fn count(&self, text: &str) -> usize {
        self.tokenizer
            .encode_fast(text, true)
            .expect("a loaded tokenizer encodes every input")
            .len()
    }

    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool {
        match self.document_over(title, text, max) {
            None => true,
            Some(doc) => self.count(&doc) <= max,
        }
    }

    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize> {
        let doc = self.document_over(title, text, max)?;
        let encoding = self
            .tokenizer
            .encode(&*doc, true)
            .expect("a loaded tokenizer encodes every input");
        if encoding.len() <= max {
            return None;
        }
        let head = DOCUMENT.head_len(title);
        // The end of the `max`-th token in the document, mapped back into
        // `text`. The estimate is exact only up to merges spanning the cut.
        Some(
            encoding.get_offsets()[max.saturating_sub(1)]
                .1
                .saturating_sub(head),
        )
    }

    fn document_tokens(&self, title: &str, text: &str) -> usize {
        self.count(&DOCUMENT.format(title, text))
    }
}
