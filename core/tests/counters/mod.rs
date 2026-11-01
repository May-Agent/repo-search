//! Test counters for `TokenCounter`, shared by the unit tests (included
//! through `#[path]` in the crate root) and the integration tests.

use repo_search_core::TokenCounter;
use repo_search_core::model::DOCUMENT;

/// A `TokenCounter` reporting zero tokens: the cap never binds, for tests of
/// rules the cap does not change.
#[allow(dead_code)]
pub struct CapOff;

impl TokenCounter for CapOff {
    fn count(&self, _: &str) -> usize {
        0
    }

    fn fits(&self, _: &str, _: &[&str], _: usize) -> bool {
        true
    }

    fn fit_end(&self, _: &str, _: &[&str], _: usize) -> Option<usize> {
        None
    }

    fn document_tokens(&self, _: &str, _: &str) -> usize {
        0
    }
}

/// A `TokenCounter` where removing a leading space raises the count: over
/// the text, each maximal run of non-whitespace chars costs 1 token when a
/// space precedes it, else 2 — a run at the very start included. The
/// title is ignored; `fit_end` guesses `0` and leaves the gallop to
/// correct it.
#[allow(dead_code)]
pub struct Glued;

#[allow(dead_code)]
impl Glued {
    /// The token count of `text`: one per non-whitespace run after a
    /// space, two per other run.
    fn glued(text: &str) -> usize {
        let mut count = 0;
        let mut prev = None;
        for c in text.chars() {
            if !c.is_whitespace() && prev.is_none_or(char::is_whitespace) {
                count += if prev == Some(' ') { 1 } else { 2 };
            }
            prev = Some(c);
        }
        count
    }
}

impl TokenCounter for Glued {
    fn count(&self, text: &str) -> usize {
        Self::glued(text)
    }

    fn fits(&self, _: &str, text: &[&str], max: usize) -> bool {
        Self::glued(&text.concat()) <= max
    }

    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize> {
        (!self.fits(title, text, max)).then_some(0)
    }

    fn document_tokens(&self, _: &str, text: &str) -> usize {
        Self::glued(text)
    }
}

/// A `TokenCounter` counting one token per `unit` chars.
#[allow(dead_code)]
pub struct CharTokens(pub usize);

impl TokenCounter for CharTokens {
    fn count(&self, text: &str) -> usize {
        text.chars().count() / self.0
    }

    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool {
        self.count(&DOCUMENT.format(title, &text.concat())) <= max
    }

    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize> {
        if self.fits(title, text, max) {
            return None;
        }
        // `max` tokens cover `max * unit` chars of the document, of which
        // the `title: {title} | text: ` head is not part of `text`.
        let head = DOCUMENT
            .before
            .chars()
            .chain(title.chars())
            .chain(DOCUMENT.between.chars())
            .count();
        let chars = max.saturating_mul(self.0).saturating_sub(head);
        let joined = text.concat();
        Some(
            joined
                .char_indices()
                .nth(chars)
                .map_or(joined.len(), |(i, _)| i),
        )
    }

    fn document_tokens(&self, title: &str, text: &str) -> usize {
        self.count(&DOCUMENT.format(title, text))
    }
}
