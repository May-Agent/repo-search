//! Test counters for `TokenCounter`, shared by the unit tests (included
//! through `#[path]` in the crate root) and the integration tests.

use repo_search_core::TokenCounter;

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
}

/// A `TokenCounter` counting one token per `unit` chars.
#[allow(dead_code)]
pub struct CharTokens(pub usize);

impl TokenCounter for CharTokens {
    fn count(&self, text: &str) -> usize {
        text.chars().count() / self.0
    }

    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool {
        let doc = format!("title: {title} | text: {}", text.concat());
        self.count(&doc) <= max
    }
}
