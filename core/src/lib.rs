//! Chunking, index store, reconcile and retrieval for `repo-search`.

pub mod chunk;
mod embed;
pub mod embed_api;
pub mod knn;
pub mod model;
pub mod qos;
pub mod reconcile;
pub mod search;
mod state;
pub mod store;

pub use state::StateDir;

/// Identifies the chunker that built an index: a hash of the chunker sources
/// and the resolved tree-sitter grammar versions, computed by `build.rs`.
/// Lowercase hex, 32 characters.
pub const CHUNKER_FINGERPRINT: &str = env!("REPO_SEARCH_CHUNKER_FINGERPRINT");

pub use embed::{BatchEmbedder, Embedder, TokenCounter, UnitVector};
pub use model::QueryTask;

// Lets the shared test counters (included from `tests/counters`) name the
// crate the same way the integration tests do.
#[cfg(test)]
extern crate self as repo_search_core;

#[cfg(test)]
#[path = "../tests/counters/mod.rs"]
pub(crate) mod counters;

#[cfg(test)]
mod fingerprint_tests {
    use super::CHUNKER_FINGERPRINT;

    #[test]
    fn the_fingerprint_is_32_lowercase_hex_chars() {
        assert_eq!(CHUNKER_FINGERPRINT.len(), 32);
        assert!(
            CHUNKER_FINGERPRINT
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
    }
}
