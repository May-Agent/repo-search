//! Chunking, index store, reconcile and retrieval for `repo-search`.

pub mod chunk;
mod embed;
pub mod knn;
pub mod reconcile;
pub mod search;
pub mod store;

pub use embed::{Embedder, QueryTask, TokenCounter, UnitVector};

// Lets the shared test counters (included from `tests/counters`) name the
// crate the same way the integration tests do.
#[cfg(test)]
extern crate self as repo_search_core;

#[cfg(test)]
#[path = "../tests/counters/mod.rs"]
pub(crate) mod counters;
