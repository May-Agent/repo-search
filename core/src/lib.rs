//! Chunking, index store, reconcile and retrieval for `repo-search`.

pub mod chunk;
mod embed;
pub mod reconcile;
pub mod store;

pub use embed::{Embedder, UnitVector};
