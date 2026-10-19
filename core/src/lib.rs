//! Chunking, index store, reconcile and retrieval for `repo-search`.

pub mod chunk;
mod embed;
pub mod knn;
pub mod store;

pub use embed::{Embedder, UnitVector};
