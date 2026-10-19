//! Chunking, index store, reconcile and retrieval for `repo-search`.

pub mod chunk;
mod embed;

pub use embed::{Embedder, UnitVector};
