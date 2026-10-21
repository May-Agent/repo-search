//! The embedding service a [`Retriever`](super::Retriever) embeds through.

use std::future::Future;

use crate::reconcile::PendingChunk;
use crate::{QueryTask, UnitVector};

/// Embeds queries and chunks without blocking the caller's executor. The
/// implementation owns the thread (or threads) that run the model, so several
/// retrievers and the indexing pipeline can share one.
pub trait EmbedService: Send + Sync {
    /// The service's failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Identifies the model build; vectors with different ids never mix.
    fn model_id(&self) -> &str;

    /// Embeds `text` as a query for `task`.
    fn query(
        &self,
        task: QueryTask,
        text: String,
    ) -> impl Future<Output = Result<UnitVector, Self::Error>> + Send;

    /// Embeds `chunks` as documents and commits a vector for each to the
    /// vector cache; resolves once every one is committed. A service that
    /// also runs indexing may commit a chunk's vector from the indexing
    /// embedder already holding it instead of embedding it again. The
    /// chunks' content hashes are distinct.
    fn documents(
        &self,
        chunks: Vec<PendingChunk>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}
