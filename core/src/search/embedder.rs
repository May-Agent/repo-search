//! The embedding service a [`Retriever`](super::Retriever) embeds through.

use std::future::Future;

use crate::reconcile::PendingChunk;
use crate::store::ContentHash;
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

    /// Embeds `chunks` as documents; one vector per chunk, in order.
    fn documents(
        &self,
        chunks: Vec<PendingChunk>,
    ) -> impl Future<Output = Result<Vec<UnitVector>, Self::Error>> + Send;

    /// The vectors [`EmbedService::documents`] returned for `hashes` are now
    /// committed to the vector cache. A service that also queues indexing
    /// work drops the matching queued items here and never earlier: a search
    /// that ends between `documents` and the commit leaves them queued.
    fn stored(
        &self,
        hashes: Vec<ContentHash>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}
