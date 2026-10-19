//! The caller's blocking [`Embedder`], run on a thread of its own so that
//! embedding a query or a batch of chunks never blocks the executor.

use std::io;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use crate::knn::set_user_interactive;
use crate::reconcile::PendingChunk;
use crate::{Embedder, QueryTask, UnitVector};

use super::Error;

/// Bound on the request queue.
const QUEUE: usize = 16;

enum Request<Err> {
    Query {
        task: QueryTask,
        text: String,
        reply: oneshot::Sender<Result<UnitVector, Err>>,
    },
    Documents {
        chunks: Vec<PendingChunk>,
        reply: oneshot::Sender<Result<Vec<UnitVector>, Err>>,
    },
}

/// Handle to the embedder thread. Requests queue in order; a request whose
/// future is dropped is still run, and its answer discarded.
pub(super) struct EmbedderHandle<E: Embedder> {
    tx: mpsc::Sender<Request<E::Error>>,
    model: String,
}

impl<E: Embedder + Send + 'static> EmbedderHandle<E> {
    /// Moves `embedder` onto a new thread, which exits when the handle drops.
    pub(super) fn spawn(mut embedder: E) -> io::Result<Self> {
        let model = embedder.model_id().to_owned();
        let (tx, mut rx) = mpsc::channel::<Request<E::Error>>(QUEUE);
        thread::Builder::new()
            .name("repo-search-embedder".into())
            .spawn(move || {
                set_user_interactive();
                while let Some(request) = rx.blocking_recv() {
                    match request {
                        Request::Query { task, text, reply } => {
                            let _ = reply.send(embedder.embed_query(task, &text));
                        }
                        Request::Documents { chunks, reply } => {
                            let _ = reply.send(
                                chunks
                                    .iter()
                                    .map(|c| embedder.embed_document(&c.title, &c.text))
                                    .collect(),
                            );
                        }
                    }
                }
            })?;
        Ok(Self { tx, model })
    }
}

impl<E: Embedder> EmbedderHandle<E> {
    /// The embedder's model id, read once at spawn.
    pub(super) fn model_id(&self) -> &str {
        &self.model
    }

    /// Embeds `text` as a query for `task`.
    pub(super) async fn query(
        &self,
        task: QueryTask,
        text: &str,
    ) -> Result<UnitVector, Error<E::Error>> {
        let (reply, rx) = oneshot::channel();
        let request = Request::Query {
            task,
            text: text.to_owned(),
            reply,
        };
        self.tx.send(request).await.map_err(|_| Error::WorkerGone)?;
        rx.await
            .map_err(|_| Error::WorkerGone)?
            .map_err(Error::Embed)
    }

    /// Embeds `chunks` as documents; one vector per chunk, in order.
    pub(super) async fn documents(
        &self,
        chunks: Vec<PendingChunk>,
    ) -> Result<Vec<UnitVector>, Error<E::Error>> {
        let (reply, rx) = oneshot::channel();
        let request = Request::Documents { chunks, reply };
        self.tx.send(request).await.map_err(|_| Error::WorkerGone)?;
        rx.await
            .map_err(|_| Error::WorkerGone)?
            .map_err(Error::Embed)
    }
}
