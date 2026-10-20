//! A blocking [`Embedder`] on a thread of its own, as an [`EmbedService`].

use std::io;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use repo_search_core::qos::{QosClass, set_current_thread};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::search::EmbedService;
use repo_search_core::{Embedder, QueryTask, UnitVector};

/// A test embedder thread that exits early is a test bug.
const GONE: &str = "the embedder thread exited";

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
pub struct EmbedThread<E: Embedder> {
    tx: mpsc::Sender<Request<E::Error>>,
    model: String,
}

impl<E: Embedder + Send + 'static> EmbedThread<E> {
    /// Moves `embedder` onto a new thread, which exits when the handle drops.
    pub fn spawn(mut embedder: E) -> io::Result<Self> {
        let model = embedder.model_id().to_owned();
        let (tx, mut rx) = mpsc::channel::<Request<E::Error>>(QUEUE);
        thread::Builder::new()
            .name("repo-search-embedder".into())
            .spawn(move || {
                set_current_thread(QosClass::UserInteractive);
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

impl<E: Embedder + Send + 'static> EmbedService for EmbedThread<E> {
    type Error = E::Error;

    fn model_id(&self) -> &str {
        &self.model
    }

    async fn query(&self, task: QueryTask, text: String) -> Result<UnitVector, Self::Error> {
        let (reply, rx) = oneshot::channel();
        let request = Request::Query { task, text, reply };
        self.tx.send(request).await.expect(GONE);
        rx.await.expect(GONE)
    }

    async fn documents(&self, chunks: Vec<PendingChunk>) -> Result<Vec<UnitVector>, Self::Error> {
        let (reply, rx) = oneshot::channel();
        let request = Request::Documents { chunks, reply };
        self.tx.send(request).await.expect(GONE);
        rx.await.expect(GONE)
    }
}
