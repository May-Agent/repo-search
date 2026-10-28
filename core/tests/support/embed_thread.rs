//! A blocking encoder on a thread of its own, as an [`EmbedService`].

use std::io;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use repo_search_core::qos::{QosClass, set_current_thread};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::search::EmbedService;
use repo_search_core::store::{ContentHash, VectorCache};
use repo_search_core::{QueryTask, UnitVector};

/// A test embedder thread that exits early is a test bug.
const GONE: &str = "the embedder thread exited";

/// Bound on the request queue.
const QUEUE: usize = 16;

/// The blocking encoder the thread runs: calls block, so the thread owns
/// them and sets its QoS.
pub trait BlockingEmbed: Send + 'static {
    /// The encoder's failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Identifies the model build; vectors with different ids never mix.
    fn model_id(&self) -> &str;

    /// Embeds the document `text` under `title`.
    ///
    /// # Errors
    ///
    /// When the encoder fails.
    fn document(&mut self, title: &str, text: &str) -> Result<UnitVector, Self::Error>;

    /// Embeds `text` as a `task` query.
    ///
    /// # Errors
    ///
    /// When the encoder fails.
    fn query(&mut self, task: QueryTask, text: &str) -> Result<UnitVector, Self::Error>;
}

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
pub struct EmbedThread<E: BlockingEmbed> {
    tx: mpsc::Sender<Request<E::Error>>,
    model: String,
    /// Where `documents` commits its vectors.
    cache: VectorCache,
}

impl<E: BlockingEmbed> EmbedThread<E> {
    /// Moves `embedder` onto a new thread, which exits when the handle
    /// drops; `documents` commits into `cache`.
    pub fn spawn(mut embedder: E, cache: VectorCache) -> io::Result<Self> {
        let model = embedder.model_id().to_owned();
        let (tx, mut rx) = mpsc::channel::<Request<E::Error>>(QUEUE);
        thread::Builder::new()
            .name("repo-search-embedder".into())
            .spawn(move || {
                set_current_thread(QosClass::UserInteractive);
                while let Some(request) = rx.blocking_recv() {
                    match request {
                        Request::Query { task, text, reply } => {
                            let _ = reply.send(embedder.query(task, &text));
                        }
                        Request::Documents { chunks, reply } => {
                            let _ = reply.send(
                                chunks
                                    .iter()
                                    .map(|c| embedder.document(&c.title, &c.text))
                                    .collect(),
                            );
                        }
                    }
                }
            })?;
        Ok(Self { tx, model, cache })
    }
}

impl<E: BlockingEmbed> EmbedService for EmbedThread<E> {
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

    async fn documents(&self, chunks: Vec<PendingChunk>) -> Result<(), Self::Error> {
        let hashes: Vec<ContentHash> = chunks.iter().map(|chunk| chunk.content_hash).collect();
        let (reply, rx) = oneshot::channel();
        let request = Request::Documents { chunks, reply };
        self.tx.send(request).await.expect(GONE);
        let vectors = rx.await.expect(GONE)?;
        self.cache
            .put_many(&self.model, hashes.into_iter().zip(vectors).collect())
            .await
            .expect("the test cache commits");
        Ok(())
    }
}
