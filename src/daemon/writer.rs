//! The per-repository vector writer.
//!
//! One task per repository buffers finished vectors and commits them with
//! `VectorCache::put_many`. Every committed batch is a checkpoint: a kill
//! loses at most the unflushed buffer.

use std::time::Duration;

use repo_search_core::UnitVector;
use repo_search_core::store::{ContentHash, Error as StoreError, VectorCache};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::Error;

/// The buffer flushes once it holds this many vectors.
const FLUSH_VECTORS: usize = 64;

/// The buffer flushes this long after its first vector.
const FLUSH_AFTER: Duration = Duration::from_secs(1);

/// Inbox capacity of a writer.
const INBOX: usize = 64;

enum WriterMsg {
    Vector(ContentHash, UnitVector),
    Flush {
        reply: oneshot::Sender<Result<(), StoreError>>,
    },
}

/// A writer task's handle; clones share the one inbox.
#[derive(Clone)]
pub struct WriterHandle {
    tx: mpsc::Sender<WriterMsg>,
}

impl std::fmt::Debug for WriterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriterHandle").finish_non_exhaustive()
    }
}

impl WriterHandle {
    /// Buffers a vector for the next flush.
    pub async fn vector(&self, hash: ContentHash, vector: UnitVector) -> Result<(), Closed> {
        self.tx
            .send(WriterMsg::Vector(hash, vector))
            .await
            .map_err(|_| Closed)
    }

    /// Buffers a vector for the next flush, from a non-async thread.
    pub fn blocking_vector(&self, hash: ContentHash, vector: UnitVector) -> Result<(), Closed> {
        self.tx
            .blocking_send(WriterMsg::Vector(hash, vector))
            .map_err(|_| Closed)
    }

    /// Flushes every buffered vector and replies once they are committed.
    ///
    /// # Errors
    ///
    /// When `put_many` failed (also reported fatal), or the writer is gone.
    pub async fn flush(&self) -> Result<(), FlushError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(WriterMsg::Flush { reply })
            .await
            .map_err(|_| FlushError::Closed)?;
        rx.await
            .map_err(|_| FlushError::Closed)?
            .map_err(FlushError::Store)
    }
}

/// The writer is gone.
#[derive(Debug, thiserror::Error)]
#[error("the repository writer is gone")]
pub struct Closed;

/// A `flush` failure.
#[derive(Debug, thiserror::Error)]
pub enum FlushError {
    /// The writer is gone.
    #[error("the repository writer is gone")]
    Closed,
    /// `put_many` failed.
    #[error("vector store: {0}")]
    Store(#[from] StoreError),
}

/// Spawns the writer task for one repository. `fatal` is the daemon's
/// shared fatal channel: a `put_many` error is reported there and the task
/// exits. `busy` is the daemon's idle-exit tracker: the task reports while
/// vectors sit unflushed, so `source` — the writer's repository id — keeps
/// the daemon alive until the commit. The returned join handle resolves
/// after the close-flush once every [`WriterHandle`] is dropped.
pub fn spawn(
    cache: VectorCache,
    model_id: String,
    fatal: mpsc::Sender<Error>,
    busy: watch::Sender<bool>,
) -> (WriterHandle, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<WriterMsg>(INBOX);
    let task = tokio::spawn(async move {
        let mut buffer: Vec<(ContentHash, UnitVector)> = Vec::new();
        let mut first_at: Option<Instant> = None;
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    None => {
                        let _ = flush(&cache, &model_id, &mut buffer, &fatal, &busy).await;
                        break;
                    }
                    Some(WriterMsg::Vector(hash, vector)) => {
                        if buffer.is_empty() {
                            first_at = Some(Instant::now());
                            busy.send_replace(true);
                        }
                        buffer.push((hash, vector));
                        if buffer.len() >= FLUSH_VECTORS
                            && flush(&cache, &model_id, &mut buffer, &fatal, &busy)
                                .await
                                .is_err()
                        {
                            break;
                        }
                    }
                    Some(WriterMsg::Flush { reply }) => {
                        let result = flush(&cache, &model_id, &mut buffer, &fatal, &busy).await;
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        if failed {
                            break;
                        }
                    }
                },
                () = async {
                    match first_at {
                        Some(at) => tokio::time::sleep_until(at + FLUSH_AFTER).await,
                        None => std::future::pending().await,
                    }
                } =>
                {
                    if flush(&cache, &model_id, &mut buffer, &fatal, &busy)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
            if buffer.is_empty() {
                first_at = None;
            }
        }
        // Whether it flushed or failed, the writer holds no unflushed work
        // once it is gone.
        busy.send_replace(false);
    });
    (WriterHandle { tx }, task)
}

/// Commits the buffer; on error it is reported fatal and returned. Empty
/// buffers are a no-op.
async fn flush(
    cache: &VectorCache,
    model_id: &str,
    buffer: &mut Vec<(ContentHash, UnitVector)>,
    fatal: &mpsc::Sender<Error>,
    busy: &watch::Sender<bool>,
) -> Result<(), StoreError> {
    if buffer.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(buffer);
    if let Err(error) = cache.put_many(model_id, batch).await {
        let _ = fatal.send(Error::Writer(error.to_string())).await;
        return Err(error);
    }
    busy.send_replace(false);
    Ok(())
}
