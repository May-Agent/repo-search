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
    /// When `put_many` failed — a full disk drops the batch — or the writer
    /// is gone.
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
/// shared fatal channel: a `put_many` error other than a full disk is
/// reported there and the task exits; a full disk drops the batch and the
/// task keeps running. `busy` is the daemon's idle-exit tracker: the task
/// reports while vectors sit unflushed, so `source` — the writer's
/// repository id — keeps the daemon alive until the commit. The returned
/// join handle resolves after the close-flush once every [`WriterHandle`]
/// is dropped.
pub fn spawn(
    cache: VectorCache,
    model_id: String,
    fatal: mpsc::Sender<Error>,
    busy: watch::Sender<bool>,
) -> (WriterHandle, tokio::task::JoinHandle<()>) {
    spawn_with(
        move |batch| {
            let (cache, model_id) = (cache.clone(), model_id.clone());
            async move { cache.put_many(&model_id, batch).await }
        },
        fatal,
        busy,
    )
}

/// `spawn` with the commit step injected, so tests answer for failures a
/// real volume needs no free space to produce.
fn spawn_with<F, Fut>(
    mut commit: F,
    fatal: mpsc::Sender<Error>,
    busy: watch::Sender<bool>,
) -> (WriterHandle, tokio::task::JoinHandle<()>)
where
    F: FnMut(Vec<(ContentHash, UnitVector)>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<(), StoreError>> + Send,
{
    let (tx, mut rx) = mpsc::channel::<WriterMsg>(INBOX);
    let task = tokio::spawn(async move {
        let mut buffer: Vec<(ContentHash, UnitVector)> = Vec::new();
        let mut first_at: Option<Instant> = None;
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    None => {
                        let _ = flush(&mut buffer, &mut commit, &fatal, &busy).await;
                        break;
                    }
                    Some(WriterMsg::Vector(hash, vector)) => {
                        if buffer.is_empty() {
                            first_at = Some(Instant::now());
                            busy.send_replace(true);
                        }
                        buffer.push((hash, vector));
                        if buffer.len() >= FLUSH_VECTORS
                            && matches!(
                                flush(&mut buffer, &mut commit, &fatal, &busy).await,
                                Err(FlushFail::Fatal(_))
                            )
                        {
                            break;
                        }
                    }
                    Some(WriterMsg::Flush { reply }) => {
                        let result = flush(&mut buffer, &mut commit, &fatal, &busy).await;
                        let fatal = matches!(result, Err(FlushFail::Fatal(_)));
                        let _ = reply.send(result.map_err(FlushFail::store_error));
                        if fatal {
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
                    if matches!(
                        flush(&mut buffer, &mut commit, &fatal, &busy).await,
                        Err(FlushFail::Fatal(_))
                    ) {
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

/// A failed flush.
enum FlushFail {
    /// A write the daemon survives — a full disk drops the batch and the
    /// writer keeps running.
    Recoverable(StoreError),
    /// Reported on the daemon's fatal channel; the writer exits.
    Fatal(StoreError),
}

impl FlushFail {
    /// The store error the reply reports either way.
    fn store_error(self) -> StoreError {
        match self {
            Self::Recoverable(error) | Self::Fatal(error) => error,
        }
    }
}

/// Commits the buffer; a full disk drops the batch and is logged, any other
/// error is reported fatal. Empty buffers are a no-op.
async fn flush<F, Fut>(
    buffer: &mut Vec<(ContentHash, UnitVector)>,
    commit: &mut F,
    fatal: &mpsc::Sender<Error>,
    busy: &watch::Sender<bool>,
) -> Result<(), FlushFail>
where
    F: FnMut(Vec<(ContentHash, UnitVector)>) -> Fut,
    Fut: std::future::Future<Output = Result<(), StoreError>>,
{
    if buffer.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(buffer);
    if let Err(error) = commit(batch).await {
        if error.is_disk_full() {
            tracing::error!(%error, "vector write dropped: disk is full");
            busy.send_replace(false);
            return Err(FlushFail::Recoverable(error));
        }
        let _ = fatal.send(Error::Writer(error.to_string())).await;
        return Err(FlushFail::Fatal(error));
    }
    busy.send_replace(false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use repo_search_core::UnitVector;
    use repo_search_core::store::{ContentHash, Error as StoreError};

    use super::*;

    /// A full disk drops the write, reports nothing fatal and the writer
    /// keeps serving once space is back.
    #[tokio::test]
    async fn a_full_write_is_dropped_and_the_writer_keeps_serving() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let commit = move |_| {
            std::future::ready(if count.fetch_add(1, Ordering::Relaxed) < 1 {
                Err(StoreError::Io(io::Error::from(io::ErrorKind::StorageFull)))
            } else {
                Ok(())
            })
        };
        let (fatal_tx, mut fatal_rx) = mpsc::channel(4);
        let (busy_tx, _busy_rx) = watch::channel(false);
        let (writer, _task) = spawn_with(commit, fatal_tx, busy_tx);

        let hash = ContentHash::of_file(b"content");
        let vector = UnitVector::normalize(vec![1.0, 2.0]);
        writer.vector(hash, vector.clone()).await.expect("queued");
        let error = writer.flush().await.expect_err("the write is dropped");
        assert!(matches!(error, FlushError::Store(_)));
        assert!(fatal_rx.try_recv().is_err(), "a full disk is not fatal");

        // Space is back: the same writer commits the next batch.
        writer
            .vector(hash, vector)
            .await
            .expect("the writer serves");
        writer.flush().await.expect("the writer recovers");
    }

    /// Any other write failure reports fatal and the writer exits.
    #[tokio::test]
    async fn a_fatal_write_error_stops_the_writer() {
        let commit = |_| {
            std::future::ready(Err(StoreError::Io(io::Error::from(
                io::ErrorKind::PermissionDenied,
            ))))
        };
        let (fatal_tx, mut fatal_rx) = mpsc::channel(4);
        let (busy_tx, _busy_rx) = watch::channel(false);
        let (writer, _task) = spawn_with(commit, fatal_tx, busy_tx);
        writer
            .vector(ContentHash::of_file(b"x"), UnitVector::normalize(vec![1.0]))
            .await
            .expect("queued");
        assert!(writer.flush().await.is_err());
        assert!(matches!(fatal_rx.try_recv(), Ok(Error::Writer(_))));
        assert!(
            writer
                .vector(ContentHash::of_file(b"y"), UnitVector::normalize(vec![1.0]))
                .await
                .is_err(),
            "the writer is gone"
        );
    }
}
