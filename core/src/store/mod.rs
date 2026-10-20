//! Index storage: a per-worktree SQLite index (`Index`) and the
//! repository-wide vector cache (`VectorCache`).
//!
//! Every `rusqlite::Connection` is owned by a dedicated thread; handles are
//! cheap clones that send requests over a `tokio` channel and await the reply.

mod index;
mod query;
mod records;
mod vectors;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use rusqlite::{Connection, TransactionBehavior};
use tokio::sync::{mpsc, oneshot};

pub use index::{Applied, FileChange, Index, Phase, StoredChunk, StoredMember, StoredMemberKind};
pub use query::{ChunkFilter, ChunkId, ChunkRow, MemberRow, Side};
pub use records::{
    ChunkRecord, ContentHash, FileState, MemberRecord, Unchunkable, UnchunkableReason, fts_body,
    split_identifiers,
};
pub use vectors::{CachedSlot, CachedVector, ModelInfo, StoredCalibration, VectorCache};

/// Bound on each actor's request queue.
const QUEUE: usize = 64;

/// How long a connection waits on a locked database before giving up.
const BUSY_TIMEOUT: Duration = Duration::from_secs(30);

/// Failures of either database.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// SQLite rejected a statement or the database file is unusable.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Creating the store directory or file failed.
    #[error("i/o: {0}")]
    Io(#[from] io::Error),

    /// The thread owning the connection exited without answering; a panic in
    /// the actor surfaces here, never as a panic in the caller.
    #[error("store actor exited without answering")]
    ActorGone,

    /// The phase was set before the index recorded its identity
    /// ([`Index::align_chunker`] creates the row the phase lives in).
    #[error("the index has no recorded identity to hold a phase")]
    NotAligned,

    /// `open` was pointed at a path that is not a directory.
    #[error("{0:?} is not a directory")]
    NotDirectory(PathBuf),

    /// The file's `user_version` neither matches nor can be initialized.
    #[error("{path:?} holds schema version {found}; this build stores version {want}")]
    BadSchemaVersion {
        /// The database file at fault.
        path: PathBuf,
        /// The `user_version` it holds.
        found: i64,
        /// The schema version this build writes.
        want: i64,
    },

    /// SQLite refused to put the shared cache into WAL mode.
    #[error("{0:?} did not switch to WAL journal mode (journal_mode = {1:?})")]
    NotWal(PathBuf, String),

    /// SQLite stores signed 64-bit integers; a field did not fit.
    #[error("integer {0} is outside sqlite's signed 64-bit range")]
    IntRange(i128),

    /// A restamp named a path the index does not hold.
    #[error("cannot restamp {0:?}: the index does not hold it")]
    NotStored(String),

    /// A blob read back from the database has a shape the writer never
    /// produces.
    #[error("corrupt {0} blob")]
    CorruptBlob(&'static str),

    /// A vector or calibration does not have the dimension the model's cache
    /// was created with.
    #[error("model {model_id:?} stores {expected}-dimensional vectors, got {got}")]
    DimMismatch {
        /// The model whose cache rejected the write.
        model_id: String,
        /// The dimension recorded when the model's first vector was stored.
        expected: usize,
        /// The dimension that was offered.
        got: usize,
    },

    /// An operation named a model with no cached vectors.
    #[error("model {0:?} has no cached vectors")]
    UnknownModel(String),

    /// A calibration counting no vectors was offered for a model; its bounds
    /// would be infinite.
    #[error("calibration of model {0:?} counts no vectors")]
    EmptyCalibration(String),
}

/// Sends `request` to an actor and awaits its reply.
async fn send_and_wait<M, T: Send>(
    tx: &mpsc::Sender<M>,
    make: impl FnOnce(oneshot::Sender<Result<T, Error>>) -> M,
) -> Result<T, Error> {
    let (reply, rx) = oneshot::channel();
    tx.send(make(reply)).await.map_err(|_| Error::ActorGone)?;
    rx.await.map_err(|_| Error::ActorGone)?
}

/// Spawns the actor thread, runs `init` — the blocking filesystem and SQLite
/// work — on it, and reports the outcome to the caller through a oneshot.
/// On success the actor enters `serve` with its connection and request queue.
async fn spawn_actor<M: Send + 'static>(
    name: &'static str,
    init: impl FnOnce() -> Result<Connection, Error> + Send + 'static,
    serve: impl FnOnce(Connection, mpsc::Receiver<M>) + Send + 'static,
) -> Result<mpsc::Sender<M>, Error> {
    let (tx, rx) = mpsc::channel(QUEUE);
    let (init_tx, init_rx) = oneshot::channel();
    thread::Builder::new().name(name.into()).spawn(move || {
        match init() {
            Ok(conn) => {
                if init_tx.send(Ok(())).is_err() {
                    return; // the caller went away; nothing to serve
                }
                serve(conn, rx);
            }
            Err(e) => {
                let _ = init_tx.send(Err(e));
            }
        }
    })?;
    init_rx.await.map_err(|_| Error::ActorGone)??;
    Ok(tx)
}

/// Opens `path` with the busy timeout set first: everything after this may
/// wait on a lock held by a sibling daemon.
fn open_conn(path: &Path) -> Result<Connection, Error> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    Ok(conn)
}

/// Brings the database to `version` if it is empty or already there.
///
/// A database already at `version` is left alone. Otherwise an exclusive
/// transaction re-reads `user_version` — a sibling may have initialized while
/// this connection waited on the lock — and an empty database
/// (`user_version` 0, no schema objects) gets `schema` and the version stamp
/// in the same transaction, so schema and version are never half-written.
///
/// Returns `Some(found)` when the file holds a version this build cannot use;
/// the caller decides whether deleting it is safe.
fn ensure_version(conn: &mut Connection, version: i64, schema: &str) -> Result<Option<i64>, Error> {
    let found: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if found == version {
        return Ok(None);
    }
    let tx = conn.transaction_with_behavior(TransactionBehavior::Exclusive)?;
    let found: i64 = tx.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if found == version {
        tx.commit()?;
        return Ok(None);
    }
    let empty: bool = tx.query_row(
        "SELECT NOT EXISTS (SELECT 1 FROM sqlite_schema)",
        [],
        |row| row.get(0),
    )?;
    if found == 0 && empty {
        tx.execute_batch(schema)?;
        tx.pragma_update(None, "user_version", version)?;
        tx.commit()?;
        return Ok(None);
    }
    tx.rollback()?;
    Ok(Some(found))
}

/// Removes a database file and its `-wal`/`-shm`/`-journal` sidecars.
fn remove_db_files(path: &Path) -> Result<(), Error> {
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let mut side = path.as_os_str().to_os_string();
        side.push(suffix);
        match fs::remove_file(PathBuf::from(side)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
