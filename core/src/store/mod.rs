//! Index storage: a per-worktree SQLite index (`Index`) and the
//! repository-wide vector cache (`VectorCache`).
//!
//! Every `rusqlite::Connection` is owned by a dedicated thread; handles are
//! cheap clones that send requests over a `tokio` channel and await the reply.

mod index;
mod query;
mod rank;
mod records;
mod vectors;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use tokio::sync::{mpsc, oneshot};

#[cfg(feature = "seed")]
pub use index::clone_index;
pub use index::{
    Applied, FileChange, Index, Phase, RowHash, StoredChunk, StoredMember, StoredMemberKind,
    StoredMeta, set_stored_phase, stored_meta,
};
pub use query::{ChunkFilter, ChunkId, ChunkRow, MemberRow, Side};
pub(crate) use records::mtime_ns;
pub use records::{
    ChunkRecord, ContentHash, FileState, MemberRecord, Unchunkable, UnchunkableReason, fts_body,
    split_identifiers,
};
#[cfg(feature = "seed")]
pub use vectors::clone_cache;
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

impl Error {
    /// The write failed because the volume is full; the batch is dropped
    /// and the daemon keeps serving.
    #[must_use]
    pub fn is_disk_full(&self) -> bool {
        match self {
            Self::Sqlite(rusqlite::Error::SqliteFailure(error, _)) => {
                error.code == rusqlite::ErrorCode::DiskFull
            }
            Self::Io(error) => error.kind() == io::ErrorKind::StorageFull,
            _ => false,
        }
    }
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

/// Opens `path` read-only with the same busy timeout: a writer serializes
/// the reads behind its commit.
fn open_read_only(path: &Path) -> Result<Connection, Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
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

/// Copies the file `from`, which is no database, to the new file `to`:
/// copy-on-write where the volume supports it (APFS `clonefile`, a Linux
/// reflink), byte for byte where it does not. A database goes through
/// [`copy_db`]: this may open the file, and closing a descriptor of a
/// database drops the POSIX locks SQLite holds for every connection of the
/// process.
#[cfg(feature = "seed")]
fn copy_file(from: &Path, to: &Path) -> Result<(), Error> {
    reflink_copy::reflink_or_copy(from, to)?;
    Ok(())
}

/// Copies the database `src`, which `conn` has open, to the new file `dst`
/// as one committed state: cloned copy-on-write with `clonefile`, its WAL
/// with it, inside a read transaction on `conn`. A writer serializes behind
/// the transaction under the busy timeout; with a rollback journal none
/// commits past it, and in WAL mode every frame it reads stays in the WAL,
/// which the copy recovers when first opened. `clonefile` takes paths, so
/// no descriptor of the database is opened outside SQLite. On a volume
/// without clones, `VACUUM INTO` writes the copy.
#[cfg(all(feature = "seed", target_os = "macos"))]
fn copy_db(conn: &mut Connection, src: &Path, dst: &Path) -> Result<(), Error> {
    let read = conn.transaction()?;
    read.query_row("SELECT count(*) FROM sqlite_schema", [], |_| Ok(()))?;
    match reflink_copy::reflink(src, dst) {
        Ok(()) => {
            let wal = sidecar(src, "-wal");
            if wal.try_exists()? {
                reflink_copy::reflink(&wal, sidecar(dst, "-wal"))?;
            }
            read.rollback()?;
            Ok(())
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::CrossesDevices
            ) =>
        {
            read.rollback()?;
            vacuum_into(conn, dst)
        }
        Err(e) => Err(e.into()),
    }
}

/// Copies the database `src`, which `conn` has open, to the new file `dst`
/// as one committed state, with `VACUUM INTO`: SQLite writes the copy, so
/// no descriptor of the database is opened outside it.
#[cfg(all(feature = "seed", not(target_os = "macos")))]
fn copy_db(conn: &mut Connection, _src: &Path, dst: &Path) -> Result<(), Error> {
    vacuum_into(conn, dst)
}

/// `VACUUM INTO dst` on `conn`: one read transaction against its database;
/// a writer serializes the copy behind its commit under the busy timeout.
#[cfg(feature = "seed")]
fn vacuum_into(conn: &Connection, dst: &Path) -> Result<(), Error> {
    let dst = dst
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path is not UTF-8"))?;
    conn.execute("VACUUM INTO ?1", [dst])?;
    Ok(())
}

/// `path` with `suffix` appended to its last component, as SQLite names a
/// database's sidecars.
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut side = path.as_os_str().to_os_string();
    side.push(suffix);
    PathBuf::from(side)
}

/// Removes a database file and its `-wal`/`-shm`/`-journal` sidecars.
fn remove_db_files(path: &Path) -> Result<(), Error> {
    for suffix in ["", "-wal", "-shm", "-journal"] {
        match fs::remove_file(sidecar(path, suffix)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn a_full_volume_is_recoverable() {
        let sqlite = Error::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(13),
            None,
        ));
        assert!(sqlite.is_disk_full(), "{sqlite:?}");
        let io = Error::Io(std::io::Error::from(std::io::ErrorKind::StorageFull));
        assert!(io.is_disk_full(), "{io:?}");
    }

    #[test]
    fn other_failures_are_not_recoverable() {
        let busy = Error::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(5),
            None,
        ));
        assert!(!busy.is_disk_full(), "{busy:?}");
        let io = Error::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(!io.is_disk_full(), "{io:?}");
        assert!(!Error::ActorGone.is_disk_full());
    }
}
