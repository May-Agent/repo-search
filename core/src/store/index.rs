//! The per-worktree index at `.repo-search/index.db`: file state, chunk
//! metadata and a contentless FTS5 table over chunk titles and bodies.

use std::fs;
use std::path::Path;

use rusqlite::{Connection, Transaction, params};
use tokio::sync::{mpsc, oneshot};

use super::{
    ChunkRecord, Error, FileState, ensure_version, open_conn, remove_db_files, send_and_wait,
    spawn_actor,
};

/// Schema generation; a database at any other version is deleted and rebuilt.
const SCHEMA_VERSION: i64 = 1;

/// One file's change inside an atomic batch.
#[derive(Debug)]
pub enum FileChange {
    /// The file's new fingerprint and current chunks; replaces whatever was
    /// stored for its path.
    Replace(FileState, Vec<ChunkRecord>),
    /// The file is gone from the worktree.
    Remove(String),
}

enum Request {
    Apply {
        changes: Vec<FileChange>,
        reply: oneshot::Sender<Result<(), Error>>,
    },
    FileStates {
        reply: oneshot::Sender<Result<Vec<FileState>, Error>>,
    },
}

/// Handle to the index actor; clones share the one connection's queue.
#[derive(Clone)]
pub struct Index {
    tx: mpsc::Sender<Request>,
}

impl Index {
    /// Opens — creating if needed — `<worktree>/.repo-search/index.db`, and
    /// creates the `.repo-search` directory with a `.gitignore` of `*`. The
    /// filesystem and SQLite work runs on the actor thread; `open` only awaits
    /// its outcome.
    ///
    /// A database found at another schema version is deleted and recreated.
    ///
    /// # Errors
    ///
    /// When `worktree` is not a directory, the store directory or `.gitignore`
    /// cannot be written, SQLite rejects the file, or the actor thread fails
    /// to spawn.
    pub async fn open(worktree: &Path) -> Result<Self, Error> {
        let worktree = worktree.to_path_buf();
        let tx = spawn_actor("repo-search-index", move || init(&worktree), serve).await?;
        Ok(Self { tx })
    }

    /// Applies `changes` in one transaction: every file lands or none do.
    ///
    /// # Errors
    ///
    /// When SQLite rejects a change — the whole batch is rolled back — or the
    /// actor is gone.
    pub async fn apply(&self, changes: Vec<FileChange>) -> Result<(), Error> {
        send_and_wait(&self.tx, |reply| Request::Apply { changes, reply }).await
    }

    /// Replaces one file's state, chunks, members and FTS rows.
    ///
    /// # Errors
    ///
    /// As [`Index::apply`].
    pub async fn replace_file(
        &self,
        state: FileState,
        chunks: Vec<ChunkRecord>,
    ) -> Result<(), Error> {
        self.apply(vec![FileChange::Replace(state, chunks)]).await
    }

    /// Drops one file's state, chunks, members and FTS rows.
    ///
    /// # Errors
    ///
    /// As [`Index::apply`].
    pub async fn remove_file(&self, path: &str) -> Result<(), Error> {
        self.apply(vec![FileChange::Remove(path.to_owned())]).await
    }

    /// Every stored file fingerprint, for reconcile to diff the worktree walk
    /// against. Empty on a fresh or rebuilt index, which tells reconcile to do
    /// a full pass.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn file_states(&self) -> Result<Vec<FileState>, Error> {
        send_and_wait(&self.tx, |reply| Request::FileStates { reply }).await
    }
}

/// Creates `.repo-search/` when missing and opens the index database. Runs on
/// the actor thread.
fn init(worktree: &Path) -> Result<Connection, Error> {
    if !worktree.is_dir() {
        return Err(Error::NotDirectory(worktree.to_path_buf()));
    }
    let dir = worktree.join(".repo-search");
    fs::create_dir_all(&dir)?;
    let gitignore = dir.join(".gitignore");
    if !gitignore.exists() {
        fs::write(&gitignore, "*\n")?;
    }
    let path = dir.join("index.db");
    let mut conn = open_conn(&path)?;
    match ensure_version(&mut conn, SCHEMA_VERSION, include_str!("index.sql"))? {
        None => Ok(conn),
        // A version-mismatched per-worktree index belongs to this worktree
        // alone; it is deleted — sidecars included — and rebuilt.
        Some(_) => {
            drop(conn);
            remove_db_files(&path)?;
            let mut conn = open_conn(&path)?;
            if let Some(found) =
                ensure_version(&mut conn, SCHEMA_VERSION, include_str!("index.sql"))?
            {
                return Err(Error::BadSchemaVersion {
                    path,
                    found,
                    want: SCHEMA_VERSION,
                });
            }
            Ok(conn)
        }
    }
}

fn serve(mut conn: Connection, mut rx: mpsc::Receiver<Request>) {
    while let Some(request) = rx.blocking_recv() {
        match request {
            Request::Apply { changes, reply } => {
                let _ = reply.send(apply(&mut conn, &changes));
            }
            Request::FileStates { reply } => {
                let _ = reply.send(file_states(&conn));
            }
        }
    }
}

fn apply(conn: &mut Connection, changes: &[FileChange]) -> Result<(), Error> {
    let tx = conn.transaction()?;
    for change in changes {
        match change {
            FileChange::Replace(state, chunks) => replace(&tx, state, chunks)?,
            FileChange::Remove(path) => remove(&tx, path)?,
        }
    }
    tx.commit()?;
    Ok(())
}

/// Deletes every stored row of `path`, FTS first: the contentless table only
/// forgets the rows it is told to by rowid.
fn remove(tx: &Transaction<'_>, path: &str) -> Result<(), Error> {
    tx.execute(
        "DELETE FROM fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    tx.execute(
        "DELETE FROM chunk_members WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    tx.execute("DELETE FROM chunks WHERE path = ?1", [path])?;
    tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    Ok(())
}

/// SQLite integers are signed 64-bit; a wider value is a rejected change, not
/// a truncation.
fn i64_of(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

fn replace(tx: &Transaction<'_>, state: &FileState, chunks: &[ChunkRecord]) -> Result<(), Error> {
    remove(tx, &state.path)?;
    tx.execute(
        "INSERT INTO files (path, size, mtime_ns, inode, content_hash) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash
        ],
    )?;
    let mut chunk_stmt = tx.prepare(
        "INSERT INTO chunks (path, start_line, end_line, title, kind, content_hash, tokens) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    let mut member_stmt = tx.prepare(
        "INSERT INTO chunk_members (chunk_id, ord, start_byte, end_byte, kind, symbol) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut fts_stmt = tx.prepare("INSERT INTO fts (rowid, title, body) VALUES (?1, ?2, ?3)")?;
    for chunk in chunks {
        chunk_stmt.execute(params![
            state.path,
            *chunk.lines.start(),
            *chunk.lines.end(),
            chunk.title,
            chunk.kind,
            chunk.content_hash,
            chunk.tokens,
        ])?;
        let chunk_id = tx.last_insert_rowid();
        for (ord, member) in chunk.members.iter().enumerate() {
            let ord = i64::try_from(ord).map_err(|_| Error::IntRange(ord as i128))?;
            let start = i64_of(member.bytes.start)?;
            let end = i64_of(member.bytes.end)?;
            member_stmt.execute(params![
                chunk_id,
                ord,
                start,
                end,
                member.kind,
                member.symbol,
            ])?;
        }
        fts_stmt.execute(params![chunk_id, chunk.title, chunk.fts_body])?;
    }
    Ok(())
}

fn file_states(conn: &Connection) -> Result<Vec<FileState>, Error> {
    let mut stmt = conn.prepare("SELECT path, size, mtime_ns, inode, content_hash FROM files")?;
    let states = stmt
        .query_map([], |row| {
            let size = row.get::<_, i64>(1)?;
            let inode = row.get::<_, i64>(3)?;
            Ok(FileState {
                path: row.get(0)?,
                size: u64::try_from(size)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, size))?,
                mtime_ns: row.get(2)?,
                inode: u64::try_from(inode)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(3, inode))?,
                content_hash: row.get(4)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(states)
}
