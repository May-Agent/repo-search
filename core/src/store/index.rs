//! The per-worktree index at `.repo-search/index.db`: file state, chunk
//! metadata and a contentless FTS5 table over chunk titles and bodies.

use std::fs;
use std::ops::{Range, RangeInclusive};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use tokio::sync::{mpsc, oneshot};

use super::{
    ChunkRecord, ContentHash, Error, FileState, ensure_version, open_conn, remove_db_files,
    send_and_wait, spawn_actor,
};

/// Schema generation; a database at any other version is deleted and rebuilt.
const SCHEMA_VERSION: i64 = 2;

/// One file's change inside an atomic batch.
#[derive(Debug)]
pub enum FileChange {
    /// The file's new fingerprint and current chunks; replaces whatever was
    /// stored for its path.
    Replace(FileState, Vec<ChunkRecord>),
    /// The file is gone from the worktree.
    Remove(String),
    /// The file's bytes are unchanged but its stat fingerprint moved; the
    /// stored chunks stay and only the fingerprint is rewritten. The path must
    /// already be stored.
    Restamp(FileState),
}

/// What an applied batch did to the chunk rows.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    /// Chunk rows deleted, including those a replaced file held before.
    pub chunks_removed: u64,
    /// Chunk rows inserted.
    pub chunks_added: u64,
}

/// Where a stored chunk's text lives: enough to rebuild it from the file's
/// bytes without parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredChunk {
    /// Worktree-relative path of the file.
    pub path: String,
    /// The chunk's title.
    pub title: String,
    /// 1-based line range, inclusive.
    pub lines: RangeInclusive<u32>,
    /// Byte ranges of the chunk's members, in order; the text is the slices
    /// joined by `\n`.
    pub ranges: Vec<Range<u64>>,
    /// The vector cache key.
    pub content_hash: ContentHash,
}

enum Request {
    Apply {
        changes: Vec<FileChange>,
        reply: oneshot::Sender<Result<Applied, Error>>,
    },
    ChunkHashes {
        reply: oneshot::Sender<Result<Vec<ContentHash>, Error>>,
    },
    ChunksByHash {
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<Vec<StoredChunk>, Error>>,
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
    pub async fn apply(&self, changes: Vec<FileChange>) -> Result<Applied, Error> {
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
        self.apply(vec![FileChange::Replace(state, chunks)])
            .await
            .map(drop)
    }

    /// Drops one file's state, chunks, members and FTS rows.
    ///
    /// # Errors
    ///
    /// As [`Index::apply`].
    pub async fn remove_file(&self, path: &str) -> Result<(), Error> {
        self.apply(vec![FileChange::Remove(path.to_owned())])
            .await
            .map(drop)
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

    /// The first stored chunk (by id) of each of `hashes`, in the order given;
    /// a hash the index does not hold yields nothing.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunks_by_hash(
        &self,
        hashes: Vec<ContentHash>,
    ) -> Result<Vec<StoredChunk>, Error> {
        send_and_wait(&self.tx, |reply| Request::ChunksByHash { hashes, reply }).await
    }

    /// The content hash of every stored chunk, one entry per chunk row in
    /// index order; reconcile checks them against the vector cache.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunk_hashes(&self) -> Result<Vec<ContentHash>, Error> {
        send_and_wait(&self.tx, |reply| Request::ChunkHashes { reply }).await
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
            Request::ChunksByHash { hashes, reply } => {
                let _ = reply.send(chunks_by_hash(&conn, &hashes));
            }
            Request::ChunkHashes { reply } => {
                let _ = reply.send(chunk_hashes(&conn));
            }
        }
    }
}

fn apply(conn: &mut Connection, changes: &[FileChange]) -> Result<Applied, Error> {
    let tx = conn.transaction()?;
    let mut applied = Applied::default();
    for change in changes {
        match change {
            FileChange::Replace(state, chunks) => {
                let (removed, added) = replace(&tx, state, chunks)?;
                applied.chunks_removed += removed;
                applied.chunks_added += added;
            }
            FileChange::Remove(path) => applied.chunks_removed += remove(&tx, path)?,
            FileChange::Restamp(state) => restamp(&tx, state)?,
        }
    }
    tx.commit()?;
    Ok(applied)
}

fn restamp(tx: &Transaction<'_>, state: &FileState) -> Result<(), Error> {
    let updated = tx.execute(
        "UPDATE files SET size = ?2, mtime_ns = ?3, inode = ?4, content_hash = ?5 \
         WHERE path = ?1",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash
        ],
    )?;
    if updated == 1 {
        Ok(())
    } else {
        Err(Error::NotStored(state.path.clone()))
    }
}

/// Deletes every stored row of `path`, FTS first: the contentless table only
/// forgets the rows it is told to by rowid. Returns the chunk rows deleted.
fn remove(tx: &Transaction<'_>, path: &str) -> Result<u64, Error> {
    tx.execute(
        "DELETE FROM fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    tx.execute(
        "DELETE FROM chunk_members WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    let removed = tx.execute("DELETE FROM chunks WHERE path = ?1", [path])?;
    tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    Ok(removed as u64)
}

/// SQLite integers are signed 64-bit; a wider value is a rejected change, not
/// a truncation.
fn i64_of(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

/// Returns `(chunk rows removed, chunk rows added)`.
fn replace(
    tx: &Transaction<'_>,
    state: &FileState,
    chunks: &[ChunkRecord],
) -> Result<(u64, u64), Error> {
    let removed = remove(tx, &state.path)?;
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
        "INSERT INTO chunks (path, start_line, end_line, title, kind, content_hash) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
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
    Ok((removed, chunks.len() as u64))
}

fn chunks_by_hash(conn: &Connection, hashes: &[ContentHash]) -> Result<Vec<StoredChunk>, Error> {
    let mut chunk_stmt = conn.prepare(
        "SELECT id, path, title, start_line, end_line FROM chunks \
         WHERE content_hash = ?1 ORDER BY id LIMIT 1",
    )?;
    let mut member_stmt = conn.prepare(
        "SELECT start_byte, end_byte FROM chunk_members WHERE chunk_id = ?1 ORDER BY ord",
    )?;
    let mut out = Vec::with_capacity(hashes.len());
    for hash in hashes {
        let row = chunk_stmt
            .query_row([hash], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u32>(3)?,
                    row.get::<_, u32>(4)?,
                ))
            })
            .optional()?;
        let Some((id, path, title, start, end)) = row else {
            continue;
        };
        let ranges = member_stmt
            .query_map([id], |row| {
                let start = row.get::<_, i64>(0)?;
                let end = row.get::<_, i64>(1)?;
                Ok(u64::try_from(start)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, start))?
                    ..u64::try_from(end)
                        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, end))?)
            })?
            .collect::<Result<_, _>>()?;
        out.push(StoredChunk {
            path,
            title,
            lines: start..=end,
            ranges,
            content_hash: *hash,
        });
    }
    Ok(out)
}

fn chunk_hashes(conn: &Connection) -> Result<Vec<ContentHash>, Error> {
    let mut stmt = conn.prepare("SELECT content_hash FROM chunks ORDER BY id")?;
    let rows = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(rows)
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
