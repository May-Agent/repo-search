//! The per-worktree index at `.repo-search/index.db`: file state, chunk
//! metadata and a contentless FTS5 table over chunk titles and bodies.

use std::fs;
use std::ops::{Range, RangeInclusive};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use tokio::sync::{mpsc, oneshot};

use super::query::{self, ChunkFilter, ChunkId, ChunkRow, Side};
use super::{
    ChunkRecord, ContentHash, Error, FileState, Unchunkable, ensure_version, open_conn,
    remove_db_files, send_and_wait, spawn_actor,
};

/// Schema generation; a database at any other version is deleted and rebuilt.
const SCHEMA_VERSION: i64 = 6;

/// One file's change inside an atomic batch.
#[derive(Debug)]
pub enum FileChange {
    /// The file's new fingerprint and current chunks; replaces whatever was
    /// stored for its path.
    Replace(FileState, Vec<ChunkRecord>),
    /// The file is gone from the worktree.
    Remove(String),
    /// The file's bytes are unchanged but its stat fingerprint or ignored flag
    /// moved; the stored chunks stay and only the fingerprint and flag are
    /// rewritten. The path must
    /// already be stored.
    Restamp(FileState),
    /// The file fails to chunk: replaces whatever was stored for its path,
    /// chunks included, with an unchunkable record.
    Unchunkable(Unchunkable),
    /// An unchunkable file's bytes are unchanged but its stat fingerprint
    /// or ignored flag moved; only those are rewritten. The path must already be
    /// recorded as unchunkable.
    RestampUnchunkable(FileState),
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
    /// The chunk's members, in order; the text is the slices joined by `\n`,
    /// a signature member emitting its slice whitespace-collapsed plus
    /// ` …\n`.
    pub members: Vec<StoredMember>,
    /// The vector cache key.
    pub content_hash: ContentHash,
}

/// One stored member's byte range and how its slice is emitted when the
/// chunk's text is rebuilt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredMember {
    /// Byte range in the file.
    pub bytes: Range<u64>,
    /// How the member's slice becomes text.
    pub kind: StoredMemberKind,
}

/// How a stored member's slice becomes text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoredMemberKind {
    /// The slice verbatim, joined by `\n` with the other members.
    Text,
    /// A split part's signature: the slice whitespace-collapsed, cut to
    /// `cut` characters, plus ` …\n`.
    Signature {
        /// Characters of the collapsed signature the prefix carries.
        cut: usize,
    },
}

enum Request {
    Apply {
        changes: Vec<FileChange>,
        reply: oneshot::Sender<Result<Applied, Error>>,
    },
    ChunkHashes {
        include_ignored: bool,
        reply: oneshot::Sender<Result<Vec<ContentHash>, Error>>,
    },
    ChunksByHash {
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<Vec<StoredChunk>, Error>>,
    },
    FileStates {
        reply: oneshot::Sender<Result<Vec<FileState>, Error>>,
    },
    HasIgnored {
        reply: oneshot::Sender<Result<bool, Error>>,
    },
    Unchunkables {
        reply: oneshot::Sender<Result<Vec<Unchunkable>, Error>>,
    },
    Align {
        chunker: String,
        model_id: String,
        reply: oneshot::Sender<Result<bool, Error>>,
    },
    Bm25 {
        query: String,
        filter: ChunkFilter,
        limit: u32,
        reply: oneshot::Sender<Result<Vec<ChunkId>, Error>>,
    },
    ByHashes {
        hashes: Vec<ContentHash>,
        filter: ChunkFilter,
        reply: oneshot::Sender<Result<Vec<Vec<ChunkId>>, Error>>,
    },
    Chunks {
        ids: Vec<ChunkId>,
        reply: oneshot::Sender<Result<Vec<ChunkRow>, Error>>,
    },
    Count {
        side: Side,
        include_ignored: bool,
        reply: oneshot::Sender<Result<u64, Error>>,
    },
    HashSides {
        reply: oneshot::Sender<Result<Vec<(ContentHash, Side)>, Error>>,
    },
    HashesMatching {
        filter: ChunkFilter,
        reply: oneshot::Sender<Result<Vec<ContentHash>, Error>>,
    },
    Matching {
        query: String,
        ids: Vec<ChunkId>,
        reply: oneshot::Sender<Result<Vec<ChunkId>, Error>>,
    },
    Kinds {
        reply: oneshot::Sender<Result<Vec<String>, Error>>,
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

    /// Whether any indexed file is present only because a pass included
    /// ignored files; with none, a search has nothing to hide.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn has_ignored(&self) -> Result<bool, Error> {
        send_and_wait(&self.tx, |reply| Request::HasIgnored { reply }).await
    }

    /// Every file recorded as unchunkable, for reconcile to diff the worktree
    /// walk against.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn unchunkables(&self) -> Result<Vec<Unchunkable>, Error> {
        send_and_wait(&self.tx, |reply| Request::Unchunkables { reply }).await
    }

    /// Records that the index is built by the chunker `fingerprint` with the
    /// token counter of `model_id`. An index stored under another identity is
    /// emptied — files, chunks, members, full-text rows and the recorded
    /// unchunkable files, which another chunker may parse, in one
    /// transaction — so the next pass re-chunks every file; vectors live in
    /// the shared cache and are untouched. A fresh index only stores the
    /// identity; a matching one is left alone.
    ///
    /// Returns whether a populated index under another identity was emptied.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the change or the actor is gone.
    pub async fn align_chunker(&self, fingerprint: &str, model_id: &str) -> Result<bool, Error> {
        send_and_wait(&self.tx, |reply| Request::Align {
            chunker: fingerprint.to_owned(),
            model_id: model_id.to_owned(),
            reply,
        })
        .await
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
    /// index order; reconcile checks them against the vector cache. Chunks of
    /// ignored files are listed only when `include_ignored`.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunk_hashes(&self, include_ignored: bool) -> Result<Vec<ContentHash>, Error> {
        send_and_wait(&self.tx, |reply| Request::ChunkHashes {
            include_ignored,
            reply,
        })
        .await
    }

    /// Up to `limit` chunks matching the FTS5 expression `query` and `filter`,
    /// best BM25 first.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the expression or the read, or the actor is gone.
    pub async fn bm25(
        &self,
        query: String,
        filter: ChunkFilter,
        limit: u32,
    ) -> Result<Vec<ChunkId>, Error> {
        send_and_wait(&self.tx, |reply| Request::Bm25 {
            query,
            filter,
            limit,
            reply,
        })
        .await
    }

    /// For each of `hashes`, in order, the chunks holding it that pass
    /// `filter`; empty for a hash this index lacks or the filter rejects.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunks_with_hashes(
        &self,
        hashes: Vec<ContentHash>,
        filter: ChunkFilter,
    ) -> Result<Vec<Vec<ChunkId>>, Error> {
        send_and_wait(&self.tx, |reply| Request::ByHashes {
            hashes,
            filter,
            reply,
        })
        .await
    }

    /// The rows of `ids`, in the order given; an id the index no longer
    /// holds is absent.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunks(&self, ids: Vec<ChunkId>) -> Result<Vec<ChunkRow>, Error> {
        send_and_wait(&self.tx, |reply| Request::Chunks { ids, reply }).await
    }

    /// The number of chunk rows on `side`, those of ignored files only when
    /// `include_ignored`.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn count(&self, side: Side, include_ignored: bool) -> Result<u64, Error> {
        send_and_wait(&self.tx, |reply| Request::Count {
            side,
            include_ignored,
            reply,
        })
        .await
    }

    /// Every distinct chunk content hash with the side its chunks are on.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn hash_sides(&self) -> Result<Vec<(ContentHash, Side)>, Error> {
        send_and_wait(&self.tx, |reply| Request::HashSides { reply }).await
    }

    /// The distinct content hashes of the chunks that pass `filter`.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn hashes_matching(&self, filter: ChunkFilter) -> Result<Vec<ContentHash>, Error> {
        send_and_wait(&self.tx, |reply| Request::HashesMatching { filter, reply }).await
    }

    /// The chunks among `ids` whose indexed text matches the FTS5 expression
    /// `query`, in id order.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the expression or the read, or the actor is gone.
    pub async fn matching(&self, query: String, ids: Vec<ChunkId>) -> Result<Vec<ChunkId>, Error> {
        send_and_wait(&self.tx, |reply| Request::Matching { query, ids, reply }).await
    }

    /// Every distinct kind the index holds, chunk kinds and member kinds
    /// alike, sorted.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn kinds(&self) -> Result<Vec<String>, Error> {
        send_and_wait(&self.tx, |reply| Request::Kinds { reply }).await
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
            Request::HasIgnored { reply } => {
                let _ = reply.send(has_ignored(&conn));
            }
            Request::Unchunkables { reply } => {
                let _ = reply.send(unchunkables(&conn));
            }
            Request::Align {
                chunker,
                model_id,
                reply,
            } => {
                let _ = reply.send(align(&mut conn, &chunker, &model_id));
            }
            Request::ChunksByHash { hashes, reply } => {
                let _ = reply.send(chunks_by_hash(&conn, &hashes));
            }
            Request::ChunkHashes {
                include_ignored,
                reply,
            } => {
                let _ = reply.send(chunk_hashes(&conn, include_ignored));
            }
            Request::Bm25 {
                query,
                filter,
                limit,
                reply,
            } => {
                let _ = reply.send(query::bm25(&conn, &query, &filter, limit));
            }
            Request::ByHashes {
                hashes,
                filter,
                reply,
            } => {
                let _ = reply.send(query::by_hashes(&conn, &hashes, &filter));
            }
            Request::Chunks { ids, reply } => {
                let _ = reply.send(query::chunks(&conn, &ids));
            }
            Request::Count {
                side,
                include_ignored,
                reply,
            } => {
                let _ = reply.send(query::count(&conn, side, include_ignored));
            }
            Request::HashSides { reply } => {
                let _ = reply.send(query::hash_sides(&conn));
            }
            Request::HashesMatching { filter, reply } => {
                let _ = reply.send(query::hashes_matching(&conn, &filter));
            }
            Request::Matching { query, ids, reply } => {
                let _ = reply.send(query::matching(&conn, &query, &ids));
            }
            Request::Kinds { reply } => {
                let _ = reply.send(query::kinds(&conn));
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
            FileChange::Unchunkable(record) => {
                applied.chunks_removed += record_unchunkable(&tx, record)?;
            }
            FileChange::RestampUnchunkable(state) => restamp_unchunkable(&tx, state)?,
        }
    }
    tx.commit()?;
    Ok(applied)
}

fn align(conn: &mut Connection, chunker: &str, model_id: &str) -> Result<bool, Error> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let stored: Option<(String, String)> = tx
        .query_row("SELECT chunker, model_id FROM meta", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .optional()?;
    if stored
        .as_ref()
        .is_some_and(|(c, m)| c == chunker && m == model_id)
    {
        return Ok(false);
    }
    let populated: bool = tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM files) OR EXISTS (SELECT 1 FROM unchunkable)",
        [],
        |row| row.get(0),
    )?;
    if populated {
        tx.execute_batch(
            "INSERT INTO fts (fts) VALUES ('delete-all');
             DELETE FROM chunk_members;
             DELETE FROM chunks;
             DELETE FROM files;
             DELETE FROM unchunkable;",
        )?;
    }
    tx.execute(
        "INSERT INTO meta (id, chunker, model_id) VALUES (1, ?1, ?2) \
         ON CONFLICT (id) DO UPDATE SET chunker = ?1, model_id = ?2",
        params![chunker, model_id],
    )?;
    tx.commit()?;
    Ok(populated)
}

fn restamp(tx: &Transaction<'_>, state: &FileState) -> Result<(), Error> {
    let updated = tx.execute(
        "UPDATE files SET size = ?2, mtime_ns = ?3, inode = ?4, content_hash = ?5, \
         ignored = ?6 WHERE path = ?1",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash,
            state.ignored
        ],
    )?;
    if updated == 1 {
        Ok(())
    } else {
        Err(Error::NotStored(state.path.clone()))
    }
}

fn restamp_unchunkable(tx: &Transaction<'_>, state: &FileState) -> Result<(), Error> {
    let updated = tx.execute(
        "UPDATE unchunkable SET size = ?2, mtime_ns = ?3, inode = ?4, content_hash = ?5, \
         ignored = ?6 WHERE path = ?1",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash,
            state.ignored
        ],
    )?;
    if updated == 1 {
        Ok(())
    } else {
        Err(Error::NotStored(state.path.clone()))
    }
}

/// Replaces every stored row of the file with its unchunkable record. Returns
/// the chunk rows deleted.
fn record_unchunkable(tx: &Transaction<'_>, record: &Unchunkable) -> Result<u64, Error> {
    let state = &record.state;
    let removed = remove(tx, &state.path)?;
    tx.execute(
        "INSERT INTO unchunkable (path, size, mtime_ns, inode, content_hash, ignored, reason) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash,
            state.ignored,
            record.reason
        ],
    )?;
    Ok(removed)
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
    tx.execute("DELETE FROM unchunkable WHERE path = ?1", [path])?;
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
        "INSERT INTO files (path, size, mtime_ns, inode, content_hash, ignored) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            state.path,
            i64_of(state.size)?,
            state.mtime_ns,
            i64_of(state.inode)?,
            state.content_hash,
            state.ignored
        ],
    )?;
    let mut chunk_stmt = tx.prepare(
        "INSERT INTO chunks (path, start_line, end_line, title, kind, content_hash) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut member_stmt = tx.prepare(
        "INSERT INTO chunk_members (chunk_id, ord, start_byte, end_byte, kind, cut, symbol) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
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
                i64_of(member.cut)?,
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
        "SELECT start_byte, end_byte, kind, cut FROM chunk_members WHERE chunk_id = ?1 ORDER BY ord",
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
        let members = member_stmt
            .query_map([id], |row| {
                let start = row.get::<_, i64>(0)?;
                let end = row.get::<_, i64>(1)?;
                let cut = row.get::<_, i64>(3)?;
                let kind = if row.get::<_, String>(2)? == crate::chunk::MemberKind::SIGNATURE_NAME {
                    StoredMemberKind::Signature {
                        cut: usize::try_from(cut)
                            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(3, cut))?,
                    }
                } else {
                    StoredMemberKind::Text
                };
                Ok(StoredMember {
                    bytes: u64::try_from(start)
                        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, start))?
                        ..u64::try_from(end)
                            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, end))?,
                    kind,
                })
            })?
            .collect::<Result<_, _>>()?;
        out.push(StoredChunk {
            path,
            title,
            lines: start..=end,
            members,
            content_hash: *hash,
        });
    }
    Ok(out)
}

fn chunk_hashes(conn: &Connection, include_ignored: bool) -> Result<Vec<ContentHash>, Error> {
    let mut stmt = conn.prepare(
        "SELECT content_hash FROM chunks WHERE ?1 OR path NOT IN \
         (SELECT path FROM files WHERE ignored = 1) ORDER BY id",
    )?;
    let rows = stmt
        .query_map([include_ignored], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Reads the six fingerprint columns starting at column 0.
fn file_state(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileState> {
    let size = row.get::<_, i64>(1)?;
    let inode = row.get::<_, i64>(3)?;
    Ok(FileState {
        path: row.get(0)?,
        size: u64::try_from(size).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, size))?,
        mtime_ns: row.get(2)?,
        inode: u64::try_from(inode)
            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(3, inode))?,
        content_hash: row.get(4)?,
        ignored: row.get(5)?,
    })
}

fn has_ignored(conn: &Connection) -> Result<bool, Error> {
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM files WHERE ignored = 1)",
        [],
        |row| row.get(0),
    )?)
}

fn file_states(conn: &Connection) -> Result<Vec<FileState>, Error> {
    let mut stmt =
        conn.prepare("SELECT path, size, mtime_ns, inode, content_hash, ignored FROM files")?;
    let states = stmt.query_map([], file_state)?.collect::<Result<_, _>>()?;
    Ok(states)
}

fn unchunkables(conn: &Connection) -> Result<Vec<Unchunkable>, Error> {
    let mut stmt = conn.prepare(
        "SELECT path, size, mtime_ns, inode, content_hash, ignored, reason FROM unchunkable \
         ORDER BY path",
    )?;
    let records = stmt
        .query_map([], |row| {
            Ok(Unchunkable {
                state: file_state(row)?,
                reason: row.get(6)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(records)
}
