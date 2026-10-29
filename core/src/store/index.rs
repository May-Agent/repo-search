//! The per-worktree index under the daemon's [`StateDir`]: file state,
//! chunk metadata and a contentless FTS5 table over chunk titles and
//! bodies.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::ops::{Range, RangeInclusive};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use tokio::sync::{mpsc, oneshot};

use super::query::{self, ChunkFilter, ChunkId, ChunkRow, Side};
use super::rank;
use super::{
    ChunkRecord, ContentHash, Error, FileState, Unchunkable, ensure_version, open_conn,
    open_read_only, remove_db_files, send_and_wait, spawn_actor,
};
use crate::StateDir;

/// Schema generation; a database at any other version is deleted and rebuilt.
const SCHEMA_VERSION: i64 = 8;

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
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Applied {
    /// Chunk rows deleted, including those a replaced file held before.
    pub chunks_removed: u64,
    /// Chunk rows inserted.
    pub chunks_added: u64,
    /// The content hashes of the deleted rows, distinct, in deletion order;
    /// a hash the batch inserted again is listed too.
    pub removed_hashes: Vec<ContentHash>,
}

/// One stored chunk row's content hash, with its file's path and ignored
/// flag and the row's side: what reconcile's books are loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowHash {
    /// Worktree-relative path of the file.
    pub path: String,
    /// The row's content hash.
    pub content_hash: ContentHash,
    /// The file's `ignored` flag.
    pub ignored: bool,
    /// The list the row's chunk belongs to.
    pub side: Side,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl StoredMemberKind {
    /// The `kind`/`cut` member columns decoded: a `signature` row emits its
    /// collapsed prefix, every other kind its slice verbatim. Every member
    /// query selects `cut` as column 3, the index the range error reports.
    pub(crate) fn from_columns(kind: &str, cut: i64) -> rusqlite::Result<Self> {
        if kind == crate::chunk::MemberKind::SIGNATURE_NAME {
            Ok(Self::Signature {
                cut: usize::try_from(cut)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(3, cut))?,
            })
        } else {
            Ok(Self::Text)
        }
    }
}

/// What an index file's `meta` row records, read without an actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredMeta {
    /// The chunker fingerprint the rows were built under.
    pub chunker: String,
    /// The model id whose token counter capped every chunk.
    pub model_id: String,
    /// The stored phase.
    pub phase: Phase,
}

/// The `meta` row of the index at `path` on a read-only connection: `None`
/// when the file is at another schema version, which [`Index::open`] would
/// discard, or recorded no identity yet. A writer on the file serializes the
/// read behind its commit under the busy timeout.
///
/// # Errors
///
/// When `path` is missing or not a database, or SQLite rejects the read.
pub fn stored_meta(path: &Path) -> Result<Option<StoredMeta>, Error> {
    let conn = open_read_only(path)?;
    let version: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version != SCHEMA_VERSION {
        return Ok(None);
    }
    match conn.query_row(
        "SELECT chunker, model_id, phase FROM meta WHERE id = 1",
        [],
        |row| {
            Ok(StoredMeta {
                chunker: row.get(0)?,
                model_id: row.get(1)?,
                phase: Phase::from_column(row.get(2)?)?,
            })
        },
    ) {
        Ok(meta) => Ok(Some(meta)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Copies the database at `src` to `dst` with `VACUUM INTO` over a
/// read-only connection, one consistent transaction against the source: a
/// writer on `src` serializes the copy behind its commit under the busy
/// timeout. `dst` must not exist; on error a partial destination may be
/// left for the caller to remove.
///
/// # Errors
///
/// When `src` cannot be read, `dst` exists or cannot be written, or `dst`
/// is not UTF-8.
pub fn vacuum_into(src: &Path, dst: &Path) -> Result<(), Error> {
    let conn = open_read_only(src)?;
    let dst = dst
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path is not UTF-8"))?;
    conn.execute("VACUUM INTO ?1", params![dst])?;
    Ok(())
}

/// Stores `phase` in the `meta` row of the index at `path` on its own
/// write connection: the seeding path stamps a copy `FirstIndex` before
/// the worktree's `Index` opens it.
///
/// # Errors
///
/// [`Error::NotAligned`] when the index has recorded no identity; or when
/// `path` cannot be opened or SQLite rejects the write.
pub fn set_stored_phase(path: &Path, phase: Phase) -> Result<(), Error> {
    let conn = open_conn(path)?;
    set_phase(&conn, phase)
}

/// How far the index has come: whether indexing has ever caught up with the
/// worktree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The first pass over the worktree has not finished.
    FirstIndex,
    /// The index has been complete once and is kept up to date.
    Maintained,
}

impl Phase {
    const fn column(self) -> i64 {
        match self {
            Self::FirstIndex => 0,
            Self::Maintained => 1,
        }
    }

    fn from_column(value: i64) -> rusqlite::Result<Self> {
        match value {
            0 => Ok(Self::FirstIndex),
            1 => Ok(Self::Maintained),
            other => Err(rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Integer,
                format!("unknown index phase {other}").into(),
            )),
        }
    }
}

enum Request {
    Apply {
        changes: Vec<FileChange>,
        reply: oneshot::Sender<Result<Applied, Error>>,
    },
    RowHashes {
        reply: oneshot::Sender<Result<Vec<RowHash>, Error>>,
    },
    ChunkHashes {
        include_ignored: bool,
        reply: oneshot::Sender<Result<Vec<ContentHash>, Error>>,
    },
    ChunksByHash {
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<Vec<StoredChunk>, Error>>,
    },
    ChunksIn {
        paths: Vec<String>,
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
    Phase {
        reply: oneshot::Sender<Result<Phase, Error>>,
    },
    SetPhase {
        phase: Phase,
        reply: oneshot::Sender<Result<(), Error>>,
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
    /// Opens — creating if needed — `worktree`'s index under `state`
    /// ([`StateDir::worktree_index`]); nothing is written under the
    /// worktree itself. `worktree` is its canonical root. The filesystem
    /// and SQLite work runs on the actor thread; `open` only awaits its
    /// outcome.
    ///
    /// A database found at another schema version is deleted and recreated.
    ///
    /// # Errors
    ///
    /// When `worktree` is not a directory, the state directory cannot be
    /// written, SQLite rejects the file, or the actor thread fails to
    /// spawn.
    pub async fn open(state: &StateDir, worktree: &Path) -> Result<Self, Error> {
        let path = state.worktree_index(worktree);
        let worktree = worktree.to_path_buf();
        let tx = spawn_actor("repo-search-index", move || init(&worktree, &path), serve).await?;
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

    /// The index's phase; [`Phase::FirstIndex`] until one is set, including
    /// before the index has recorded its identity.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn phase(&self) -> Result<Phase, Error> {
        send_and_wait(&self.tx, |reply| Request::Phase { reply }).await
    }

    /// Stores the index's phase. A chunker change keeps it: the re-chunked
    /// files are embedded in the class the stored phase already names.
    ///
    /// # Errors
    ///
    /// [`Error::NotAligned`] before [`Index::align_chunker`] has run; or when
    /// SQLite rejects the change or the actor is gone.
    pub async fn set_phase(&self, phase: Phase) -> Result<(), Error> {
        send_and_wait(&self.tx, |reply| Request::SetPhase { phase, reply }).await
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

    /// Every stored chunk of `paths`, ordered by path then id, with its
    /// members; a path the index does not hold yields nothing.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn chunks_in(&self, paths: Vec<String>) -> Result<Vec<StoredChunk>, Error> {
        send_and_wait(&self.tx, |reply| Request::ChunksIn { paths, reply }).await
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

    /// Every stored chunk row as its path, content hash and its file's
    /// ignored flag, in index order.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn row_hashes(&self) -> Result<Vec<RowHash>, Error> {
        send_and_wait(&self.tx, |reply| Request::RowHashes { reply }).await
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

/// Creates the index's state directory when missing, opens the index
/// database and registers `exact_bm25` on the connection. Runs on the actor
/// thread.
fn init(worktree: &Path, path: &Path) -> Result<Connection, Error> {
    if !worktree.is_dir() {
        return Err(Error::NotDirectory(worktree.to_path_buf()));
    }
    fs::create_dir_all(path.parent().expect("worktree_index names index.db"))?;
    let conn = open_schema(path)?;
    rank::register(&conn)?;
    Ok(conn)
}

/// Opens the index database at `path` at this build's schema version.
fn open_schema(path: &Path) -> Result<Connection, Error> {
    let mut conn = open_conn(path)?;
    match ensure_version(&mut conn, SCHEMA_VERSION, include_str!("index.sql"))? {
        None => Ok(conn),
        // A version-mismatched per-worktree index belongs to this worktree
        // alone; it is deleted — sidecars included — and rebuilt.
        Some(_) => {
            drop(conn);
            remove_db_files(path)?;
            let mut conn = open_conn(path)?;
            if let Some(found) =
                ensure_version(&mut conn, SCHEMA_VERSION, include_str!("index.sql"))?
            {
                return Err(Error::BadSchemaVersion {
                    path: path.to_path_buf(),
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
            Request::Phase { reply } => {
                let _ = reply.send(phase(&conn));
            }
            Request::SetPhase { phase, reply } => {
                let _ = reply.send(set_phase(&conn, phase));
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
            Request::ChunksIn { paths, reply } => {
                let _ = reply.send(chunks_in(&conn, &paths));
            }
            Request::ChunkHashes {
                include_ignored,
                reply,
            } => {
                let _ = reply.send(chunk_hashes(&conn, include_ignored));
            }
            Request::RowHashes { reply } => {
                let _ = reply.send(row_hashes(&conn));
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
    let mut seen = HashSet::new();
    let mut removed = |applied: &mut Applied, hashes: Vec<ContentHash>| {
        applied.chunks_removed += hashes.len() as u64;
        for hash in hashes {
            if seen.insert(hash) {
                applied.removed_hashes.push(hash);
            }
        }
    };
    for change in changes {
        match change {
            FileChange::Replace(state, chunks) => {
                let (hashes, added) = replace(&tx, state, chunks)?;
                removed(&mut applied, hashes);
                applied.chunks_added += added;
            }
            FileChange::Remove(path) => removed(&mut applied, remove(&tx, path)?),
            FileChange::Restamp(state) => restamp(&tx, state)?,
            FileChange::Unchunkable(record) => {
                removed(&mut applied, record_unchunkable(&tx, record)?);
            }
            FileChange::RestampUnchunkable(state) => restamp_unchunkable(&tx, state)?,
        }
    }
    tx.commit()?;
    Ok(applied)
}

fn phase(conn: &Connection) -> Result<Phase, Error> {
    let stored: Option<i64> = conn
        .query_row("SELECT phase FROM meta", [], |row| row.get(0))
        .optional()?;
    Ok(stored
        .map(Phase::from_column)
        .transpose()?
        .unwrap_or(Phase::FirstIndex))
}

fn set_phase(conn: &Connection, phase: Phase) -> Result<(), Error> {
    match conn.execute("UPDATE meta SET phase = ?1", params![phase.column()])? {
        1 => Ok(()),
        _ => Err(Error::NotAligned),
    }
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
             UPDATE bm25_stats SET rows = 0, title_tokens = 0, body_tokens = 0;
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
/// the deleted rows' content hashes.
fn record_unchunkable(
    tx: &Transaction<'_>,
    record: &Unchunkable,
) -> Result<Vec<ContentHash>, Error> {
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
/// forgets the rows it is told to by rowid. The rows' sizes leave the BM25
/// statistics before the rows leave `fts`. Returns the deleted rows' content
/// hashes.
fn remove(tx: &Transaction<'_>, path: &str) -> Result<Vec<ContentHash>, Error> {
    adjust_stats(tx, -fts_sizes(tx, path)?)?;
    tx.execute(
        "DELETE FROM fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    tx.execute(
        "DELETE FROM chunk_members WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?1)",
        [path],
    )?;
    let removed = tx
        .prepare("DELETE FROM chunks WHERE path = ?1 RETURNING content_hash")?
        .query_map([path], |row| row.get::<_, ContentHash>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    tx.execute("DELETE FROM unchunkable WHERE path = ?1", [path])?;
    Ok(removed)
}

/// SQLite integers are signed 64-bit; a wider value is a rejected change, not
/// a truncation.
fn i64_of(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

/// Returns `(deleted rows' content hashes, chunk rows added)`.
fn replace(
    tx: &Transaction<'_>,
    state: &FileState,
    chunks: &[ChunkRecord],
) -> Result<(Vec<ContentHash>, u64), Error> {
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
    adjust_stats(tx, fts_sizes(tx, &state.path)?)?;
    Ok((removed, chunks.len() as u64))
}

/// The number of `fts` columns: `title` and `body`.
const FTS_COLUMNS: usize = 2;

/// The `fts` rows of a set of chunks: how many, and their token count in each
/// column.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct FtsSizes {
    rows: i64,
    tokens: [i64; FTS_COLUMNS],
}

impl std::ops::Neg for FtsSizes {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            rows: -self.rows,
            tokens: self.tokens.map(|t| -t),
        }
    }
}

/// The sizes of the `fts` rows of `path`'s stored chunks, read from the
/// table's `fts_docsize` shadow table: the per-row column sizes FTS5 itself
/// wrote on insert and reads for `bm25()`'s row lengths.
fn fts_sizes(tx: &Transaction<'_>, path: &str) -> Result<FtsSizes, Error> {
    let mut stmt = tx
        .prepare("SELECT d.sz FROM fts_docsize d JOIN chunks c ON d.id = c.id WHERE c.path = ?1")?;
    let mut rows = stmt.query([path])?;
    let mut sizes = FtsSizes::default();
    while let Some(row) = rows.next()? {
        let blob = row
            .get_ref(0)?
            .as_blob()
            .map_err(|_| Error::CorruptBlob("fts_docsize"))?;
        let columns = docsize_columns(blob)?;
        sizes.rows += 1;
        for (total, column) in sizes.tokens.iter_mut().zip(columns) {
            *total += column;
        }
    }
    Ok(sizes)
}

/// Decodes one `fts_docsize.sz` blob: each column's token count as an SQLite
/// varint, in column order, filling the blob exactly
/// (`fts5StorageDecodeSizeArray`).
fn docsize_columns(blob: &[u8]) -> Result<[i64; FTS_COLUMNS], Error> {
    const CORRUPT: Error = Error::CorruptBlob("fts_docsize");
    let mut columns = [0; FTS_COLUMNS];
    let mut rest = blob;
    for column in &mut columns {
        let (value, len) = sqlite_varint(rest).ok_or(CORRUPT)?;
        *column = i64::try_from(value).map_err(|_| CORRUPT)?;
        rest = &rest[len..];
    }
    if rest.is_empty() {
        Ok(columns)
    } else {
        Err(CORRUPT)
    }
}

/// Reads one SQLite varint from the front of `bytes`: up to eight bytes of
/// seven bits each, high bit set while more follow, then a ninth byte of
/// eight bits (`sqlite3Fts5GetVarint`). Returns the value and its length, or
/// `None` when `bytes` ends inside it.
fn sqlite_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for (i, &byte) in bytes.iter().enumerate().take(9) {
        if i == 8 {
            return Some(((value << 8) | u64::from(byte), 9));
        }
        value = (value << 7) | u64::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Adds `sizes` to the BM25 statistics row; negated sizes take rows out.
fn adjust_stats(tx: &Transaction<'_>, sizes: FtsSizes) -> Result<(), Error> {
    let [title, body] = sizes.tokens;
    tx.execute(
        "UPDATE bm25_stats SET rows = rows + ?1, title_tokens = title_tokens + ?2, \
         body_tokens = body_tokens + ?3",
        params![sizes.rows, title, body],
    )?;
    Ok(())
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
        let members = stored_members(&mut member_stmt, id)?;
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

/// The members of chunk `id`, in order, mapped from their columns.
fn stored_members(stmt: &mut rusqlite::Statement<'_>, id: i64) -> Result<Vec<StoredMember>, Error> {
    let members = stmt
        .query_map([id], |row| {
            let start = row.get::<_, i64>(0)?;
            let end = row.get::<_, i64>(1)?;
            let kind = StoredMemberKind::from_columns(&row.get::<_, String>(2)?, row.get(3)?)?;
            Ok(StoredMember {
                bytes: u64::try_from(start)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, start))?
                    ..u64::try_from(end)
                        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, end))?,
                kind,
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(members)
}

fn chunks_in(conn: &Connection, paths: &[String]) -> Result<Vec<StoredChunk>, Error> {
    let mut chunk_stmt = conn.prepare(
        "SELECT id, title, start_line, end_line, content_hash FROM chunks \
         WHERE path = ?1 ORDER BY id",
    )?;
    let mut member_stmt = conn.prepare(
        "SELECT start_byte, end_byte, kind, cut FROM chunk_members WHERE chunk_id = ?1 ORDER BY ord",
    )?;
    let mut out = Vec::new();
    for path in paths {
        let rows = chunk_stmt
            .query_map([path.as_str()], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u32>(2)?,
                    row.get::<_, u32>(3)?,
                    row.get::<_, ContentHash>(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        for (id, title, start, end, content_hash) in rows {
            let members = stored_members(&mut member_stmt, id)?;
            out.push(StoredChunk {
                path: path.clone(),
                title,
                lines: start..=end,
                members,
                content_hash,
            });
        }
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

fn row_hashes(conn: &Connection) -> Result<Vec<RowHash>, Error> {
    let mut stmt = conn.prepare(
        "SELECT c.path, c.content_hash, f.ignored, c.kind = ?1 FROM chunks c \
         JOIN files f ON f.path = c.path ORDER BY c.id",
    )?;
    let rows = stmt
        .query_map([query::DOC_KIND], |row| {
            let docs: bool = row.get(3)?;
            Ok(RowHash {
                path: row.get(0)?,
                content_hash: row.get(1)?,
                ignored: row.get(2)?,
                side: if docs { Side::Docs } else { Side::Code },
            })
        })?
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

#[cfg(all(test, feature = "lang-rust"))]
#[path = "bm25_tests.rs"]
mod bm25_tests;
