//! The repository-wide vector cache at `<git common dir>/repo-search/
//! vectors.v1.db`: content-hash -> fp16 vector, keyed per model id and shared
//! by every worktree's daemons at once through WAL. The schema version lives
//! in the file name, so different versions never touch the same file and the
//! store never deletes a cache another daemon may hold.

use std::fs;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use tokio::sync::{mpsc, oneshot};

use crate::UnitVector;

use super::{ContentHash, Error, ensure_version, open_conn, send_and_wait, spawn_actor};

/// Schema generation, part of the file name: a new version starts a new cache file.
const SCHEMA_VERSION: i64 = 1;

/// A cached embedding: its key plus the stored fp16 little-endian components.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedVector {
    /// The chunk content hash this vector belongs to.
    pub content_hash: ContentHash,
    fp16: Box<[u8]>,
}

impl CachedVector {
    /// The stored fp16 bytes: two little-endian bytes per component, what the
    /// in-memory KNN layout is built from.
    #[must_use]
    pub fn fp16_bytes(&self) -> &[u8] {
        &self.fp16
    }

    /// The components decoded back to f32, at fp16 precision.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        self.fp16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect()
    }
}

enum Request {
    GetMany {
        model_id: String,
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<Vec<CachedVector>, Error>>,
    },
    PutMany {
        model_id: String,
        vectors: Vec<(ContentHash, UnitVector)>,
        reply: oneshot::Sender<Result<(), Error>>,
    },
    Scan {
        model_id: String,
        after: Option<ContentHash>,
        limit: i64,
        reply: oneshot::Sender<Result<Vec<CachedVector>, Error>>,
    },
}

/// Handle to the vector-cache actor; clones share the one connection's queue.
/// Separate `VectorCache::open`s — in this or other daemon processes — share
/// the database file through WAL.
#[derive(Clone)]
pub struct VectorCache {
    tx: mpsc::Sender<Request>,
}

impl VectorCache {
    /// Opens — creating if needed — `<git_common_dir>/repo-search/
    /// vectors.v1.db`. Resolving the git common dir is the caller's job. The
    /// filesystem and SQLite work runs on the actor thread; `open` only awaits
    /// its outcome.
    ///
    /// The schema version is part of the file name; a `vectors.v1.db` whose
    /// `user_version` is not 1 is corrupt and [`Error::BadSchemaVersion`] is
    /// returned — the shared cache is never deleted by the store.
    ///
    /// # Errors
    ///
    /// When `git_common_dir` is not a directory, the store directory cannot be
    /// written, the file is a corrupt cache, WAL mode is refused, SQLite
    /// rejects the file, or the actor thread fails to spawn.
    pub async fn open(git_common_dir: &Path) -> Result<Self, Error> {
        let git_common_dir = git_common_dir.to_path_buf();
        let tx = spawn_actor("repo-search-vectors", move || init(&git_common_dir), serve).await?;
        Ok(Self { tx })
    }

    /// The vectors cached under `model_id` at `hashes`; misses are absent.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn get_many(
        &self,
        model_id: &str,
        hashes: &[ContentHash],
    ) -> Result<Vec<CachedVector>, Error> {
        send_and_wait(&self.tx, |reply| Request::GetMany {
            model_id: model_id.to_owned(),
            hashes: hashes.to_vec(),
            reply,
        })
        .await
    }

    /// Stores `vectors` under `model_id` in one transaction, replacing entries
    /// already cached at their content hash.
    ///
    /// # Errors
    ///
    /// When SQLite rejects a write — the whole batch is rolled back — or the
    /// actor is gone.
    pub async fn put_many(
        &self,
        model_id: &str,
        vectors: Vec<(ContentHash, UnitVector)>,
    ) -> Result<(), Error> {
        send_and_wait(&self.tx, |reply| Request::PutMany {
            model_id: model_id.to_owned(),
            vectors,
            reply,
        })
        .await
    }

    /// Up to `limit` vectors stored under `model_id` with `content_hash >
    /// after` (`None` starts at the first), ordered by content hash —
    /// keyset-paged reads the in-memory KNN layout is built from. Each page is
    /// an ordinary request/reply, so the actor never waits on a consumer.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn scan(
        &self,
        model_id: &str,
        after: Option<ContentHash>,
        limit: u32,
    ) -> Result<Vec<CachedVector>, Error> {
        send_and_wait(&self.tx, |reply| Request::Scan {
            model_id: model_id.to_owned(),
            after,
            limit: i64::from(limit),
            reply,
        })
        .await
    }
}

/// Creates `repo-search/` when missing and opens the cache database. Runs on
/// the actor thread.
fn init(git_common_dir: &Path) -> Result<Connection, Error> {
    if !git_common_dir.is_dir() {
        return Err(Error::NotDirectory(git_common_dir.to_path_buf()));
    }
    let dir = git_common_dir.join("repo-search");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("vectors.v{SCHEMA_VERSION}.db"));
    let mut conn = open_conn(&path)?;
    if let Some(found) = ensure_version(&mut conn, SCHEMA_VERSION, include_str!("vectors.sql"))? {
        return Err(Error::BadSchemaVersion {
            path,
            found,
            want: SCHEMA_VERSION,
        });
    }
    let mode: String =
        conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(Error::NotWal(path, mode));
    }
    Ok(conn)
}

fn serve(mut conn: Connection, mut rx: mpsc::Receiver<Request>) {
    while let Some(request) = rx.blocking_recv() {
        match request {
            Request::GetMany {
                model_id,
                hashes,
                reply,
            } => {
                let _ = reply.send(get_many(&mut conn, &model_id, &hashes));
            }
            Request::PutMany {
                model_id,
                vectors,
                reply,
            } => {
                let _ = reply.send(put_many(&mut conn, &model_id, &vectors));
            }
            Request::Scan {
                model_id,
                after,
                limit,
                reply,
            } => {
                let _ = reply.send(scan(&conn, &model_id, after, limit));
            }
        }
    }
}

/// One prepared point lookup per hash inside a single read transaction.
fn get_many(
    conn: &mut Connection,
    model_id: &str,
    hashes: &[ContentHash],
) -> Result<Vec<CachedVector>, Error> {
    let tx = conn.transaction()?;
    let mut hits = Vec::new();
    {
        let mut stmt = tx.prepare(
            "SELECT content_hash, vector FROM vectors \
             WHERE model_id = ?1 AND content_hash = ?2",
        )?;
        for hash in hashes {
            if let Some((content_hash, blob)) = stmt
                .query_row(params![model_id, *hash], |row| {
                    Ok((row.get::<_, ContentHash>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .optional()?
            {
                hits.push(cached_vector(content_hash, blob)?);
            }
        }
    }
    tx.commit()?;
    Ok(hits)
}

fn put_many(
    conn: &mut Connection,
    model_id: &str,
    vectors: &[(ContentHash, UnitVector)],
) -> Result<(), Error> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO vectors (model_id, content_hash, vector) \
             VALUES (?1, ?2, ?3)",
        )?;
        for (hash, vector) in vectors {
            let mut fp16 = Vec::with_capacity(vector.as_slice().len() * 2);
            for &x in vector.as_slice() {
                fp16.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
            }
            stmt.execute(params![model_id, *hash, fp16])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn scan(
    conn: &Connection,
    model_id: &str,
    after: Option<ContentHash>,
    limit: i64,
) -> Result<Vec<CachedVector>, Error> {
    // An empty blob sorts before every stored key, so `after = None` reads
    // from the first row.
    let cursor: Vec<u8> = after.map_or_else(Vec::new, |h| h.as_bytes().to_vec());
    let mut stmt = conn.prepare(
        "SELECT content_hash, vector FROM vectors \
         WHERE model_id = ?1 AND content_hash > ?2 ORDER BY content_hash LIMIT ?3",
    )?;
    let mut rows = stmt.query(params![model_id, cursor, limit])?;
    let mut page = Vec::new();
    while let Some(row) = rows.next()? {
        page.push(cached_vector(row.get(0)?, row.get(1)?)?);
    }
    Ok(page)
}

fn cached_vector(hash: ContentHash, blob: Vec<u8>) -> Result<CachedVector, Error> {
    if !blob.len().is_multiple_of(2) {
        return Err(Error::CorruptBlob("vector"));
    }
    Ok(CachedVector {
        content_hash: hash,
        fp16: blob.into_boxed_slice(),
    })
}
