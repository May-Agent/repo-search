//! The repository-wide vector cache under the daemon's [`StateDir`]
//! ([`StateDir::repo_cache`]):
//! `vectors.v1.db` maps (model id, content hash) to a slot, and each model's
//! fp16 vectors live in a flat file `vectors.v1.<model hash>.f16` where row
//! `slot` sits at byte offset `slot * dim * 2`. Keyed per model id and shared
//! by every worktree's daemons at once through WAL. The schema version lives
//! in the file names, so different versions never touch the same files and
//! the store never deletes a cache another daemon may hold.
//!
//! Slots are allocated inside SQLite's write transaction and rows are written
//! positionally, so concurrent daemons never overlap: the write lock
//! serialises allocation. Rows are written and synced before the transaction
//! commits. A crash or rollback in between leaves unreferenced bytes at slots
//! at or past `next_slot`, which a later batch overwrites. The invariants
//! readers rely on: a committed row is never rewritten, only committed slots
//! (those the `vectors` table references) are ever read, and no reference
//! spans uncommitted bytes, so the mapped file is only ever viewed one
//! committed row at a time.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use numkong::f16;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use tokio::sync::{mpsc, oneshot};

use crate::{StateDir, UnitVector};

use super::{ContentHash, Error, ensure_version, open_conn, send_and_wait, spawn_actor};

/// Schema generation, part of the file names: a new version starts new cache files.
const SCHEMA_VERSION: i64 = 1;

/// A cached embedding: its key, its slot in the model's file, and the stored
/// fp16 little-endian components read from that row.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedVector {
    /// The chunk content hash this vector belongs to.
    pub content_hash: ContentHash,
    /// The row of the model's fp16 file holding this vector.
    pub slot: u64,
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
            .map(|c| f16(u16::from_le_bytes(*c)).to_f32())
            .collect()
    }
}

/// A stored vector's key and slot, what the in-memory KNN layout is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedSlot {
    /// The chunk content hash.
    pub content_hash: ContentHash,
    /// The row of the model's fp16 file holding the vector.
    pub slot: u64,
}

/// What the cache knows about one model's vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Components per vector.
    pub dim: usize,
    /// How many distinct content hashes are cached.
    pub vectors: u64,
    /// The first unallocated row of the fp16 file: every cached vector sits
    /// in a slot below it, and a vector stored later gets a slot at or above.
    pub next_slot: u64,
    /// The flat fp16 file; row `slot` is at byte offset `slot * dim * 2`.
    pub file: PathBuf,
}

/// The per-dimension bounds the in-memory int8 codes of a model are quantized
/// with, and the cached vector count they were taken over.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredCalibration {
    /// Per-dimension minimum.
    pub lo: Box<[f32]>,
    /// Per-dimension maximum.
    pub hi: Box<[f32]>,
    /// How many vectors the bounds were computed from.
    pub count: u64,
}

impl StoredCalibration {
    /// Empty bounds over `dim` dimensions: no vectors counted yet.
    #[must_use]
    pub fn empty(dim: usize) -> Self {
        Self {
            lo: vec![f32::INFINITY; dim].into(),
            hi: vec![f32::NEG_INFINITY; dim].into(),
            count: 0,
        }
    }

    /// Widens the bounds to cover `row` and counts it.
    ///
    /// # Panics
    ///
    /// When `row` is not as long as the bounds.
    pub fn add(&mut self, row: &[f16]) {
        self.widen(row.iter().map(|x| x.to_f32()));
    }

    /// Widens the bounds to cover a row already converted to `f32`.
    fn widen(&mut self, row: impl ExactSizeIterator<Item = f32>) {
        assert_eq!(row.len(), self.lo.len(), "row dimension");
        for ((x, lo), hi) in row.zip(&mut self.lo).zip(&mut self.hi) {
            *lo = lo.min(x);
            *hi = hi.max(x);
        }
        self.count += 1;
    }
}

enum Request {
    GetMany {
        model_id: String,
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<Vec<CachedVector>, Error>>,
    },
    Cached {
        model_id: String,
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Result<HashSet<ContentHash>, Error>>,
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
        reply: oneshot::Sender<Result<Vec<CachedSlot>, Error>>,
    },
    Since {
        model_id: String,
        from_slot: u64,
        reply: oneshot::Sender<Result<Vec<CachedSlot>, Error>>,
    },
    Model {
        model_id: String,
        reply: oneshot::Sender<Result<Option<ModelInfo>, Error>>,
    },
    Calibration {
        model_id: String,
        reply: oneshot::Sender<Result<Option<StoredCalibration>, Error>>,
    },
    Bounds {
        model_id: String,
        reply: oneshot::Sender<Result<StoredCalibration, Error>>,
    },
    SetCalibration {
        model_id: String,
        calibration: StoredCalibration,
        reply: oneshot::Sender<Result<(), Error>>,
    },
}

/// Handle to the vector-cache actor; clones share the one connection's queue.
/// Separate `VectorCache::open`s — in this or other daemon processes — share
/// the database and the fp16 files.
#[derive(Clone)]
pub struct VectorCache {
    tx: mpsc::Sender<Request>,
}

impl VectorCache {
    /// Opens — creating if needed — the repository's cache under `state`
    /// ([`StateDir::repo_cache`]); nothing is written under the worktree
    /// `git_common_dir` belongs to. `git_common_dir` is the repository's
    /// canonical git common dir — resolving it is the caller's job. The
    /// filesystem and SQLite work runs on the actor thread; `open` only
    /// awaits its outcome.
    ///
    /// The schema version is part of the file name; a `vectors.v1.db` whose
    /// `user_version` is not 1 is corrupt and [`Error::BadSchemaVersion`] is
    /// returned — the shared cache is never deleted by the store.
    ///
    /// # Errors
    ///
    /// When `git_common_dir` is not a directory, the state directory cannot be
    /// written, the file is a corrupt cache, WAL mode is refused, SQLite
    /// rejects the file, or the actor thread fails to spawn.
    pub async fn open(state: &StateDir, git_common_dir: &Path) -> Result<Self, Error> {
        let dir = state.repo_cache(git_common_dir);
        let git_common_dir = git_common_dir.to_path_buf();
        let serve_dir = dir.clone();
        let tx = spawn_actor(
            "repo-search-vectors",
            move || init(&git_common_dir, &dir),
            move |conn, rx| serve(conn, &serve_dir, rx),
        )
        .await?;
        Ok(Self { tx })
    }

    /// The vectors cached under `model_id` at `hashes`, with their rows read
    /// from the fp16 file; misses are absent.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read, the fp16 file cannot be read, or the
    /// actor is gone.
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

    /// The subset of `hashes` that has a cached vector under `model_id`.
    /// Answered from the SQLite index alone; the fp16 file is never read.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn cached(
        &self,
        model_id: &str,
        hashes: &[ContentHash],
    ) -> Result<HashSet<ContentHash>, Error> {
        send_and_wait(&self.tx, |reply| Request::Cached {
            model_id: model_id.to_owned(),
            hashes: hashes.to_vec(),
            reply,
        })
        .await
    }

    /// Stores `vectors` under `model_id` in one transaction: new content
    /// hashes get the next slots and their rows are written and synced
    /// before the commit. A hash already cached keeps its stored vector.
    ///
    /// # Errors
    ///
    /// When a vector's dimension differs from the one recorded for the model,
    /// SQLite rejects a write — the whole batch is rolled back — the fp16
    /// file cannot be written, or the actor is gone.
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

    /// Up to `limit` entries stored under `model_id` with `content_hash >
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
    ) -> Result<Vec<CachedSlot>, Error> {
        send_and_wait(&self.tx, |reply| Request::Scan {
            model_id: model_id.to_owned(),
            after,
            limit: i64::from(limit),
            reply,
        })
        .await
    }

    /// Every entry stored under `model_id` at slot `from_slot` or later,
    /// ordered by slot: the vectors added since a reader last saw
    /// [`ModelInfo::next_slot`] equal to `from_slot`.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn since(&self, model_id: &str, from_slot: u64) -> Result<Vec<CachedSlot>, Error> {
        send_and_wait(&self.tx, |reply| Request::Since {
            model_id: model_id.to_owned(),
            from_slot,
            reply,
        })
        .await
    }

    /// The dimension, vector count and fp16 file of `model_id`; `None` before
    /// its first vector is stored.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read or the actor is gone.
    pub async fn model(&self, model_id: &str) -> Result<Option<ModelInfo>, Error> {
        send_and_wait(&self.tx, |reply| Request::Model {
            model_id: model_id.to_owned(),
            reply,
        })
        .await
    }

    /// The calibration stored for `model_id`, if one was set.
    ///
    /// # Errors
    ///
    /// When SQLite rejects the read, the stored blobs are malformed, or the
    /// actor is gone.
    pub async fn calibration(&self, model_id: &str) -> Result<Option<StoredCalibration>, Error> {
        send_and_wait(&self.tx, |reply| Request::Calibration {
            model_id: model_id.to_owned(),
            reply,
        })
        .await
    }

    /// The per-dimension bounds over every vector cached under `model_id`,
    /// computed on the actor thread, which owns the file and database reads.
    ///
    /// # Errors
    ///
    /// When the model has no vectors ([`Error::UnknownModel`]), SQLite rejects
    /// the read, the fp16 file cannot be read, or the actor is gone.
    pub async fn bounds(&self, model_id: &str) -> Result<StoredCalibration, Error> {
        send_and_wait(&self.tx, |reply| Request::Bounds {
            model_id: model_id.to_owned(),
            reply,
        })
        .await
    }

    /// Replaces the calibration of `model_id`.
    ///
    /// # Errors
    ///
    /// When the model has no vectors ([`Error::UnknownModel`]), the
    /// calibration counts no vectors ([`Error::EmptyCalibration`]), a bound's
    /// length differs from the model's dimension, SQLite rejects the write, or
    /// the actor is gone.
    pub async fn set_calibration(
        &self,
        model_id: &str,
        calibration: StoredCalibration,
    ) -> Result<(), Error> {
        send_and_wait(&self.tx, |reply| Request::SetCalibration {
            model_id: model_id.to_owned(),
            calibration,
            reply,
        })
        .await
    }
}

/// Creates the repository's cache directory under the state root when
/// missing and opens the cache database. Runs on the actor thread.
fn init(git_common_dir: &Path, dir: &Path) -> Result<Connection, Error> {
    if !git_common_dir.is_dir() {
        return Err(Error::NotDirectory(git_common_dir.to_path_buf()));
    }
    fs::create_dir_all(dir)?;
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

fn serve(mut conn: Connection, dir: &Path, mut rx: mpsc::Receiver<Request>) {
    while let Some(request) = rx.blocking_recv() {
        match request {
            Request::GetMany {
                model_id,
                hashes,
                reply,
            } => {
                let _ = reply.send(get_many(&mut conn, dir, &model_id, &hashes));
            }
            Request::Cached {
                model_id,
                hashes,
                reply,
            } => {
                let _ = reply.send(cached(&mut conn, &model_id, &hashes));
            }
            Request::PutMany {
                model_id,
                vectors,
                reply,
            } => {
                let _ = reply.send(put_many(&mut conn, dir, &model_id, &vectors));
            }
            Request::Scan {
                model_id,
                after,
                limit,
                reply,
            } => {
                let _ = reply.send(scan(&conn, &model_id, after, limit));
            }
            Request::Since {
                model_id,
                from_slot,
                reply,
            } => {
                let _ = reply.send(since(&conn, &model_id, from_slot));
            }
            Request::Model { model_id, reply } => {
                let _ = reply.send(model(&conn, dir, &model_id));
            }
            Request::Calibration { model_id, reply } => {
                let _ = reply.send(calibration(&conn, &model_id));
            }
            Request::Bounds { model_id, reply } => {
                let _ = reply.send(bounds(&conn, dir, &model_id));
            }
            Request::SetCalibration {
                model_id,
                calibration,
                reply,
            } => {
                let _ = reply.send(set_calibration(&conn, &model_id, &calibration));
            }
        }
    }
}

/// The fp16 file of `model_id`: the first 16 hex characters of the BLAKE3 of
/// the id keep arbitrary ids valid file names.
fn fp16_path(dir: &Path, model_id: &str) -> PathBuf {
    let digest = blake3::hash(model_id.as_bytes());
    let prefix = digest
        .as_bytes()
        .first_chunk::<8>()
        .expect("BLAKE3 digest is 32 bytes");
    dir.join(format!(
        "vectors.v{SCHEMA_VERSION}.{:016x}.f16",
        u64::from_be_bytes(*prefix)
    ))
}

fn usize_of(value: i64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

fn u64_of(value: i64) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

fn i64_of(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::IntRange(i128::from(value)))
}

/// A model's `(dim, next_slot)`.
fn model_row(conn: &Connection, model_id: &str) -> Result<Option<(usize, u64)>, Error> {
    let row: Option<(i64, i64)> = conn
        .query_row(
            "SELECT dim, next_slot FROM models WHERE model_id = ?1",
            params![model_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    row.map(|(dim, next)| Ok((usize_of(dim)?, u64_of(next)?)))
        .transpose()
}

/// Byte offset of row `slot`.
fn row_offset(slot: u64, dim: usize) -> u64 {
    slot * dim as u64 * 2
}

/// One prepared point lookup per hash inside a single read transaction, then
/// each hit's row read from the fp16 file.
fn get_many(
    conn: &mut Connection,
    dir: &Path,
    model_id: &str,
    hashes: &[ContentHash],
) -> Result<Vec<CachedVector>, Error> {
    let tx = conn.transaction()?;
    let Some((dim, _)) = model_row(&tx, model_id)? else {
        return Ok(Vec::new());
    };
    let mut slots = Vec::new();
    {
        let mut stmt =
            tx.prepare("SELECT slot FROM vectors WHERE model_id = ?1 AND content_hash = ?2")?;
        for hash in hashes {
            if let Some(slot) = stmt
                .query_row(params![model_id, *hash], |row| row.get::<_, i64>(0))
                .optional()?
            {
                slots.push((*hash, u64_of(slot)?));
            }
        }
    }
    tx.commit()?;
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    let file = File::open(fp16_path(dir, model_id))?;
    slots
        .into_iter()
        .map(|(content_hash, slot)| {
            let mut fp16 = vec![0; dim * 2];
            file.read_exact_at(&mut fp16, row_offset(slot, dim))?;
            Ok(CachedVector {
                content_hash,
                slot,
                fp16: fp16.into_boxed_slice(),
            })
        })
        .collect()
}

/// Membership only: one existence probe per hash inside a single read
/// transaction, touching no fp16 row.
fn cached(
    conn: &mut Connection,
    model_id: &str,
    hashes: &[ContentHash],
) -> Result<HashSet<ContentHash>, Error> {
    let tx = conn.transaction()?;
    let mut found = HashSet::with_capacity(hashes.len());
    {
        let mut stmt = tx.prepare(
            "SELECT EXISTS(SELECT 1 FROM vectors WHERE model_id = ?1 AND content_hash = ?2)",
        )?;
        for hash in hashes {
            if stmt.query_row(params![model_id, *hash], |row| row.get::<_, bool>(0))? {
                found.insert(*hash);
            }
        }
    }
    tx.commit()?;
    Ok(found)
}

fn put_many(
    conn: &mut Connection,
    dir: &Path,
    model_id: &str,
    vectors: &[(ContentHash, UnitVector)],
) -> Result<(), Error> {
    let Some((_, first)) = vectors.first() else {
        return Ok(());
    };
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (dim, mut next_slot) = match model_row(&tx, model_id)? {
        Some(row) => row,
        None => {
            let dim = first.as_slice().len();
            tx.execute(
                "INSERT INTO models (model_id, dim, next_slot) VALUES (?1, ?2, 0)",
                params![model_id, i64_of(dim as u64)?],
            )?;
            (dim, 0)
        }
    };
    if let Some((_, vector)) = vectors.iter().find(|(_, v)| v.as_slice().len() != dim) {
        return Err(Error::DimMismatch {
            model_id: model_id.to_owned(),
            expected: dim,
            got: vector.as_slice().len(),
        });
    }

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(fp16_path(dir, model_id))?;
    let allocated_from = next_slot;
    {
        let mut present =
            tx.prepare("SELECT 1 FROM vectors WHERE model_id = ?1 AND content_hash = ?2")?;
        let mut insert =
            tx.prepare("INSERT INTO vectors (model_id, content_hash, slot) VALUES (?1, ?2, ?3)")?;
        let mut row = Vec::with_capacity(dim * 2);
        for (hash, vector) in vectors {
            if present.exists(params![model_id, *hash])? {
                continue;
            }
            row.clear();
            for &x in vector.as_slice() {
                row.extend_from_slice(&f16::from_f32(x).0.to_le_bytes());
            }
            file.write_all_at(&row, row_offset(next_slot, dim))?;
            insert.execute(params![model_id, *hash, i64_of(next_slot)?])?;
            next_slot += 1;
        }
    }
    if next_slot > allocated_from {
        tx.execute(
            "UPDATE models SET next_slot = ?2 WHERE model_id = ?1",
            params![model_id, i64_of(next_slot)?],
        )?;
        file.sync_data()?;
    }
    tx.commit()?;
    Ok(())
}

fn scan(
    conn: &Connection,
    model_id: &str,
    after: Option<ContentHash>,
    limit: i64,
) -> Result<Vec<CachedSlot>, Error> {
    // An empty blob sorts before every stored key, so `after = None` reads
    // from the first row.
    let cursor: Vec<u8> = after.map_or_else(Vec::new, |h| h.as_bytes().to_vec());
    let mut stmt = conn.prepare(
        "SELECT content_hash, slot FROM vectors \
         WHERE model_id = ?1 AND content_hash > ?2 ORDER BY content_hash LIMIT ?3",
    )?;
    let mut rows = stmt.query(params![model_id, cursor, limit])?;
    let mut page = Vec::new();
    while let Some(row) = rows.next()? {
        page.push(CachedSlot {
            content_hash: row.get(0)?,
            slot: u64_of(row.get(1)?)?,
        });
    }
    Ok(page)
}

fn since(conn: &Connection, model_id: &str, from_slot: u64) -> Result<Vec<CachedSlot>, Error> {
    let mut stmt = conn.prepare(
        "SELECT content_hash, slot FROM vectors \
         WHERE model_id = ?1 AND slot >= ?2 ORDER BY slot",
    )?;
    let mut rows = stmt.query(params![model_id, i64_of(from_slot)?])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(CachedSlot {
            content_hash: row.get(0)?,
            slot: u64_of(row.get(1)?)?,
        });
    }
    Ok(out)
}

fn model(conn: &Connection, dir: &Path, model_id: &str) -> Result<Option<ModelInfo>, Error> {
    let Some((dim, next_slot)) = model_row(conn, model_id)? else {
        return Ok(None);
    };
    // Slots are allocated densely and never deleted: the first unallocated
    // slot is the vector count.
    Ok(Some(ModelInfo {
        dim,
        vectors: next_slot,
        next_slot,
        file: fp16_path(dir, model_id),
    }))
}

fn f32s_to_blob(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_to_f32s(blob: &[u8], dim: usize, what: &'static str) -> Result<Box<[f32]>, Error> {
    if blob.len() != dim * 4 {
        return Err(Error::CorruptBlob(what));
    }
    Ok(blob
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn calibration(conn: &Connection, model_id: &str) -> Result<Option<StoredCalibration>, Error> {
    type Row = (i64, Option<Vec<u8>>, Option<Vec<u8>>, Option<i64>);
    let row: Option<Row> = conn
        .query_row(
            "SELECT dim, calib_lo, calib_hi, calib_count FROM models WHERE model_id = ?1",
            params![model_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    let Some((dim, Some(lo), Some(hi), Some(count))) = row else {
        return Ok(None);
    };
    let dim = usize_of(dim)?;
    Ok(Some(StoredCalibration {
        lo: blob_to_f32s(&lo, dim, "calibration lo")?,
        hi: blob_to_f32s(&hi, dim, "calibration hi")?,
        count: u64_of(count)?,
    }))
}

/// Bounds over every committed row of `model_id`; only slots the `vectors`
/// table references are read.
fn bounds(conn: &Connection, dir: &Path, model_id: &str) -> Result<StoredCalibration, Error> {
    let Some((dim, _)) = model_row(conn, model_id)? else {
        return Err(Error::UnknownModel(model_id.to_owned()));
    };
    let file = File::open(fp16_path(dir, model_id))?;
    let mut stmt = conn.prepare("SELECT slot FROM vectors WHERE model_id = ?1 ORDER BY slot")?;
    let mut slots = stmt.query(params![model_id])?;
    let mut bounds = StoredCalibration::empty(dim);
    let mut bytes = vec![0; dim * 2];
    let mut row = vec![f16(0); dim];
    let mut wide = vec![0.0f32; dim];
    while let Some(slot) = slots.next()? {
        file.read_exact_at(&mut bytes, row_offset(u64_of(slot.get(0)?)?, dim))?;
        for (x, c) in row.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *x = f16(u16::from_le_bytes(*c));
        }
        numkong::cast(&row, &mut wide).expect("row and buffer are both dim long");
        bounds.widen(wide.iter().copied());
    }
    Ok(bounds)
}

fn set_calibration(
    conn: &Connection,
    model_id: &str,
    calibration: &StoredCalibration,
) -> Result<(), Error> {
    let Some((dim, _)) = model_row(conn, model_id)? else {
        return Err(Error::UnknownModel(model_id.to_owned()));
    };
    if calibration.count == 0 {
        return Err(Error::EmptyCalibration(model_id.to_owned()));
    }
    for bound in [&calibration.lo, &calibration.hi] {
        if bound.len() != dim {
            return Err(Error::DimMismatch {
                model_id: model_id.to_owned(),
                expected: dim,
                got: bound.len(),
            });
        }
    }
    conn.execute(
        "UPDATE models SET calib_lo = ?2, calib_hi = ?3, calib_count = ?4 WHERE model_id = ?1",
        params![
            model_id,
            f32s_to_blob(&calibration.lo),
            f32s_to_blob(&calibration.hi),
            i64_of(calibration.count)?
        ],
    )?;
    Ok(())
}
