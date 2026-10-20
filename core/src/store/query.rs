//! Read queries over the index that retrieval needs: BM25 candidates, chunks
//! of a content hash, chunk rows with their members, and per-side counts.

use std::ops::{Range, RangeInclusive};

use rusqlite::{Connection, named_params, params};

use crate::chunk::ChunkKind;

use super::{ContentHash, Error};

/// The kind string of a documentation chunk: the one chunker kind the store
/// itself gives meaning to. It is the one source of the `doc` string; SQL
/// receives it as a bound parameter, never as a literal.
const DOC_KIND: &str = match ChunkKind::Doc.static_name() {
    Some(name) => name,
    None => panic!("ChunkKind::Doc has a static name"),
};

/// The two result lists: each is searched, fused and ranked on its own.
///
/// A chunk is on the docs side exactly when its kind is [`ChunkKind::Doc`];
/// module docs (`docs`) and configuration chunks are code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    /// Source code (and configuration).
    Code,
    /// Documentation sections.
    Docs,
}

impl Side {
    /// The side a chunk of `kind` belongs to.
    #[must_use]
    pub fn of_kind(kind: &str) -> Self {
        if kind == DOC_KIND {
            Self::Docs
        } else {
            Self::Code
        }
    }
}

/// A chunk's row id in the index it was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId(pub i64);

/// Restricts a chunk query to one side and to the user's filters. The path
/// set is resolved by the caller (globs are matched outside SQLite); file
/// extensions, kinds and the symbol glob are matched in SQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkFilter {
    /// Which list the chunks belong to.
    pub side: Side,
    /// Whether chunks of ignored files are visible (`--include-ignored`).
    pub include_ignored: bool,
    /// When set, only chunks of exactly these worktree-relative paths.
    pub paths: Option<Vec<String>>,
    /// When non-empty, only chunks in files with one of these extensions:
    /// what the `--lang` filters select, matched by SQLite `LIKE` on the path
    /// (ASCII case-insensitive, as [`Language::of_path`](crate::chunk::Language::of_path)).
    pub extensions: Vec<&'static str>,
    /// When non-empty, only chunks whose kind, or one of whose members' kinds,
    /// is listed.
    pub kinds: Vec<String>,
    /// When set, only chunks with a member whose symbol path matches this
    /// SQLite `GLOB`.
    pub symbol: Option<String>,
}

impl ChunkFilter {
    /// Whether the filter admits every chunk of its side: no path set,
    /// extension list, kinds or symbol restricts it, so a side-specific index
    /// needs no mask.
    #[must_use]
    pub fn is_unrestricted(&self) -> bool {
        self.include_ignored
            && self.paths.is_none()
            && self.extensions.is_empty()
            && self.kinds.is_empty()
            && self.symbol.is_none()
    }
}

/// A filter's SQL parameters: JSON arrays that `json_each` expands.
pub(super) struct FilterParams {
    docs: bool,
    include_ignored: bool,
    doc_kind: &'static str,
    paths: Option<String>,
    extensions: Option<String>,
    kinds: Option<String>,
    symbol: Option<String>,
}

impl ChunkFilter {
    pub(super) fn params(&self) -> FilterParams {
        fn json<T: serde::Serialize + ?Sized>(v: &T) -> String {
            serde_json::to_string(v).expect("strings serialize")
        }
        FilterParams {
            docs: self.side == Side::Docs,
            include_ignored: self.include_ignored,
            doc_kind: DOC_KIND,
            paths: self.paths.as_deref().map(json),
            extensions: (!self.extensions.is_empty()).then(|| json(&self.extensions)),
            kinds: (!self.kinds.is_empty()).then(|| json(self.kinds.as_slice())),
            symbol: self.symbol.clone(),
        }
    }
}

/// One member of a stored chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberRow {
    /// Byte range in the file.
    pub bytes: Range<u64>,
    /// The member's syntactic kind.
    pub kind: String,
    /// The member's symbol path; empty when it has none.
    pub symbol: String,
}

/// A stored chunk with everything needed to rebuild its text from the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRow {
    /// The row id.
    pub id: ChunkId,
    /// Worktree-relative path.
    pub path: String,
    /// The chunk title.
    pub title: String,
    /// The chunk kind.
    pub kind: String,
    /// 1-based inclusive line range.
    pub lines: RangeInclusive<u32>,
    /// The vector cache key.
    pub content_hash: ContentHash,
    /// The members, in order.
    pub members: Vec<MemberRow>,
}

/// BM25 order, best first, of the chunks matching the FTS5 `query`.
pub(super) fn bm25(
    conn: &Connection,
    query: &str,
    filter: &ChunkFilter,
    limit: u32,
) -> Result<Vec<ChunkId>, Error> {
    let p = filter.params();
    let mut stmt = conn.prepare(concat!(
        include_str!("bm25_head.sql"),
        include_str!("filter.sql"),
        include_str!("bm25_tail.sql"),
    ))?;
    let ids = stmt
        .query_map(
            named_params! {
                ":query": query, ":limit": limit, ":docs": p.docs, ":include_ignored": p.include_ignored, ":doc_kind": p.doc_kind,
                ":paths": p.paths, ":extensions": p.extensions, ":kinds": p.kinds,
                ":symbol": p.symbol,
            },
            |row| row.get(0).map(ChunkId),
        )?
        .collect::<Result<_, _>>()?;
    Ok(ids)
}

/// For each hash in order, the ids of the chunks holding it that pass
/// `filter`; a hash this index does not hold, or the filter rejects, yields
/// an empty list.
pub(super) fn by_hashes(
    conn: &Connection,
    hashes: &[ContentHash],
    filter: &ChunkFilter,
) -> Result<Vec<Vec<ChunkId>>, Error> {
    let p = filter.params();
    let mut stmt = conn.prepare(concat!(
        include_str!("by_hash_head.sql"),
        include_str!("filter.sql"),
        include_str!("by_hash_tail.sql"),
    ))?;
    hashes
        .iter()
        .map(|hash| {
            let ids = stmt
                .query_map(
                    named_params! {
                        ":hash": hash, ":docs": p.docs, ":include_ignored": p.include_ignored, ":doc_kind": p.doc_kind,
                        ":paths": p.paths, ":extensions": p.extensions, ":kinds": p.kinds,
                        ":symbol": p.symbol,
                    },
                    |row| row.get(0).map(ChunkId),
                )?
                .collect::<Result<_, _>>()?;
            Ok(ids)
        })
        .collect()
}

/// The rows of `ids`, in the order given; an id the index does not hold is
/// absent.
pub(super) fn chunks(conn: &Connection, ids: &[ChunkId]) -> Result<Vec<ChunkRow>, Error> {
    let mut chunk_stmt = conn.prepare(
        "SELECT path, title, kind, start_line, end_line, content_hash FROM chunks WHERE id = ?1",
    )?;
    let mut member_stmt = conn.prepare(
        "SELECT start_byte, end_byte, kind, symbol FROM chunk_members \
         WHERE chunk_id = ?1 ORDER BY ord",
    )?;
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let row = chunk_stmt.query_row([id.0], |row| {
            Ok(ChunkRow {
                id,
                path: row.get(0)?,
                title: row.get(1)?,
                kind: row.get(2)?,
                lines: row.get(3)?..=row.get(4)?,
                content_hash: row.get(5)?,
                members: Vec::new(),
            })
        });
        let mut row = match row {
            Ok(row) => row,
            Err(rusqlite::Error::QueryReturnedNoRows) => continue,
            Err(e) => return Err(e.into()),
        };
        row.members = member_stmt
            .query_map([id.0], |m| {
                let start: i64 = m.get(0)?;
                let end: i64 = m.get(1)?;
                Ok(MemberRow {
                    bytes: u64::try_from(start)
                        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, start))?
                        ..u64::try_from(end)
                            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, end))?,
                    kind: m.get(2)?,
                    symbol: m.get(3)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        out.push(row);
    }
    Ok(out)
}

/// The number of chunk rows on `side`, those of ignored files only when
/// `include_ignored`.
pub(super) fn count(conn: &Connection, side: Side, include_ignored: bool) -> Result<u64, Error> {
    let n: i64 = conn.query_row(
        "SELECT count(*) FROM chunks c WHERE (c.kind = ?1) = ?2 AND (?3 OR c.path NOT IN \
         (SELECT path FROM files WHERE ignored = 1))",
        params![DOC_KIND, side == Side::Docs, include_ignored],
        |row| row.get(0),
    )?;
    Ok(u64::try_from(n).expect("a count is non-negative"))
}

/// Every distinct chunk content hash with its side.
pub(super) fn hash_sides(conn: &Connection) -> Result<Vec<(ContentHash, Side)>, Error> {
    let mut stmt = conn.prepare("SELECT DISTINCT content_hash, kind = ?1 FROM chunks")?;
    let rows = stmt
        .query_map([DOC_KIND], |row| {
            let docs: bool = row.get(1)?;
            Ok((row.get(0)?, if docs { Side::Docs } else { Side::Code }))
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// The distinct content hashes of the chunks passing `filter`: the rows a
/// vector scan may return.
pub(super) fn hashes_matching(
    conn: &Connection,
    filter: &ChunkFilter,
) -> Result<Vec<ContentHash>, Error> {
    let p = filter.params();
    let mut stmt = conn.prepare(concat!(
        include_str!("hashes_head.sql"),
        include_str!("filter.sql"),
    ))?;
    let hashes = stmt
        .query_map(
            named_params! {
                ":docs": p.docs, ":include_ignored": p.include_ignored, ":doc_kind": p.doc_kind, ":paths": p.paths,
                ":extensions": p.extensions, ":kinds": p.kinds, ":symbol": p.symbol,
            },
            |row| row.get(0),
        )?
        .collect::<Result<_, _>>()?;
    Ok(hashes)
}

/// The members of `ids` whose chunk text matches the FTS5 expression `query`,
/// in id order.
pub(super) fn matching(
    conn: &Connection,
    query: &str,
    ids: &[ChunkId],
) -> Result<Vec<ChunkId>, Error> {
    let ids_json = serde_json::to_string(&ids.iter().map(|id| id.0).collect::<Vec<_>>())
        .expect("integers serialize");
    let mut stmt = conn.prepare(include_str!("matching.sql"))?;
    let found = stmt
        .query_map(named_params! { ":query": query, ":ids": ids_json }, |row| {
            row.get(0).map(ChunkId)
        })?
        .collect::<Result<_, _>>()?;
    Ok(found)
}

/// Every distinct kind the index holds, chunk kinds and member kinds alike.
pub(super) fn kinds(conn: &Connection) -> Result<Vec<String>, Error> {
    let mut stmt =
        conn.prepare("SELECT kind FROM chunks UNION SELECT kind FROM chunk_members ORDER BY kind")?;
    let kinds = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(kinds)
}
