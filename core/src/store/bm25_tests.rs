//! `exact_bm25` and the BM25 statistics the index keeps for it, over the
//! chunks of the crate's Rust fixtures.

use std::collections::BTreeMap;

use rusqlite::Connection;

use super::{FileChange, align, apply, init};
use crate::chunk::{Chunker, Rust};
use crate::counters::CapOff;
use crate::reconcile::chunk_with;
use crate::store::query::{self, ChunkFilter, Side};
use crate::store::{ChunkRecord, ContentHash, FileState, Unchunkable, UnchunkableReason};

const OVERSIZED_FN: &str = include_str!("../../tests/fixtures/oversized_fn.rs");
const OVERSIZED_IMPL: &str = include_str!("../../tests/fixtures/oversized_impl.rs");
const SPLIT_SCOPES: &str = include_str!("../../tests/fixtures/split_scopes.rs");

/// Single-term, AND, OR, phrase and column-filtered queries over terms the
/// fixtures hold in many, few and one chunk.
const QUERIES: &[&str] = &[
    "\"padding\"",
    "\"usize\"",
    "\"self\"",
    "\"inline\"",
    "\"preamble\"",
    "\"padding\" OR \"inline\"",
    "\"member\" OR \"deep\" OR \"big\"",
    "\"padding\" \"self\"",
    "\"padding comment\"",
    "title : \"pad\"",
];

const CHUNKER: &str = "fixture-chunker";
const MODEL: &str = "fixture-model";

/// The statistics arguments `exact_bm25` takes, as `bm25_head.sql` passes
/// them.
const STATS_ARGS: &str = "(SELECT rows FROM bm25_stats), (SELECT title_tokens FROM bm25_stats), \
     (SELECT body_tokens FROM bm25_stats)";

/// An index database in its own tempdir, opened as the actor opens it.
struct Fixture {
    conn: Connection,
    _dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut conn = init(dir.path(), &dir.path().join("index.db")).unwrap();
        align(&mut conn, CHUNKER, MODEL).unwrap();
        Self { conn, _dir: dir }
    }

    /// A fresh index holding exactly `files`, written in one batch.
    fn fresh(files: &BTreeMap<&str, &str>) -> Self {
        let mut fx = Self::new();
        fx.apply(files.iter().map(|(path, src)| replace(path, src)).collect());
        fx
    }

    fn apply(&mut self, changes: Vec<FileChange>) {
        apply(&mut self.conn, &changes).unwrap();
    }

    /// The `bm25_stats` row: rows, title tokens, body tokens.
    fn stats(&self) -> [i64; 3] {
        self.conn
            .query_row(
                "SELECT rows, title_tokens, body_tokens FROM bm25_stats",
                [],
                |row| Ok([row.get(0)?, row.get(1)?, row.get(2)?]),
            )
            .unwrap()
    }

    /// The statistics counted independently of the store: the `fts` rows
    /// and, per column, the token instances `fts5vocab` reads from the full-
    /// text index. Exact only for a table that has never seen a delete.
    fn vocab_stats(&self) -> [i64; 3] {
        self.conn
            .execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab USING fts5vocab(main, fts, 'col')",
            )
            .unwrap();
        self.conn
            .query_row(
                "SELECT (SELECT count(*) FROM fts_docsize), \
                 (SELECT total(cnt) FROM temp.vocab WHERE col = 'title'), \
                 (SELECT total(cnt) FROM temp.vocab WHERE col = 'body')",
                [],
                |row| {
                    Ok([
                        row.get(0)?,
                        row.get::<_, f64>(1)? as i64,
                        row.get::<_, f64>(2)? as i64,
                    ])
                },
            )
            .unwrap()
    }

    /// Every chunk matching `query`, in the production keyword order, as
    /// `(path, first line, title, score)`.
    fn keyword(&self, query: &str) -> Vec<(String, u32, String, f64)> {
        let scored = query::bm25_scored(&self.conn, query, &unrestricted(), 10_000).unwrap();
        let ids: Vec<_> = scored.iter().map(|&(id, _)| id).collect();
        let rows = query::chunks(&self.conn, &ids).unwrap();
        assert_eq!(rows.len(), scored.len());
        rows.into_iter()
            .zip(scored)
            .map(|(row, (_, score))| (row.path, *row.lines.start(), row.title, score))
            .collect()
    }

    /// `(rowid, built-in bm25, exact_bm25)` of every row matching `query`,
    /// with `weights` appended to both calls.
    fn both_scores(&self, query: &str, weights: &str) -> Vec<(i64, f64, f64)> {
        let sql = format!(
            "SELECT rowid, bm25(fts{weights}), exact_bm25(fts, {STATS_ARGS}{weights}) \
             FROM fts WHERE fts MATCH ?1 ORDER BY rowid"
        );
        self.conn
            .prepare(&sql)
            .unwrap()
            .query_map([query], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// The built-in `bm25()` of every chunk matching `query`, keyed as
    /// [`Fixture::keyword`], sorted by key.
    fn builtin(&self, query: &str) -> Vec<(String, u32, String, f64)> {
        let mut out: Vec<_> = self
            .conn
            .prepare(
                "SELECT c.path, c.start_line, c.title, bm25(fts) FROM fts \
                 JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?1",
            )
            .unwrap()
            .query_map([query], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        out.sort_by(|a, b| (&a.0, a.1, &a.2).cmp(&(&b.0, b.1, &b.2)));
        out
    }
}

fn unrestricted() -> ChunkFilter {
    ChunkFilter {
        side: Side::Code,
        include_ignored: true,
        paths: None,
        extensions: Vec::new(),
        kinds: Vec::new(),
        symbol: None,
    }
}

fn file_state(path: &str, src: &str) -> FileState {
    FileState {
        path: path.into(),
        size: src.len() as u64,
        mtime_ns: 1_700_000_000_000_000_000,
        inode: 7,
        content_hash: ContentHash::of_file(src.as_bytes()),
        ignored: false,
    }
}

fn records(path: &str, src: &str) -> Vec<ChunkRecord> {
    chunk_with(&mut Chunker::new(Rust), path, src, &CapOff)
        .unwrap()
        .into_iter()
        .map(|(record, _)| record)
        .collect()
}

fn replace(path: &str, src: &str) -> FileChange {
    FileChange::Replace(file_state(path, src), records(path, src))
}

/// The files every index of these tests ends with.
fn final_files() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("src/a.rs", SPLIT_SCOPES),
        ("src/b.rs", OVERSIZED_FN),
        ("src/c.rs", OVERSIZED_IMPL),
    ])
}

/// Brings `fx` to [`final_files`] through every write path that deletes
/// `fts` rows: replacing a file's chunks, removing a file, recording a file
/// as unchunkable, and re-chunking the files a second time.
fn edit_to_final(fx: &mut Fixture) {
    fx.apply(vec![
        replace("src/a.rs", OVERSIZED_IMPL),
        replace("src/b.rs", OVERSIZED_FN),
        replace("src/gone.rs", SPLIT_SCOPES),
        replace("src/broken.rs", OVERSIZED_FN),
    ]);
    fx.apply(vec![
        replace("src/a.rs", SPLIT_SCOPES),
        FileChange::Remove("src/gone.rs".into()),
        FileChange::Unchunkable(Unchunkable {
            state: file_state("src/broken.rs", OVERSIZED_FN),
            reason: UnchunkableReason::NotUtf8,
        }),
    ]);
    fx.apply(vec![
        replace("src/b.rs", OVERSIZED_FN),
        replace("src/c.rs", OVERSIZED_IMPL),
        FileChange::Remove("src/broken.rs".into()),
    ]);
}

/// On a table that has never seen a delete, `exact_bm25` over the stored
/// statistics scores every row as the built-in `bm25()` does, with and
/// without column weights: the formula is SQLite's.
#[test]
fn exact_bm25_scores_as_builtin_bm25_before_any_delete() {
    let fx = Fixture::fresh(&final_files());
    assert_eq!(fx.stats(), fx.vocab_stats());
    for query in QUERIES {
        for weights in ["", ", 1.0", ", 2.5, 0.5", ", 0.0, 3.0"] {
            let rows = fx.both_scores(query, weights);
            assert!(!rows.is_empty(), "{query} matches nothing");
            for (rowid, builtin, exact) in rows {
                assert!(
                    (builtin - exact).abs() <= 1e-12 * builtin.abs().max(1.0),
                    "{query}{weights}: row {rowid} built-in {builtin} exact {exact}"
                );
            }
        }
    }
}

/// The issue's done-when: an index brought to the final files through
/// replacements and removals ranks every keyword query exactly as a fresh
/// build of those files, while the built-in `bm25()` of the same maintained
/// table has drifted.
#[test]
fn maintained_index_ranks_as_a_fresh_build() {
    let mut maintained = Fixture::new();
    edit_to_final(&mut maintained);
    let fresh = Fixture::fresh(&final_files());
    assert_eq!(maintained.stats(), fresh.stats());

    let mut drifted = false;
    for query in QUERIES {
        let got = maintained.keyword(query);
        let want = fresh.keyword(query);
        assert!(!want.is_empty(), "{query} matches nothing");
        // Scores in rank order are identical; equal scores may sit in
        // either index's id order, so the keyed results compare as sets.
        let scores = |rows: &[(String, u32, String, f64)]| -> Vec<u64> {
            rows.iter().map(|r| r.3.to_bits()).collect()
        };
        assert_eq!(scores(&got), scores(&want), "{query}: scores in rank order");
        let keyed = |mut rows: Vec<(String, u32, String, f64)>| {
            rows.sort_by(|a, b| (&a.0, a.1, &a.2).cmp(&(&b.0, b.1, &b.2)));
            rows.into_iter()
                .map(|(path, line, title, score)| (path, line, title, score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(keyed(got), keyed(want), "{query}: chunks and their scores");
        drifted |= maintained.builtin(query) != fresh.builtin(query);
    }
    assert!(
        drifted,
        "the built-in bm25() of the maintained table should have drifted"
    );
}

/// The statistics equal those of a fresh build of the same files after every
/// write path: insert, replace, remove, unchunkable, and the chunker
/// realign's `delete-all` followed by a full re-index.
#[test]
fn stats_stay_exact_through_every_fts_write() {
    let check = |fx: &Fixture, files: &BTreeMap<&str, &str>, step: &str| {
        let fresh = Fixture::fresh(files);
        assert_eq!(fresh.stats(), fresh.vocab_stats(), "{step}: fresh build");
        assert_eq!(fx.stats(), fresh.stats(), "{step}");
    };
    let mut fx = Fixture::new();
    let mut files = BTreeMap::new();
    check(&fx, &files, "empty");

    fx.apply(vec![
        replace("src/a.rs", OVERSIZED_IMPL),
        replace("src/b.rs", OVERSIZED_FN),
    ]);
    files.extend([("src/a.rs", OVERSIZED_IMPL), ("src/b.rs", OVERSIZED_FN)]);
    check(&fx, &files, "insert");

    fx.apply(vec![replace("src/a.rs", SPLIT_SCOPES)]);
    files.insert("src/a.rs", SPLIT_SCOPES);
    check(&fx, &files, "replace");

    fx.apply(vec![FileChange::Remove("src/b.rs".into())]);
    files.remove("src/b.rs");
    check(&fx, &files, "remove");

    fx.apply(vec![FileChange::Unchunkable(Unchunkable {
        state: file_state("src/a.rs", SPLIT_SCOPES),
        reason: UnchunkableReason::NotUtf8,
    })]);
    files.remove("src/a.rs");
    check(&fx, &files, "unchunkable");

    fx.apply(vec![
        replace("src/c.rs", OVERSIZED_IMPL),
        replace("src/a.rs", SPLIT_SCOPES),
    ]);
    files.extend([("src/c.rs", OVERSIZED_IMPL), ("src/a.rs", SPLIT_SCOPES)]);
    check(&fx, &files, "re-add");

    assert!(align(&mut fx.conn, "another-chunker", MODEL).unwrap());
    check(&fx, &BTreeMap::new(), "realign");

    fx.apply(
        final_files()
            .iter()
            .map(|(path, src)| replace(path, src))
            .collect(),
    );
    check(&fx, &final_files(), "re-index after realign");
}

/// The SQLite varint decoder reads the one- to nine-byte forms.
#[test]
fn sqlite_varints_decode() {
    use super::sqlite_varint;
    assert_eq!(sqlite_varint(&[0x00]), Some((0, 1)));
    assert_eq!(sqlite_varint(&[0x7f, 0xff]), Some((127, 1)));
    assert_eq!(sqlite_varint(&[0x81, 0x00]), Some((128, 2)));
    assert_eq!(sqlite_varint(&[0xff, 0x7f]), Some((16_383, 2)));
    assert_eq!(sqlite_varint(&[0x81, 0x80, 0x00]), Some((16_384, 3)));
    assert_eq!(sqlite_varint(&[0xff; 9]), Some((u64::MAX, 9)));
    assert_eq!(sqlite_varint(&[0x81]), None);
    assert_eq!(sqlite_varint(&[]), None);
}
