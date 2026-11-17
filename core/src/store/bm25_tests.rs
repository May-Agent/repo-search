//! The built-in `bm25()` of an index maintained through the indexer's
//! writes against a fresh build of the same rows, over the chunks of the
//! crate's Rust fixtures.

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
}

fn unrestricted() -> ChunkFilter {
    ChunkFilter {
        side: Side::Code,
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

/// An index brought to the final files through replacements and removals
/// ranks every keyword query, through the production query, exactly as a
/// fresh build of those files.
#[test]
fn maintained_index_ranks_as_a_fresh_build() {
    let mut maintained = Fixture::new();
    edit_to_final(&mut maintained);
    let fresh = Fixture::fresh(&final_files());

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
    }
}

/// A chunk's identity independent of its rowid: path, first line, title.
type ChunkKey = (String, u32, String);

/// Every chunk matching `query` with the built-in `bm25(fts{weights})` score,
/// read straight from `fts`, in rank order.
fn builtin_ranking(fx: &Fixture, query: &str, weights: &str) -> Vec<(ChunkKey, f64)> {
    let sql = format!(
        "SELECT c.path, c.start_line, c.title, bm25(fts{weights}) AS score \
         FROM fts JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?1 \
         ORDER BY score, c.path, c.start_line, c.title"
    );
    fx.conn
        .prepare(&sql)
        .unwrap()
        .query_map([query], |row| {
            Ok(((row.get(0)?, row.get(1)?, row.get(2)?), row.get(3)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// The built-in `bm25()` of `maintained` ranks every query as that of `fresh`:
/// the same chunks, each with a score equal within f64 tolerance, in an order
/// whose score sequence is equal within the same tolerance.
fn assert_ranks_alike(maintained: &Fixture, fresh: &Fixture, step: &str) {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
    for query in QUERIES {
        for weights in ["", ", 1.0", ", 2.5, 0.5", ", 0.0, 3.0"] {
            let got = builtin_ranking(maintained, query, weights);
            let want = builtin_ranking(fresh, query, weights);
            assert_eq!(got.len(), want.len(), "{step}: {query}{weights}: matches");
            for ((_, g), (_, w)) in got.iter().zip(&want) {
                assert!(
                    close(*g, *w),
                    "{step}: {query}{weights}: rank-order score {g} vs {w}"
                );
            }
            let keyed = |rows: Vec<(ChunkKey, f64)>| -> BTreeMap<ChunkKey, f64> {
                rows.into_iter().collect()
            };
            let (got, want) = (keyed(got), keyed(want));
            assert_eq!(
                got.keys().collect::<Vec<_>>(),
                want.keys().collect::<Vec<_>>(),
                "{step}: {query}{weights}: matching chunks"
            );
            for (key, g) in &got {
                assert!(
                    close(*g, want[key]),
                    "{step}: {query}{weights}: {key:?} scores {g} vs {}",
                    want[key]
                );
            }
        }
    }
}

/// On the bundled SQLite, an index maintained through the indexer's writes
/// ranks with the built-in `bm25()` as a fresh build of its final rows.
#[test]
fn builtin_bm25_of_a_maintained_index_ranks_as_a_fresh_build() {
    let mut maintained = Fixture::new();
    edit_to_final(&mut maintained);
    let fresh = Fixture::fresh(&final_files());
    for query in QUERIES {
        assert!(
            !builtin_ranking(&fresh, query, "").is_empty(),
            "{query} matches nothing"
        );
    }
    assert_ranks_alike(&maintained, &fresh, "final");
}

/// The same after every write path: insert, replace, remove, unchunkable, and
/// the chunker realign's `delete-all` followed by a full re-index.
#[test]
fn builtin_bm25_ranks_as_a_fresh_build_after_every_fts_write() {
    let check = |fx: &Fixture, files: &BTreeMap<&str, &str>, step: &str| {
        assert_ranks_alike(fx, &Fixture::fresh(files), step);
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
