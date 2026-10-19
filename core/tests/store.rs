use std::collections::HashMap;
use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use futures_executor::block_on;
use repo_search_core::UnitVector;
use repo_search_core::store::{
    self, ChunkRecord, ContentHash, FileChange, FileState, Index, MemberRecord, VectorCache,
    fts_body, split_identifiers,
};

fn state(path: &str, content: &[u8]) -> FileState {
    FileState {
        path: path.into(),
        size: content.len() as u64,
        mtime_ns: 1_700_000_000_000_000_000,
        inode: 42,
        content_hash: ContentHash::of_file(content),
    }
}

fn chunk(lines: RangeInclusive<u32>, title: &str, kind: &str, text: &str) -> ChunkRecord {
    ChunkRecord {
        lines,
        title: title.into(),
        kind: kind.into(),
        members: vec![MemberRecord {
            bytes: 0..text.len() as u64,
            kind: kind.into(),
            symbol: title.into(),
        }],
        content_hash: ContentHash::of_chunk(title, text),
        tokens: 10,
        fts_body: fts_body(text),
    }
}

fn index_db(worktree: &Path) -> PathBuf {
    worktree.join(".repo-search/index.db")
}

/// FTS rowids matching `query`, read through a second connection to check what
/// the store actually committed.
fn fts_rowids(db: &Path, query: &str) -> Vec<i64> {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.prepare("SELECT rowid FROM fts WHERE fts MATCH ?1 ORDER BY rowid")
        .unwrap()
        .query_map([query], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// `(chunks, chunk_members)` row counts stored for `path`, read through a
/// second connection.
fn row_counts(db: &Path, path: &str) -> (i64, i64) {
    let conn = rusqlite::Connection::open(db).unwrap();
    let chunks: i64 = conn
        .query_row(
            "SELECT count(*) FROM chunks WHERE path = ?1",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    let members: i64 = conn
        .query_row(
            "SELECT count(*) FROM chunk_members \
             WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?1)",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    (chunks, members)
}

#[test]
fn fts5_is_compiled_into_the_bundled_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(x)")
        .unwrap();
}

#[test]
fn split_identifiers_matches_the_python_reference() {
    let cases = [
        ("ComputedSignal", "Computed Signal"),
        ("HTTPServer", "HTTP Server"),
        ("parseURL2Value", "parse URL2 Value"),
        ("snake_case", ""),
        (
            "fn HTTPRequest2xx parseHTML a I x9 ABc",
            "HTTP Request2xx parse HTML A Bc",
        ),
        ("already split words", ""),
        ("camel2Case3Word", "camel2 Case3 Word"),
        ("URL", ""),
    ];
    for (input, want) in cases {
        assert_eq!(split_identifiers(input), want, "input: {input:?}");
    }
}

#[test]
fn chunk_hash_depends_on_the_title() {
    let text = "fn f() {}";
    assert_ne!(
        ContentHash::of_chunk("fn f", text),
        ContentHash::of_chunk("fn g", text)
    );
    assert_ne!(
        ContentHash::of_file(b"abc"),
        ContentHash::of_chunk("abc", "abc")
    );
}

#[test]
fn fresh_index_reports_empty_and_creates_gitignore() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    assert!(block_on(index.file_states()).unwrap().is_empty());
    assert_eq!(
        fs::read_to_string(dir.path().join(".repo-search/.gitignore")).unwrap(),
        "*\n"
    );
}

#[test]
fn replace_and_remove_keep_fts_and_rows_in_sync() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    let db = index_db(dir.path());

    let v1 = chunk(
        1..=3,
        "fn alpha",
        "fn",
        "fn alpha() { ComputedSignal::new() }",
    );
    block_on(index.replace_file(state("src/a.rs", b"v1"), vec![v1])).unwrap();
    let states = block_on(index.file_states()).unwrap();
    assert_eq!(states, vec![state("src/a.rs", b"v1")]);
    assert!(!fts_rowids(&db, "computed").is_empty());
    assert_eq!(row_counts(&db, "src/a.rs"), (1, 1));

    let v2 = chunk(1..=3, "fn alpha", "fn", "fn alpha() { beta() }");
    block_on(index.replace_file(state("src/a.rs", b"v2"), vec![v2])).unwrap();
    assert!(fts_rowids(&db, "computed").is_empty());
    assert!(!fts_rowids(&db, "beta").is_empty());
    assert_eq!(row_counts(&db, "src/a.rs"), (1, 1));
    assert_eq!(
        block_on(index.file_states()).unwrap(),
        vec![state("src/a.rs", b"v2")]
    );

    block_on(index.remove_file("src/a.rs")).unwrap();
    assert!(fts_rowids(&db, "beta").is_empty());
    assert_eq!(row_counts(&db, "src/a.rs"), (0, 0));
    assert!(block_on(index.file_states()).unwrap().is_empty());
}

#[test]
fn camel_case_identifiers_are_searchable() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    let db = index_db(dir.path());

    let text = "impl HTTPServer { fn parseURL2Value(&self) {} }";
    let c = chunk(10..=14, "impl HTTPServer", "impl", text);
    block_on(index.replace_file(state("src/http.rs", text.as_bytes()), vec![c])).unwrap();

    for term in ["http", "server", "parse", "value"] {
        assert!(
            !fts_rowids(&db, term).is_empty(),
            "term {term} should match"
        );
    }
    assert!(fts_rowids(&db, "websocket").is_empty());
}

#[test]
fn a_batch_applies_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    let db = index_db(dir.path());

    let good = FileChange::Replace(
        state("src/a.rs", b"a"),
        vec![chunk(1..=1, "fn a", "fn", "fn a() {}")],
    );
    let mut bad_state = state("src/b.rs", b"b");
    bad_state.size = u64::MAX; // outside sqlite's i64 range: the change must fail
    let err =
        block_on(index.apply(vec![good, FileChange::Replace(bad_state, vec![])])).unwrap_err();
    assert!(matches!(err, store::Error::IntRange(_)), "got {err:?}");
    assert!(block_on(index.file_states()).unwrap().is_empty());
    assert_eq!(row_counts(&db, "src/a.rs"), (0, 0));
    assert_eq!(row_counts(&db, "src/b.rs"), (0, 0));
}

#[test]
fn vector_cache_is_shared_between_worktrees() {
    let worktree_a = tempfile::tempdir().unwrap();
    let worktree_b = tempfile::tempdir().unwrap();
    let common = tempfile::tempdir().unwrap();
    let index_a = block_on(Index::open(worktree_a.path())).unwrap();
    let index_b = block_on(Index::open(worktree_b.path())).unwrap();
    let cache_a = block_on(VectorCache::open(common.path())).unwrap();
    let cache_b = block_on(VectorCache::open(common.path())).unwrap();

    // Both worktrees index a file holding the same chunk; one content hash
    // covers both.
    let text = "fn shared() { ComputedSignal::new() }";
    let chunk_a = chunk(1..=3, "fn shared", "fn", text);
    let chunk_b = chunk_a.clone();
    let hash = chunk_a.content_hash;
    block_on(index_a.replace_file(state("src/a.rs", b"a"), vec![chunk_a])).unwrap();
    block_on(index_b.replace_file(state("src/b.rs", b"b"), vec![chunk_b])).unwrap();

    let vector = UnitVector::normalize(vec![1.0, -2.0, 3.0]);
    block_on(cache_a.put_many("embedgemma2-coreml", vec![(hash, vector)])).unwrap();

    // Worktree B's daemon — a second connection — sees the hit through WAL.
    let hits = block_on(cache_b.get_many("embedgemma2-coreml", &[hash])).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content_hash, hash);

    // Other model ids and other hashes miss.
    assert!(
        block_on(cache_b.get_many("other-model", &[hash]))
            .unwrap()
            .is_empty()
    );
    assert!(
        block_on(cache_b.get_many(
            "embedgemma2-coreml",
            &[ContentHash::of_chunk("fn b", "fn b() {}")]
        ))
        .unwrap()
        .is_empty()
    );
}

#[test]
fn scan_pages_interleaved_with_writes_return_every_vector_once() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();

    let vector = |seed: usize| UnitVector::normalize(vec![seed as f32, 1.0, -1.0]);
    let hash_of = |i: usize| ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}"));

    let initial: Vec<(ContentHash, UnitVector)> =
        (0..100).map(|i| (hash_of(i), vector(i))).collect();
    let initial_hashes: Vec<ContentHash> = initial.iter().map(|(h, _)| *h).collect();
    block_on(cache.put_many("m", initial)).unwrap();

    let mut seen: HashMap<ContentHash, usize> = HashMap::new();
    let mut cursor = None;
    let mut page_no = 0;
    loop {
        let page = block_on(cache.scan("m", cursor, 20)).unwrap();
        let done = page.len() < 20;
        if let Some(last) = page.last() {
            cursor = Some(last.content_hash);
        }
        for v in page {
            *seen.entry(v.content_hash).or_default() += 1;
        }
        page_no += 1;
        // Writes interleave with paging: new keys and re-puts of keys the
        // cursor already passed both land.
        if page_no <= 3 {
            let extra: Vec<(ContentHash, UnitVector)> = (0..10)
                .map(|i| {
                    (
                        ContentHash::of_chunk("fn w", &format!("fn w{page_no}x{i}() {{}}")),
                        vector(i),
                    )
                })
                .collect();
            block_on(cache.put_many("m", extra)).unwrap();
            block_on(cache.put_many("m", vec![(initial_hashes[page_no], vector(999))])).unwrap();
        }
        if done {
            break;
        }
    }

    // Every original vector exactly once; keys inserted mid-scan appear at
    // most once (a key below the cursor is legitimately skipped).
    for hash in &initial_hashes {
        assert_eq!(seen.get(hash), Some(&1), "hash {hash} seen wrong count");
    }
    assert!(seen.values().all(|&n| n == 1));
}

#[test]
fn cache_runs_in_wal_mode() {
    let common = tempfile::tempdir().unwrap();
    let _cache = block_on(VectorCache::open(common.path())).unwrap();
    let db = common.path().join("repo-search/vectors.v1.db");
    assert!(db.exists());
    let conn = rusqlite::Connection::open(&db).unwrap();
    let mode: String = conn
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
}

#[test]
fn vectors_round_trip_at_fp16_precision() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();

    let vector = UnitVector::normalize(vec![1.0, -2.0, 3.0, 0.5]);
    let hash = ContentHash::of_chunk("fn v", "fn v() {}");
    block_on(cache.put_many("m", vec![(hash, vector.clone())])).unwrap();

    let hits = block_on(cache.get_many("m", &[hash])).unwrap();
    assert_eq!(hits[0].fp16_bytes().len(), vector.as_slice().len() * 2);
    for (got, want) in hits[0].to_f32().iter().zip(vector.as_slice()) {
        // fp16 normals carry at most 2^-11 relative error.
        assert!(
            (got - want).abs() <= want.abs() * 0.001 + 1e-7,
            "fp16 round trip: got {got}, want {want}"
        );
    }
}

#[test]
fn a_version_mismatch_rebuilds_the_index_and_removes_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    block_on(index.replace_file(
        state("src/a.rs", b"a"),
        vec![chunk(1..=1, "fn a", "fn", "fn a() {}")],
    ))
    .unwrap();
    assert_eq!(block_on(index.file_states()).unwrap().len(), 1);
    drop(index);

    let db = index_db(dir.path());
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.pragma_update(None, "user_version", 99_i64).unwrap();
    drop(conn);
    // Stale sidecars a crashed journal mode could have left behind.
    for suffix in ["-wal", "-shm", "-journal"] {
        fs::write(db.with_file_name(format!("index.db{suffix}")), b"stale").unwrap();
    }

    let index = block_on(Index::open(dir.path())).unwrap();
    assert!(block_on(index.file_states()).unwrap().is_empty());
    for suffix in ["-wal", "-shm", "-journal"] {
        assert!(
            !db.with_file_name(format!("index.db{suffix}")).exists(),
            "stale index.db{suffix} survived the rebuild"
        );
    }
}

#[test]
fn a_foreign_version_on_the_shared_cache_is_rejected() {
    let common = tempfile::tempdir().unwrap();
    let dir = common.path().join("repo-search");
    fs::create_dir_all(&dir).unwrap();
    let db = dir.join("vectors.v1.db");
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute_batch("CREATE TABLE not_ours (x)").unwrap();
    conn.pragma_update(None, "user_version", 42_i64).unwrap();
    drop(conn);

    let err = block_on(VectorCache::open(common.path())).err().unwrap();
    match err {
        store::Error::BadSchemaVersion { path, found, want } => {
            assert_eq!(path, db);
            assert_eq!((found, want), (42, 1));
        }
        other => panic!("expected BadSchemaVersion, got {other:?}"),
    }
}
