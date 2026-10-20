use numkong::f16;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use futures_executor::block_on;
use repo_search_core::UnitVector;
use repo_search_core::store::{
    self, ChunkRecord, ContentHash, FileChange, FileState, Index, MemberRecord, StoredCalibration,
    VectorCache, fts_body, split_identifiers,
};

fn state(path: &str, content: &[u8]) -> FileState {
    FileState {
        path: path.into(),
        size: content.len() as u64,
        mtime_ns: 1_700_000_000_000_000_000,
        inode: 42,
        content_hash: ContentHash::of_file(content),
        ignored: false,
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
            cut: 0,
            symbol: title.into(),
        }],
        content_hash: ContentHash::of_chunk(title, text),
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
fn cached_reports_membership_without_the_fp16_file() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let hit = ContentHash::of_chunk("fn a", "fn a() {}");
    let miss = ContentHash::of_chunk("fn b", "fn b() {}");
    block_on(cache.put_many("m", vec![(hit, UnitVector::normalize(vec![1.0, 2.0]))])).unwrap();

    // Removing the fp16 file proves the query reads only the SQLite index.
    for entry in fs::read_dir(common.path().join("repo-search")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "f16") {
            fs::remove_file(path).unwrap();
        }
    }
    let found = block_on(cache.cached("m", &[hit, miss, hit])).unwrap();
    assert_eq!(found, HashSet::from([hit]));
    assert!(block_on(cache.cached("other", &[hit])).unwrap().is_empty());
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

fn fp16_of(vector: &UnitVector) -> Vec<u8> {
    vector
        .as_slice()
        .iter()
        .flat_map(|&x| f16::from_f32(x).0.to_le_bytes())
        .collect()
}

#[test]
fn vectors_live_in_a_flat_fp16_file_at_slot_offsets() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    assert!(block_on(cache.model("m")).unwrap().is_none());

    let vectors: Vec<(ContentHash, UnitVector)> = (0..5)
        .map(|i| {
            (
                ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}")),
                UnitVector::normalize(vec![i as f32 + 1.0, -1.0, 0.5, 2.0]),
            )
        })
        .collect();
    block_on(cache.put_many("m", vectors.clone())).unwrap();

    let info = block_on(cache.model("m")).unwrap().unwrap();
    assert_eq!((info.dim, info.vectors), (4, 5));
    let name = info.file.file_name().unwrap().to_str().unwrap();
    let id_hash = name
        .strip_prefix("vectors.v1.")
        .and_then(|n| n.strip_suffix(".f16"))
        .unwrap();
    assert_eq!(id_hash.len(), 16);
    assert!(id_hash.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(
        info.file.parent().unwrap(),
        common.path().join("repo-search")
    );

    let bytes = fs::read(&info.file).unwrap();
    assert_eq!(bytes.len(), 5 * 4 * 2);
    let hashes: Vec<ContentHash> = vectors.iter().map(|(h, _)| *h).collect();
    for hit in block_on(cache.get_many("m", &hashes)).unwrap() {
        let (_, vector) = vectors
            .iter()
            .find(|(h, _)| *h == hit.content_hash)
            .unwrap();
        let at = hit.slot as usize * 8;
        assert_eq!(&bytes[at..at + 8], fp16_of(vector));
        assert_eq!(hit.fp16_bytes(), fp16_of(vector));
    }

    // Slots are dense and the scan reports them.
    let mut slots: Vec<u64> = block_on(cache.scan("m", None, 100))
        .unwrap()
        .iter()
        .map(|s| s.slot)
        .collect();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1, 2, 3, 4]);

    // An id that is not a file name is safe, and its file is its own.
    block_on(cache.put_many("org/model:1 ../x", vectors[..1].to_vec())).unwrap();
    let other = block_on(cache.model("org/model:1 ../x")).unwrap().unwrap();
    assert_ne!(other.file, info.file);
    assert_eq!(other.file.parent(), info.file.parent());
}

#[test]
fn a_cached_hash_keeps_its_slot_and_vector() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let hash = ContentHash::of_chunk("fn v", "fn v() {}");
    let first = UnitVector::normalize(vec![1.0, 2.0]);
    block_on(cache.put_many("m", vec![(hash, first.clone())])).unwrap();
    block_on(cache.put_many("m", vec![(hash, UnitVector::normalize(vec![-3.0, 1.0]))])).unwrap();

    let hits = block_on(cache.get_many("m", &[hash])).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].slot, 0);
    assert_eq!(hits[0].fp16_bytes(), fp16_of(&first));
    assert_eq!(block_on(cache.model("m")).unwrap().unwrap().vectors, 1);
}

#[test]
fn a_vector_of_another_dimension_rolls_the_batch_back() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let ok = ContentHash::of_chunk("fn a", "fn a() {}");
    block_on(cache.put_many("m", vec![(ok, UnitVector::normalize(vec![1.0, 2.0]))])).unwrap();

    let fresh = ContentHash::of_chunk("fn b", "fn b() {}");
    let bad = ContentHash::of_chunk("fn c", "fn c() {}");
    let err = block_on(cache.put_many(
        "m",
        vec![
            (fresh, UnitVector::normalize(vec![1.0, 1.0])),
            (bad, UnitVector::normalize(vec![1.0, 1.0, 1.0])),
        ],
    ))
    .unwrap_err();
    assert!(
        matches!(
            err,
            store::Error::DimMismatch {
                expected: 2,
                got: 3,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        block_on(cache.get_many("m", &[ok, fresh, bad]))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(block_on(cache.model("m")).unwrap().unwrap().vectors, 1);

    // The slot the failed batch would have taken goes to the next writer.
    block_on(cache.put_many("m", vec![(fresh, UnitVector::normalize(vec![1.0, 1.0]))])).unwrap();
    let hits = block_on(cache.get_many("m", &[fresh])).unwrap();
    assert_eq!(hits[0].slot, 1);
}

#[test]
fn concurrent_writers_never_overlap_slots() {
    let common = tempfile::tempdir().unwrap();
    let a = block_on(VectorCache::open(common.path())).unwrap();
    let b = block_on(VectorCache::open(common.path())).unwrap();
    let vector_of =
        |i: usize| UnitVector::normalize(vec![i as f32 + 1.0, (i % 7) as f32 - 3.0, 0.25]);
    let hash_of = |i: usize| ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}"));

    // Writer A owns evens, B odds, and both race on the shared 0..40.
    std::thread::scope(|scope| {
        for (cache, parity) in [(&a, 0), (&b, 1)] {
            scope.spawn(move || {
                for chunk in (0..400)
                    .filter(|i| i % 2 == parity)
                    .collect::<Vec<_>>()
                    .chunks(10)
                {
                    let batch = chunk.iter().map(|&i| (hash_of(i), vector_of(i))).collect();
                    block_on(cache.put_many("m", batch)).unwrap();
                    let shared = (0..40).map(|i| (hash_of(i), vector_of(i))).collect();
                    block_on(cache.put_many("m", shared)).unwrap();
                }
            });
        }
    });

    let info = block_on(a.model("m")).unwrap().unwrap();
    assert_eq!(info.vectors, 400);
    assert_eq!(fs::metadata(&info.file).unwrap().len(), 400 * 3 * 2);
    let hashes: Vec<ContentHash> = (0..400).map(hash_of).collect();
    let hits = block_on(b.get_many("m", &hashes)).unwrap();
    assert_eq!(hits.len(), 400);
    let mut slots = HashSet::new();
    for hit in hits {
        let i = (0..400).find(|&i| hash_of(i) == hit.content_hash).unwrap();
        assert_eq!(hit.fp16_bytes(), fp16_of(&vector_of(i)), "row {i}");
        assert!(slots.insert(hit.slot), "slot {} allocated twice", hit.slot);
    }
}

#[test]
fn calibration_is_stored_per_model() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let hash = ContentHash::of_chunk("fn v", "fn v() {}");
    let calibration = || StoredCalibration {
        lo: vec![-1.0, 0.5].into(),
        hi: vec![1.0, 0.75].into(),
        count: 12,
    };

    let err = block_on(cache.set_calibration("m", calibration())).unwrap_err();
    assert!(matches!(err, store::Error::UnknownModel(_)), "{err:?}");

    block_on(cache.put_many("m", vec![(hash, UnitVector::normalize(vec![1.0, 2.0]))])).unwrap();
    assert!(block_on(cache.calibration("m")).unwrap().is_none());
    block_on(cache.set_calibration("m", calibration())).unwrap();
    assert_eq!(
        block_on(cache.calibration("m")).unwrap(),
        Some(calibration())
    );

    let short = StoredCalibration {
        lo: vec![0.0].into(),
        hi: vec![1.0].into(),
        count: 1,
    };
    let err = block_on(cache.set_calibration("m", short)).unwrap_err();
    assert!(
        matches!(
            err,
            store::Error::DimMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(block_on(cache.calibration("other")).unwrap().is_none());
}

#[test]
fn bounds_cover_every_committed_vector_at_fp16_precision() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();

    let err = block_on(cache.bounds("m")).unwrap_err();
    assert!(matches!(err, store::Error::UnknownModel(_)), "{err:?}");

    let vectors = [
        vec![3.0, -1.0, 2.0],
        vec![-2.0, 4.0, 0.5],
        vec![1.0, 1.0, -3.0],
    ];
    let batch: Vec<(ContentHash, UnitVector)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            (
                ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}")),
                UnitVector::normalize(v.clone()),
            )
        })
        .collect();
    let stored: Vec<Vec<f32>> = batch
        .iter()
        .map(|(_, v)| {
            v.as_slice()
                .iter()
                .map(|x| f16::from_f32(*x).to_f32())
                .collect()
        })
        .collect();
    block_on(cache.put_many("m", batch)).unwrap();

    let bounds = block_on(cache.bounds("m")).unwrap();
    assert_eq!(bounds.count, 3);
    for d in 0..3 {
        let column = stored.iter().map(|v| v[d]);
        assert_eq!(bounds.lo[d], column.clone().fold(f32::INFINITY, f32::min));
        assert_eq!(bounds.hi[d], column.fold(f32::NEG_INFINITY, f32::max));
    }
}

#[test]
fn a_calibration_over_no_vectors_is_rejected() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let hash = ContentHash::of_chunk("fn v", "fn v() {}");
    block_on(cache.put_many("m", vec![(hash, UnitVector::normalize(vec![1.0, 2.0]))])).unwrap();
    let err = block_on(cache.set_calibration("m", StoredCalibration::empty(2))).unwrap_err();
    assert!(matches!(err, store::Error::EmptyCalibration(_)), "{err:?}");
    assert!(block_on(cache.calibration("m")).unwrap().is_none());
}

#[test]
fn restamping_an_unstored_path_is_a_named_error() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    let err = block_on(index.apply(vec![FileChange::Restamp(state("nope.rs", b"x"))]))
        .err()
        .unwrap();
    assert!(matches!(err, store::Error::NotStored(p) if p == "nope.rs"));
}

#[test]
fn chunks_by_hash_returns_the_first_row_with_member_ranges() {
    let dir = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(dir.path())).unwrap();
    let text = "fn a() {}";
    block_on(index.replace_file(
        state("src/a.rs", b"a"),
        vec![chunk(3..=3, "fn a", "fn", text)],
    ))
    .unwrap();
    let hash = ContentHash::of_chunk("fn a", text);
    let missing = ContentHash::of_chunk("other", "x");
    let found = block_on(index.chunks_by_hash(vec![missing, hash])).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, "src/a.rs");
    assert_eq!(found[0].title, "fn a");
    assert_eq!(found[0].lines, 3..=3);
    assert_eq!(
        found[0].members,
        vec![repo_search_core::store::StoredMember {
            bytes: 0..text.len() as u64,
            kind: repo_search_core::store::StoredMemberKind::Text,
        }]
    );
}

#[test]
fn since_lists_the_vectors_stored_from_a_slot_on() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let batch = |range: std::ops::Range<usize>| -> Vec<(ContentHash, UnitVector)> {
        range
            .map(|i| {
                (
                    ContentHash::of_chunk("t", &i.to_string()),
                    UnitVector::normalize(vec![1.0, i as f32 + 1.0]),
                )
            })
            .collect()
    };
    block_on(cache.put_many("m", batch(0..3))).unwrap();
    let info = block_on(cache.model("m")).unwrap().unwrap();
    assert_eq!((info.vectors, info.next_slot), (3, 3));
    assert!(
        block_on(cache.since("m", info.next_slot))
            .unwrap()
            .is_empty()
    );

    block_on(cache.put_many("m", batch(2..6))).unwrap();
    let after = block_on(cache.model("m")).unwrap().unwrap();
    assert_eq!((after.vectors, after.next_slot), (6, 6));
    let new = block_on(cache.since("m", info.next_slot)).unwrap();
    assert_eq!(
        new.iter().map(|s| s.slot).collect::<Vec<_>>(),
        [3, 4, 5],
        "slots in order"
    );
    assert_eq!(
        new.iter().map(|s| s.content_hash).collect::<Vec<_>>(),
        batch(3..6).iter().map(|(h, _)| *h).collect::<Vec<_>>()
    );
    assert!(block_on(cache.since("other", 0)).unwrap().is_empty());
}

#[test]
fn chunk_queries_bind_the_doc_kind_and_expose_filters() {
    use repo_search_core::chunk::ChunkKind;
    use repo_search_core::store::{ChunkFilter, Side};

    let tree = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(tree.path())).unwrap();
    let doc_kind = ChunkKind::Doc.static_name().unwrap();
    let files = [
        ("a.rs", "function_item", "alpha beta"),
        ("b.rs", "struct_item", "alpha gamma"),
        ("c.md", doc_kind, "alpha docs"),
    ];
    for (path, kind, text) in files {
        let record = chunk(1..=3, &format!("{path} › {text}"), kind, text);
        block_on(index.replace_file(state(path, text.as_bytes()), vec![record])).unwrap();
    }
    let filter = |side, paths: Option<Vec<&str>>, kinds: Vec<&str>| ChunkFilter {
        include_ignored: true,
        side,
        paths: paths.map(|p| p.into_iter().map(str::to_owned).collect()),
        extensions: Vec::new(),
        kinds: kinds.into_iter().map(str::to_owned).collect(),
        symbol: None,
    };
    assert!(filter(Side::Code, None, vec![]).is_unrestricted());
    assert!(!filter(Side::Code, Some(vec![]), vec![]).is_unrestricted());

    // The sides split on the reserved kind.
    assert_eq!(block_on(index.count(Side::Code, true)).unwrap(), 2);
    assert_eq!(block_on(index.count(Side::Docs, true)).unwrap(), 1);
    let sides = block_on(index.hash_sides()).unwrap();
    assert_eq!(sides.iter().filter(|(_, s)| *s == Side::Docs).count(), 1);

    let hash_of = |path: &str, text: &str| ContentHash::of_chunk(&format!("{path} › {text}"), text);
    let code = block_on(index.hashes_matching(filter(Side::Code, None, vec![]))).unwrap();
    assert_eq!(
        code.into_iter().collect::<HashSet<_>>(),
        HashSet::from([
            hash_of("a.rs", "alpha beta"),
            hash_of("b.rs", "alpha gamma")
        ])
    );
    let narrowed =
        block_on(index.hashes_matching(filter(Side::Code, Some(vec!["b.rs"]), vec![]))).unwrap();
    assert_eq!(narrowed, [hash_of("b.rs", "alpha gamma")]);
    let by_kind =
        block_on(index.hashes_matching(filter(Side::Code, None, vec!["function_item"]))).unwrap();
    assert_eq!(by_kind, [hash_of("a.rs", "alpha beta")]);
    let docs = block_on(index.hashes_matching(filter(Side::Docs, None, vec![]))).unwrap();
    assert_eq!(docs, [hash_of("c.md", "alpha docs")]);

    // Kinds the index holds, for `--kind` validation.
    assert_eq!(
        block_on(index.kinds()).unwrap(),
        ["doc", "function_item", "struct_item"]
    );

    // Which of some chunks spell a phrase.
    let all =
        block_on(index.bm25("\"alpha\"".to_owned(), filter(Side::Code, None, vec![]), 10)).unwrap();
    assert_eq!(all.len(), 2);
    let phrase = block_on(index.matching("\"alpha gamma\"".to_owned(), all.clone())).unwrap();
    assert_eq!(phrase.len(), 1);
    assert!(
        block_on(index.matching("\"alpha gamma\"".to_owned(), vec![]))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn the_extension_filter_is_ascii_case_insensitive() {
    use repo_search_core::chunk::ChunkKind;
    use repo_search_core::store::{ChunkFilter, Side};

    let tree = tempfile::tempdir().unwrap();
    let index = block_on(Index::open(tree.path())).unwrap();
    let doc_kind = ChunkKind::Doc.static_name().unwrap();
    for (path, kind, text) in [
        ("README.MD", doc_kind, "alpha docs"),
        ("a.rs", "function_item", "alpha code"),
    ] {
        let record = chunk(1..=3, &format!("{path} › {text}"), kind, text);
        block_on(index.replace_file(state(path, text.as_bytes()), vec![record])).unwrap();
    }
    // `--lang markdown` carries `md`: `README.MD` is markdown because the
    // SQL `LIKE` on the path is ASCII case-insensitive.
    let filter = ChunkFilter {
        include_ignored: true,
        side: Side::Docs,
        paths: None,
        extensions: vec!["md"],
        kinds: Vec::new(),
        symbol: None,
    };
    let docs = block_on(index.hashes_matching(filter)).unwrap();
    assert_eq!(
        docs,
        [ContentHash::of_chunk(
            "README.MD › alpha docs",
            "alpha docs"
        )]
    );
}
