//! These fixtures write `.rs` files: without `lang-rust` nothing scans.
#![cfg(feature = "lang-rust")]

use std::fs;
use std::sync::Arc;

mod counters;

use counters::CapOff;
use futures_executor::block_on;
use repo_search_core::reconcile::{Options, PendingChunk, Reconciled, SkipReason, reconcile};
use repo_search_core::store::{ContentHash, Index, VectorCache};
use repo_search_core::{CHUNKER_FINGERPRINT, UnitVector};
use tempfile::TempDir;

const MODEL: &str = "test-model";

struct Fixture {
    tree: TempDir,
    _common: TempDir,
    index: Index,
    cache: VectorCache,
}

impl Fixture {
    fn new() -> Self {
        let tree = tempfile::tempdir().unwrap();
        let common = tempfile::tempdir().unwrap();
        let index = block_on(Index::open(tree.path())).unwrap();
        let cache = block_on(VectorCache::open(common.path())).unwrap();
        Self {
            tree,
            _common: common,
            index,
            cache,
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.tree.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn run(&self, options: Options) -> Reconciled {
        self.run_as(&self.index, CHUNKER_FINGERPRINT, options)
    }

    fn run_as(&self, index: &Index, chunker: &str, options: Options) -> Reconciled {
        block_on(reconcile(
            index,
            &self.cache,
            MODEL,
            chunker,
            self.tree.path(),
            options,
            Arc::new(CapOff),
        ))
        .unwrap()
    }

    fn pass(&self) -> Reconciled {
        self.run(Options::default())
    }

    /// Stores a vector for every pending chunk, as an embedder would.
    fn embed(&self, pending: &[PendingChunk]) {
        let vectors = pending
            .iter()
            .map(|c| (c.content_hash, UnitVector::normalize(vec![1.0, 2.0])))
            .collect();
        block_on(self.cache.put_many(MODEL, vectors)).unwrap();
    }

    fn db(&self) -> std::path::PathBuf {
        self.tree.path().join(".repo-search/index.db")
    }

    fn query<T: rusqlite::types::FromSql>(&self, sql: &str) -> T {
        rusqlite::Connection::open(self.db())
            .unwrap()
            .query_row(sql, [], |row| row.get(0))
            .unwrap()
    }

    fn fts_hits(&self, term: &str) -> i64 {
        rusqlite::Connection::open(self.db())
            .unwrap()
            .query_row("SELECT count(*) FROM fts WHERE fts MATCH ?1", [term], |r| {
                r.get(0)
            })
            .unwrap()
    }
}

fn paths(pending: &[PendingChunk]) -> Vec<&str> {
    pending.iter().map(|c| c.path.as_str()).collect()
}

#[test]
fn first_pass_indexes_everything_and_reports_missing_vectors() {
    let fx = Fixture::new();
    fx.write("src/a.rs", "fn alpha() { 1; }\n\nstruct Gamma;\n");
    fx.write("src/b.rs", "fn beta() { 2; }\n");
    fx.write("notes.xyz", "unknown extension\n");

    let first = fx.pass();
    let r = &first.report;
    assert_eq!((r.files_scanned, r.files_added), (2, 2));
    assert_eq!(r.files_changed(), 2);
    assert!(r.chunks_added >= 2);
    assert_eq!(r.chunks_total, r.chunks_added);
    assert_eq!(r.chunks_with_vector, 0);
    assert_eq!(r.vector_coverage(), 0.0);
    assert!(!first.pending.is_empty());
    assert!(first.pending.iter().all(|c| !c.text.is_empty()));
    assert_eq!(fx.fts_hits("alpha"), 1);

    fx.embed(&first.pending);
    let second = fx.pass();
    assert!(second.pending.is_empty());
    assert_eq!(second.report.vector_coverage(), 1.0);
    assert_eq!(second.report.chunks_with_vector, second.report.chunks_total);
}

#[test]
fn an_edit_replaces_only_that_files_chunks() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.pass();
    fx.embed(&first.pending);

    fx.write("a.rs", "fn alpha() { 1; }\n\nfn delta() { 4; }\n");
    let second = fx.pass();
    assert_eq!(second.report.files_modified, 1);
    assert_eq!(second.report.files_added, 0);
    assert_eq!(second.report.files_removed, 0);
    assert!(second.report.chunks_removed >= 1 && second.report.chunks_added >= 1);
    assert_eq!(fx.fts_hits("delta"), 1);
    // b.rs kept its vector; only a.rs's new chunks are pending.
    assert!(!second.pending.is_empty());
    assert!(paths(&second.pending).iter().all(|p| *p == "a.rs"));
    assert!(second.report.chunks_with_vector > 0);
}

#[test]
fn add_and_delete_are_applied() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.pass();

    fx.write("new.rs", "fn fresh() { 3; }\n");
    let added = fx.pass();
    assert_eq!(added.report.files_added, 1);
    assert_eq!(fx.fts_hits("fresh"), 1);

    fs::remove_file(fx.tree.path().join("a.rs")).unwrap();
    let removed = fx.pass();
    assert_eq!(removed.report.files_removed, 1);
    assert!(removed.report.chunks_removed >= 1);
    assert_eq!(fx.fts_hits("alpha"), 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 1);
    assert!(paths(&removed.pending).iter().all(|p| *p == "new.rs"));
}

#[test]
fn a_rename_drops_the_old_path_and_adds_the_new() {
    let fx = Fixture::new();
    fx.write("old.rs", "fn moved() { 5; }\n");
    fx.pass();

    fs::rename(fx.tree.path().join("old.rs"), fx.tree.path().join("new.rs")).unwrap();
    let r = fx.pass();
    assert_eq!(r.report.files_added, 1);
    assert_eq!(r.report.files_removed, 1);
    assert_eq!(
        fx.query::<i64>("SELECT count(*) FROM files WHERE path = 'old.rs'"),
        0
    );
    assert_eq!(
        fx.query::<i64>("SELECT count(*) FROM files WHERE path = 'new.rs'"),
        1
    );
    assert_eq!(fx.fts_hits("moved"), 1);
    assert_eq!(r.report.chunks_total, r.report.chunks_added);
    // Titles carry the path, so the old vectors do not apply: the new path's
    // chunks are all pending, with the moved text.
    assert!(!r.pending.is_empty());
    assert!(paths(&r.pending).iter().all(|p| *p == "new.rs"));
    assert!(r.pending.iter().any(|c| c.text.contains("moved")));
}

#[test]
fn a_touched_file_with_the_same_bytes_is_not_rechunked() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    let first = fx.pass();
    fx.embed(&first.pending);
    let chunk_ids: String = fx.query("SELECT group_concat(id) FROM chunks");

    // Same bytes, new inode and mtime, as an editor's atomic save leaves.
    let path = fx.tree.path().join("a.rs");
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    fs::write(&path, bytes).unwrap();

    let r = fx.pass();
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(r.report.files_changed(), 0);
    assert_eq!((r.report.chunks_added, r.report.chunks_removed), (0, 0));
    assert_eq!(
        fx.query::<String>("SELECT group_concat(id) FROM chunks"),
        chunk_ids
    );
    // The fingerprint was refreshed, so the next pass reads nothing.
    assert_eq!(fx.pass().report.files_restamped, 0);
}

#[test]
fn ignored_files_are_skipped_unless_asked_for() {
    let fx = Fixture::new();
    fx.write(".gitignore", "target/\nsecret.rs\n");
    fx.write("kept.rs", "fn kept() {}\n");
    fx.write("secret.rs", "fn hidden() {}\n");
    fx.write("target/gen.rs", "fn generated() {}\n");
    fx.write(".git/hooks/x.rs", "fn inner() {}\n");

    let plain = fx.pass();
    assert_eq!(plain.report.files_scanned, 1);
    assert_eq!(fx.fts_hits("hidden"), 0);
    assert_eq!(fx.fts_hits("generated"), 0);

    let all = fx.run(Options {
        include_ignored: true,
    });
    assert_eq!(all.report.files_added, 2);
    assert_eq!(fx.fts_hits("hidden"), 1);
    assert_eq!(fx.fts_hits("generated"), 1);
    assert_eq!(fx.fts_hits("inner"), 0, ".git is never indexed");

    // Back to the default: the ignored files leave the index.
    let again = fx.pass();
    assert_eq!(again.report.files_removed, 2);
    assert_eq!(fx.fts_hits("hidden"), 0);
}

#[test]
fn a_pass_with_no_change_writes_nothing() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.pass();
    fx.embed(&first.pending);
    fx.pass();

    let before = fs::metadata(fx.db()).unwrap().modified().unwrap();
    let ids: String = fx.query("SELECT group_concat(id) FROM chunks");
    let r = fx.pass();
    assert_eq!(r.report.files_changed(), 0);
    assert_eq!(r.report.files_restamped, 0);
    assert_eq!((r.report.chunks_added, r.report.chunks_removed), (0, 0));
    assert!(r.pending.is_empty());
    assert_eq!(fs::metadata(fx.db()).unwrap().modified().unwrap(), before);
    assert_eq!(
        fx.query::<String>("SELECT group_concat(id) FROM chunks"),
        ids
    );
}

#[test]
fn a_failing_transaction_lands_nothing() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    // a.rs is applied first, then b.rs aborts the batch.
    let conn = rusqlite::Connection::open(fx.db()).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER boom BEFORE INSERT ON files WHEN NEW.path = 'b.rs' \
         BEGIN SELECT RAISE(ABORT, 'boom'); END;",
    )
    .unwrap();

    let err = block_on(reconcile(
        &fx.index,
        &fx.cache,
        MODEL,
        CHUNKER_FINGERPRINT,
        fx.tree.path(),
        Options::default(),
        Arc::new(CapOff),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("boom"), "{err}");
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM chunks"), 0);
    assert_eq!(fx.fts_hits("alpha"), 0);

    conn.execute_batch("DROP TRIGGER boom").unwrap();
    let ok = fx.pass();
    assert_eq!(ok.report.files_added, 2);
    assert_eq!(fx.fts_hits("alpha"), 1);
    assert_eq!(fx.fts_hits("beta"), 1);
}

#[test]
fn unreadable_text_is_reported_and_dropped() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.pass();
    fs::write(fx.tree.path().join("a.rs"), [0xff, 0xfe, 0x00]).unwrap();

    let r = fx.pass();
    assert_eq!(r.report.skipped.len(), 1);
    assert_eq!(r.report.skipped[0].path, "a.rs");
    assert_eq!(r.report.skipped[0].reason, SkipReason::NotUtf8);
    assert_eq!(r.report.files_removed, 1);
    assert_eq!(r.report.chunks_total, 0);
}

#[test]
fn chunks_of_untouched_files_without_vectors_are_returned() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    let first = fx.pass();
    assert!(!first.pending.is_empty());
    // The embedder never ran; a later pass still lists the chunks, with text.
    let second = fx.pass();
    assert_eq!(second.report.files_changed(), 0);
    assert_eq!(second.pending, first.pending);
    assert!(second.pending.iter().all(|c| c.path == "a.rs"));
    assert!(second.pending.iter().any(|c| c.text.contains("alpha")));
}

#[test]
fn pending_text_comes_from_stored_ranges_without_rechunking() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    let first = fx.pass();
    let chunk = &first.pending[0];

    // Give the stored row a title and hash the chunker would never produce.
    let title = "hand-made title";
    let hash = ContentHash::of_chunk(title, &chunk.text);
    rusqlite::Connection::open(fx.db())
        .unwrap()
        .execute(
            "UPDATE chunks SET title = ?1, content_hash = ?2 WHERE content_hash = ?3",
            rusqlite::params![title, hash, chunk.content_hash],
        )
        .unwrap();

    let second = fx.pass();
    assert_eq!(second.report.files_changed(), 0);
    assert_eq!(second.pending.len(), 1);
    assert_eq!(second.pending[0].title, title);
    assert_eq!(second.pending[0].content_hash, hash);
    assert_eq!(second.pending[0].text, chunk.text);
}

#[test]
fn dot_directories_are_indexed() {
    let fx = Fixture::new();
    fx.write(".github/scripts/tool.rs", "fn dotted() {}\n");
    fx.write(".hidden.rs", "fn secret_dot() {}\n");
    let r = fx.pass();
    assert_eq!(r.report.files_added, 2);
    assert_eq!(fx.fts_hits("dotted"), 1);
    assert_eq!(fx.fts_hits("secret_dot"), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn a_path_that_is_not_utf8_is_reported() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fx = Fixture::new();
    fx.write("ok.rs", "fn fine() {}\n");
    fs::write(
        fx.tree.path().join(OsStr::from_bytes(b"bad\xff.rs")),
        "fn odd() {}\n",
    )
    .unwrap();
    let r = fx.pass();
    assert_eq!(r.report.files_scanned, 1);
    assert_eq!(r.report.skipped.len(), 1);
    assert_eq!(r.report.skipped[0].reason, SkipReason::NotUtf8Path);
    assert!(r.report.skipped[0].path.starts_with("bad"));
}

#[test]
fn a_missing_worktree_is_an_error_and_leaves_the_index_untouched() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    fx.pass();
    let before: i64 = fx.query("SELECT count(*) FROM chunks");
    assert!(before > 0);

    let missing = fx.tree.path().join("no-such-subdir");
    let err = block_on(reconcile(
        &fx.index,
        &fx.cache,
        MODEL,
        CHUNKER_FINGERPRINT,
        &missing,
        Options::default(),
        Arc::new(CapOff),
    ))
    .unwrap_err();
    assert!(
        matches!(err, repo_search_core::reconcile::Error::Walk(_)),
        "{err:?}"
    );
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM chunks"), before);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 2);
}

#[test]
fn rebuilt_pending_text_equals_the_chunkers_for_packed_crlf_non_ascii() {
    use repo_search_core::chunk::{ChunkKind, Chunker, FileChunker, Rust};

    let src = "// caf\u{e9} \u{65e5}\u{672c}\u{8a9e}\r\nfn a() { \"na\u{ef}ve\"; }\r\n\r\nfn b() { \"\u{4e16}\u{754c}\"; }\r\n\r\nfn c() { 3; }\r\n";
    let fx = Fixture::new();
    fx.write("p.rs", src);
    let first = fx.pass();

    let mut expected = Vec::new();
    let mut packed = false;
    for chunk in Chunker::new(Rust).chunk_file("p.rs", src, &CapOff).unwrap() {
        packed |= chunk.kind == ChunkKind::Packed && chunk.members.len() > 1;
        let mut text = String::new();
        chunk.write_text(src, &mut text);
        expected.push((ContentHash::of_chunk(&chunk.title, &text), text));
    }
    assert!(packed, "fixture must produce a packed multi-member chunk");

    let second = fx.pass();
    assert_eq!(second.report.files_changed(), 0);
    assert_eq!(second.pending, first.pending);
    let mut got: Vec<_> = second
        .pending
        .iter()
        .map(|c| (c.content_hash, c.text.clone()))
        .collect();
    got.sort_by_key(|(_, t)| t.clone());
    expected.sort_by_key(|(_, t)| t.clone());
    assert_eq!(got, expected);
    assert!(second.pending.iter().any(|c| c.text.contains("\r\n")));
}

#[test]
fn documentation_and_configuration_are_chunked_by_extension() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("docs/guide.md", "# Guide\n\n## Setup\ninstall the widget\n");
    fx.write("Cargo.toml", "[package]\nname = \"demo\"\n");
    fx.write("conf.yaml", "server:\n  port: 80\n");
    fx.write("package.json", "{\"name\": \"demo\"}");
    fx.write("image.png", "not read");

    let r = fx.pass();
    assert_eq!(r.report.files_scanned, 5);
    assert!(r.report.skipped.is_empty());
    assert_eq!(fx.fts_hits("widget"), 1);
    let doc_title: String =
        fx.query("SELECT title FROM chunks WHERE path = 'docs/guide.md' AND kind = 'doc'");
    assert_eq!(doc_title, "docs/guide.md › Guide › Setup");
    let member_kind: String = fx.query(
        "SELECT m.kind FROM chunk_members m JOIN chunks c ON c.id = m.chunk_id \
         WHERE c.path = 'docs/guide.md'",
    );
    assert_eq!(member_kind, "doc");
    let config_chunks: i64 = fx.query("SELECT count(*) FROM chunk_members WHERE kind = 'config'");
    assert_eq!(config_chunks, 3);
}

#[test]
fn malformed_configuration_is_reported_and_dropped() {
    let fx = Fixture::new();
    fx.write("c.json", "{\"a\": 1}");
    fx.pass();
    fx.write("c.json", "{\"a\": ");
    // A different size moves the fingerprint.
    let r = fx.pass();
    assert_eq!(r.report.skipped.len(), 1);
    assert_eq!(r.report.skipped[0].path, "c.json");
    assert_eq!(r.report.skipped[0].reason, SkipReason::Malformed);
    assert_eq!(r.report.files_removed, 1);
    assert_eq!(r.report.chunks_total, 0);
}

#[test]
fn another_chunker_rechunks_and_reuses_cached_vectors() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n\nstruct Gamma;\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.run_as(&fx.index, "chunker-a", Options::default());
    assert!(!first.report.rechunked);
    fx.embed(&first.pending);
    // Mark the stored rows; a rebuilt index must not carry the mark.
    rusqlite::Connection::open(fx.db())
        .unwrap()
        .execute("UPDATE chunks SET title = 'stale'", [])
        .unwrap();
    let total = first.report.chunks_total;

    let reopened = block_on(Index::open(fx.tree.path())).unwrap();
    let second = fx.run_as(&reopened, "chunker-b", Options::default());
    let r = &second.report;
    assert!(r.rechunked);
    assert_eq!((r.files_added, r.files_modified), (2, 0));
    assert_eq!(r.chunks_added, total);
    assert_eq!(r.chunks_total, total);
    assert_eq!(r.chunks_with_vector, total);
    assert!(second.pending.is_empty(), "no chunk needs embedding again");
    assert_eq!(fx.fts_hits("alpha"), 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM fts"), total as i64);
    assert_eq!(
        fx.query::<i64>("SELECT count(*) FROM chunks WHERE title = 'stale'"),
        0,
        "the chunk rows were rebuilt"
    );

    let third = fx.run_as(&reopened, "chunker-b", Options::default());
    assert!(!third.report.rechunked);
    assert_eq!(third.report.files_changed(), 0);
    assert_eq!(third.report.chunks_added, 0);
}

#[test]
fn the_same_chunker_reopens_without_rechunking() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.run_as(&fx.index, "chunker-a", Options::default());
    let reopened = block_on(Index::open(fx.tree.path())).unwrap();
    let again = fx.run_as(&reopened, "chunker-a", Options::default());
    assert!(!again.report.rechunked);
    assert_eq!(again.report.chunks_added, 0);
    assert_eq!(again.report.files_changed(), 0);
}
