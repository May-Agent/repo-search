//! These fixtures write `.rs` files: without `lang-rust` nothing scans.
#![cfg(feature = "lang-rust")]

use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::sync::Arc;

mod counters;

use counters::CapOff;
use futures_executor::block_on;
use repo_search_core::reconcile::{
    Books, Changes, Counts, Options, PendingChunk, Presence, Reads, Reconciled, Reconciler, Report,
    SkipReason,
};
use repo_search_core::store::{ContentHash, Index, Side, VectorCache};
use repo_search_core::{CHUNKER_FINGERPRINT, UnitVector};
use tempfile::TempDir;

const MODEL: &str = "test-model";

/// One pass as the old API returned it, plus its changes: `pending` is
/// the reconciler's `missing` taken right after the pass.
struct Pass {
    report: Report,
    changes: Option<Changes>,
    presence: Presence,
    pending: Vec<PendingChunk>,
}

struct Fixture {
    tree: TempDir,
    _common: TempDir,
    index: Index,
    cache: VectorCache,
    /// The reconciler of `index`, kept across passes as the daemon keeps it.
    reconciler: RefCell<Reconciler>,
    /// The pairs the presences of `reconciler`'s passes add up to, as a
    /// retriever follows them.
    followed: RefCell<HashSet<(ContentHash, Side)>>,
}

impl Fixture {
    fn new() -> Self {
        let tree = tempfile::tempdir().unwrap();
        let common = tempfile::tempdir().unwrap();
        let index = block_on(Index::open(tree.path())).unwrap();
        let cache = block_on(VectorCache::open(common.path())).unwrap();
        let reconciler = RefCell::new(Reconciler::new(
            index.clone(),
            cache.clone(),
            MODEL,
            tree.path(),
        ));
        Self {
            tree,
            _common: common,
            index,
            cache,
            reconciler,
            followed: RefCell::new(HashSet::new()),
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.tree.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn run(&self, options: Options) -> Pass {
        self.run_chunker(CHUNKER_FINGERPRINT, options)
    }

    /// A pass of the fixture's reconciler under `chunker`, followed.
    fn run_chunker(&self, chunker: &str, options: Options) -> Pass {
        let pass = self.run_with(&mut self.reconciler.borrow_mut(), chunker, options);
        self.follow(&pass.presence);
        pass
    }

    /// A pass of `reconciler`, then its `missing` read the way the feeder
    /// reads it.
    fn run_with(&self, reconciler: &mut Reconciler, chunker: &str, options: Options) -> Pass {
        let reconciled = self.reconcile_with(reconciler, chunker, options);
        let pending = self.missing_with(reconciler, options.include_ignored);
        Pass {
            report: reconciled.report,
            changes: reconciled.changes,
            presence: reconciled.presence,
            pending,
        }
    }

    /// One pass of `reconciler`. Every pass is also checked: the books it
    /// keeps are the books a fresh load reads, its counts are what a
    /// recount of every row finds, and a whole presence is the index's
    /// pairs.
    fn reconcile_with(
        &self,
        reconciler: &mut Reconciler,
        chunker: &str,
        options: Options,
    ) -> Reconciled {
        let reconciled =
            block_on(reconciler.reconcile(chunker, options, Arc::new(CapOff))).unwrap();
        let (loaded, _) = block_on(Books::load(&self.index, &self.cache, MODEL)).unwrap();
        assert_eq!(
            reconciler.books(),
            Some(&loaded),
            "the books a pass keeps are the books a load reads"
        );
        assert_eq!(
            (
                reconciled.report.chunks_total,
                reconciled.report.chunks_with_vector,
                reconciled.report.doc_chunks
            ),
            self.recount(options.include_ignored),
            "the counts are a full recount's"
        );
        match &reconciled.presence {
            Presence::Whole(pairs) => assert_eq!(
                pairs.iter().copied().collect::<HashSet<_>>(),
                self.pairs(),
                "a whole presence is every pair"
            ),
            Presence::Changed { appeared, vanished } => assert!(
                appeared.iter().all(|pair| !vanished.contains(pair)),
                "a pair is in one list"
            ),
        }
        reconciled
    }

    /// Applies a presence of the fixture's reconciler to `followed`, as a
    /// retriever applies it, and checks that it gives the index's pairs.
    fn follow(&self, presence: &Presence) {
        let mut followed = self.followed.borrow_mut();
        match presence {
            Presence::Whole(pairs) => *followed = pairs.iter().copied().collect(),
            Presence::Changed { appeared, vanished } => {
                for pair in vanished {
                    assert!(followed.remove(pair), "a vanished pair was held");
                }
                for pair in appeared {
                    assert!(followed.insert(*pair), "an appeared pair was not held");
                }
            }
        }
        assert_eq!(
            *followed,
            self.pairs(),
            "the followed presences are the index's pairs"
        );
    }

    /// Every `(hash, side)` pair the index holds, read with SQL.
    fn pairs(&self) -> HashSet<(ContentHash, Side)> {
        let conn = rusqlite::Connection::open(self.db()).unwrap();
        let mut stmt = conn
            .prepare("SELECT DISTINCT content_hash, kind = 'doc' FROM chunks")
            .unwrap();
        stmt.query_map([], |row| {
            let docs: bool = row.get(1)?;
            Ok((row.get(0)?, if docs { Side::Docs } else { Side::Code }))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
    }

    /// `missing` of `reconciler` with its texts, as the feeder reads it.
    /// Checked: the hashes are what a recount of every row against the
    /// cache finds, and every listed hash whose text the worktree still
    /// holds comes back once.
    fn missing_with(
        &self,
        reconciler: &mut Reconciler,
        include_ignored: bool,
    ) -> Vec<PendingChunk> {
        let missing = block_on(reconciler.missing(include_ignored)).unwrap();
        let listed: HashSet<ContentHash> = missing.hashes().iter().copied().collect();
        assert_eq!(listed.len(), missing.hashes().len(), "listed once each");
        assert_eq!(
            listed,
            self.uncached(include_ignored),
            "the books list what a full recount finds"
        );
        let pending = block_on(missing.pending(Vec::new())).unwrap().chunks;
        assert!(
            pending.iter().all(|c| listed.contains(&c.content_hash)),
            "only listed hashes come back"
        );
        pending
    }

    /// The distinct hashes of the scope's rows with no cached vector,
    /// recounted from every row.
    fn uncached(&self, include_ignored: bool) -> HashSet<ContentHash> {
        let distinct: Vec<ContentHash> = block_on(self.index.chunk_hashes(include_ignored))
            .unwrap()
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let cached = block_on(self.cache.cached(MODEL, &distinct)).unwrap();
        distinct
            .into_iter()
            .filter(|hash| !cached.contains(hash))
            .collect()
    }

    /// `counts` of the fixture's reconciler, checked against a recount.
    fn counts(&self, include_ignored: bool) -> Counts {
        let counts = block_on(self.reconciler.borrow_mut().counts(include_ignored)).unwrap();
        let (total, with_vector, _) = self.recount(include_ignored);
        assert_eq!(
            (counts.chunks_total, counts.chunks_with_vector),
            (total, with_vector),
            "counts are a full recount's"
        );
        counts
    }

    /// Runs only a pass of the fixture's reconciler, which therefore reads
    /// no file outside the pass itself, and follows it.
    fn reconcile_only(&self, options: Options) -> Reconciled {
        let reconciled = self.reconcile_with(
            &mut self.reconciler.borrow_mut(),
            CHUNKER_FINGERPRINT,
            options,
        );
        self.follow(&reconciled.presence);
        reconciled
    }

    /// A new reconciler of `index` over the fixture's tree and cache.
    fn reconciler_for(&self, index: &Index) -> Reconciler {
        Reconciler::new(index.clone(), self.cache.clone(), MODEL, self.tree.path())
    }

    /// Replaces the fixture's reconciler with one holding no books, as
    /// rows changed outside it require.
    fn forget_books(&self) {
        *self.reconciler.borrow_mut() = self.reconciler_for(&self.index);
    }

    /// `(rows, rows with a cached vector, docs rows)` in the scope, counted
    /// from every stored row the way a pass counted before it kept books.
    fn recount(&self, include_ignored: bool) -> (u64, u64, u64) {
        let conn = rusqlite::Connection::open(self.db()).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT content_hash, kind = 'doc' FROM chunks WHERE ?1 OR path NOT IN \
                 (SELECT path FROM files WHERE ignored = 1)",
            )
            .unwrap();
        let rows: Vec<(ContentHash, bool)> = stmt
            .query_map([include_ignored], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let docs = rows.iter().filter(|(_, docs)| *docs).count() as u64;
        let rows: Vec<ContentHash> = rows.into_iter().map(|(hash, _)| hash).collect();
        let distinct: Vec<ContentHash> = rows
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let cached = block_on(self.cache.cached(MODEL, &distinct)).unwrap();
        let with_vector = rows.iter().filter(|hash| cached.contains(hash)).count();
        (rows.len() as u64, with_vector as u64, docs)
    }

    fn pass(&self) -> Pass {
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

    // Back to the default: the ignored files stay stored but are hidden.
    let again = fx.pass();
    assert_eq!(again.report.files_removed, 0);
    assert_eq!(again.report.files_scanned, 1);
    assert_eq!(again.report.chunks_total, 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 3);
    assert_eq!(
        fx.query::<i64>("SELECT count(*) FROM files WHERE ignored = 1"),
        2
    );
}

const WITH_IGNORED: Options = Options {
    include_ignored: true,
};

/// A tree with one normal file and one ignored file, indexed with the flag
/// and embedded.
fn ignored_fixture() -> Fixture {
    let fx = Fixture::new();
    fx.write(".gitignore", "secret.rs\n");
    fx.write("kept.rs", "fn kept() {}\n");
    fx.write("secret.rs", "fn hidden() {}\n");
    let first = fx.run(WITH_IGNORED);
    assert_eq!(first.report.files_added, 2, "kept.rs and secret.rs");
    assert!(paths(&first.pending).contains(&"secret.rs"));
    fx.embed(&first.pending);
    fx
}

#[test]
fn alternating_passes_keep_ignored_rows_without_rework() {
    let fx = ignored_fixture();
    let ignored =
        |fx: &Fixture| fx.query::<i64>("SELECT ignored FROM files WHERE path = 'secret.rs'");
    assert_eq!(ignored(&fx), 1);
    let both = fx.run(WITH_IGNORED).report.chunks_total;

    // Without the flag nothing is removed or re-chunked, the ignored chunks
    // are not counted and none is queued for embedding.
    let plain = fx.pass();
    let r = &plain.report;
    assert_eq!(
        (r.files_removed, r.files_chunked, r.chunks_added),
        (0, 0, 0)
    );
    assert_eq!(r.chunks_removed, 0);
    assert_eq!(r.chunks_total, 1);
    assert!(r.chunks_total < both);
    assert!(plain.pending.is_empty());
    assert_eq!(ignored(&fx), 1);

    // An ignored file edited meanwhile is not read by a pass without the flag
    // and never queued; the flagged pass picks the edit up.
    fx.write("secret.rs", "fn hidden() {}\nfn extra() {}\n");
    let plain = fx.pass();
    assert_eq!(plain.report.files_chunked, 0);
    assert!(plain.pending.is_empty());
    let again = fx.run(WITH_IGNORED);
    assert_eq!(again.report.files_modified, 1);
    assert!(paths(&again.pending).iter().all(|p| *p == "secret.rs"));
    fx.embed(&again.pending);

    // The next flagged pass has nothing to chunk or embed.
    let settled = fx.run(WITH_IGNORED);
    assert_eq!(settled.report.files_changed(), 0);
    assert_eq!(settled.report.files_chunked, 0);
    assert_eq!(settled.report.chunks_added, 0);
    assert!(settled.pending.is_empty());
    assert_eq!(
        settled.report.chunks_with_vector,
        settled.report.chunks_total
    );
}

#[test]
fn an_unignored_file_is_flipped_without_rechunking() {
    let fx = ignored_fixture();
    let chunk_ids: String = fx.query("SELECT group_concat(id) FROM chunks");
    fx.write(".gitignore", "");
    let r = fx.pass();
    assert_eq!(r.report.files_scanned, 2, "kept.rs and secret.rs");
    assert_eq!((r.report.files_chunked, r.report.files_removed), (0, 0));
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(
        fx.query::<i64>("SELECT ignored FROM files WHERE path = 'secret.rs'"),
        0
    );
    assert_eq!(
        fx.query::<String>("SELECT group_concat(id) FROM chunks"),
        chunk_ids
    );
    assert!(r.pending.is_empty());
    assert_eq!(r.report.chunks_with_vector, r.report.chunks_total);
}

#[test]
fn a_newly_ignored_file_is_flagged_by_a_pass_that_includes_ignored() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() {}\n");
    fx.pass();
    fx.write(".gitignore", "a.rs\n");
    let r = fx.run(WITH_IGNORED);
    assert_eq!((r.report.files_chunked, r.report.files_removed), (0, 0));
    assert_eq!(
        fx.query::<i64>("SELECT ignored FROM files WHERE path = 'a.rs'"),
        1
    );
    // A pass without the flag no longer sees it, and keeps it.
    let plain = fx.pass();
    assert_eq!(plain.report.files_removed, 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 1);
}

#[test]
fn an_ignored_file_is_removed_when_deleted_and_a_flagged_pass_runs() {
    let fx = ignored_fixture();
    fs::remove_file(fx.tree.path().join("secret.rs")).unwrap();

    // Not walked without the flag: the row waits for the next flagged pass.
    let plain = fx.pass();
    assert_eq!(plain.report.files_removed, 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 2);

    let flagged = fx.run(WITH_IGNORED);
    assert_eq!(flagged.report.files_removed, 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 1);
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

    let err = block_on(fx.reconciler.borrow_mut().reconcile(
        CHUNKER_FINGERPRINT,
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
    // Rows changed outside the reconciler leave its books wrong.
    fx.forget_books();

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
    let err = block_on(
        Reconciler::new(fx.index.clone(), fx.cache.clone(), MODEL, &missing).reconcile(
            CHUNKER_FINGERPRINT,
            Options::default(),
            Arc::new(CapOff),
        ),
    )
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
    assert!(
        matches!(&r.report.skipped[0].reason, SkipReason::Malformed(error) if !error.is_empty()),
        "{:?}",
        r.report.skipped[0].reason
    );
    assert_eq!(r.report.files_removed, 1);
    assert_eq!(r.report.chunks_total, 0);
}

fn malformed_error(r: &Pass) -> &str {
    match &r.report.skipped[0].reason {
        SkipReason::Malformed(error) => error,
        other => panic!("expected a malformed file, got {other:?}"),
    }
}

#[test]
fn a_malformed_file_is_recorded_and_not_parsed_again() {
    let fx = Fixture::new();
    fx.write("ok.json", "{\"a\": 1}");
    fx.write("c.json", "{\"a\": ");

    let first = fx.pass();
    assert_eq!(first.report.files_chunked, 2);
    let error = malformed_error(&first).to_owned();
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 1);
    assert_eq!(
        fx.query::<String>("SELECT reason FROM unchunkable"),
        format!("malformed:{error}")
    );

    // Nothing changed: the recorded file is neither read nor parsed, yet is
    // still reported.
    let second = fx.pass();
    assert_eq!(second.report.files_chunked, 0);
    assert_eq!(second.report.files_restamped, 0);
    assert_eq!(second.report.skipped.len(), 1);
    assert_eq!(second.report.skipped[0].path, "c.json");
    assert_eq!(malformed_error(&second), error);

    // Fixing the file chunks it and clears its row.
    fx.write("c.json", "{\"a\": 2, \"b\": 3}");
    let third = fx.pass();
    assert_eq!(third.report.files_chunked, 1);
    assert_eq!(third.report.files_added, 1);
    assert!(third.report.skipped.is_empty());
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 2);
}

#[test]
fn a_malformed_file_with_moved_stat_is_restamped_not_parsed() {
    let fx = Fixture::new();
    fx.write("c.json", "{\"a\": ");
    fx.pass();
    let before: i64 = fx.query("SELECT inode FROM unchunkable");

    // Same bytes, new inode and mtime.
    let path = fx.tree.path().join("c.json");
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    fs::write(&path, bytes).unwrap();

    let r = fx.pass();
    assert_eq!(r.report.files_chunked, 0);
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(r.report.skipped.len(), 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 1);
    let after: i64 = fx.query("SELECT inode FROM unchunkable");
    assert_ne!(before, after, "the fingerprint was rewritten");
    assert_eq!(fx.pass().report.files_restamped, 0);
}

#[test]
fn a_changed_malformed_file_is_tried_again_and_updates_its_row() {
    let fx = Fixture::new();
    fx.write("c.json", "{\"a\": ");
    let first = fx.pass();
    fx.write("c.json", "{\"a\": 1,, ");
    let r = fx.pass();
    assert_eq!(r.report.files_chunked, 1);
    assert_eq!(r.report.skipped.len(), 1);
    assert_ne!(malformed_error(&r), malformed_error(&first));
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 1);
}

#[test]
fn a_deleted_unchunkable_file_loses_its_row() {
    let fx = Fixture::new();
    fx.write("c.json", "{\"a\": ");
    fx.write("b.rs", "fn b() {}\n");
    fs::write(fx.tree.path().join("a.rs"), [0xff, 0xfe, 0x00]).unwrap();
    let first = fx.pass();
    assert_eq!(first.report.skipped.len(), 2);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 2);

    let again = fx.pass();
    assert_eq!(again.report.files_chunked, 0);
    assert_eq!(again.report.skipped.len(), 2);

    fs::remove_file(fx.tree.path().join("c.json")).unwrap();
    fs::remove_file(fx.tree.path().join("a.rs")).unwrap();
    let r = fx.pass();
    assert!(r.report.skipped.is_empty());
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 0);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM files"), 1);
}

#[test]
fn another_chunker_rechunks_and_reuses_cached_vectors() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n\nstruct Gamma;\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.run_chunker("chunker-a", Options::default());
    assert!(!first.report.rechunked);
    fx.embed(&first.pending);
    // Mark the stored rows; a rebuilt index must not carry the mark.
    rusqlite::Connection::open(fx.db())
        .unwrap()
        .execute("UPDATE chunks SET title = 'stale'", [])
        .unwrap();
    let total = first.report.chunks_total;

    let mut reopened = fx.reconciler_for(&block_on(Index::open(fx.tree.path())).unwrap());
    let second = fx.run_with(&mut reopened, "chunker-b", Options::default());
    assert!(second.changes.is_none(), "a rechunk reports no changes");
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

    let third = fx.run_with(&mut reopened, "chunker-b", Options::default());
    assert!(!third.report.rechunked);
    assert_eq!(third.report.files_changed(), 0);
    assert_eq!(third.report.chunks_added, 0);
}

#[test]
fn the_same_chunker_reopens_without_rechunking() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.run_chunker("chunker-a", Options::default());
    let mut reopened = fx.reconciler_for(&block_on(Index::open(fx.tree.path())).unwrap());
    let again = fx.run_with(&mut reopened, "chunker-a", Options::default());
    assert!(!again.report.rechunked);
    assert_eq!(again.report.chunks_added, 0);
    assert_eq!(again.report.files_changed(), 0);
}

#[test]
fn another_chunker_retries_recorded_unchunkable_files() {
    let fx = Fixture::new();
    fx.write("ok.json", "{\"a\": 1}");
    fx.write("c.json", "{\"a\": ");
    fx.run_chunker("chunker-a", Options::default());
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 1);

    // Unchanged files under a new chunker: the recorded failure is parsed
    // again rather than trusted, since the new chunker may accept it.
    let r = fx.run_chunker("chunker-b", Options::default());
    assert!(r.report.rechunked);
    assert_eq!(r.report.files_chunked, 2);
    assert_eq!(r.report.skipped.len(), 1);
    assert_eq!(fx.query::<i64>("SELECT count(*) FROM unchunkable"), 1);
}

// DEV-PROBE (migrated to the final form with the implementation): a no-change
// pass must not read a pending file. On dev `reconcile` rebuilds every missing
// chunk's text and fails here with `Error::Io` (PermissionDenied on b.rs).
#[test]
fn a_no_change_pass_reads_no_file_while_vectors_are_missing() {
    use std::os::unix::fs::PermissionsExt;

    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.pass();
    assert!(!first.pending.is_empty());
    let b = fx.tree.path().join("b.rs");
    // ctime only: size, mtime and inode hold, so nothing changed.
    fs::set_permissions(&b, fs::Permissions::from_mode(0o000)).unwrap();
    let r = fx.reconcile_only(Options::default());
    fs::set_permissions(&b, fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(r.report.files_changed(), 0);
    assert_eq!(r.report.files_chunked, 0);
    assert_eq!(r.changes, Some(Changes::default()));
    assert_eq!(r.report.chunks_with_vector, 0);
    assert_eq!(r.report.chunks_total, first.report.chunks_total);
}

/// The content hashes of `pending`'s chunks on `path`.
fn hashes_on(pending: &[PendingChunk], path: &str) -> std::collections::HashSet<ContentHash> {
    pending
        .iter()
        .filter(|c| c.path == path)
        .map(|c| c.content_hash)
        .collect()
}

#[test]
fn an_edit_offers_that_files_chunks_and_drops_its_old_ones() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn gamma() { 3; }\n");
    let first = fx.pass();
    let a0 = hashes_on(&first.pending, "a.rs");
    let b0 = hashes_on(&first.pending, "b.rs");
    assert!(!a0.is_empty() && !b0.is_empty());

    fx.write("a.rs", "fn alpha() { 10; }\n");
    let r = fx.reconcile_only(Options::default());
    let c = r.changes.expect("a plain pass reports its changes");
    assert!(!c.added.is_empty());
    assert!(c.added.iter().all(|chunk| chunk.path == "a.rs"));
    assert!(c.added.iter().any(|chunk| chunk.text.contains("10")));
    let dropped: std::collections::HashSet<_> = c.dropped.iter().copied().collect();
    assert_eq!(dropped, a0);
    let added: std::collections::HashSet<_> =
        c.added.iter().map(|chunk| chunk.content_hash).collect();
    assert!(
        b0.iter()
            .all(|h| !added.contains(h) && !dropped.contains(h))
    );

    // Convergence: `missing` after the pass is the old set minus the drops
    // plus the additions.
    let next: std::collections::HashSet<_> =
        fx.pass().pending.iter().map(|c| c.content_hash).collect();
    let mut expect: std::collections::HashSet<_> =
        first.pending.iter().map(|c| c.content_hash).collect();
    for hash in &a0 {
        expect.remove(hash);
    }
    expect.extend(added);
    assert_eq!(next, expect);
}

#[test]
fn an_unchanged_chunk_of_an_edited_file_stays_wanted() {
    let big = |name: &str| {
        format!(
            "pub fn {name}() -> usize {{ \"{}\".len() }}\n\n",
            "x".repeat(500)
        )
    };
    let fx = Fixture::new();
    fx.write("a.rs", &(big("alpha") + &big("beta")));
    let first = fx.pass();
    let hash_of = |needle: &str| {
        first
            .pending
            .iter()
            .find(|c| c.text.contains(needle))
            .unwrap_or_else(|| panic!("a pending chunk holds {needle}"))
            .content_hash
    };
    let alpha = hash_of("fn alpha");
    let beta = hash_of("fn beta");
    assert_ne!(alpha, beta, "each function is its own chunk");

    fx.write("a.rs", &(big("alpha") + &big("beta_two")));
    let c = fx
        .reconcile_only(Options::default())
        .changes
        .expect("a plain pass reports its changes");
    assert!(c.dropped.contains(&beta));
    assert!(!c.dropped.contains(&alpha));
    let added: std::collections::HashSet<_> =
        c.added.iter().map(|chunk| chunk.content_hash).collect();
    assert!(added.contains(&alpha), "the unchanged chunk is re-offered");
    assert!(c.added.iter().any(|chunk| chunk.text.contains("beta_two")));
}

#[test]
fn a_deleted_file_drops_its_hashes() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.pass();
    let b0 = hashes_on(&first.pending, "b.rs");
    assert!(!b0.is_empty());

    fs::remove_file(fx.tree.path().join("b.rs")).unwrap();
    let c = fx
        .reconcile_only(Options::default())
        .changes
        .expect("a plain pass reports its changes");
    assert!(c.added.is_empty());
    let dropped: std::collections::HashSet<_> = c.dropped.iter().copied().collect();
    assert_eq!(dropped, b0);
}

#[test]
fn a_restamped_file_offers_its_uncached_chunks_again() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    let first = fx.pass();
    let a0 = hashes_on(&first.pending, "a.rs");
    assert!(!a0.is_empty());

    // Same bytes, new inode and mtime, as an editor's atomic save leaves.
    let path = fx.tree.path().join("a.rs");
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    fs::write(&path, bytes).unwrap();

    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(r.report.files_chunked, 0);
    let c = r.changes.expect("a plain pass reports its changes");
    let added: std::collections::HashSet<_> =
        c.added.iter().map(|chunk| chunk.content_hash).collect();
    assert_eq!(added, a0);
    for chunk in &c.added {
        let first_chunk = first
            .pending
            .iter()
            .find(|p| p.content_hash == chunk.content_hash)
            .expect("the restamped file's chunks are its stored ones");
        assert_eq!(chunk.text, first_chunk.text);
    }
    assert!(c.dropped.is_empty());
}

#[test]
fn an_unignored_file_offers_its_uncached_chunks() {
    let fx = Fixture::new();
    fx.write(".gitignore", "secret.rs\n");
    fx.write("kept.rs", "fn kept() {}\n");
    fx.write("secret.rs", "fn hidden() {}\n");
    let first = fx.run(WITH_IGNORED);
    let secret = hashes_on(&first.pending, "secret.rs");
    assert!(!secret.is_empty());

    fx.write(".gitignore", "");
    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_restamped, 1);
    let c = r.changes.expect("a plain pass reports its changes");
    let added: std::collections::HashSet<_> =
        c.added.iter().map(|chunk| chunk.content_hash).collect();
    assert_eq!(added, secret);
}

#[test]
fn a_scope_true_pass_drops_a_deleted_ignored_file() {
    let fx = Fixture::new();
    fx.write(".gitignore", "secret.rs\n");
    fx.write("kept.rs", "fn kept() {}\n");
    fx.write("secret.rs", "fn hidden() {}\n");
    let first = fx.run(WITH_IGNORED);
    let secret = hashes_on(&first.pending, "secret.rs");
    assert!(!secret.is_empty());

    fs::remove_file(fx.tree.path().join("secret.rs")).unwrap();
    let c = fx
        .reconcile_only(WITH_IGNORED)
        .changes
        .expect("a plain pass reports its changes");
    let dropped: std::collections::HashSet<_> = c.dropped.iter().copied().collect();
    assert_eq!(dropped, secret);
}

/// `count` files whose chunks hash apart from every other file's.
fn write_modules(fx: &Fixture, count: usize) {
    for i in 0..count {
        fx.write(
            &format!("m{i}.rs"),
            &format!("fn alpha_{i}() {{ {i}; }}\n\nfn beta_{i}() {{ {i}; }}\n\nstruct Gamma{i};\n"),
        );
    }
}

#[test]
fn a_no_change_pass_reads_no_chunk_row_and_looks_up_no_hash() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    let total = first.report.chunks_total;
    assert!(total >= 20);
    // A fresh index: no row to load, and each new hash is looked up once.
    assert_eq!(
        first.report.reads,
        Reads {
            chunk_rows: 0,
            cache_probes: first.pending.len() as u64,
            cache_slots: 0
        }
    );

    let (half, _) = first.pending.split_at(first.pending.len() / 2);
    fx.embed(half);
    // Nothing changed but vectors arrived: only the slot log is read.
    let second = fx.reconcile_only(Options::default());
    assert_eq!(second.report.files_changed(), 0);
    assert_eq!(
        second.report.reads,
        Reads {
            chunk_rows: 0,
            cache_probes: 0,
            cache_slots: half.len() as u64
        }
    );
    assert_eq!(second.report.chunks_total, total);
    assert!(second.report.chunks_with_vector > 0);

    // Nothing changed and nothing arrived: the pass reads nothing, however
    // many chunks the index holds.
    let third = fx.reconcile_only(Options::default());
    assert_eq!(third.report.reads, Reads::default());
    assert_eq!(
        third.report.chunks_with_vector,
        second.report.chunks_with_vector
    );
}

#[test]
fn a_pass_looks_up_only_the_hashes_of_the_file_it_changed() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    fx.embed(&first.pending);
    fx.reconcile_only(Options::default());

    fx.write(
        "m3.rs",
        "fn alpha_3() { 3; }\n\nfn changed_3() { 33; }\n\nstruct Gamma3;\n",
    );
    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_modified, 1);
    let reads = r.report.reads;
    assert_eq!((reads.chunk_rows, reads.cache_slots), (0, 0));
    // The file's rows were dropped and added again, so each of its hashes
    // is new to the books; no other file's hash is looked up.
    assert_eq!(reads.cache_probes, r.report.chunks_added);
    assert!(reads.cache_probes < r.report.chunks_total);
    let added = r.changes.expect("a plain pass reports its changes").added;
    assert!(!added.is_empty());
    assert!(added.iter().all(|chunk| chunk.path == "m3.rs"));
}

#[test]
fn a_restamped_file_reads_only_its_own_rows() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    fx.embed(&first.pending);
    fx.reconcile_only(Options::default());
    let rows: i64 = fx.query("SELECT count(*) FROM chunks WHERE path = 'm5.rs'");

    // Same bytes, new inode and mtime, as an editor's atomic save leaves.
    let path = fx.tree.path().join("m5.rs");
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    fs::write(&path, bytes).unwrap();

    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(
        r.report.reads,
        Reads {
            chunk_rows: rows as u64,
            cache_probes: 0,
            cache_slots: 0
        }
    );
}

#[test]
fn an_unignored_file_moves_its_rows_into_the_plain_scope() {
    let fx = ignored_fixture();
    let plain = fx.reconcile_only(Options::default());
    assert_eq!(plain.report.chunks_total, 1);
    let secret: i64 = fx.query("SELECT count(*) FROM chunks WHERE path = 'secret.rs'");

    fx.write(".gitignore", "");
    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_restamped, 1);
    assert_eq!(r.report.chunks_total, 1 + secret as u64);
    assert_eq!(r.report.chunks_with_vector, r.report.chunks_total);
    assert_eq!(
        r.report.reads,
        Reads {
            chunk_rows: secret as u64,
            cache_probes: 0,
            cache_slots: 0
        }
    );
}

#[test]
fn a_reconciler_without_books_loads_them_once() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    fx.embed(&first.pending);
    let rows: i64 = fx.query("SELECT count(*) FROM chunks");
    let distinct: i64 = fx.query("SELECT count(DISTINCT content_hash) FROM chunks");

    let mut fresh = fx.reconciler_for(&fx.index);
    let loaded = fx.run_with(&mut fresh, CHUNKER_FINGERPRINT, Options::default());
    assert_eq!(
        loaded.report.reads,
        Reads {
            chunk_rows: rows as u64,
            cache_probes: distinct as u64,
            cache_slots: 0
        }
    );
    assert_eq!(loaded.report.chunks_with_vector, loaded.report.chunks_total);

    let again = fx.run_with(&mut fresh, CHUNKER_FINGERPRINT, Options::default());
    assert_eq!(again.report.reads, Reads::default());
}

#[test]
fn a_rechunk_starts_the_books_over() {
    let fx = Fixture::new();
    write_modules(&fx, 5);
    let first = fx.run_chunker("chunker-a", Options::default());
    fx.embed(&first.pending);
    fx.run_chunker("chunker-a", Options::default());
    let distinct: i64 = fx.query("SELECT count(DISTINCT content_hash) FROM chunks");

    let r = fx.run_chunker("chunker-b", Options::default());
    assert!(r.report.rechunked);
    // The emptied index loads no row; every re-chunked hash is new to the
    // books and looked up once.
    assert_eq!(
        r.report.reads,
        Reads {
            chunk_rows: 0,
            cache_probes: distinct as u64,
            cache_slots: 0
        }
    );
    assert_eq!(r.report.chunks_with_vector, r.report.chunks_total);
}

#[test]
fn a_failed_pass_drops_the_books() {
    let fx = Fixture::new();
    fx.write("a.rs", "fn alpha() { 1; }\n");
    fx.write("b.rs", "fn beta() { 2; }\n");
    fx.pass();
    let rows: i64 = fx.query("SELECT count(*) FROM chunks");
    let conn = rusqlite::Connection::open(fx.db()).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER boom BEFORE INSERT ON chunks WHEN NEW.path = 'b.rs' \
         BEGIN SELECT RAISE(ABORT, 'boom'); END;",
    )
    .unwrap();
    fx.write("b.rs", "fn beta() { 22; }\n");

    let err = block_on(fx.reconciler.borrow_mut().reconcile(
        CHUNKER_FINGERPRINT,
        Options::default(),
        Arc::new(CapOff),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("boom"), "{err}");
    assert!(
        fx.reconciler.borrow().books().is_none(),
        "a failed pass drops the books"
    );

    conn.execute_batch("DROP TRIGGER boom").unwrap();
    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.files_modified, 1);
    assert_eq!(
        r.report.reads.chunk_rows, rows as u64,
        "the next pass loads the books again"
    );
}

#[test]
fn a_pass_reports_only_the_pairs_it_changed() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    fx.write(
        "guide.md",
        "# Guide\n\n## Stacks\n\nStacks align their children.\n\n## Values\n\nValues live in the environment.\n",
    );
    let first = fx.pass();
    // A new reconciler reads its books whole: every pair, docs included.
    let Presence::Whole(pairs) = &first.presence else {
        panic!("a first pass is whole: {:?}", first.presence);
    };
    assert!(pairs.iter().any(|(_, side)| *side == Side::Docs));
    assert!(first.report.doc_chunks > 0);
    let pairs = pairs.len();

    // Nothing changed: an empty presence, however many pairs the index holds.
    let unchanged = fx.reconcile_only(Options::default());
    assert_eq!(
        unchanged.presence,
        Presence::Changed {
            appeared: Vec::new(),
            vanished: Vec::new()
        }
    );

    // One file changed: only its pairs move. `alpha_3` and `Gamma3` keep
    // their hashes, so they neither appear nor vanish.
    fx.write(
        "m3.rs",
        "fn alpha_3() { 3; }\n\nfn changed_3() { 33; }\n\nstruct Gamma3;\n",
    );
    let r = fx.reconcile_only(Options::default());
    let Presence::Changed { appeared, vanished } = &r.presence else {
        panic!("a pass with books reports changes: {:?}", r.presence);
    };
    assert!(!appeared.is_empty() && !vanished.is_empty());
    assert!(appeared.len() + vanished.len() < pairs);
    assert!(appeared.len() as u64 <= r.report.chunks_added);
    assert!(vanished.len() as u64 <= r.report.chunks_removed);
}

#[test]
fn a_pass_after_reloaded_books_reports_every_pair() {
    let fx = Fixture::new();
    write_modules(&fx, 5);
    fx.pass();
    fx.reconcile_only(Options::default());

    // Books read whole between passes, as `counts` does without them.
    fx.forget_books();
    fx.counts(false);
    let r = fx.reconcile_only(Options::default());
    assert!(matches!(r.presence, Presence::Whole(_)), "{:?}", r.presence);
    assert_eq!(r.report.reads, Reads::default(), "counts loaded the books");

    let again = fx.reconcile_only(Options::default());
    assert!(matches!(again.presence, Presence::Changed { .. }));
}

#[test]
fn counts_between_passes_read_only_the_vectors_stored_since() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    assert!(first.report.chunks_total >= 20);

    // Nothing stored since the pass: nothing is read, whatever the index
    // holds.
    let idle = fx.counts(false);
    assert_eq!(idle.reads, Reads::default());
    assert_eq!(idle.chunks_with_vector, 0);

    // Vectors stored since: only those are read, from the slot log.
    let (half, _) = first.pending.split_at(first.pending.len() / 2);
    fx.embed(half);
    let after = fx.counts(false);
    assert_eq!(
        after.reads,
        Reads {
            chunk_rows: 0,
            cache_probes: 0,
            cache_slots: half.len() as u64
        }
    );
    assert!(after.chunks_with_vector > 0);
    assert_eq!(fx.counts(false).reads, Reads::default());

    // The next pass picks up from where `counts` left the books.
    let r = fx.reconcile_only(Options::default());
    assert_eq!(r.report.reads, Reads::default());
    assert_eq!(r.report.chunks_with_vector, after.chunks_with_vector);
}

#[test]
fn counts_without_books_read_them_whole_once() {
    let fx = Fixture::new();
    write_modules(&fx, 20);
    let first = fx.pass();
    fx.embed(&first.pending);
    let rows: i64 = fx.query("SELECT count(*) FROM chunks");
    let distinct: i64 = fx.query("SELECT count(DISTINCT content_hash) FROM chunks");

    fx.forget_books();
    let loaded = fx.counts(false);
    assert_eq!(
        loaded.reads,
        Reads {
            chunk_rows: rows as u64,
            cache_probes: distinct as u64,
            cache_slots: 0
        }
    );
    assert_eq!(loaded.chunks_with_vector, loaded.chunks_total);
    assert_eq!(fx.counts(false).reads, Reads::default());
}

#[test]
fn counts_follow_the_scope() {
    let fx = ignored_fixture();
    fx.reconcile_only(Options::default());
    let secret: i64 = fx.query("SELECT count(*) FROM chunks WHERE path = 'secret.rs'");
    let plain = fx.counts(false);
    let all = fx.counts(true);
    assert_eq!(all.chunks_total, plain.chunks_total + secret as u64);
    assert_eq!(all.reads, Reads::default());
}

/// What a resync reads once every chunk but those of `m0.rs` to `m2.rs`
/// has a vector, over `modules` modules: its listing at the barrier, then
/// its settling after one more vector was stored, and the number of chunks
/// left and stored.
fn resync_reads(modules: usize) -> (Reads, Reads, usize, usize) {
    let fx = Fixture::new();
    write_modules(&fx, modules);
    let first = fx.pass();
    let (left, rest): (Vec<PendingChunk>, Vec<PendingChunk>) = first
        .pending
        .into_iter()
        .partition(|c| ["m0.rs", "m1.rs", "m2.rs"].contains(&c.path.as_str()));
    assert!(left.len() >= 3);
    fx.embed(&rest);
    // The barrier the resync belongs to books the vectors stored since.
    fx.reconcile_only(Options::default());

    let mut reconciler = fx.reconciler.borrow_mut();
    let missing = block_on(reconciler.missing(false)).unwrap();
    let listing = missing.reads;
    let listed: HashSet<ContentHash> = missing.hashes().iter().copied().collect();
    assert_eq!(listed, fx.uncached(false));
    assert_eq!(
        listed,
        left.iter().map(|c| c.content_hash).collect::<HashSet<_>>()
    );

    // One more vector lands before the feeder's flush: settling drops it,
    // and a text the pass already holds is not rebuilt.
    fx.embed(&left[..1]);
    let known = left[left.len() - 1..].to_vec();
    let pending = block_on(missing.pending(known)).unwrap();
    assert_eq!(pending.chunks, left[1..], "the texts a pass would offer");
    (listing, pending.reads, left.len(), rest.len())
}

#[test]
fn a_resync_reads_nothing_for_the_chunks_that_have_a_vector() {
    let small = resync_reads(5);
    let large = resync_reads(60);
    // The large tree has over ten times the chunks with a vector...
    assert!(large.3 >= 10 * small.3, "{} vs {}", large.3, small.3);
    assert_eq!(large.2, small.2);
    // ...and every read is the same: the listing reads nothing, settling
    // reads the one vector stored since and the rows of the hashes left
    // that the pass held no text for.
    let rebuilt = small.2 as u64 - 2;
    for (listing, settling, _, _) in [small, large] {
        assert_eq!(listing, Reads::default());
        assert_eq!(
            settling,
            Reads {
                chunk_rows: rebuilt,
                cache_probes: 0,
                cache_slots: 1
            }
        );
    }
}

#[test]
fn a_scope_flip_resync_lists_the_ignored_chunks_from_the_books() {
    let fx = Fixture::new();
    fx.write(".gitignore", "secret.rs\n");
    fx.write("kept.rs", "fn kept() {}\n");
    fx.write("secret.rs", "fn hidden() {}\n");
    let plain = fx.pass();
    assert_eq!(paths(&plain.pending), ["kept.rs"]);
    fx.embed(&plain.pending);
    // The flip's barrier indexes the ignored file and looks up only its
    // hashes; its resync then reads nothing.
    let flip = fx.reconcile_only(Options {
        include_ignored: true,
    });
    let mut reconciler = fx.reconciler.borrow_mut();
    let missing = block_on(reconciler.missing(true)).unwrap();
    assert_eq!(missing.reads, Reads::default());
    let listed: HashSet<ContentHash> = missing.hashes().iter().copied().collect();
    assert_eq!(listed, fx.uncached(true));
    assert!(!listed.is_empty());
    let pending = block_on(missing.pending(Vec::new())).unwrap();
    assert_eq!(paths(&pending.chunks), ["secret.rs"]);
    assert_eq!(
        pending.chunks,
        flip.changes
            .expect("a plain pass reports its changes")
            .added
    );
    // The plain scope wants nothing.
    let plain = block_on(reconciler.missing(false)).unwrap();
    assert!(plain.hashes().is_empty());
    assert_eq!(plain.reads, Reads::default());
}

#[test]
fn a_resync_without_books_loads_them_whole_once() {
    let fx = Fixture::new();
    write_modules(&fx, 10);
    let first = fx.pass();
    let (half, _) = first.pending.split_at(first.pending.len() / 2);
    fx.embed(half);
    let rows: i64 = fx.query("SELECT count(*) FROM chunks");
    let distinct: i64 = fx.query("SELECT count(DISTINCT content_hash) FROM chunks");

    // A reopened index's owner holds no books: the resync loads them.
    fx.forget_books();
    let mut reconciler = fx.reconciler.borrow_mut();
    let missing = block_on(reconciler.missing(false)).unwrap();
    assert_eq!(
        missing.reads,
        Reads {
            chunk_rows: rows as u64,
            cache_probes: distinct as u64,
            cache_slots: 0
        }
    );
    assert_eq!(
        missing.hashes().iter().copied().collect::<HashSet<_>>(),
        fx.uncached(false)
    );
    // The books stay for the next resync and the next pass.
    let again = block_on(reconciler.missing(false)).unwrap();
    assert_eq!(again.reads, Reads::default());
    drop(reconciler);
    let pass = fx.reconcile_only(Options::default());
    assert_eq!(pass.report.reads, Reads::default());
    assert!(matches!(pass.presence, Presence::Whole(_)));
}
