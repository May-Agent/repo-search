//! End to end: a small synthetic worktree, the real index and cache, and a
//! fake bag-of-words embedder.

use std::fs;
use std::future::Future;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

mod counters;
mod support;

use counters::CapOff;
use futures_executor::block_on;
use repo_search_core::reconcile::{Options, PendingChunk, missing, reconcile};
use repo_search_core::search::{
    Channel, Error, FilterError, Filters, Mode, ON_THE_SPOT, Reconciliation, Request, Results,
    Retriever, TokenCounter,
};
use repo_search_core::store::{
    ChunkFilter, ChunkRecord, ContentHash, FileState, Index, MemberRecord, Side, VectorCache,
    fts_body,
};
use repo_search_core::{CHUNKER_FINGERPRINT, Embedder, QueryTask, UnitVector};
use support::EmbedThread;
use tempfile::TempDir;

const MODEL: &str = "fake-bag-of-words";
const DIM: usize = 64;

/// Hashes each lowercase word of the text onto one of `DIM` axes.
fn bag_of_words(text: &str) -> UnitVector {
    let mut v = vec![0.0f32; DIM];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
    {
        let hash = blake3::hash(word.to_lowercase().as_bytes());
        v[usize::from(hash.as_bytes()[0]) % DIM] += 1.0;
    }
    if v.iter().all(|x| *x == 0.0) {
        v[0] = 1.0;
    }
    UnitVector::normalize(v)
}

#[derive(Debug, thiserror::Error)]
#[error("fake embedder failure")]
struct FakeError;

/// What the embedder saw; shared with the test because the retriever moves
/// the embedder onto its own thread.
#[derive(Default)]
struct Seen {
    queries: Mutex<Vec<QueryTask>>,
    documents: AtomicUsize,
}

impl Seen {
    fn queries(&self) -> Vec<QueryTask> {
        self.queries.lock().unwrap().clone()
    }

    fn documents(&self) -> usize {
        self.documents.load(Ordering::SeqCst)
    }
}

#[derive(Default)]
struct FakeEmbedder {
    seen: Arc<Seen>,
}

impl Embedder for FakeEmbedder {
    type Error = FakeError;

    fn model_id(&self) -> &str {
        MODEL
    }

    fn embed_document(&mut self, title: &str, text: &str) -> Result<UnitVector, FakeError> {
        self.seen.documents.fetch_add(1, Ordering::SeqCst);
        Ok(bag_of_words(&format!("{title} {text}")))
    }

    fn embed_query(&mut self, task: QueryTask, query: &str) -> Result<UnitVector, FakeError> {
        self.seen.queries.lock().unwrap().push(task);
        Ok(bag_of_words(query))
    }
}

/// The vector the fake embedder gives a pending chunk.
fn document_vector(chunk: &PendingChunk) -> UnitVector {
    FakeEmbedder::default()
        .embed_document(&chunk.title, &chunk.text)
        .unwrap()
}

struct Words;

impl TokenCounter for Words {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

/// Which chunks have a vector when the query runs.
enum Vectors {
    All,
    None,
    /// The first `count` lexical candidates of the code query `query`.
    Candidates {
        query: &'static str,
        count: usize,
    },
}

struct Fixture {
    tree: TempDir,
    _common: TempDir,
    index: Index,
    retriever: Retriever<EmbedThread<FakeEmbedder>>,
    seen: Arc<Seen>,
    reconciliation: Reconciliation,
    cache: VectorCache,
    /// Every chunk of the code files, with its text.
    pending: Vec<PendingChunk>,
}

/// A doc comment long enough that no item of the fixture is packed with a
/// neighbour.
fn padding() -> String {
    "/// Padding line that keeps this item above the packing limit.\n".repeat(10)
}

fn rust_item(summary: &str, code: &str) -> String {
    format!("/// {summary}\n{}{code}\n\n", padding())
}

const ENV_RS: &str = "environment.rs";

fn environment_source() -> String {
    [
        rust_item(
            "A scope that holds named values.",
            "pub struct Environment {\n    values: Vec<(String, i32)>,\n}",
        ),
        rust_item(
            "Stores a value into the environment under a key.",
            "pub fn insert_value(env: &mut Environment, key: String, value: i32) {\n    env.values.push((key, value));\n}",
        ),
    ]
    .concat()
}

fn layout_source() -> String {
    rust_item(
        "Aligns the children of a vertical stack along its cross axis.",
        "pub fn stack_alignment(children: usize) -> usize {\n    children\n}",
    )
}

fn signal_source() -> String {
    rust_item(
        "A signal derived from other signals.",
        "pub struct ComputedSignal {\n    value: i32,\n}",
    )
}

fn ui_source() -> String {
    rust_item(
        "Builds the widgets.",
        "pub fn widgets() {\n    // ComputedSignal ComputedSignal ComputedSignal ComputedSignal\n    // ComputedSignal ComputedSignal ComputedSignal ComputedSignal\n    let _ = ComputedSignal::new();\n}",
    )
}

const GUIDE: &str = "# Guide\n\n## Stacks\nStacks align their children along the vertical axis of the stack.\n\n## Values\nValues are stored into the environment with insert_value under a key.\n";

/// The code filter that admits everything on the code side.
fn code_filter() -> ChunkFilter {
    ChunkFilter {
        include_ignored: true,
        side: Side::Code,
        paths: None,
        extensions: Vec::new(),
        kinds: Vec::new(),
        symbol: None,
    }
}

impl Fixture {
    fn new(vectors: &Vectors, with_docs: bool) -> Self {
        Self::build(
            &[
                (ENV_RS, environment_source()),
                ("layout.rs", layout_source()),
                ("signal.rs", signal_source()),
                ("ui.rs", ui_source()),
            ]
            .map(|(name, text)| (name.to_owned(), text)),
            vectors,
            with_docs,
        )
    }

    /// A worktree of `files` (names under `src/`), indexed for real.
    fn build(files: &[(String, String)], vectors: &Vectors, with_docs: bool) -> Self {
        let tree = tempfile::tempdir().unwrap();
        let common = tempfile::tempdir().unwrap();
        let src = tree.path().join("src");
        fs::create_dir_all(&src).unwrap();
        for (name, text) in files {
            fs::write(src.join(name), text).unwrap();
        }

        let index = block_on(Index::open(tree.path())).unwrap();
        let cache = block_on(VectorCache::open(common.path())).unwrap();
        let reconciled = block_on(reconcile(
            &index,
            &cache,
            MODEL,
            CHUNKER_FINGERPRINT,
            tree.path(),
            Options::default(),
            Arc::new(CapOff),
        ))
        .unwrap();
        let code_pending = block_on(missing(
            &index,
            &cache,
            MODEL,
            tree.path(),
            Options::default(),
            Vec::new(),
        ))
        .unwrap();
        let mut pending = code_pending.clone();
        let mut total = reconciled.report.chunks_total;
        let changed = reconciled.report.files_changed();

        if with_docs {
            let docs = tree.path().join("docs");
            fs::create_dir_all(&docs).unwrap();
            fs::write(docs.join("guide.md"), GUIDE).unwrap();
            let sections = [
                ("Guide › Stacks", "## Stacks"),
                ("Guide › Values", "## Values"),
            ];
            let records: Vec<ChunkRecord> = sections
                .iter()
                .map(|(heading, marker)| {
                    let start = GUIDE.find(marker).unwrap();
                    let end = GUIDE[start + 1..]
                        .find("\n## ")
                        .map_or(GUIDE.len(), |e| start + 1 + e + 1);
                    let text = &GUIDE[start..end];
                    let title = format!("docs/guide.md › {heading}");
                    let first_line = GUIDE[..start].lines().count() as u32 + 1;
                    let last_line = first_line + text.trim_end().lines().count() as u32 - 1;
                    let hash = ContentHash::of_chunk(&title, text);
                    pending.push(PendingChunk {
                        path: "docs/guide.md".to_owned(),
                        title: title.clone(),
                        text: text.to_owned(),
                        lines: first_line..=last_line,
                        content_hash: hash,
                    });
                    ChunkRecord {
                        lines: first_line..=last_line,
                        title,
                        kind: "doc".to_owned(),
                        members: vec![MemberRecord {
                            bytes: start as u64..end as u64,
                            kind: "doc".to_owned(),
                            cut: 0,
                            symbol: String::new(),
                        }],
                        content_hash: hash,
                        fts_body: fts_body(text),
                    }
                })
                .collect();
            total += records.len() as u64;
            block_on(index.replace_file(
                FileState {
                    path: "docs/guide.md".to_owned(),
                    size: GUIDE.len() as u64,
                    mtime_ns: 0,
                    inode: 0,
                    content_hash: ContentHash::of_file(GUIDE.as_bytes()),
                    ignored: false,
                },
                records,
            ))
            .unwrap();
        }

        let embedded = match vectors {
            Vectors::All => {
                let batch: Vec<_> = pending
                    .iter()
                    .map(|p| (p.content_hash, document_vector(p)))
                    .collect();
                let n = batch.len() as u64;
                block_on(cache.put_many(MODEL, batch)).unwrap();
                n
            }
            Vectors::None => 0,
            Vectors::Candidates { query, count } => {
                let terms = repo_search_core::search::QueryTerms::parse(query);
                let ids =
                    block_on(index.bm25(terms.fts_match().unwrap(), code_filter(), 200)).unwrap();
                let rows = block_on(index.chunks(ids)).unwrap();
                let chosen: Vec<_> = rows
                    .iter()
                    .take(*count)
                    .map(|row| {
                        let chunk = pending
                            .iter()
                            .find(|p| p.content_hash == row.content_hash)
                            .unwrap();
                        (row.content_hash, document_vector(chunk))
                    })
                    .collect();
                let n = chosen.len() as u64;
                block_on(cache.put_many(MODEL, chosen)).unwrap();
                n
            }
        };

        let reconciliation = Reconciliation {
            commit: "abc123".to_owned(),
            changed_files: changed + u64::from(with_docs),
            chunks_total: total,
            chunks_with_vector: embedded.min(total),
            unparsable: Vec::new(),
        };
        let embedder = FakeEmbedder::default();
        let seen = Arc::clone(&embedder.seen);
        let retriever = Retriever::new(
            index.clone(),
            cache.clone(),
            EmbedThread::spawn(embedder).unwrap(),
            PathBuf::from(tree.path()),
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap();
        Self {
            tree,
            _common: common,
            index,
            retriever,
            seen,
            reconciliation,
            cache,
            pending: code_pending,
        }
    }

    fn search(&mut self, request: &Request) -> Results {
        self.try_search(request).unwrap()
    }

    fn try_search(&mut self, request: &Request) -> Result<Results, Error<FakeError>> {
        block_on(
            self.retriever
                .search(request, self.reconciliation.clone(), &Words),
        )
    }
}

fn paths<C>(list: &Option<Vec<repo_search_core::search::FileHit<C>>>) -> Vec<&str> {
    list.iter().flatten().map(|hit| hit.path.as_str()).collect()
}

#[test]
fn hybrid_answers_with_a_code_list_and_a_docs_list() {
    let mut fixture = Fixture::new(&Vectors::All, true);
    let results = fixture.search(&Request::new(
        "where is a value stored into the environment under a key",
    ));

    assert_eq!(paths(&results.code)[0], "src/environment.rs");
    assert_eq!(paths(&results.docs), ["docs/guide.md"]);
    let top = &results.code.as_ref().unwrap()[0].chunks[0];
    assert_eq!(top.channels, [Channel::Bm25, Channel::Vector]);
    assert!(top.text.contains("insert_value"));
    assert_eq!(top.symbol.as_deref(), Some("insert_value"));
    let doc = &results.docs.as_ref().unwrap()[0].chunks[0];
    assert_eq!(doc.heading, "Guide › Values");
    assert!(doc.text.starts_with("## Values"));

    // The code query used the code prompt and the docs query the QA prompt.
    let queries = fixture.retriever_embedder_queries();
    assert_eq!(queries.len(), 2);
    assert!(queries.contains(&QueryTask::CodeRetrieval));
    assert!(queries.contains(&QueryTask::QuestionAnswering));

    // Everything had a vector: nothing was embedded on the spot.
    assert_eq!(fixture.retriever_embedder_documents(), 0);

    let text = results.to_text();
    assert!(text.starts_with("Code:\nsrc/environment.rs\n"), "{text}");
    assert!(text.contains("\nDocs:\ndocs/guide.md\n"), "{text}");
    assert!(
        text.ends_with("-- fresh @ abc123 +5 changed files reconciled; vectors 100%\n"),
        "{text}"
    );
    let json: serde_json::Value = serde_json::from_str(&results.to_json()).unwrap();
    assert!(json["code"][0]["chunks"][0]["channels"].is_array());
    assert_eq!(json["freshness"]["commit"], "abc123");
}

#[test]
fn lexical_mode_runs_no_embedding() {
    let mut fixture = Fixture::new(&Vectors::None, true);
    let results = fixture.search(&Request {
        mode: Mode::Lexical,
        ..Request::new("stack alignment")
    });
    assert_eq!(paths(&results.code), ["src/layout.rs"]);
    let chunk = &results.code.as_ref().unwrap()[0].chunks[0];
    assert_eq!(chunk.channels, [Channel::Bm25]);
    assert!(fixture.retriever_embedder_queries().is_empty());
    assert_eq!(fixture.retriever_embedder_documents(), 0);
}

#[test]
fn semantic_mode_uses_only_the_vector_channel() {
    let mut fixture = Fixture::new(&Vectors::All, true);
    let results = fixture.search(&Request {
        mode: Mode::Semantic,
        ..Request::new("stack alignment children")
    });
    let all_vector = results
        .code
        .iter()
        .flatten()
        .flat_map(|f| &f.chunks)
        .all(|c| c.channels == [Channel::Vector]);
    assert!(all_vector);
    assert_eq!(paths(&results.code)[0], "src/layout.rs");
}

#[test]
fn candidates_without_vectors_are_embedded_on_the_spot_once() {
    let mut fixture = Fixture::new(&Vectors::None, true);
    let request = Request::new("stack alignment of children");

    let first = fixture.search(&request);
    let embedded = fixture.retriever_embedder_documents();
    assert!(embedded > 0, "BM25 candidates must be embedded");
    // Only chunks that matched lexically were embedded.
    assert!(embedded < 8, "{embedded}");
    assert_eq!(paths(&first.code)[0], "src/layout.rs");
    let top = &first.code.as_ref().unwrap()[0].chunks[0];
    assert_eq!(top.channels, [Channel::Bm25, Channel::Vector]);
    assert!(first.freshness.vectors.fraction() > 0.0 && first.freshness.vectors.fraction() < 1.0);

    // A second query finds them loaded and embeds nothing again.
    let again = fixture.search(&request);
    assert_eq!(fixture.retriever_embedder_documents(), embedded);
    assert_eq!(paths(&again.code), paths(&first.code));
}

#[test]
fn chunks_still_without_a_vector_are_found_lexically() {
    let mut fixture = Fixture::new(&Vectors::None, false);
    let results = fixture.search(&Request::new("ComputedSignal"));
    // Neither candidate list is empty although the index held no vector.
    assert!(!results.is_empty());
}

#[test]
fn the_docs_query_is_embedded_only_when_docs_are_in_scope() {
    let mut without_docs = Fixture::new(&Vectors::All, false);
    let results = without_docs.search(&Request::new("stack alignment"));
    assert!(results.docs.is_none());
    assert_eq!(
        without_docs.retriever_embedder_queries(),
        [QueryTask::CodeRetrieval]
    );

    let mut with_docs = Fixture::new(&Vectors::All, true);
    with_docs.search(&Request {
        filters: Filters {
            kinds: vec!["fn".to_owned()],
            ..Filters::default()
        },
        ..Request::new("stack alignment")
    });
    assert_eq!(
        with_docs.retriever_embedder_queries(),
        [QueryTask::CodeRetrieval]
    );
}

#[test]
fn filters_restrict_and_drop_lists() {
    let mut fixture = Fixture::new(&Vectors::All, true);
    let query = "environment value stored under a key stack alignment";

    let docs_only = fixture.search(&Request {
        filters: Filters {
            langs: vec!["markdown".to_owned()],
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert!(docs_only.code.is_none());
    assert_eq!(paths(&docs_only.docs), ["docs/guide.md"]);

    let kind_doc = fixture.search(&Request {
        filters: Filters {
            kinds: vec!["doc".to_owned()],
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert!(kind_doc.code.is_none() && kind_doc.docs.is_some());

    let structs = fixture.search(&Request {
        filters: Filters {
            kinds: vec!["struct".to_owned()],
            ..Filters::default()
        },
        ..Request::new("environment signal")
    });
    assert!(structs.docs.is_none());
    let chunks: Vec<_> = structs
        .code
        .iter()
        .flatten()
        .flat_map(|f| &f.chunks)
        .collect();
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|c| c.text.contains("pub struct")));

    let symbol = fixture.search(&Request {
        filters: Filters {
            symbol: Some("insert_*".to_owned()),
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert!(symbol.docs.is_none());
    assert_eq!(paths(&symbol.code), ["src/environment.rs"]);

    let path = fixture.search(&Request {
        filters: Filters {
            paths: vec!["src/layout.rs".to_owned()],
            langs: vec!["rust".to_owned()],
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert_eq!(paths(&path.code), ["src/layout.rs"]);
    assert!(path.docs.is_none());

    let dir = fixture.search(&Request {
        filters: Filters {
            paths: vec!["docs".to_owned()],
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert!(dir.code.is_none());
    assert_eq!(paths(&dir.docs), ["docs/guide.md"]);
}

#[test]
fn a_definition_ranks_above_a_use_when_the_query_names_the_symbol() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    let results = fixture.search(&Request {
        mode: Mode::Lexical,
        ..Request::new("ComputedSignal")
    });
    let code = results.code.unwrap();
    // The use mentions the name far more often, so it outranks the definition
    // lexically; the definition is placed just ahead of it and takes its
    // score, so scores keep descending down the list.
    assert_eq!(code[0].path, "src/signal.rs");
    assert_eq!(code[1].path, "src/ui.rs");
    assert!(code[0].score >= code[1].score);
}

#[test]
fn plain_words_do_not_promote_definitions() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    // `widgets` is an ordinary word here; the use of ComputedSignal keeps its
    // lexical rank.
    let results = fixture.search(&Request {
        mode: Mode::Lexical,
        ..Request::new("computedsignal widgets")
    });
    assert_eq!(paths(&results.code)[0], "src/ui.rs");
}

#[test]
fn the_budget_limits_output_and_files_only_lists_paths() {
    let mut fixture = Fixture::new(&Vectors::All, true);
    let query = "environment stack alignment value key children signal";
    let full = fixture.search(&Request {
        budget: 100_000,
        ..Request::new(query)
    });
    let chunk_count = |r: &Results| -> usize {
        r.code
            .iter()
            .flatten()
            .map(|f| f.chunks.len())
            .chain(r.docs.iter().flatten().map(|f| f.chunks.len()))
            .sum()
    };
    let small = fixture.search(&Request {
        budget: 60,
        ..Request::new(query)
    });
    assert!(chunk_count(&small) < chunk_count(&full));
    assert!(chunk_count(&small) >= 1);

    let files = fixture.search(&Request {
        budget: 100_000,
        files_only: true,
        ..Request::new(query)
    });
    assert_eq!(chunk_count(&files), 0);
    assert!(paths(&files.code).len() >= 3);
    let text = files.to_text();
    assert!(text.contains("Code:\n"));
    assert!(!text.contains("pub fn"), "{text}");
}

#[test]
fn a_file_changed_after_the_barrier_is_an_error() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    fs::write(
        fixture.tree.path().join("src").join("layout.rs"),
        "// rewritten\n",
    )
    .unwrap();
    let err = fixture
        .try_search(&Request::new("stack alignment"))
        .unwrap_err();
    assert!(matches!(err, Error::Stale { .. }), "{err}");
}

#[test]
fn an_empty_query_and_unknown_filters_are_errors() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    assert!(matches!(
        fixture.try_search(&Request::new("   ")),
        Err(Error::EmptyQuery)
    ));
    let unknown = Request {
        filters: Filters {
            langs: vec!["cobol".to_owned()],
            ..Filters::default()
        },
        ..Request::new("stack")
    };
    assert!(matches!(
        fixture.try_search(&unknown),
        Err(Error::Filter(_))
    ));
}

#[test]
fn an_unknown_kind_is_a_usage_error_not_an_empty_answer() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    let kind = |name: &str| Request {
        filters: Filters {
            kinds: vec![name.to_owned()],
            ..Filters::default()
        },
        ..Request::new("stack alignment")
    };
    let err = fixture.try_search(&kind("nonsense_kind")).unwrap_err();
    assert!(
        matches!(err, Error::Filter(FilterError::UnknownKind(ref k)) if k == "nonsense_kind"),
        "{err}"
    );
    // A short name the index has no chunk of, and a kind it holds, are not.
    assert!(fixture.search(&kind("union")).is_empty());
    assert!(!fixture.search(&kind("function_item")).is_empty());
}

#[test]
fn only_the_units_the_answer_reaches_are_read() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    let query = "stack alignment children";
    let all = fixture.search(&Request {
        budget: 100_000,
        ..Request::new(query)
    });
    let last = all.code.as_ref().unwrap().last().unwrap().path.clone();
    let first = paths(&all.code)[0].to_owned();
    assert_ne!(first, last);

    // The last-ranked file changes: a small budget never reaches its chunk,
    // although the chunk was a candidate; a large one does.
    fs::write(fixture.tree.path().join(&last), "// rewritten\n").unwrap();
    let small = fixture.search(&Request {
        budget: 5,
        ..Request::new(query)
    });
    assert_eq!(paths(&small.code), [first.as_str()]);
    let err = fixture
        .try_search(&Request {
            budget: 100_000,
            ..Request::new(query)
        })
        .unwrap_err();
    assert!(
        matches!(err, Error::Stale { ref path, .. } if *path == last),
        "{err}"
    );

    // A files-only answer reads no text at all.
    let files = fixture.search(&Request {
        files_only: true,
        budget: 100_000,
        ..Request::new(query)
    });
    assert!(paths(&files.code).contains(&last.as_str()));
}

#[test]
fn the_on_the_spot_window_is_the_first_candidates_not_the_first_missing() {
    const FILES: usize = 120;
    let files: Vec<(String, String)> = (0..FILES)
        .map(|i| {
            (
                format!("w{i:03}.rs"),
                format!("pub fn widget_{i:03}() {{\n    let _ = {i};\n}}\n"),
            )
        })
        .collect();
    // Every file matches `widget` equally, so the lexical order is the id
    // order. The first 30 already have a vector.
    let mut fixture = Fixture::build(
        &files,
        &Vectors::Candidates {
            query: "widget",
            count: 30,
        },
        false,
    );
    assert_eq!(fixture.reconciliation.chunks_with_vector, 30);
    let results = fixture.search(&Request::new("widget"));
    // The window is the first ON_THE_SPOT candidates: 30 have a vector and the
    // other 20 are embedded, not the first 50 that lack one.
    assert_eq!(fixture.retriever_embedder_documents(), ON_THE_SPOT - 30);
    let vectors = results.freshness.vectors.fraction();
    assert!(
        (vectors - (30.0 + 20.0) / FILES as f64).abs() < 1e-9,
        "{vectors}"
    );
}

#[test]
fn the_vector_channel_returns_the_best_chunks_within_a_filter() {
    // 260 files whose words are the query's, and one that shares none: the
    // global top 200 holds no chunk of the odd one out.
    let mut files: Vec<(String, String)> = (0..260)
        .map(|i| {
            (
                format!("near{i:03}.rs"),
                format!("pub fn alpha_beta_gamma_{i:03}() {{}}\n"),
            )
        })
        .collect();
    files.push((
        "far.rs".to_owned(),
        "pub fn unrelated_thing() {}\n".to_owned(),
    ));
    let mut fixture = Fixture::build(&files, &Vectors::All, false);
    let query = "alpha beta gamma";

    let unfiltered = fixture.search(&Request {
        mode: Mode::Semantic,
        budget: 100_000,
        ..Request::new(query)
    });
    assert!(!paths(&unfiltered.code).contains(&"src/far.rs"));

    let filtered = fixture.search(&Request {
        mode: Mode::Semantic,
        filters: Filters {
            paths: vec!["src/far.rs".to_owned()],
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert_eq!(paths(&filtered.code), ["src/far.rs"]);
    let chunk = &filtered.code.as_ref().unwrap()[0].chunks[0];
    assert_eq!(chunk.channels, [Channel::Vector]);

    // The same through the symbol and kind filters.
    let by_symbol = fixture.search(&Request {
        mode: Mode::Semantic,
        filters: Filters {
            symbol: Some("unrelated_*".to_owned()),
            ..Filters::default()
        },
        ..Request::new(query)
    });
    assert_eq!(paths(&by_symbol.code), ["src/far.rs"]);
}

#[test]
fn vectors_stored_after_a_query_are_found_by_the_next() {
    let files: Vec<(String, String)> = (0..40)
        .map(|i| {
            (
                format!("w{i:03}.rs"),
                format!("pub fn widget_{i:03}() {{\n    let _ = {i};\n}}\n"),
            )
        })
        .collect();
    let mut fixture = Fixture::build(
        &files,
        &Vectors::Candidates {
            query: "widget",
            count: 30,
        },
        false,
    );
    let all = Request {
        mode: Mode::Semantic,
        budget: 100_000,
        ..Request::new("widget")
    };
    assert_eq!(paths(&fixture.search(&all).code).len(), 30);

    // The background pass embeds the rest: 40 vectors against the 30 the
    // index was calibrated on, so the new ones are found by their slots.
    let batch: Vec<_> = fixture
        .pending
        .iter()
        .map(|p| (p.content_hash, document_vector(p)))
        .collect();
    block_on(fixture.cache.put_many(MODEL, batch)).unwrap();
    assert_eq!(paths(&fixture.search(&all).code).len(), 40);
}

#[test]
fn a_search_dropped_half_way_leaves_the_retriever_usable() {
    let mut fixture = Fixture::new(&Vectors::None, true);
    let request = Request::new("stack alignment of children");
    for polls in 1..6 {
        let reconciliation = fixture.reconciliation.clone();
        let mut search = Box::pin(fixture.retriever.search(&request, reconciliation, &Words));
        let mut cx = Context::from_waker(Waker::noop());
        for _ in 0..polls {
            if let Poll::Ready(done) = search.as_mut().poll(&mut cx) {
                done.unwrap();
                break;
            }
            // Let the worker threads answer before the next poll.
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        drop(search);
        let results = fixture.search(&request);
        assert_eq!(paths(&results.code)[0], "src/layout.rs");
    }
}

#[test]
fn a_query_without_matches_is_empty() {
    let mut fixture = Fixture::new(&Vectors::All, true);
    let results = fixture.search(&Request {
        mode: Mode::Lexical,
        ..Request::new("zzyzx")
    });
    assert!(results.is_empty());
}

#[test]
fn a_content_hash_shared_by_code_and_docs_is_searched_on_both_sides() {
    let mut fixture = Fixture::new(&Vectors::All, false);
    // docs/shared.md holds, byte for byte, the title and text of the
    // layout.rs chunk: one content hash living on both sides. Its vector must
    // be searched for the code list and for the docs list.
    let shared = fixture
        .pending
        .iter()
        .find(|p| p.path == "src/layout.rs")
        .unwrap()
        .clone();
    let docs_dir = fixture.tree.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    fs::write(docs_dir.join("shared.md"), &shared.text).unwrap();
    block_on(fixture.index.replace_file(
        FileState {
            path: "docs/shared.md".to_owned(),
            size: shared.text.len() as u64,
            mtime_ns: 0,
            inode: 0,
            content_hash: ContentHash::of_file(shared.text.as_bytes()),
            ignored: false,
        },
        vec![ChunkRecord {
            lines: 1..=shared.text.lines().count() as u32,
            title: shared.title.clone(),
            kind: "doc".to_owned(),
            members: vec![MemberRecord {
                bytes: 0..shared.text.len() as u64,
                kind: "doc".to_owned(),
                cut: 0,
                symbol: String::new(),
            }],
            content_hash: shared.content_hash,
            fts_body: fts_body(&shared.text),
        }],
    ))
    .unwrap();

    // Semantic mode isolates the vector channel: a lexical hit cannot mask a
    // missing row in the docs index.
    let results = fixture.search(&Request {
        mode: Mode::Semantic,
        budget: 100_000,
        ..Request::new("stack alignment children")
    });
    assert_eq!(paths(&results.docs), ["docs/shared.md"]);
    assert!(paths(&results.code).contains(&"src/layout.rs"));
}

#[test]
fn a_stale_on_the_spot_candidate_stays_lexical_only() {
    let mut fixture = Fixture::new(&Vectors::None, false);
    // A top lexical candidate's file changed after the barrier: its text
    // cannot be read for on-the-spot embedding. The query must still answer
    // and the other candidates still get embedded.
    fs::write(
        fixture.tree.path().join("src").join("layout.rs"),
        "// rewritten\n",
    )
    .unwrap();
    let results = fixture.search(&Request {
        files_only: true,
        ..Request::new("stack environment")
    });
    assert_eq!(
        paths(&results.code),
        ["src/environment.rs", "src/layout.rs"]
    );
    assert!(fixture.retriever_embedder_documents() > 0);
}

#[test]
fn a_configuration_language_counts_as_code() {
    let files = [
        (
            "app.rs".to_owned(),
            rust_item("Reads the port.", "pub fn port() -> u16 {\n    8080\n}"),
        ),
        (
            "settings.toml".to_owned(),
            "[server]\nport = 8080\n".to_owned(),
        ),
    ];
    let mut fixture = Fixture::build(&files, &Vectors::All, false);
    let results = fixture.search(&Request {
        filters: Filters {
            langs: vec!["toml".to_owned()],
            ..Filters::default()
        },
        ..Request::new("server port")
    });
    assert!(results.docs.is_none());
    assert_eq!(paths(&results.code), ["src/settings.toml"]);
}

impl Fixture {
    fn retriever_embedder_queries(&self) -> Vec<QueryTask> {
        self.seen.queries()
    }

    fn retriever_embedder_documents(&self) -> usize {
        self.seen.documents()
    }
}

#[test]
fn alternating_searches_show_ignored_files_only_when_asked() {
    let tree = tempfile::tempdir().unwrap();
    let common = tempfile::tempdir().unwrap();
    fs::write(tree.path().join(".gitignore"), "secret.rs\n").unwrap();
    fs::write(
        tree.path().join("kept.rs"),
        rust_item(
            "Counts the widgets.",
            "pub fn widget_count() -> usize {\n    1\n}",
        ),
    )
    .unwrap();
    fs::write(
        tree.path().join("secret.rs"),
        rust_item(
            "Counts the widgets in secret.",
            "pub fn secret_widgets() -> usize {\n    2\n}",
        ),
    )
    .unwrap();
    let index = block_on(Index::open(tree.path())).unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    let pass = |include_ignored| {
        block_on(reconcile(
            &index,
            &cache,
            MODEL,
            CHUNKER_FINGERPRINT,
            tree.path(),
            Options { include_ignored },
            Arc::new(CapOff),
        ))
        .unwrap()
    };
    let embed = |pending: &[PendingChunk]| {
        let batch = pending
            .iter()
            .map(|p| (p.content_hash, document_vector(p)))
            .collect();
        block_on(cache.put_many(MODEL, batch)).unwrap();
    };
    let embedder = FakeEmbedder::default();
    let seen = Arc::clone(&embedder.seen);
    let mut retriever = Retriever::new(
        index.clone(),
        cache.clone(),
        EmbedThread::spawn(embedder).unwrap(),
        PathBuf::from(tree.path()),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    let mut search = |include_ignored: bool, mode: Mode, files_only: bool| {
        let reconciled = pass(include_ignored);
        assert!(
            block_on(missing(
                &index,
                &cache,
                MODEL,
                tree.path(),
                Options { include_ignored },
                Vec::new()
            ))
            .unwrap()
            .is_empty(),
            "everything is embedded"
        );
        let report = &reconciled.report;
        assert_eq!(report.chunks_total, if include_ignored { 2 } else { 1 });
        assert_eq!(report.chunks_with_vector, report.chunks_total);
        let request = Request {
            mode,
            files_only,
            include_ignored,
            budget: 100_000,
            ..Request::new("widgets count")
        };
        block_on(retriever.search(
            &request,
            Reconciliation::new("abc123", &reconciled.report),
            &Words,
        ))
        .unwrap()
    };

    let _first = pass(true);
    let first_pending = block_on(missing(
        &index,
        &cache,
        MODEL,
        tree.path(),
        Options {
            include_ignored: true,
        },
        Vec::new(),
    ))
    .unwrap();
    assert!(first_pending.iter().any(|p| p.path == "secret.rs"));
    embed(&first_pending);

    for mode in [Mode::Lexical, Mode::Semantic, Mode::Hybrid] {
        for files_only in [false, true] {
            let plain = search(false, mode, files_only);
            assert_eq!(paths(&plain.code), ["kept.rs"], "{mode:?} {files_only}");
            let all = search(true, mode, files_only);
            let mut found = paths(&all.code);
            found.sort_unstable();
            assert_eq!(found, ["kept.rs", "secret.rs"], "{mode:?} {files_only}");
        }
    }
    assert_eq!(
        seen.documents.load(Ordering::SeqCst),
        0,
        "nothing re-embedded"
    );
}

/// A function whose body is over the line-split byte limit, followed by a
/// small sibling. With the cap off, `lazy_rows` line-splits in two and its
/// second part carries a `signature` member (the collapsed multi-line
/// signature plus ` …\n`). With `rows` small enough the second part is
/// under the packing limit and packs with `row_height`.
fn split_fn_source(rows: usize) -> String {
    let mut src = String::from(
        "pub fn lazy_rows(\n    items: &[Item],\n    window: usize,\n) -> Vec<Row> {\n    let mut rows = Vec::new();\n",
    );
    for i in 1..=rows {
        src.push_str(&format!(
            "    rows.push(render_row(items, window, {i:02}));\n"
        ));
    }
    src.push_str("    rows\n}\n\npub fn row_height() -> usize {\n    1\n}\n");
    src
}

/// The indexed text of the chunk whose title contains `title`.
fn indexed_text<'a>(fixture: &'a Fixture, title: &str) -> &'a PendingChunk {
    fixture
        .pending
        .iter()
        .find(|p| p.title.contains(title))
        .unwrap_or_else(|| panic!("no chunk titled {title}"))
}

/// Searches a fully indexed worktree of `src` and checks that the chunk
/// titled `title` reads back as exactly the text that was hashed at index
/// time.
fn assert_reads_back_after_a_full_index(src: String, title: &str) {
    let mut fixture = Fixture::build(&[("lazy.rs".to_owned(), src)], &Vectors::All, false);
    let expected = indexed_text(&fixture, title).clone();
    assert!(
        expected
            .text
            .starts_with("pub fn lazy_rows( items: &[Item], window: usize, ) -> Vec<Row> …\n"),
        "{:?}",
        expected.text
    );
    let results = fixture
        .try_search(&Request::new("render a list of rows lazily"))
        .expect("an unchanged chunk reads back after a full index");
    let chunk = results
        .code
        .iter()
        .flatten()
        .flat_map(|f| &f.chunks)
        .find(|c| c.lines == [*expected.lines.start(), *expected.lines.end()])
        .unwrap_or_else(|| panic!("{title} is not in the answer"));
    assert_eq!(chunk.text, expected.text);
}

#[test]
fn a_split_part_reads_back_after_a_full_index() {
    // `lazy_rows (part 2/2)` alone: members `[signature, text]`.
    assert_reads_back_after_a_full_index(split_fn_source(60), "fn lazy_rows (part 2/2)");
}

#[test]
fn a_packed_split_tail_reads_back_after_a_full_index() {
    // `lazy_rows (part 2/2), fn row_height`: members `[signature, text, text]`.
    assert_reads_back_after_a_full_index(
        split_fn_source(52),
        "fn lazy_rows (part 2/2), fn row_height",
    );
}

#[test]
fn a_split_part_without_a_vector_is_embedded_on_the_spot() {
    let mut fixture = Fixture::build(
        &[("lazy.rs".to_owned(), split_fn_source(60))],
        &Vectors::None,
        false,
    );
    // Both parts of `lazy_rows` match; neither has a vector yet.
    let candidates = fixture
        .pending
        .iter()
        .filter(|p| p.text.contains("rows.push"))
        .count();
    assert_eq!(candidates, 2);
    fixture.search(&Request {
        files_only: true,
        ..Request::new("rows push render")
    });
    assert_eq!(fixture.retriever_embedder_documents(), candidates);
}
