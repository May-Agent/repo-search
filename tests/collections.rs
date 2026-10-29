//! Document collections through the daemon and the library client: a
//! collection is filled by upserting `(id, title, text)` documents and
//! searched with the worktree pipeline's own chunking, vectors, keyword
//! index and fusion. The daemon runs in process over the stub embedding
//! server, as in `daemon.rs`; vectors there are seeded by text, not
//! meaning, so every ranking assertion rests on what the keyword channel
//! alone decides (a document no other shares a query word with).

#![cfg(target_os = "macos")]

#[allow(dead_code)] // each test file uses its own slice of the fixtures
mod support;

use std::sync::mpsc;
use std::time::Duration;

use repo_search::daemon;
use repo_search_client::client::{Client, ClientError};
use repo_search_client::protocol::{
    Answer, CollectionReport, CollectionSearch, Document, ErrorKind, Final, Format, WireMode,
};
use repo_search_core::StateDir;
use repo_search_server::stub::StubBackend;
use support::client::{final_of, hello, request, search};
use support::daemon::{Daemon, Model, daemon};
use support::repo::{self, TestRepo};
use support::tokens::StubTokens;

async fn start() -> Daemon {
    let tmp = tempfile::tempdir().expect("tempdir");
    daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Cached,
        StubTokens::default(),
        Duration::from_secs(600),
        daemon::LOAD_BOUND,
    )
    .await
}

fn document(id: &str, title: &str, text: &str) -> Document {
    Document {
        id: id.to_owned(),
        title: title.to_owned(),
        text: text.to_owned(),
    }
}

async fn client(d: &Daemon) -> Client {
    Client::connect(&d.socket, &hello())
        .await
        .expect("connect to the daemon")
}

async fn upsert(d: &Daemon, collection: &str, documents: Vec<Document>) -> CollectionReport {
    client(d)
        .await
        .upsert_documents(collection, documents)
        .await
        .expect("the upsert is answered")
}

async fn delete(d: &Daemon, collection: &str, ids: &[&str]) -> CollectionReport {
    client(d)
        .await
        .delete_documents(collection, ids.iter().map(|id| (*id).to_owned()).collect())
        .await
        .expect("the delete is answered")
}

fn query(collection: &str, text: &str, mode: WireMode) -> CollectionSearch {
    CollectionSearch {
        collection: collection.to_owned(),
        query: text.to_owned(),
        mode,
        budget: 4000,
        ids_only: false,
        format: Format::Json,
    }
}

async fn ask(d: &Daemon, collection: &str, text: &str, mode: WireMode) -> Answer {
    client(d)
        .await
        .search_collection(query(collection, text, mode))
        .await
        .expect("the search is answered")
}

/// The ids of an answer's documents, best first.
fn ids(answer: &Answer) -> Vec<String> {
    let json: serde_json::Value = serde_json::from_str(&answer.output).expect("JSON output");
    json.get("docs")
        .and_then(|docs| docs.as_array())
        .map(|docs| {
            docs.iter()
                .map(|doc| doc["path"].as_str().expect("a path").to_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Chunks the daemon sent to the embedding server since the last call.
fn embeds(d: &Daemon) -> usize {
    d.records
        .try_iter()
        .filter(|record| record.hash.is_some())
        .count()
}

/// The keyword statistics and row counts of a collection's index, read
/// straight from its database: `(bm25 rows, title tokens, body tokens,
/// chunk rows, fts rows)`.
fn keyword_statistics(state: &StateDir, collection: &str) -> (i64, i64, i64, i64, i64) {
    let conn = rusqlite::Connection::open_with_flags(
        state.collection_index(collection),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("the collection's index opens");
    let stats = conn
        .query_row(
            "SELECT rows, title_tokens, body_tokens FROM bm25_stats",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("the statistics row");
    let count = |table: &str| -> i64 {
        conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .expect("a count")
    };
    (
        stats.0,
        stats.1,
        stats.2,
        count("chunks"),
        count("fts_docsize"),
    )
}

fn corpus() -> Vec<Document> {
    vec![
        document(
            "kb:password",
            "Resetting a forgotten password",
            "## Steps\n\nOpen the account page and choose the reset link to receive a new \
             password by email.\n\n## Limits\n\nA reset link expires after one hour.",
        ),
        document(
            "kb:invoices",
            "Downloading invoices",
            "## Billing\n\nEvery invoice is available as a PDF from the billing tab.",
        ),
        document(
            "kb:storage",
            "Storage quotas",
            "## Quotas\n\nEach workspace includes fifty gigabytes of storage.",
        ),
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upserted_documents_are_returned_ranked_for_a_natural_language_query() {
    let d = start().await;
    let report = upsert(&d, "support", corpus()).await;
    assert_eq!(report.changed, 3);
    assert_eq!(report.unchanged, 0);
    assert!(report.chunks_total >= 3, "every document is chunked");
    assert_eq!(report.chunks_with_vector, report.chunks_total);
    assert_eq!(
        report.embedded, report.chunks_total,
        "every new chunk is embedded once"
    );

    let answer = ask(
        &d,
        "support",
        "how do I reset my forgotten password?",
        WireMode::Hybrid,
    )
    .await;
    assert!(answer.has_results);
    let found = ids(&answer);
    assert_eq!(
        found.first().map(String::as_str),
        Some("kb:password"),
        "the document that answers the question ranks first: {found:?}"
    );
    let json: serde_json::Value = serde_json::from_str(&answer.output).unwrap();
    let chunk = &json["docs"][0]["chunks"][0];
    assert!(
        chunk["heading"]
            .as_str()
            .unwrap()
            .starts_with("Resetting a forgotten password"),
        "a chunk's heading path starts with the document's title: {chunk}"
    );
    assert!(chunk["text"].as_str().unwrap().contains("reset link"));

    let ids_only = client(&d)
        .await
        .search_collection(CollectionSearch {
            ids_only: true,
            ..query("support", "forgotten password", WireMode::Lexical)
        })
        .await
        .expect("an ids-only search is answered");
    assert_eq!(ids(&ids_only), ["kb:password"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unchanged_document_embeds_nothing_an_update_replaces_chunks_and_a_delete_leaves_none() {
    let d = start().await;
    let first = upsert(&d, "support", corpus()).await;
    assert_eq!(embeds(&d), first.chunks_total as usize);
    let (rows, _, _, chunk_rows, fts_rows) = keyword_statistics(&d.state, "support");
    assert_eq!(rows as u64, first.chunks_total);
    assert_eq!((chunk_rows, fts_rows), (rows, rows));

    // The same documents again: nothing is chunked or embedded.
    let again = upsert(&d, "support", corpus()).await;
    assert_eq!((again.changed, again.unchanged, again.embedded), (0, 3, 0));
    assert_eq!(again.chunks_total, first.chunks_total);
    assert_eq!(embeds(&d), 0, "an unchanged upsert reaches the embedder");
    assert_eq!(keyword_statistics(&d.state, "support").0, rows);

    // An update replaces the document's chunks and only those.
    let mut documents = corpus();
    documents[1] = document(
        "kb:invoices",
        "Downloading invoices",
        "## Receipts\n\nEvery receipt can be exported as a spreadsheet.",
    );
    let updated = upsert(&d, "support", documents).await;
    assert_eq!((updated.changed, updated.unchanged), (1, 2));
    assert_eq!(updated.embedded, 1, "the replaced section is embedded");
    assert_eq!(embeds(&d), 1);
    assert_eq!(updated.chunks_with_vector, updated.chunks_total);
    let old = ask(&d, "support", "billing tab invoice", WireMode::Lexical).await;
    assert!(!old.has_results, "the old text no longer matches");
    let new = ask(&d, "support", "receipt spreadsheet", WireMode::Lexical).await;
    assert_eq!(ids(&new), ["kb:invoices"]);
    let (rows, _, _, chunk_rows, fts_rows) = keyword_statistics(&d.state, "support");
    assert_eq!(rows as u64, updated.chunks_total);
    assert_eq!((chunk_rows, fts_rows), (rows, rows));

    // A deleted document no longer matches and leaves no keyword
    // statistics.
    let removed = delete(&d, "support", &["kb:password", "kb:missing"]).await;
    assert_eq!(
        (removed.changed, removed.unchanged, removed.embedded),
        (1, 1, 0)
    );
    assert_eq!(embeds(&d), 0);
    let gone = ask(&d, "support", "forgotten password reset", WireMode::Lexical).await;
    assert!(!gone.has_results);
    let (rows, _, _, chunk_rows, fts_rows) = keyword_statistics(&d.state, "support");
    assert_eq!(rows as u64, removed.chunks_total);
    assert_eq!((chunk_rows, fts_rows), (rows, rows));

    let rest = delete(&d, "support", &["kb:invoices", "kb:storage"]).await;
    assert_eq!((rest.changed, rest.chunks_total), (2, 0));
    assert_eq!(
        keyword_statistics(&d.state, "support"),
        (0, 0, 0, 0, 0),
        "an empty collection holds no keyword statistics"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn collections_and_worktree_indexes_never_return_each_others_documents() {
    let d = start().await;
    let repo: TestRepo = repo::repo(&[
        (
            "README.md".to_owned(),
            "# Gadget\n\n## Overview\n\nThe gadget calibrates the zeppelin rudder.".to_owned(),
        ),
        (
            "src/lib.rs".to_owned(),
            repo::rust_fn("calibrate_rudder", 500),
        ),
    ]);
    // The same id in both collections, with different documents.
    upsert(
        &d,
        "alpha",
        vec![
            document(
                "shared",
                "Alpha note",
                "The alpha archive keeps wombat records.",
            ),
            document("a-only", "Alpha extra", "Alpha stores marmalade recipes."),
        ],
    )
    .await;
    upsert(
        &d,
        "beta",
        vec![document(
            "shared",
            "Beta note",
            "The beta archive keeps narwhal records.",
        )],
    )
    .await;

    for (collection, own, foreign) in [
        ("alpha", "wombat", ["narwhal", "zeppelin"]),
        ("beta", "narwhal", ["wombat", "zeppelin"]),
    ] {
        let hit = ask(&d, collection, own, WireMode::Lexical).await;
        assert_eq!(ids(&hit), ["shared"], "{collection} finds its own {own}");
        for word in foreign {
            let miss = ask(&d, collection, word, WireMode::Lexical).await;
            assert!(!miss.has_results, "{collection} does not know {word}");
        }
        // The vector channel returns neighbours whatever the query; they
        // are still this collection's.
        let hybrid = ask(&d, collection, "records archive", WireMode::Hybrid).await;
        let json: serde_json::Value = serde_json::from_str(&hybrid.output).unwrap();
        for found in ids(&hybrid) {
            assert!(["shared", "a-only"].contains(&found.as_str()), "{found}");
        }
        let text = json.to_string().to_lowercase();
        let other = if collection == "alpha" {
            "narwhal"
        } else {
            "wombat"
        };
        assert!(
            !text.contains(other),
            "{collection} returned {other}: {text}"
        );
    }

    // The worktree sees none of the collections' words, and the
    // collections none of the worktree's.
    for word in ["wombat", "narwhal", "marmalade"] {
        let events = request(&d.socket, &hello(), lexical(&repo, word)).await;
        match final_of(&events) {
            Final::Answer(answer) => assert!(!answer.has_results, "the worktree knows {word}"),
            other => panic!("expected an Answer, got {other:?}"),
        }
    }
    let events = request(&d.socket, &hello(), lexical(&repo, "zeppelin")).await;
    match final_of(&events) {
        Final::Answer(answer) => assert!(answer.has_results, "the worktree finds its own word"),
        other => panic!("expected an Answer, got {other:?}"),
    }
    let events = request(&d.socket, &hello(), search(&repo.root, "records archive")).await;
    match final_of(&events) {
        Final::Answer(answer) => {
            let output = answer.output.to_lowercase();
            for word in ["wombat", "narwhal", "shared", "a-only"] {
                assert!(!output.contains(word), "the worktree returned {word}");
            }
        }
        other => panic!("expected an Answer, got {other:?}"),
    }
    // Writing to a collection left the worktree's tree alone.
    assert!(
        repo::clean(&repo.root),
        "the daemon wrote under the worktree"
    );
}

fn lexical(repo: &TestRepo, word: &str) -> repo_search_client::protocol::Request {
    let mut request = search(&repo.root, word);
    let repo_search_client::protocol::Request::Search(inner) = &mut request else {
        unreachable!("search builds a Search request")
    };
    inner.mode = WireMode::Lexical;
    request
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_malformed_write_is_refused() {
    let d = start().await;
    for (documents, what) in [
        (vec![document("", "t", "x")], "an empty id"),
        (
            vec![document("a", "t", "x"), document("a", "u", "y")],
            "a repeated id",
        ),
    ] {
        match client(&d)
            .await
            .upsert_documents("support", documents)
            .await
        {
            Err(ClientError::Refused(report)) => {
                assert_eq!(report.kind, ErrorKind::BadRequest, "{what}")
            }
            other => panic!("{what} should be refused, got {other:?}"),
        }
    }
    match client(&d).await.upsert_documents("", vec![]).await {
        Err(ClientError::Refused(report)) => assert_eq!(report.kind, ErrorKind::BadRequest),
        other => panic!("an empty collection name should be refused, got {other:?}"),
    }
    // Nothing was written.
    let answer = ask(&d, "support", "anything", WireMode::Lexical).await;
    assert!(!answer.has_results);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn searching_a_never_written_collection_fails_and_creates_nothing() {
    let d = start().await;
    let missing = d.state.collection("ghost");
    match client(&d)
        .await
        .search_collection(query("ghost", "anything", WireMode::Lexical))
        .await
    {
        Err(ClientError::Refused(report)) => {
            assert_eq!(report.kind, ErrorKind::NoSuchCollection);
            assert!(report.message.contains("ghost"), "{}", report.message);
        }
        other => panic!("searching a missing collection should be refused, got {other:?}"),
    }
    // Deleting from it fails the same way.
    match client(&d)
        .await
        .delete_documents("ghost", vec!["a".to_owned()])
        .await
    {
        Err(ClientError::Refused(report)) => {
            assert_eq!(report.kind, ErrorKind::NoSuchCollection)
        }
        other => panic!("deleting from a missing collection should be refused, got {other:?}"),
    }
    assert!(!missing.exists(), "a read or delete created {missing:?}");
    // A second refusal shows the first left nothing in the registry.
    assert!(
        client(&d)
            .await
            .search_collection(query("ghost", "anything", WireMode::Lexical))
            .await
            .is_err()
    );

    upsert(&d, "ghost", vec![document("a", "Title", "anchovy pizza")]).await;
    assert!(missing.exists());
    let answer = ask(&d, "ghost", "anchovy", WireMode::Lexical).await;
    assert_eq!(ids(&answer), vec!["a".to_owned()]);
}
