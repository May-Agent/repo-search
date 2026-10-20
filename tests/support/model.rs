//! Helpers for the real-model tests (`tests/daemon_model.rs`,
//! `tests/launchd.rs`), included through `#[path]` like `stub.rs` is for
//! `scheduler.rs`: the model-cache layout, the corpus clone, the
//! request/event plumbing and the kill-and-resume sequence pieces. Each
//! target uses a subset.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use futures_util::StreamExt;
use repo_search::client::{Client, Responses};
use repo_search::model_store::{ModelSpec, ModelStore};
use repo_search::protocol::{
    Event, Final, Format, Progress, Request, SearchRequest, WireMode, WirePhase,
};
use repo_search_core::store::{Index, VectorCache};

/// A single event's deadline: a cold model load and a reconcile can take a
/// while, and the tests assert structure, not latency.
pub const TIMEOUT: Duration = Duration::from_secs(180);

/// `REPO_SEARCH_MODEL_DIR`: the extracted model archive.
pub fn model_dir() -> PathBuf {
    let dir = std::env::var_os("REPO_SEARCH_MODEL_DIR")
        .map(PathBuf::from)
        .expect("REPO_SEARCH_MODEL_DIR names the extracted model archive");
    assert!(
        dir.is_dir(),
        "REPO_SEARCH_MODEL_DIR is not a directory: {}",
        dir.display()
    );
    dir
}

/// `REPO_SEARCH_CORPUS`'s first entry: a checkout the tests clone, never
/// write to.
pub fn corpus() -> PathBuf {
    let corpus =
        std::env::var("REPO_SEARCH_CORPUS").expect("REPO_SEARCH_CORPUS names the corpus checkouts");
    let first = corpus.split(':').next().expect("an entry exists");
    let dir = PathBuf::from(first);
    assert!(
        dir.is_dir(),
        "REPO_SEARCH_CORPUS's first entry is not a directory: {first}"
    );
    dir
}

/// Clones the corpus into `tmp` with `git clone --depth 1 --quiet`, so the
/// vector cache starts empty, and returns the worktree root. A shallow
/// clone is all the daemon needs — `git rev-parse` plus the working tree —
/// and the only clone that works when the checkout is itself shallow or a
/// `blob:none` partial clone, whose missing objects make a full
/// `upload-pack` enumeration fail.
pub fn clone_corpus(tmp: &tempfile::TempDir) -> PathBuf {
    let corpus = corpus();
    let worktree = tmp.path().join("corpus");
    let status = Command::new("git")
        .args(["clone", "--depth", "1", "--quiet"])
        .arg(&corpus)
        .arg(&worktree)
        .status()
        .expect("git clone runs");
    assert!(status.success(), "git clone {corpus:?} failed: {status}");
    worktree
}

/// A `ModelStore` rooted at `root` whose `<sha256>/` entry is a symlink to
/// `REPO_SEARCH_MODEL_DIR`, so `ensure` returns at once.
pub fn model_store(root: PathBuf) -> ModelStore {
    let spec = ModelSpec::embedding_gemma_2();
    std::fs::create_dir_all(&root).expect("the model cache root");
    std::os::unix::fs::symlink(model_dir(), root.join(spec.sha256.hex()))
        .expect("the model dir symlink");
    ModelStore::new(root, spec).expect("model store")
}

/// A `search` request for `query` on `worktree`.
pub fn search(worktree: &Path, query: &str) -> Request {
    Request::Search(SearchRequest {
        worktree: worktree.to_path_buf(),
        query: query.to_owned(),
        mode: WireMode::Hybrid,
        paths: vec![],
        langs: vec![],
        kinds: vec![],
        symbol: None,
        budget: 4000,
        files_only: false,
        include_ignored: false,
        format: Format::Json,
    })
}

/// Sends `request` and collects events to the `Final` frame.
pub async fn request(socket: &Path, request: Request) -> Vec<Event> {
    let client = Client::connect(socket)
        .await
        .expect("connect to the daemon");
    let responses = client.request(request).await.expect("request sends");
    tokio::time::timeout(TIMEOUT, responses.collect::<Vec<_>>())
        .await
        .expect("the reply stream does not stall")
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("every frame decodes")
}

/// The events' `Final`.
pub fn final_of(events: &[Event]) -> &Final {
    match events.last() {
        Some(Event::Final(f)) => f,
        other => panic!("the last event is the Final frame, got {other:?}"),
    }
}

/// Sends `request` and returns the live event stream, uncollected.
pub async fn stream_request(socket: &Path, request: Request) -> Responses {
    let client = Client::connect(socket).await.expect("connect");
    client.request(request).await.expect("request sends")
}

/// The next frame on a live stream within `TIMEOUT`.
pub async fn next_event(stream: &mut Responses) -> Event {
    tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("a frame arrives")
        .expect("the stream stays open")
        .expect("the frame decodes")
}

/// Opens an `index --wait` stream for `worktree` on `socket`.
pub async fn index_wait(socket: &Path, worktree: &Path) -> Responses {
    stream_request(
        socket,
        Request::Index {
            worktree: worktree.to_path_buf(),
            wait: true,
        },
    )
    .await
}

/// Reads coverage frames until `min` committed vectors are reported and
/// returns the reported count.
pub async fn await_coverage(stream: &mut Responses, min: u64) -> u64 {
    loop {
        match next_event(stream).await {
            Event::Progress(Progress::Coverage { with_vector, .. }) => {
                if with_vector >= min {
                    return with_vector;
                }
            }
            Event::Final(f) => panic!("the index wait ended early: {f:?}"),
            _ => {}
        }
    }
}

/// The count of `worktree`'s chunks whose `model_id` vectors are committed
/// in the vector cache — what `Coverage` reports, read straight from disk.
pub async fn committed(worktree: &Path, common_dir: &Path, model_id: &str) -> u64 {
    let index = Index::open(worktree).await.expect("index opens");
    let hashes = index.chunk_hashes(false).await.expect("chunk hashes");
    let cache = VectorCache::open(common_dir).await.expect("cache opens");
    let cached = cache.cached(model_id, &hashes).await.expect("cached reads");
    // Count chunk rows, as coverage does: two chunks with identical content
    // share one cached vector, so the distinct count would fall short.
    hashes.iter().filter(|hash| cached.contains(hash)).count() as u64
}

/// The resume assertions shared by `daemon_host_coreml` and the launchd
/// agent, run against the replacement daemon answering on `socket`: the
/// phase the kill left behind is `first_index`, its index stream resumes at
/// `at_least` committed vectors, and coverage grows by another 200. The
/// stream is dropped without waiting for the full index.
pub async fn assert_resume(socket: &Path, worktree: &Path, at_least: u64) {
    let events = request(
        socket,
        Request::Status {
            worktree: worktree.to_path_buf(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::FirstIndex,
        "the kill left the phase at first-index"
    );

    let mut stream = index_wait(socket, worktree).await;
    let mut first: Option<u64> = None;
    loop {
        match next_event(&mut stream).await {
            Event::Progress(Progress::Coverage { with_vector, .. }) => match first {
                None => {
                    assert!(
                        with_vector >= at_least,
                        "the resumed coverage starts at the committed count: \
                         {with_vector} < {at_least}"
                    );
                    first = Some(with_vector);
                }
                Some(_) if with_vector >= at_least + 200 => break,
                Some(_) => {}
            },
            Event::Final(Final::Indexed { with_vector, .. }) => {
                assert!(
                    with_vector >= at_least + 200,
                    "the index ended before coverage grew by 200: {with_vector}"
                );
                break;
            }
            Event::Final(other) => panic!("the resumed index wait failed: {other:?}"),
            _ => {}
        }
    }
}
