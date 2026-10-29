//! Helpers for the real-model tests (`tests/daemon_model.rs`,
//! `tests/launchd.rs`), included through `#[path]` like `stub.rs` is for
//! `scheduler.rs`: the model-cache layout, the corpus clone and the
//! kill-and-resume sequence pieces. The request/event plumbing is
//! `client.rs`'s, reached here as `super::client`: a `#[path]` includer
//! declares its own `mod client` at the same level, and `mod support`
//! provides `support::client`. Each target uses a subset.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use repo_search::model_store::{ModelSpec, ModelStore, TokenizerSpec, TokenizerStore};
use repo_search_client::client::Responses;
use repo_search_client::protocol::{Event, Final, Hello, Progress, Request, WirePhase};
use repo_search_core::StateDir;
use repo_search_core::store::{Index, VectorCache};

use super::client;

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

/// A `TokenizerStore` rooted at `root` whose cached path is a copy of the
/// archive's `tokenizer.json`, so `load` returns at once.
pub fn tokenizer_store(root: PathBuf) -> TokenizerStore {
    let spec = TokenizerSpec::embedding_gemma_2();
    let store = TokenizerStore::new(root, spec).expect("tokenizer store");
    let path = store.path();
    std::fs::create_dir_all(path.parent().unwrap()).expect("the tokenizer dir");
    std::fs::copy(model_dir().join("tokenizer.json"), &path).expect("the tokenizer copy lands");
    store
}

/// Opens an `index --wait` stream for `worktree` on `socket`.
pub async fn index_wait(socket: &Path, hello: &Hello, worktree: &Path) -> Responses {
    client::stream_request(
        socket,
        hello,
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
        match client::next_event(stream).await {
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
pub async fn committed(
    state: &StateDir,
    worktree: &Path,
    common_dir: &Path,
    model_id: &str,
) -> u64 {
    // The daemon fingerprints the canonical paths it resolved.
    let worktree = worktree.canonicalize().expect("the worktree canonicalizes");
    let common_dir = common_dir
        .canonicalize()
        .expect("the common dir canonicalizes");
    let index = Index::open(state, &worktree).await.expect("index opens");
    let hashes = index.chunk_hashes(false).await.expect("chunk hashes");
    let cache = VectorCache::open(state, &common_dir)
        .await
        .expect("cache opens");
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
pub async fn assert_resume(socket: &Path, hello: &Hello, worktree: &Path, at_least: u64) {
    let events = client::request(
        socket,
        hello,
        Request::Status {
            worktree: worktree.to_path_buf(),
        },
    )
    .await;
    let status = match client::final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::FirstIndex,
        "the kill left the phase at first-index"
    );

    let mut stream = index_wait(socket, hello, worktree).await;
    let mut first: Option<u64> = None;
    loop {
        match client::next_event(&mut stream).await {
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
