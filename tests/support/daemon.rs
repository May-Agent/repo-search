//! The in-process daemon harness: `daemon::run` on a socket with the stub
//! backend and a temp-dir model store, shared by the daemon tests and the
//! fake launchd (which supplies an already-bound listener).
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use repo_search::daemon::{self, Config, Error};
use repo_search::model_store::{ModelSpec, ModelStore, Sha256Digest};
use repo_search::protocol::BuildId;
use sha2::Digest;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;

use super::client;
use super::stub::{Record, StubBackend};

/// The test daemon: a socket in a temp dir, the stub backend's record log
/// and the run task's handle.
pub struct Daemon {
    pub socket: PathBuf,
    /// The store's archive dir: `<models>/<sha>` appears only on a
    /// completed download.
    pub model_dir: PathBuf,
    pub records: mpsc::Receiver<Record>,
    pub backend: Arc<StubBackend>,
    pub join: JoinHandle<Result<(), Error>>,
    _tmp: tempfile::TempDir,
}

/// How the model cache is laid out for the store.
pub enum Model {
    /// `models/<sha>/` exists, so `ensure` returns at once.
    Cached,
    /// The dir is absent and the spec points at `url` serving `body`.
    Served { url: String, sha: Sha256Digest },
}

pub fn digest_of(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::from_hex(&hex::encode(sha2::Sha256::digest(bytes))).expect("hex digest")
}

/// A fixed model id for the cached-model tests.
pub fn sha_7() -> Sha256Digest {
    Sha256Digest::from_hex(&"07".repeat(32)).expect("hex digest")
}

/// The `ModelStore` a `Model` describes, rooted at `root`.
pub fn model_store(root: PathBuf, model: &Model) -> ModelStore {
    let spec = match model {
        Model::Cached => ModelSpec {
            url: "http://127.0.0.1:1/unused".into(),
            sha256: sha_7(),
        },
        Model::Served { url, sha } => ModelSpec {
            url: url.clone(),
            sha256: *sha,
        },
    };
    let store = ModelStore::new(root, spec).expect("model store");
    if matches!(model, Model::Cached) {
        std::fs::create_dir_all(store.dir()).expect("cached model dir");
    }
    store
}

/// Starts the stub daemon on an already-bound `listener` — the caller owns
/// the socket path. `runtime` spawns the run task, so a non-tokio thread
/// (the fake launchd) can start it too.
pub fn spawn_on(
    runtime: &tokio::runtime::Handle,
    listener: std::os::unix::net::UnixListener,
    model_store: ModelStore,
    backend: Arc<StubBackend>,
    idle_exit: Duration,
    build: BuildId,
) -> JoinHandle<Result<(), Error>> {
    listener
        .set_nonblocking(true)
        .expect("the listener goes nonblocking");
    let listener = UnixListener::from_std(listener).expect("the listener adopts");
    runtime.spawn(daemon::run(Config {
        listener,
        idle_exit,
        model_store,
        backend,
        build,
    }))
}

/// Starts a daemon with `backend`; the record log drains into `records`.
/// The daemon's build is the test binary's, matching `client::hello()`.
pub async fn daemon(
    tmp: tempfile::TempDir,
    mut backend: StubBackend,
    model: Model,
    idle_exit: Duration,
) -> Daemon {
    let root = tmp.path().join("models");
    let store = model_store(root, &model);
    let model_dir = store.dir();
    let socket = tmp.path().join("daemon.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).expect("socket binds");
    let (records_tx, records) = mpsc::channel();
    backend.records_to(records_tx);
    let backend = Arc::new(backend);
    let join = spawn_on(
        &tokio::runtime::Handle::current(),
        listener,
        store,
        backend.clone(),
        idle_exit,
        client::build(),
    );
    Daemon {
        socket,
        model_dir,
        records,
        backend,
        join,
        _tmp: tmp,
    }
}
