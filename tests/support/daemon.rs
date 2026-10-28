//! The in-process daemon harness: `repo_search_server::spawn` with the
//! stub backend on a `server.sock`, then `daemon::run` against it — the
//! same pair production runs under launchd — shared by the daemon tests
//! and the fake launchd (which supplies the already-bound listeners).
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use repo_search::daemon::counter::Tokens;
use repo_search::daemon::server::{Embeddings, Endpoint, ServerClient};
use repo_search::daemon::{self, Config, Error};
use repo_search::model_store::{ModelSpec, ModelStore, Sha256Digest};
use repo_search::protocol::BuildId;
use repo_search_core::StateDir;
use repo_search_core::embed_api::Urgency;
use repo_search_server::chip::ChipClass;
use repo_search_server::loader::ModelState;
use repo_search_server::stub::{Record, StubBackend};
use sha2::Digest;
use tokio::net::UnixListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::client;

/// The test daemon and server: sockets in a temp dir, the stub backend's
/// record log and the two run tasks' handles.
pub struct Daemon {
    pub socket: PathBuf,
    /// The socket the daemon reaches the server over.
    pub server_socket: PathBuf,
    /// The daemon's state root, `tmp/state`: indexes and caches live here.
    pub state: StateDir,
    /// The server store's archive dir: `<models>/<sha>` appears only on a
    /// completed download.
    pub model_dir: PathBuf,
    pub records: mpsc::Receiver<Record>,
    pub backend: Arc<StubBackend>,
    pub join: JoinHandle<Result<(), Error>>,
    pub server: JoinHandle<Result<(), repo_search_server::Error>>,
    /// The server's model state: once `Ready`, its scheduler's queue
    /// depth tells a test which requests reached the server.
    pub server_state: watch::Receiver<ModelState<StubBackend>>,
    _tmp: tempfile::TempDir,
}

impl Daemon {
    /// Waits until the server's scheduler holds exactly `items` queued
    /// items — waiting for hardware, not on it. The daemon's requests
    /// race each other to the server, so a test that needs the whole
    /// window queued before the ANE or GPU pulls waits here first.
    pub async fn await_server_queued(&self, items: u64) {
        let mut queued = self.server_model().await.scheduler.queued();
        tokio::time::timeout(client::TIMEOUT, async {
            while *queued.borrow_and_update() != items {
                queued
                    .changed()
                    .await
                    .expect("the server's scheduler lives");
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the server's queue reaches {items} items"));
    }

    /// The server's model once its loader publishes `Ready`.
    async fn server_model(&self) -> Arc<repo_search_server::loader::Model<StubBackend>> {
        let mut state = self.server_state.clone();
        tokio::time::timeout(client::TIMEOUT, async {
            loop {
                if let ModelState::Ready(model) = &*state.borrow_and_update() {
                    return model.clone();
                }
                state.changed().await.expect("the server's loader lives");
            }
        })
        .await
        .expect("the server's model becomes ready")
    }

    /// Sends one embedding request of the test's own straight to the
    /// server at `urgency` once its model is ready, through the daemon's
    /// production client: a filler a gated instance holds, so the
    /// daemon's requests queue behind it in a known order. The returned
    /// task resolves once the filler is answered.
    pub async fn filler(&self, urgency: u8) -> JoinHandle<()> {
        self.server_model().await;
        let client = ServerClient::new(&Endpoint::Unix(self.server_socket.clone()))
            .expect("the filler's client builds");
        tokio::spawn(async move {
            client
                .embed(vec!["filler".to_owned()], Urgency::new(urgency))
                .await
                .expect("the filler is answered");
        })
    }
}

/// How the server's model cache is laid out for its store.
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

/// Everything but the socket a stub daemon starts with.
pub struct Setup<T: Tokens> {
    /// The tokenizer source the daemon's link loads.
    pub tokens: T,
    /// Where the daemon reaches the embedding server.
    pub server: Endpoint,
    /// The model-id the server's store reports — the daemon's probe
    /// expects it.
    pub model_id: String,
    /// The daemon's state root.
    pub state: StateDir,
    /// Idle exit.
    pub idle_exit: Duration,
    /// The link bound.
    pub load_bound: Duration,
    /// The daemon's build identity.
    pub build: BuildId,
}

/// Everything but the socket a stub server starts with.
pub struct ServerSetup {
    /// The model cache the server's loader downloads from.
    pub model_store: ModelStore,
    /// The chip class the server's scheduler reads; `Wide` keeps the GPU
    /// on stage-1 work.
    pub chip: ChipClass,
    /// The stub embedder.
    pub backend: Arc<StubBackend>,
    /// Idle exit.
    pub idle_exit: Duration,
    /// The model-load bound.
    pub load_bound: Duration,
}

/// Starts the stub daemon on an already-bound `listener` — the caller owns
/// the socket path. `runtime` spawns the run task, so a non-tokio thread
/// (the fake launchd) can start it too; `from_std` registers the fd with
/// the reactor, so it runs inside the spawned task.
pub fn spawn_on<T: Tokens>(
    runtime: &tokio::runtime::Handle,
    listener: std::os::unix::net::UnixListener,
    setup: Setup<T>,
) -> JoinHandle<Result<(), Error>> {
    let Setup {
        tokens,
        server,
        model_id,
        state,
        idle_exit,
        load_bound,
        build,
    } = setup;
    runtime.spawn(async move {
        listener
            .set_nonblocking(true)
            .expect("the listener goes nonblocking");
        let listener = UnixListener::from_std(listener).expect("the listener adopts");
        daemon::run(Config {
            listener,
            idle_exit,
            load_bound,
            tokens,
            server,
            model_id,
            state,
            build,
        })
        .await
    })
}

/// Starts the stub server on an already-bound `listener`, as
/// [`spawn_on`] does for the daemon.
pub fn spawn_server_on(
    runtime: &tokio::runtime::Handle,
    listener: std::os::unix::net::UnixListener,
    setup: ServerSetup,
) -> JoinHandle<Result<(), repo_search_server::Error>> {
    runtime.spawn(async move { repo_search_server::run(server_config(listener, setup)).await })
}

/// The unix-only, tokenless server `Config` a `ServerSetup` describes.
/// It adopts `listener` into the runtime, so it runs inside one.
fn server_config(
    listener: std::os::unix::net::UnixListener,
    setup: ServerSetup,
) -> repo_search_server::Config<StubBackend, ModelStore> {
    let ServerSetup {
        model_store,
        chip,
        backend,
        idle_exit,
        load_bound,
    } = setup;
    listener
        .set_nonblocking(true)
        .expect("the listener goes nonblocking");
    repo_search_server::Config {
        unix: UnixListener::from_std(listener).expect("the listener adopts"),
        tcp: Vec::new(),
        token: None,
        idle_exit,
        load_bound,
        model: model_store,
        chip,
        backend,
    }
}

/// Starts a server with `backend` and `model` on `<tmp>/server.sock`,
/// then the daemon against it; the record log drains into `records`.
/// The daemon's build is the test binary's, matching `client::hello()`.
pub async fn daemon<T: Tokens>(
    tmp: tempfile::TempDir,
    mut backend: StubBackend,
    model: Model,
    tokens: T,
    idle_exit: Duration,
    load_bound: Duration,
) -> Daemon {
    let store = model_store(tmp.path().join("models"), &model);
    let model_dir = store.dir();
    let state = StateDir::new(tmp.path().join("state"));
    let socket = tmp.path().join("daemon.sock");
    let server_socket = tmp.path().join("server.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).expect("socket binds");
    let server_listener =
        std::os::unix::net::UnixListener::bind(&server_socket).expect("server socket binds");
    let (records_tx, records) = mpsc::channel();
    backend.records_to(records_tx);
    let backend = Arc::new(backend);
    let server = repo_search_server::spawn(server_config(
        server_listener,
        ServerSetup {
            model_store: store,
            chip: ChipClass::Wide,
            backend: backend.clone(),
            // `idle_exit` is the daemon's: nothing relaunches this server
            // as launchd does, so it keeps the production idle exit,
            // which `server/tests/api.rs` covers.
            idle_exit: repo_search_server::IDLE_EXIT,
            load_bound: repo_search_server::LOAD_BOUND,
        },
    ))
    .await
    .expect("the stub server starts");
    let runtime = tokio::runtime::Handle::current();
    let join = spawn_on(
        &runtime,
        listener,
        Setup {
            tokens,
            server: Endpoint::Unix(server_socket.clone()),
            // `/v1/models` always lists `MODEL_ID`.
            model_id: repo_search_core::model::MODEL_ID.to_owned(),
            state: state.clone(),
            idle_exit,
            load_bound,
            build: client::build(),
        },
    );
    Daemon {
        socket,
        server_socket,
        state,
        model_dir,
        records,
        backend,
        join,
        server: server.task,
        server_state: server.state,
        _tmp: tmp,
    }
}
