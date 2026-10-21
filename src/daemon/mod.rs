//! The `search` daemon: the embedding pipeline and, behind it, the model
//! loader, the worktree registry and the socket server.

pub mod ane;
pub mod coreml;
pub mod coverage;
pub mod feed;
pub mod gpu;
pub mod model;
pub mod registry;
pub mod scheduler;
pub mod watch;
pub mod worktree;
pub mod writer;

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::SinkExt;
use futures_util::stream::StreamExt;
use repo_search_core::{BatchEmbedder, Embedder, TokenCounter};
use repo_search_coreml::GpuBlocked;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot};
use tokio_util::codec::LinesCodecError;

use crate::model_store::ModelStore;
use crate::protocol::{
    self, BuildId, ErrorKind, ErrorReport, Event, Final, Hello, PROTOCOL, Progress, Request, framed,
};

use model::{LoaderMsg, ModelState};
use registry::{GetError, RegistryHandle};
use worktree::OwnerMsg;

/// Host headroom for the GPU embedder: `Ok(())` when the GPU instance may run.
pub type Headroom = Result<(), GpuBlocked>;

/// The counter for chunking and queue ordering, with the model's bucket
/// lengths in ascending order.
pub type CounterLoad<C, E> = Result<(C, Box<[usize]>), E>;

/// The model backend behind the daemon's embedding pipeline.
///
/// The seam that lets the scheduler-level tests run every component against a
/// stub embedder. Production is [`coreml::CoreMlBackend`].
pub trait Backend: Send + Sync + 'static {
    /// The single-chunk instance on the ANE thread, shared by queries and
    /// indexing.
    type Ane: Embedder + Send + 'static;
    /// The batch instance on the GPU thread; serves stage-1 indexing only.
    type Gpu: BatchEmbedder + Send + 'static;
    /// Counts the model's tokens for chunking and queue ordering.
    type Counter: TokenCounter + 'static;
    /// The backend's model-load failure.
    type LoadError: std::error::Error + Send + Sync + 'static;
    /// The headroom monitor's guard; dropping it stops the monitor.
    type Monitor: Send + 'static;

    /// The counter for chunking and queue ordering, and the model's bucket
    /// lengths, ascending.
    ///
    /// # Errors
    ///
    /// When the model directory cannot be read.
    fn load_counter(&self, dir: &Path) -> CounterLoad<Self::Counter, Self::LoadError>;

    /// Opens the ANE instance; called on the ANE thread, which owns the
    /// returned embedder. Implementations warm every bucket before returning
    /// so the first query does not pay a compile. `stage` is already `Open`
    /// on entry; an implementation sends [`LoadStage::WarmUp`] once the
    /// model is open and the warm-up embeds begin, so a load that outlives
    /// the bound names the stage it stopped in.
    ///
    /// # Errors
    ///
    /// When the model cannot be loaded.
    fn open_ane(
        &self,
        dir: &Path,
        model_id: &str,
        stage: &tokio::sync::watch::Sender<LoadStage>,
    ) -> Result<Self::Ane, Self::LoadError>;

    /// Opens the GPU instance; called on the GPU thread, which owns the
    /// returned embedder.
    ///
    /// # Errors
    ///
    /// [`GpuOpen::Blocked`] when the host has no GPU headroom — the headroom
    /// may have changed between the check and the open — and
    /// [`GpuOpen::Failed`] for every other load failure, which is fatal.
    fn open_gpu(&self, dir: &Path, model_id: &str) -> Result<Self::Gpu, GpuOpen<Self::LoadError>>;

    /// Calls `on_change` once with the current headroom, then on every
    /// power-source or thermal change.
    fn watch_headroom(&self, on_change: Box<dyn Fn(Headroom) + Send + Sync>) -> Self::Monitor;
}

/// The outcome of opening the GPU instance.
#[derive(Debug)]
pub enum GpuOpen<E> {
    /// The host refuses the GPU instance right now: not fatal, retry later.
    Blocked(GpuBlocked),
    /// The model could not be loaded: fatal.
    Failed(E),
}

/// A failure that shuts the daemon down. Delivered to `run` over the fatal
/// channel every component shares.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An [`Embedder`] or [`BatchEmbedder`] call, or an `open_gpu`, failed.
    /// There is no retry and no fallback to the other instance.
    #[error("embedder failed: {0}")]
    Embedder(String),
    /// A writer's `put_many` failed.
    #[error("vector writer failed: {0}")]
    Writer(String),
    /// The listener or the signal handler failed.
    #[error("daemon I/O: {0}")]
    Io(#[from] io::Error),
}

/// A connection that sends no `Hello` within this long is closed.
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);

/// The daemon exits after this long with no connections and no work.
pub const IDLE_EXIT: Duration = Duration::from_secs(600);

/// The model load — counter, open and warm-up — must report within this
/// long. A cache-hit open takes seconds; a miss pays the system's ANE
/// compile, which is unbounded under contention and was observed past
/// 180 s. The bound is generous enough for a cold compile on a slow host
/// while still failing a wedged load instead of waiting on it forever.
pub const LOAD_BOUND: Duration = Duration::from_secs(300);

/// Where the ANE thread's load is, reported over `AneThread`'s `stage`
/// watch so a load that outlives [`LOAD_BOUND`] names the stage it stopped
/// in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStage {
    /// `load_counter` is reading the tokenizer and the manifest.
    Counter,
    /// `open_ane` is inside the compiled-model open — the ANE compile when
    /// the system's compiled-model cache misses.
    Open,
    /// The model is open; the per-bucket warm-up embeds are running.
    WarmUp,
}

impl std::fmt::Display for LoadStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Counter => "the counter load",
            Self::Open => "the model open",
            Self::WarmUp => "the bucket warm-up",
        })
    }
}

/// A busy-source registration for the idle-exit tracker. Each source owns
/// its `watch::Sender<bool>` — state, not transitions, so nothing is lost —
/// and the aggregator receives the receiver once over this channel.
pub struct BusySource {
    /// The source's current busy state; changes are pushed, never polled.
    pub state: tokio::sync::watch::Receiver<bool>,
}

/// The daemon's configuration.
pub struct Config<B: Backend> {
    /// The socket to accept on; launchd's in production, a bound one in
    /// tests.
    pub listener: UnixListener,
    /// Exit after this long with no connections and no work.
    pub idle_exit: Duration,
    /// Fail the model load past this bound, naming the stage it stopped
    /// in; [`LOAD_BOUND`] in production.
    pub load_bound: Duration,
    /// The model cache; the loader starts `ensure` at once.
    pub model_store: ModelStore,
    /// The embedding backend.
    pub backend: Arc<B>,
    /// The identity of the executable file this process runs; a `Hello`
    /// naming another build is a `ProtocolMismatch`.
    pub build: BuildId,
}

/// What a connection task tells the accept loop.
enum ConnMsg {
    /// The connection closed.
    Closed,
    /// A protocol-mismatch `Hello`: shut down so the next connection starts
    /// the binary that is on disk.
    Mismatch,
}

/// Runs the daemon until idle exit, SIGTERM or a fatal error.
///
/// Shutdown — on any trigger — stops accepting, broadcasts `stopping`,
/// closes the registry (dropping the owners' inboxes and watchers, and the
/// writer handles they hold), tells the scheduler to stop (which answers
/// the embedding threads' pulls with `Stop`), then awaits the writer
/// tasks: with every `WriterHandle` dropped, each writer's close-flush
/// commits the buffered vectors. The connection tasks are awaited last.
pub async fn run<B: Backend>(config: Config<B>) -> Result<(), Error> {
    // The signal stream exists before anything is spawned: a failure here
    // leaves nothing to shut down.
    let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let started = Instant::now();
    let (fatal_tx, mut fatal_rx) = mpsc::channel::<Error>(64);
    let (busy_tx, mut busy_rx) = busy_channel();
    let (stopping_tx, stopping_rx) = tokio::sync::watch::channel(false);
    let (conn_tx, mut conn_rx) = mpsc::channel::<ConnMsg>(64);
    let backend = config.backend;
    let build = config.build;
    let model_id = config.model_store.model_id();

    // Headroom monitor: the callback may run on any thread.
    let (headroom_tx, headroom_rx) =
        tokio::sync::watch::channel(Err(GpuBlocked::PowerSourceUnknown));
    let _monitor = backend.watch_headroom(Box::new(move |h| {
        let _ = headroom_tx.send_replace(h);
    }));

    let (loader_busy_tx, loader_busy_rx) = tokio::sync::watch::channel(false);
    busy_tx
        .send(BusySource {
            state: loader_busy_rx,
        })
        .await
        .expect("the busy aggregator outlives run");
    let loader = model::spawn(
        backend,
        Arc::new(config.model_store),
        config.load_bound,
        headroom_rx,
        loader_busy_tx,
        busy_tx.clone(),
        fatal_tx.clone(),
    );
    let registry = registry::spawn(registry::RegistryConfig {
        model_id,
        model: loader.state.clone(),
        started,
        stopping: stopping_rx.clone(),
        busy: busy_tx.clone(),
        fatal: fatal_tx.clone(),
        marker: std::marker::PhantomData,
    });

    let mut conns = 0u64;
    let mut conn_tasks = tokio::task::JoinSet::new();
    // The deadline ticks only while the daemon is idle: no connections and
    // nothing busy.
    let mut deadline: Option<Instant> = Some(Instant::now() + config.idle_exit);
    let mut outcome: Result<(), Error> = Ok(());
    'accept: loop {
        let sleep = async {
            match deadline {
                Some(at) => tokio::time::sleep_until(at.into()).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            accepted = config.listener.accept() => match accepted {
                Ok((stream, _)) => {
                    conns += 1;
                    conn_tasks.spawn(connection(
                        stream,
                        loader.state.clone(),
                        loader.tx.clone(),
                        registry.clone(),
                        stopping_rx.clone(),
                        conn_tx.clone(),
                        build,
                    ));
                }
                // Every accept error is fatal but goes through the
                // shutdown tail: the writers still flush.
                Err(error) => {
                    outcome = Err(error.into());
                    break 'accept;
                }
            },
            msg = conn_rx.recv() => match msg {
                Some(ConnMsg::Closed) => {
                    conns = conns.checked_sub(1).expect("a close without an open")
                }
                Some(ConnMsg::Mismatch) | None => break 'accept,
            },
            changed = busy_rx.changed() => {
                if changed.is_err() {
                    break 'accept;
                }
            }
            _ = signal.recv() => break 'accept,
            _ = sleep => {
                if conns == 0 && !*busy_rx.borrow() {
                    break 'accept;
                }
            }
            fatal = fatal_rx.recv() => {
                outcome = Err(fatal.expect("run holds a fatal sender"));
                break 'accept;
            }
        }
        deadline = if conns == 0 && !*busy_rx.borrow() {
            Some(Instant::now() + config.idle_exit)
        } else {
            None
        };
    }

    // Shutdown: watchers die with their owners, the threads pull `Stop`,
    // and the writers flush once every handle is dropped. Connections end
    // quickly once owners answer `ShuttingDown` and exit; they are awaited
    // so no task is cancelled mid-write.
    let _ = stopping_tx.send_replace(true);
    let writers = registry.shutdown().await;
    let (reply, rx) = oneshot::channel();
    let _ = loader.tx.send(LoaderMsg::Shutdown { reply }).await;
    let _ = rx.await;
    for task in writers {
        let _ = task.await;
    }
    // A finishing connection must not block on a close report nobody reads.
    drop(conn_rx);
    while conn_tasks.join_next().await.is_some() {}
    outcome
}

/// One wire message as a frame line: compact JSON, always serializable.
fn line(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("wire types serialize")
}

/// The busy aggregator: one task folding every source's state `watch` —
/// merged in a `SelectAll` — into the `watch` the accept loop reads.
/// Sources report state, so a latest value can never be dropped.
fn busy_channel() -> (mpsc::Sender<BusySource>, tokio::sync::watch::Receiver<bool>) {
    use futures_util::stream::{SelectAll, Stream};

    enum State {
        Live(tokio::sync::watch::Receiver<bool>),
        Dead,
    }

    fn stream(
        id: u64,
        rx: tokio::sync::watch::Receiver<bool>,
    ) -> impl Stream<Item = (u64, Option<bool>)> {
        futures_util::stream::unfold((id, State::Live(rx)), |(id, state)| async move {
            match state {
                State::Live(mut rx) => match rx.changed().await {
                    Ok(()) => {
                        let state = *rx.borrow();
                        Some(((id, Some(state)), (id, State::Live(rx))))
                    }
                    // The source is gone; report it once, then end.
                    Err(_) => Some(((id, None), (id, State::Dead))),
                },
                State::Dead => None,
            }
        })
    }

    let (tx, mut rx) = mpsc::channel::<BusySource>(64);
    let (busy_tx, busy_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        // Sources are keyed by registration, not by source id: a retried
        // load registers the scheduler again, and the old stream's death
        // must not remove the new entry.
        let mut sources: HashMap<u64, bool> = HashMap::new();
        let mut streams = SelectAll::new();
        let mut next = 0u64;
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    Some(BusySource { state }) => {
                        next += 1;
                        sources.insert(next, *state.borrow());
                        streams.push(stream(next, state).boxed());
                    }
                    None if streams.is_empty() => return,
                    None => {}
                },
                event = streams.next(), if !streams.is_empty() => match event {
                    Some((id, Some(state))) => {
                        sources.insert(id, state);
                    }
                    Some((id, None)) => {
                        sources.remove(&id);
                    }
                    None => {}
                },
            }
            let any = sources.values().any(|b| *b);
            let _ = busy_tx.send_if_modified(|current| {
                if *current != any {
                    *current = any;
                    true
                } else {
                    false
                }
            });
        }
    });
    (tx, busy_rx)
}

/// One connection: the handshake, the one request, then events until the
/// `Final` frame.
async fn connection<B: Backend>(
    stream: UnixStream,
    model: tokio::sync::watch::Receiver<ModelState<B>>,
    loader: mpsc::Sender<LoaderMsg>,
    registry: RegistryHandle,
    stopping: tokio::sync::watch::Receiver<bool>,
    conn: mpsc::Sender<ConnMsg>,
    build: BuildId,
) {
    let mut framed = framed(stream);
    let end = serve(&mut framed, &model, &loader, &registry, &stopping, build).await;
    let _ = conn.send(end).await;
}

async fn serve<B: Backend>(
    framed: &mut protocol::Connection,
    model: &tokio::sync::watch::Receiver<ModelState<B>>,
    loader: &mpsc::Sender<LoaderMsg>,
    registry: &RegistryHandle,
    stopping: &tokio::sync::watch::Receiver<bool>,
    build: BuildId,
) -> ConnMsg {
    // A client that never sends `Hello` must not hold the daemon alive.
    let frame = match tokio::time::timeout(HELLO_TIMEOUT, read_frame(framed, stopping)).await {
        Ok(Some(frame)) => frame,
        _ => return ConnMsg::Closed,
    };
    let hello = match serde_json::from_str::<Hello>(&frame) {
        Ok(hello) => hello,
        Err(_) => {
            send_final(
                framed,
                ErrorKind::BadRequest,
                "expected a Hello frame".into(),
            )
            .await;
            return ConnMsg::Closed;
        }
    };
    let ours = Hello::new(build);
    if hello != ours {
        send_final(
            framed,
            ErrorKind::ProtocolMismatch,
            format!(
                "client is protocol {} {} build {}, daemon is {PROTOCOL} {} build {}",
                hello.protocol, hello.version, hello.build, ours.version, ours.build
            ),
        )
        .await;
        return ConnMsg::Mismatch;
    }
    if framed.send(line(&ours)).await.is_err() {
        return ConnMsg::Closed;
    }
    let Some(frame) = read_frame(framed, stopping).await else {
        return ConnMsg::Closed;
    };
    let request = match serde_json::from_str::<Request>(&frame) {
        Ok(request) => request,
        Err(_) => {
            send_final(
                framed,
                ErrorKind::BadRequest,
                "expected a Request frame".into(),
            )
            .await;
            return ConnMsg::Closed;
        }
    };

    // The request's own event stream once it reaches an owner; before that
    // the connection writes progress itself.
    let (events_tx, mut events_rx) = mpsc::channel::<Event>(64);
    let owner_msg = match &request {
        Request::Search(_) | Request::Index { .. } => {
            match await_model(framed, model, loader, stopping).await {
                ModelWait::Ready => {}
                ModelWait::Stop => return ConnMsg::Closed,
                ModelWait::Failed => return ConnMsg::Closed,
            }
            let worktree = match &request {
                Request::Search(search) => &search.worktree,
                Request::Index { worktree, .. } => worktree,
                Request::Status { .. } => unreachable!(),
            };
            match get_owner(registry, framed, worktree.clone(), stopping).await {
                Some(handle) => match request {
                    Request::Search(request) => {
                        handle
                            .dispatch(OwnerMsg::Search {
                                request,
                                events: events_tx,
                            })
                            .await
                    }
                    Request::Index { wait, .. } => {
                        handle
                            .dispatch(OwnerMsg::Index {
                                wait,
                                events: events_tx,
                            })
                            .await
                    }
                    Request::Status { .. } => unreachable!(),
                },
                None => return ConnMsg::Closed,
            }
        }
        Request::Status { worktree } => {
            match get_owner(registry, framed, worktree.clone(), stopping).await {
                Some(handle) => {
                    handle
                        .dispatch(OwnerMsg::Status { events: events_tx })
                        .await
                }
                None => return ConnMsg::Closed,
            }
        }
    };
    if owner_msg.is_err() {
        send_final(
            framed,
            ErrorKind::ShuttingDown,
            "the daemon is shutting down".into(),
        )
        .await;
        return ConnMsg::Closed;
    }

    // Forward events until the Final. The protocol sends nothing after the
    // `Request`, so any frame or EOF from the client means it is gone: drop
    // `events_rx` and let the owner's `events.closed()` fire.
    loop {
        tokio::select! {
            event = events_rx.recv() => {
                let Some(event) = event else {
                    // The owner went away without answering.
                    send_final(
                        framed,
                        ErrorKind::ShuttingDown,
                        "the daemon is shutting down".into(),
                    )
                    .await;
                    break;
                };
                let last = matches!(event, Event::Final(_));
                let mut frame_line = line(&event);
                if frame_line.len() > protocol::MAX_FRAME {
                    frame_line = line(&error_too_large());
                }
                if framed.send(frame_line).await.is_err() {
                    break;
                }
                if last {
                    break;
                }
            }
            frame = framed.next() => {
                if matches!(frame, Some(Err(LinesCodecError::MaxLineLengthExceeded))) {
                    send_final(framed, ErrorKind::TooLarge, "frame exceeds the maximum length".into()).await;
                }
                break;
            }
        }
    }
    ConnMsg::Closed
}

enum ModelWait {
    Ready,
    Stop,
    Failed,
}

/// Waits out the model load, streaming progress, sending the loader a
/// `Retry` once on the first `Failed`.
async fn await_model<B: Backend>(
    framed: &mut protocol::Connection,
    model: &tokio::sync::watch::Receiver<ModelState<B>>,
    loader: &mpsc::Sender<LoaderMsg>,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> ModelWait {
    let mut model = model.clone();
    let mut stopping = stopping.clone();
    let mut retried = false;
    loop {
        enum Snap {
            Ready,
            Send(Progress),
            Failed(String),
            Fatal(String),
        }
        let snap = match &*model.borrow_and_update() {
            ModelState::Ready(_) => Snap::Ready,
            ModelState::Absent | ModelState::Loading => Snap::Send(Progress::ModelLoading),
            ModelState::Downloading { received, total } => Snap::Send(Progress::ModelDownload {
                received: *received,
                total: *total,
            }),
            ModelState::Failed(message) => Snap::Failed(message.clone()),
            ModelState::Fatal(message) => Snap::Fatal(message.clone()),
        };
        match snap {
            Snap::Ready => return ModelWait::Ready,
            Snap::Send(progress) => {
                if framed.send(line(&Event::Progress(progress))).await.is_err() {
                    return ModelWait::Stop;
                }
            }
            Snap::Failed(message) => {
                if retried {
                    send_final(framed, ErrorKind::Model, message).await;
                    return ModelWait::Failed;
                }
                retried = true;
                let _ = loader.send(LoaderMsg::Retry).await;
            }
            // A fatal failure is never retried: the load error is the
            // answer, and `run` is already on its way out.
            Snap::Fatal(message) => {
                send_final(framed, ErrorKind::Model, message).await;
                return ModelWait::Failed;
            }
        }
        tokio::select! {
            changed = model.changed() => if changed.is_err() { return ModelWait::Stop },
            _ = framed.next() => return ModelWait::Stop,
            _ = stopping.changed() => {
                // A published fatal failure outranks the generic shutdown
                // error: `Fatal` precedes the fatal channel, which precedes
                // `stopping`, so it is always already there.
                let message = match &*model.borrow() {
                    ModelState::Fatal(message) => Some(message.clone()),
                    _ => None,
                };
                match message {
                    Some(message) => send_final(framed, ErrorKind::Model, message).await,
                    None => {
                        send_final(framed, ErrorKind::ShuttingDown, "the daemon is shutting down".into())
                            .await
                    }
                }
                return ModelWait::Stop;
            }
        }
    }
}

fn error_too_large() -> Event {
    Event::Final(Final::Error(ErrorReport {
        kind: ErrorKind::TooLarge,
        message: "the answer exceeds the maximum frame".into(),
    }))
}

/// Resolves a request's worktree path in the connection task — `git
/// rev-parse` plus canonicalization and metadata, concurrently per
/// connection and off the registry actor.
async fn resolve_worktree(worktree: &Path) -> Result<registry::ResolvedWorktree, GetError> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args([
            "rev-parse",
            "--path-format=absolute",
            "--show-toplevel",
            "--git-common-dir",
            "--absolute-git-dir",
        ])
        .output()
        .await
        .map_err(|e| GetError::Internal(e.to_string()))?;
    if !output.status.success() {
        return Err(GetError::NotAWorktree(worktree.to_path_buf()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let mut line = |what: &str| -> Result<PathBuf, GetError> {
        lines
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| GetError::Internal(format!("git rev-parse printed no {what}")))
    };
    let toplevel = line("toplevel")?;
    let common_dir = line("git-common-dir")?;
    let git_dir = line("git-dir")?;
    let (requested, toplevel, common_dir, git_dir) = tokio::join!(
        tokio::fs::canonicalize(worktree),
        tokio::fs::canonicalize(toplevel),
        tokio::fs::canonicalize(common_dir),
        tokio::fs::canonicalize(git_dir),
    );
    let requested = requested.map_err(|_| GetError::NotAWorktree(worktree.to_path_buf()))?;
    let path = |result: io::Result<PathBuf>| result.map_err(|e| GetError::Internal(e.to_string()));
    let toplevel = path(toplevel)?;
    let common_dir = path(common_dir)?;
    let git_dir = path(git_dir)?;
    use std::os::unix::fs::MetadataExt;
    let meta = tokio::fs::metadata(&toplevel)
        .await
        .map_err(|e| GetError::Internal(e.to_string()))?;
    Ok(registry::ResolvedWorktree {
        requested,
        toplevel,
        common_dir,
        git_dir,
        root_id: (meta.dev(), meta.ino()),
    })
}

async fn get_owner(
    registry: &RegistryHandle,
    framed: &mut protocol::Connection,
    worktree: PathBuf,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> Option<registry::OwnerHandle> {
    let resolved = match resolve_worktree(&worktree).await {
        Ok(resolved) => resolved,
        Err(GetError::NotAWorktree(_)) => {
            send_final(
                framed,
                ErrorKind::NotAWorktree,
                format!("{} is not a worktree root", worktree.display()),
            )
            .await;
            return None;
        }
        Err(GetError::Internal(message)) => {
            send_final(framed, ErrorKind::Internal, message).await;
            return None;
        }
    };
    let mut stopping = stopping.clone();
    tokio::select! {
        owner = registry.get(resolved) => match owner {
            Some(Ok(handle)) => Some(handle),
            Some(Err(GetError::NotAWorktree(_))) => {
                send_final(framed, ErrorKind::NotAWorktree, format!("{} is not a worktree root", worktree.display())).await;
                None
            }
            Some(Err(GetError::Internal(message))) => {
                send_final(framed, ErrorKind::Internal, message).await;
                None
            }
            None => {
                send_final(framed, ErrorKind::ShuttingDown, "the daemon is shutting down".into()).await;
                None
            }
        },
        _ = stopping.changed() => {
            send_final(framed, ErrorKind::ShuttingDown, "the daemon is shutting down".into()).await;
            None
        }
    }
}

/// Reads one frame, or `None` on close, error or shutdown.
async fn read_frame(
    framed: &mut protocol::Connection,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> Option<String> {
    let mut stopping = stopping.clone();
    tokio::select! {
        frame = framed.next() => match frame {
            Some(Ok(line)) => Some(line),
            Some(Err(LinesCodecError::MaxLineLengthExceeded)) => {
                send_final(framed, ErrorKind::TooLarge, "frame exceeds the maximum length".into()).await;
                None
            }
            _ => None,
        },
        _ = stopping.changed() => None,
    }
}

async fn send_final(framed: &mut protocol::Connection, kind: ErrorKind, message: String) {
    let event = Event::Final(Final::Error(ErrorReport { kind, message }));
    let _ = framed.send(line(&event)).await;
}
