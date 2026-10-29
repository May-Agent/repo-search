//! The `search` daemon: the embedding pipeline and, behind it, the model
//! loader, the worktree registry and the socket server.

pub mod collection;
pub mod counter;
pub mod coverage;
pub mod feed;
pub mod link;
pub mod listen;
pub mod registry;
pub mod scheduler;
pub mod server;
pub mod spelling;
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
use repo_search_core::StateDir;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio_util::codec::LinesCodecError;

use repo_search_client::protocol::{
    self, BuildId, ErrorKind, ErrorReport, Event, Final, Hello, PROTOCOL, Progress, Request, framed,
};

use collection::CollectionMsg;
use counter::Tokens;
use link::{EmbeddingState, LoaderMsg};
use registry::{GetError, RegistryHandle};
use server::{Endpoint, ServerClient};
use worktree::OwnerMsg;

/// A failure that shuts the daemon down. Delivered to `run` over the fatal
/// channel every component shares.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An embed request to the embedding server failed past its retry,
    /// or the server answered a non-2xx status.
    #[error("embedding server failed: {0}")]
    Server(String),
    /// A writer's `put_many` failed.
    #[error("vector writer failed: {0}")]
    Writer(String),
    /// The listener failed.
    #[error("daemon I/O: {0}")]
    Io(#[from] io::Error),
    /// `run` could not start; the queued connections were answered first.
    #[error("{0}")]
    Startup(#[from] StartupError),
}

/// A failure after socket activation that keeps the daemon from serving.
/// `to_string` is both the log line and the message [`refuse`] sends.
#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    /// launchd handed over a number of `Listener` descriptors other than 1.
    #[error("launchd returned {0} Listener fds, expected exactly 1")]
    ListenerCount(usize),
    /// The running binary could not be identified.
    #[error("could not identify the running binary: {0}")]
    Build(#[source] io::Error),
    /// The tokenizer cache could not be opened.
    #[error("could not open the model cache: {0}")]
    ModelStore(#[source] crate::model_store::Error),
    /// The server client could not be built.
    #[error("could not build the embedding server client: {0}")]
    ServerClient(#[source] reqwest::Error),
    /// The state directory could not be resolved.
    #[error("could not resolve the state directory: {0}")]
    StateDir(#[source] io::Error),
    /// The `Listener` fd could not be adopted into the runtime.
    #[error("could not adopt the Listener fd: {0}")]
    Adopt(#[source] io::Error),
    /// The `--server-token-file` could not be read.
    #[error("could not read the token file {}: {source}", .path.display())]
    TokenFile {
        /// The file the daemon was told to read.
        path: PathBuf,
        /// The read's failure.
        source: io::Error,
    },
    /// The SIGTERM handler could not be installed.
    #[error("could not install the SIGTERM handler: {0}")]
    Signal(#[source] io::Error),
}

/// Answers every connection already queued on `listener` with a
/// [`ErrorKind::Startup`] error, so no client is left waiting on a daemon
/// that is about to exit. `listener` must be nonblocking.
pub async fn refuse(listener: std::os::unix::net::UnixListener, error: &StartupError) {
    answer_queued(listener, ErrorKind::Startup, error.to_string()).await
}

/// Answers every connection already queued on `listener` with a `kind`
/// error carrying `message`, so no client is left waiting on a daemon
/// that is about to exit. `listener` must be nonblocking.
///
/// Each connection's first frame is read (and ignored) before the answer,
/// so the client's `Hello` write cannot hit a closed connection.
pub(crate) async fn answer_queued(
    listener: std::os::unix::net::UnixListener,
    kind: ErrorKind,
    message: String,
) {
    let mut streams = Vec::new();
    loop {
        match listener.accept() {
            Ok((stream, _)) => streams.push(stream),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
            Err(e) => {
                tracing::error!(error = %e, "could not accept a pending connection");
                break;
            }
        }
    }
    // Keeps `read_frame`'s shutdown arm pending for the whole refusal.
    let (_stop, stopping) = tokio::sync::watch::channel(false);
    let mut answers = tokio::task::JoinSet::new();
    for stream in streams {
        let message = message.clone();
        let stopping = stopping.clone();
        answers.spawn(async move {
            let stream = match stream
                .set_nonblocking(true)
                .and_then(|()| UnixStream::from_std(stream))
            {
                Ok(stream) => stream,
                Err(e) => {
                    tracing::error!(error = %e, "could not adopt a pending connection");
                    return;
                }
            };
            let mut framed = framed(stream);
            let _ = read_hello(&mut framed, &stopping).await;
            send_final(&mut framed, kind, message).await;
        });
    }
    let mut answered = 0usize;
    while let Some(done) = answers.join_next().await {
        if done.is_ok() {
            answered += 1;
        }
    }
    tracing::info!(answered, kind = ?kind, "answered pending connections");
}

/// A connection that sends no `Hello` within this long is closed; it also
/// bounds the wait for the frame of a connection the stop answered early.
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);

/// The daemon exits after this long with no connections and no work.
pub const IDLE_EXIT: Duration = Duration::from_secs(600);

/// `launchctl bootout` returns before launchd finishes the teardown; the
/// agent's process exit is awaited this long before `bootstrap`. On Linux
/// it is the old daemon's shutdown tail a replacement waits out.
pub const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// `launchctl bootstrap` returns once the job is loaded; launchd binds the
/// `SockPathName` a moment later, awaited this long. On Linux it is the
/// new daemon's startup after the old one is gone.
pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(10);

/// The bound on one link attempt — the tokenizer download plus the wait
/// for the server to report the model ready: the server's own load bound
/// plus slack for the tokenizer fetch and the probe round trips.
pub const LOAD_BOUND: Duration =
    Duration::from_secs(repo_search_core::embed_api::LOAD_BOUND.as_secs() + 30);

/// A busy-source registration for the idle-exit tracker. Each source owns
/// its `watch::Sender<bool>` — state, not transitions, so nothing is lost —
/// and the aggregator receives the receiver once over this channel.
pub struct BusySource {
    /// The source's current busy state; changes are pushed, never polled.
    pub state: tokio::sync::watch::Receiver<bool>,
}

/// The daemon's configuration.
pub struct Config<T: Tokens, L: listen::Listener> {
    /// The socket to accept on: launchd's [`Activated`](listen::Activated)
    /// in macOS production, a [`Bound`](listen::Bound) the daemon owns on
    /// Linux and in tests.
    pub listener: L,
    /// The SIGTERM stream the accept loop polls. The caller installs it
    /// before it reports readiness, so a signal landing before `run`
    /// polls it is buffered in the stream, never defaulted.
    pub signal: tokio::signal::unix::Signal,
    /// Exit after this long with no connections and no work.
    pub idle_exit: Duration,
    /// Fail the link — tokenizer download plus the server-readiness wait
    /// — past this bound; [`LOAD_BOUND`] in production.
    pub load_bound: Duration,
    /// The tokenizer source the link's `load` starts on at once.
    pub tokens: T,
    /// Where the embedding server listens; `--server-socket` or
    /// `--server-addr` in production.
    pub server: Endpoint,
    /// The model-id vectors are stored under and the server is expected
    /// to list; `MODEL_ID` in production, the stub server's in tests.
    pub model_id: String,
    /// The daemon's state root: the per-worktree indexes and the
    /// per-repository vector caches live under it, never under a
    /// worktree.
    pub state: StateDir,
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
/// The SIGTERM stream arrives on the config, installed by the caller
/// before it reports readiness: a signal landing before `run` first
/// polls it is buffered there, so the shutdown tail still runs.
///
/// Shutdown — on any trigger — closes the listener first (a bound socket
/// is unlinked and its queued connections answered `ShuttingDown`),
/// broadcasts `stopping`, closes the registry (dropping the owners'
/// inboxes and watchers, and the writer handles they hold), tells the
/// scheduler to stop (which answers the embedding threads' pulls with
/// `Stop`), then awaits the writer
/// tasks: with every `WriterHandle` dropped, each writer's close-flush
/// commits the buffered vectors. The connection tasks are awaited last.
/// What [`close`](listen::Listener::close) returns — a `Bound`
/// listener's lifetime lock — is held until `run` returns, so the
/// lock's release stays after every flush; in the production entry
/// points that is the process's exit.
pub async fn run<T: Tokens, L: listen::Listener>(config: Config<T, L>) -> Result<(), Error> {
    let mut signal = config.signal;
    let started = Instant::now();
    let (fatal_tx, mut fatal_rx) = mpsc::channel::<Error>(64);
    let (busy_tx, mut busy_rx) = busy_channel();
    let (stopping_tx, stopping_rx) = tokio::sync::watch::channel(false);
    let (conn_tx, mut conn_rx) = mpsc::channel::<ConnMsg>(64);
    let build = config.build;

    let server = match ServerClient::new(&config.server) {
        Ok(client) => Arc::new(client),
        Err(error) => {
            let error = StartupError::ServerClient(error);
            let _held = config.listener.refuse(&error).await?;
            return Err(error.into());
        }
    };

    let (loader_busy_tx, loader_busy_rx) = tokio::sync::watch::channel(false);
    busy_tx
        .send(BusySource {
            state: loader_busy_rx,
        })
        .await
        .expect("the busy aggregator outlives run");
    let loader = link::spawn(link::LinkConfig {
        tokens: config.tokens,
        server: server.clone(),
        model_id: config.model_id.clone(),
        bound: config.load_bound,
        busy: loader_busy_tx,
        busy_reg: busy_tx.clone(),
        fatal: fatal_tx.clone(),
    });
    let registry = registry::spawn(registry::RegistryConfig {
        state: config.state,
        model_id: config.model_id,
        model: loader.state.clone(),
        started,
        stopping: stopping_rx.clone(),
        busy: busy_tx.clone(),
        fatal: fatal_tx.clone(),
        server,
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
                Ok(stream) => {
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

    // The listener goes first, whatever the trigger: nothing new arrives
    // while the tail works, and a `Bound` socket answers its queued
    // connections `ShuttingDown` and stops existing. What `close` hands
    // back — a `Bound`'s lifetime lock — is held until `run` returns, so
    // the next daemon's acquire still waits on this tail.
    let held = config.listener.close().await;
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
    drop(held);
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
async fn connection<T: Tokens>(
    stream: UnixStream,
    model: tokio::sync::watch::Receiver<EmbeddingState<T>>,
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

async fn serve<T: Tokens>(
    framed: &mut protocol::Connection,
    model: &tokio::sync::watch::Receiver<EmbeddingState<T>>,
    loader: &mpsc::Sender<LoaderMsg>,
    registry: &RegistryHandle,
    stopping: &tokio::sync::watch::Receiver<bool>,
    build: BuildId,
) -> ConnMsg {
    let frame = match read_hello(framed, stopping).await {
        Read::Frame(frame) => frame,
        Read::Gone => return ConnMsg::Closed,
        Read::Stopping => {
            answer_stop(framed, model).await;
            return ConnMsg::Closed;
        }
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
    let frame = match read_frame(framed, stopping).await {
        Read::Frame(frame) => frame,
        Read::Gone => return ConnMsg::Closed,
        Read::Stopping => {
            answer_stop(framed, model).await;
            return ConnMsg::Closed;
        }
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
                _ => unreachable!("matched above"),
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
                    _ => unreachable!("matched above"),
                },
                None => return ConnMsg::Closed,
            }
        }
        Request::UpsertDocuments { .. }
        | Request::DeleteDocuments { .. }
        | Request::SearchCollection(_) => {
            match await_model(framed, model, loader, stopping).await {
                ModelWait::Ready => {}
                ModelWait::Stop => return ConnMsg::Closed,
                ModelWait::Failed => return ConnMsg::Closed,
            }
            let creates = matches!(request, Request::UpsertDocuments { .. });
            let (name, msg) = match request {
                Request::UpsertDocuments {
                    collection,
                    documents,
                } => (
                    collection,
                    CollectionMsg::Upsert {
                        documents,
                        events: events_tx,
                    },
                ),
                Request::DeleteDocuments { collection, ids } => (
                    collection,
                    CollectionMsg::Delete {
                        ids,
                        events: events_tx,
                    },
                ),
                Request::SearchCollection(request) => (
                    request.collection.clone(),
                    CollectionMsg::Search {
                        request,
                        events: events_tx,
                    },
                ),
                _ => unreachable!("matched above"),
            };
            if name.is_empty() {
                send_final(
                    framed,
                    ErrorKind::BadRequest,
                    "a collection name is empty".into(),
                )
                .await;
                return ConnMsg::Closed;
            }
            match get_collection(registry, framed, name, creates, stopping).await {
                Some(handle) => handle.dispatch(msg).await,
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
async fn await_model<T: Tokens>(
    framed: &mut protocol::Connection,
    model: &tokio::sync::watch::Receiver<EmbeddingState<T>>,
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
            EmbeddingState::Ready(_) => Snap::Ready,
            EmbeddingState::Absent | EmbeddingState::Loading => Snap::Send(Progress::ModelLoading),
            EmbeddingState::Downloading { received, total } => {
                Snap::Send(Progress::ModelDownload {
                    received: *received,
                    total: *total,
                })
            }
            EmbeddingState::Failed(message) => Snap::Failed(message.clone()),
            EmbeddingState::Fatal(message) => Snap::Fatal(message.clone()),
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
                let (kind, message) = stop_report(&model);
                send_final(framed, kind, message).await;
                return ModelWait::Stop;
            }
        }
    }
}

/// The error a connection is answered with when the daemon stops under
/// it. A published fatal failure outranks the generic shutdown error:
/// `Fatal` precedes the fatal channel, which precedes `stopping`, so it is
/// always already there.
fn stop_report<T: Tokens>(
    model: &tokio::sync::watch::Receiver<EmbeddingState<T>>,
) -> (ErrorKind, String) {
    match &*model.borrow() {
        EmbeddingState::Fatal(message) => (ErrorKind::Model, message.clone()),
        _ => (
            ErrorKind::ShuttingDown,
            "the daemon is shutting down".into(),
        ),
    }
}

/// Answers a connection the stop caught before its `Hello` or `Request`
/// arrived, then waits within [`HELLO_TIMEOUT`] for that frame or the
/// close: the client writes it regardless, and closing first would turn
/// the write into `EPIPE` before the answer is read.
async fn answer_stop<T: Tokens>(
    framed: &mut protocol::Connection,
    model: &tokio::sync::watch::Receiver<EmbeddingState<T>>,
) {
    let (kind, message) = stop_report(model);
    send_final(framed, kind, message).await;
    let _ = tokio::time::timeout(HELLO_TIMEOUT, framed.next()).await;
}

fn error_too_large() -> Event {
    Event::Final(Final::Error(ErrorReport {
        kind: ErrorKind::TooLarge,
        message: "the answer exceeds the maximum frame".into(),
    }))
}

/// Resolves a request's worktree path in the connection task — `git
/// rev-parse` plus canonicalization to the on-disk spelling and metadata,
/// concurrently per connection and off the registry actor.
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
    // Canonicalizing to the on-disk spelling is blocking file I/O; it runs
    // off the executor as `tokio::fs::canonicalize` did.
    let canonical = |path: &Path| {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || spelling::canonical(&path))
    };
    let (requested, toplevel, common_dir, git_dir) = tokio::join!(
        canonical(worktree),
        canonical(&toplevel),
        canonical(&common_dir),
        canonical(&git_dir),
    );
    let path = |result: Result<io::Result<PathBuf>, tokio::task::JoinError>| {
        result
            .map_err(|e| GetError::Internal(e.to_string()))
            .and_then(|r| r.map_err(|e| GetError::Internal(e.to_string())))
    };
    let requested = requested
        .map_err(|e| GetError::Internal(e.to_string()))?
        .map_err(|_| GetError::NotAWorktree(worktree.to_path_buf()))?;
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

async fn get_collection(
    registry: &RegistryHandle,
    framed: &mut protocol::Connection,
    name: String,
    create: bool,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> Option<registry::CollectionHandle> {
    let mut stopping = stopping.clone();
    tokio::select! {
        owner = registry.collection(name, create) => match owner {
            Some(Ok(handle)) => Some(handle),
            Some(Err(error @ GetError::NoSuchCollection(_))) => {
                send_final(framed, ErrorKind::NoSuchCollection, error.to_string()).await;
                None
            }
            Some(Err(GetError::Internal(message))) => {
                send_final(framed, ErrorKind::Internal, message).await;
                None
            }
            Some(Err(GetError::NotAWorktree(_))) => unreachable!("a collection is no worktree"),
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
        Err(GetError::NoSuchCollection(_)) => unreachable!("a worktree is no collection"),
    };
    let mut stopping = stopping.clone();
    tokio::select! {
        owner = registry.get(resolved) => match owner {
            Some(Ok(handle)) => Some(handle),
            Some(Err(GetError::NotAWorktree(_))) => {
                send_final(framed, ErrorKind::NotAWorktree, format!("{} is not a worktree root", worktree.display())).await;
                None
            }
            Some(Err(GetError::NoSuchCollection(_))) => unreachable!("a worktree is no collection"),
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

/// A frame read's outcome.
enum Read {
    /// The peer's frame.
    Frame(String),
    /// Closed, failed, too large (already answered) or past its deadline:
    /// nothing left to answer.
    Gone,
    /// The daemon began stopping before the frame arrived; the connection
    /// is still owed an answer.
    Stopping,
}

/// Reads the first frame within [`HELLO_TIMEOUT`]: a client that never
/// sends `Hello` must not hold the daemon alive.
async fn read_hello(
    framed: &mut protocol::Connection,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> Read {
    tokio::time::timeout(HELLO_TIMEOUT, read_frame(framed, stopping))
        .await
        .unwrap_or(Read::Gone)
}

/// Reads one frame, unless the peer goes first or the daemon stops.
async fn read_frame(
    framed: &mut protocol::Connection,
    stopping: &tokio::sync::watch::Receiver<bool>,
) -> Read {
    let mut stopping = stopping.clone();
    tokio::select! {
        frame = framed.next() => match frame {
            Some(Ok(line)) => Read::Frame(line),
            Some(Err(LinesCodecError::MaxLineLengthExceeded)) => {
                send_final(framed, ErrorKind::TooLarge, "frame exceeds the maximum length".into()).await;
                Read::Gone
            }
            _ => Read::Gone,
        },
        _ = stopping.changed() => Read::Stopping,
    }
}

async fn send_final(framed: &mut protocol::Connection, kind: ErrorKind, message: String) {
    let event = Event::Final(Final::Error(ErrorReport { kind, message }));
    let _ = framed.send(line(&event)).await;
}
