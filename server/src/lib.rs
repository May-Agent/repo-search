//! The repo-search embedding server: the model, its Core ML instances and
//! the hardware scheduling that used to live in the daemon, served behind
//! an OpenAI-compatible HTTP API.
#![cfg(target_os = "macos")]

use std::io;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::backend::{Backend, LoadStage};
use crate::chip::ChipClass;

pub mod ane;
pub mod backend;
pub mod chip;
pub mod coreml;
pub mod gpu;
pub mod http;
pub mod loader;
pub mod scheduler;
#[cfg(feature = "stub")]
pub mod stub;

pub use loader::{Loader, LoaderConfig, LoaderMsg, Model, ModelSource, ModelState};
pub use scheduler::{GpuState, SchedulerHandle};

/// Exit after this long with no request in flight, nothing queued and the
/// loader not busy.
pub const IDLE_EXIT: Duration = Duration::from_secs(600);

/// The cap on the model load; past it the load fails fatally with the
/// stage it stopped in.
pub const LOAD_BOUND: Duration = Duration::from_secs(300);

/// The request body cap: 32 MiB.
pub const BODY_LIMIT: usize = 32 << 20;

/// A failure that shuts the server down. Delivered to `run` over the
/// fatal channel every component shares.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An `embed` or `embed_batch` call, or an `open_gpu`, failed. There
    /// is no retry and no fallback to the other instance.
    #[error("embedder failed: {0}")]
    Embedder(String),
    /// The listener or the signal handler failed.
    #[error("server I/O: {0}")]
    Io(#[from] io::Error),
    /// A listener's configuration is refused at startup.
    #[error("{0}")]
    Listener(String),
}

/// A busy-source registration for the idle-exit tracker. Each source owns
/// its `watch::Sender<bool>` — state, not transitions, so nothing is lost
/// — and the aggregator receives the receiver once over this channel.
pub struct BusySource {
    /// The source's current busy state; changes are pushed, never polled.
    pub state: watch::Receiver<bool>,
}

/// What the server is built from.
pub struct Config<B: Backend, M: ModelSource> {
    /// The unix socket the daemon talks over; launchd's in production, a
    /// bound one in tests.
    pub unix: tokio::net::UnixListener,
    /// Loopback TCP listeners; empty when the server is unix-only.
    pub tcp: Vec<tokio::net::TcpListener>,
    /// The bearer token TCP listeners require; `None` means no auth.
    /// The unix socket needs none — its 0600 permissions are the check.
    pub token: Option<String>,
    /// Exit after this long idle.
    pub idle_exit: Duration,
    /// Fail the model load past this bound, naming the stage it stopped
    /// in; [`LOAD_BOUND`] in production.
    pub load_bound: Duration,
    /// The model source the loader downloads from.
    pub model: M,
    /// The host's chip class, read once at startup: on a base chip the GPU
    /// is kept off normal work.
    pub chip: ChipClass,
    /// The embedding backend.
    pub backend: Arc<B>,
}

/// A listener `refuse_http` drains: the std listeners socket activation
/// handed over, or one already adopted into tokio and handed back.
pub enum Listener {
    /// A unix-socket listener.
    Unix(std::os::unix::net::UnixListener),
    /// A TCP listener.
    Tcp(std::net::TcpListener),
}

/// How long a refused connection may take to send its request head
/// before the server closes it: the refusal exists so queued clients
/// are answered, not so a failed server waits on them.
const REFUSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Answers every connection already queued on `listeners` with a 503 and
/// the error envelope, so no client is left waiting on a server that is
/// about to exit. Listeners must be nonblocking.
pub async fn refuse_http(listeners: Vec<Listener>, error: &str) {
    let body = http::error_json("server_error", error, Some("server_start_failed"));
    let mut tasks = tokio::task::JoinSet::new();
    for listener in listeners {
        let body = body.clone();
        tasks.spawn(async move {
            match listener {
                Listener::Unix(listener) => refuse_loop(listener, body).await,
                Listener::Tcp(listener) => refuse_loop(listener, body).await,
            }
        });
    }
    tasks.join_all().await;
}

/// The unix/TCP `accept`, unified over the raw and adopted stream types.
trait Accept {
    /// The accepted std stream.
    type Raw;
    /// `Raw` adopted into the runtime: the stream hyper serves.
    type Stream: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static;
    /// The next queued connection; `WouldBlock` ends the drain.
    fn accept(&self) -> io::Result<Self::Raw>;
    /// Adopts `raw` into the runtime: nonblocking, then `from_std`.
    fn from_std(raw: Self::Raw) -> io::Result<Self::Stream>;
}

impl Accept for std::os::unix::net::UnixListener {
    type Raw = std::os::unix::net::UnixStream;
    type Stream = tokio::net::UnixStream;
    fn accept(&self) -> io::Result<Self::Raw> {
        self.accept().map(|(stream, _)| stream)
    }
    fn from_std(raw: Self::Raw) -> io::Result<Self::Stream> {
        raw.set_nonblocking(true)?;
        tokio::net::UnixStream::from_std(raw)
    }
}

impl Accept for std::net::TcpListener {
    type Raw = std::net::TcpStream;
    type Stream = tokio::net::TcpStream;
    fn accept(&self) -> io::Result<Self::Raw> {
        self.accept().map(|(stream, _)| stream)
    }
    fn from_std(raw: Self::Raw) -> io::Result<Self::Stream> {
        raw.set_nonblocking(true)?;
        tokio::net::TcpStream::from_std(raw)
    }
}

/// Drains `listener`'s queue — `WouldBlock` ends it — and serves each
/// connection one bounded 503. A per-connection failure is logged and
/// skipped; only an empty queue or an `accept` failure ends the drain,
/// so one bad fd never leaves later queued clients unanswered.
async fn refuse_loop<L: Accept>(listener: L, body: String) {
    let mut conns = tokio::task::JoinSet::new();
    loop {
        let raw = match listener.accept() {
            Ok(raw) => raw,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            // A per-connection failure — one fd's — never ends the
            // drain: every later queued client is still answered.
            Err(error) => {
                tracing::error!(%error, "could not accept a pending connection");
                continue;
            }
        };
        let stream = match L::from_std(raw) {
            Ok(stream) => stream,
            Err(error) => {
                tracing::error!(%error, "could not adopt a pending connection");
                continue;
            }
        };
        let body = body.clone();
        conns.spawn(async move { serve_refusal(stream, body).await });
    }
    conns.join_all().await;
}

/// Serves `stream` one 503 `connection: close` reply through hyper. The
/// header timeout bounds the wait on a client that connects and sends
/// nothing, so a failed server still exits.
async fn serve_refusal<S>(stream: S, body: String)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let service = hyper::service::service_fn(move |_| {
        let body = body.clone();
        async move { Ok::<_, std::convert::Infallible>(http::refuse_503(body)) }
    });
    let conn = hyper::server::conn::http1::Builder::new()
        .timer(hyper_util::rt::TokioTimer::new())
        .header_read_timeout(Some(REFUSE_TIMEOUT))
        .serve_connection(hyper_util::rt::TokioIo::new(stream), service);
    let _ = conn.await;
}

/// The watches the server's handlers serve — the model's state, its
/// load stage and the GPU instance's state — plus the task driving the
/// server to its exit. [`spawn`] returns them so a caller can wait on
/// what `/status` and `/v1/` report instead of polling HTTP.
pub struct Running<B: Backend> {
    /// The model's state.
    pub state: watch::Receiver<ModelState<B>>,
    /// The `open_ane` stage while the model loads.
    pub stage: watch::Receiver<LoadStage>,
    /// The GPU instance's state.
    pub gpu: watch::Receiver<GpuState>,
    /// Drives the server; resolves at idle exit, SIGTERM or a fatal
    /// error.
    pub task: tokio::task::JoinHandle<Result<(), Error>>,
}

/// Runs the server until idle exit, SIGTERM or a fatal error.
///
/// Accept loops on the unix socket and every TCP listener feed connection
/// tasks into a `GracefulShutdown` registry, so an exit trigger closes
/// keep-alive connections after their in-flight request. The idle
/// deadline ticks only while nothing is busy: the loader downloading or
/// opening the model, the scheduler holding queued or in-flight work, or
/// a request task running. Shutdown drains the connections first so
/// in-flight requests complete, then stops the loader — which stops the
/// scheduler and joins the embedding threads.
pub async fn run<B: Backend, M: ModelSource>(config: Config<B, M>) -> Result<(), Error> {
    spawn(config)
        .await?
        .task
        .await
        .expect("the server task panicked")
}

/// Starts the server and returns the watches its handlers serve;
/// [`Running::task`] drives it to the exit [`run`] awaits.
///
/// # Errors
///
/// [`Error::Listener`] when a TCP listener without a token is not
/// loopback — its queued connections are refused first — and
/// [`Error::Io`] on listener or signal-handler failures.
pub async fn spawn<B: Backend, M: ModelSource>(config: Config<B, M>) -> Result<Running<B>, Error> {
    if config.token.is_none() {
        let mut non_loopback = Vec::new();
        for listener in &config.tcp {
            let addr = listener.local_addr()?;
            if !addr.ip().is_loopback() {
                non_loopback.push(addr);
            }
        }
        if let Some(&addr) = non_loopback.first() {
            let message = format!("tcp listener {addr} is not loopback and no token is configured");
            // Queued connections get the same 503 the other startup
            // failures send — dropping the listeners would reset them.
            // `into_std` hands each back the handle the drain needs.
            let mut listeners = Vec::new();
            match config.unix.into_std() {
                Ok(listener) => listeners.push(Listener::Unix(listener)),
                Err(error) => {
                    tracing::error!(%error, "could not un-adopt the unix listener");
                }
            }
            for listener in config.tcp {
                match listener.into_std() {
                    Ok(listener) => listeners.push(Listener::Tcp(listener)),
                    Err(error) => {
                        tracing::error!(%error, "could not un-adopt a tcp listener");
                    }
                }
            }
            refuse_http(listeners, &message).await;
            return Err(Error::Listener(message));
        }
    }
    let signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let (fatal_tx, fatal_rx) = mpsc::channel::<Error>(64);
    let (sched_fatal_tx, sched_fatal_rx) = mpsc::channel::<String>(64);
    // The scheduler's fatal channel carries messages; fold them into
    // `Error` here so `run` reports one error type.
    let fold = fatal_tx.clone();
    tokio::spawn(async move {
        let mut rx = sched_fatal_rx;
        while let Some(error) = rx.recv().await {
            if fold.send(Error::Embedder(error)).await.is_err() {
                return;
            }
        }
    });
    let (busy_tx, busy_rx) = busy_channel();
    let conns = http::serve(config.unix, config.tcp);

    // Headroom monitor: the callback may run on any thread.
    let (headroom_tx, headroom_rx) =
        watch::channel(Err(repo_search_coreml::GpuBlocked::PowerSourceUnknown));
    let _monitor = config.backend.watch_headroom(Box::new(move |h| {
        let _ = headroom_tx.send_replace(h);
    }));

    let (loader_busy_tx, loader_busy_rx) = watch::channel(false);
    busy_tx
        .send(BusySource {
            state: loader_busy_rx,
        })
        .await
        .expect("the busy aggregator outlives run");
    let (gpu_state_tx, gpu_state_rx) = watch::channel(GpuState::Absent { blocked: None });
    let loader = loader::spawn(LoaderConfig {
        backend: config.backend,
        model: Arc::new(config.model),
        bound: config.load_bound,
        chip: config.chip,
        headroom: headroom_rx,
        busy: loader_busy_tx,
        busy_reg: busy_tx.clone(),
        fatal: sched_fatal_tx,
        gpu_state: gpu_state_tx,
    });
    let (in_flight, in_flight_source, _in_flight_count) = http::InFlight::new();
    busy_tx
        .send(in_flight_source)
        .await
        .expect("the busy aggregator outlives run");
    let shared = Arc::new(http::Shared {
        state: loader.state.clone(),
        loader: loader.tx.clone(),
        gpu: gpu_state_rx,
        stage: loader.stage.clone(),
        token: config.token,
        next_request: std::sync::atomic::AtomicU64::new(0),
        in_flight: Arc::new(in_flight),
    });
    let accepts = conns.run(shared.clone());
    let task = tokio::spawn(drive(
        accepts,
        loader.tx.clone(),
        fatal_rx,
        busy_rx,
        signal,
        config.idle_exit,
    ));
    Ok(Running {
        state: shared.state.clone(),
        stage: shared.stage.clone(),
        gpu: shared.gpu.clone(),
        task,
    })
}

/// The accept loop's tail: the idle/fatal/signal select, then the
/// ordered shutdown — connections drain first so in-flight requests
/// complete, then the loader stops the scheduler and joins the
/// embedding threads.
async fn drive(
    accepts: http::Accepts,
    loader: mpsc::Sender<LoaderMsg>,
    mut fatal_rx: mpsc::Receiver<Error>,
    mut busy_rx: watch::Receiver<bool>,
    mut signal: tokio::signal::unix::Signal,
    idle_exit: Duration,
) -> Result<(), Error> {
    let mut deadline: Option<std::time::Instant> = Some(std::time::Instant::now() + idle_exit);
    let mut outcome: Result<(), Error> = Ok(());
    let exit = 'accept: loop {
        let sleep = async {
            match deadline {
                Some(at) => tokio::time::sleep_until(at.into()).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = sleep => break 'accept true,
            changed = busy_rx.changed() => {
                if changed.is_err() {
                    break 'accept true;
                }
            }
            fatal = fatal_rx.recv() => {
                if let Some(error) = fatal {
                    outcome = Err(error);
                    break 'accept false;
                }
            }
            _ = signal.recv() => break 'accept false,
        }
        deadline = if *busy_rx.borrow() {
            None
        } else {
            Some(std::time::Instant::now() + idle_exit)
        };
    };
    if exit {
        tracing::info!("idle exit");
    }
    // Shutdown: the graceful drain finishes in-flight requests — they
    // need the pipeline — then the loader stops the scheduler and joins
    // the threads.
    accepts.shutdown().await;
    let (stopped, stopping) = tokio::sync::oneshot::channel();
    let _ = loader.send(LoaderMsg::Shutdown { reply: stopped }).await;
    let _ = stopping.await;
    outcome
}

/// The busy aggregator: one task folding every source's state `watch` —
/// the loader's, the scheduler's and the request tasks' — into the single
/// `Receiver` the idle deadline reads.
fn busy_channel() -> (mpsc::Sender<BusySource>, watch::Receiver<bool>) {
    let (tx, mut rx) = mpsc::channel::<BusySource>(16);
    let (busy_tx, busy_rx) = watch::channel(false);
    tokio::spawn(async move {
        let mut sources: Vec<watch::Receiver<bool>> = Vec::new();
        loop {
            let busy = sources.iter().any(|s| *s.borrow());
            busy_tx.send_replace(busy);
            tokio::select! {
                source = rx.recv() => match source {
                    Some(source) => sources.push(source.state),
                    None => break,
                },
                _ = wait_any(&mut sources) => {}
            }
        }
    });
    (tx, busy_rx)
}

/// Resolves when a source reports a change or a source's sender went away
/// — in which case it is dropped — then returns. Pending forever on an
/// empty set.
async fn wait_any(sources: &mut Vec<watch::Receiver<bool>>) {
    if sources.is_empty() {
        return std::future::pending().await;
    }
    let (out, i) = {
        let mut futures: Vec<_> = sources.iter_mut().map(|s| Box::pin(s.changed())).collect();
        std::future::poll_fn(|cx| {
            for (i, f) in futures.iter_mut().enumerate() {
                if let std::task::Poll::Ready(out) = f.as_mut().poll(cx) {
                    return std::task::Poll::Ready((out, i));
                }
            }
            std::task::Poll::Pending
        })
        .await
    };
    if out.is_err() {
        sources.remove(i);
    }
}
