//! The `search` binary: a thin dispatcher — `search --daemon` runs the
//! launchd daemon, everything else is the client over the daemon socket.

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("the search binary runs on macOS and Linux only");

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use repo_search::cli::{Cli, Env, Invocation, Terminal};

fn main() -> ExitCode {
    match Cli::parse().invocation() {
        Invocation::Daemon {
            idle_exit,
            reconcile_wait,
            build,
            model_cache,
            state_dir,
            server_socket,
            listen,
            ready_fd,
            server_addr,
            server_token_file,
        } => daemon_main(
            Timing {
                idle_exit,
                reconcile_wait,
            },
            listen,
            ready_fd,
            DaemonArgs {
                build,
                model_cache,
                state_dir,
                server: ServerArgs {
                    socket: server_socket,
                    addr: server_addr,
                    token_file: server_token_file,
                },
            },
        ),
        #[cfg(target_os = "macos")]
        Invocation::Server {
            idle_exit,
            model_cache,
            token_file,
            build,
            ..
        } => server_main(idle_exit, model_cache, token_file, build),
        #[cfg(not(target_os = "macos"))]
        Invocation::Server {
            idle_exit,
            token_file,
            build,
            model_cache,
            model_dir,
            socket,
            addr,
            ready_fd,
        } => ort_server_main(OrtServer {
            idle_exit,
            token_file,
            build,
            model_dir,
            model_cache,
            socket,
            addr,
            ready_fd,
        }),
        client => client_main(client),
    }
}

/// The client entry point: `Env` is gathered before the runtime exists,
/// and the request runs on a current-thread runtime — the client has one
/// request in flight, so a multi-thread runtime would only add threads.
fn client_main(invocation: Invocation) -> ExitCode {
    let env = match Env::user() {
        Ok(env) => env,
        Err(error) => {
            let _ = std::io::stderr().write_all(format!("search: {error}\n").as_bytes());
            return ExitCode::from(2);
        }
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the tokio runtime builds");
    let mut term = Terminal::stdio();
    runtime
        .block_on(repo_search::cli::run(invocation, env, &mut term))
        .into()
}

/// The `search --daemon` entry point.
///
/// Tracing goes to stderr at `INFO`, which the agent's `StandardErrorPath`
/// captures; nothing in the daemon prints. The two-worker runtime runs at
/// user-initiated QoS — its tasks only route messages, and every blocking
/// call runs on a thread the daemon or core owns. The `Listener` socket
/// comes from launchd; exactly one fd is expected, and any other count
/// means the binary runs under a plist it was not built for. Every check
/// that can fail runs after socket activation: a failure there is logged
/// and then answered to each connection already queued on the socket
/// (`ErrorKind::Startup`), so no client waits on a daemon that has exited.
/// Only the failures that precede the socket — the runtime and the
/// activation itself — are logged alone. Exit is 0 on idle exit or SIGTERM
/// and 1 on a fatal error.
///
/// Off macOS there is no launchd: `--listen` is required, and its absence
/// is a startup error reported on the ready fd.
fn daemon_main(
    timing: Timing,
    listen: Option<PathBuf>,
    ready_fd: Option<std::os::fd::RawFd>,
    args: DaemonArgs,
) -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_start(|| {
            repo_search_core::qos::set_current_thread(
                repo_search_core::qos::QosClass::UserInitiated,
            );
        })
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(%error, "could not build the tokio runtime");
            return fail_ready(
                ready_fd,
                &format!("could not build the tokio runtime: {error}"),
            );
        }
    };

    // `--listen` hands the daemon a socket it owns; its absence keeps
    // launchd activation on macOS.
    if let Some(path) = listen {
        return bound_main(runtime, timing, ready_fd, args, path);
    }

    #[cfg(target_os = "macos")]
    {
        activated_main(runtime, timing, ready_fd, args)
    }
    #[cfg(not(target_os = "macos"))]
    {
        tracing::error!("--daemon needs --listen off macOS");
        fail_ready(ready_fd, "--daemon needs --listen off macOS")
    }
}

/// The launchd entry point: the `Listener` socket comes from socket
/// activation, and the startup failures answer the connections already
/// queued on it.
#[cfg(target_os = "macos")]
fn activated_main(
    runtime: tokio::runtime::Runtime,
    timing: Timing,
    ready_fd: Option<std::os::fd::RawFd>,
    args: DaemonArgs,
) -> ExitCode {
    use futures_util::future::join_all;
    use repo_search::daemon::listen::Activated;
    use repo_search::daemon::{self, Config};
    use repo_search::launchd;
    use std::os::unix::net::UnixListener;

    let fds = match launchd::activate_socket("Listener") {
        Ok(fds) if fds.is_empty() => {
            tracing::error!("launchd returned no Listener fd");
            return fail_ready(ready_fd, "launchd returned no Listener fd");
        }
        Ok(fds) => fds,
        Err(error) => {
            tracing::error!(%error, "launchd socket activation failed");
            return fail_ready(
                ready_fd,
                &format!("launchd socket activation failed: {error}"),
            );
        }
    };
    let listeners = fds
        .into_iter()
        .map(|fd| {
            let listener = UnixListener::from(fd);
            listener.set_nonblocking(true).map(|()| listener)
        })
        .collect::<Result<Vec<_>, _>>();
    let listeners = match listeners {
        Ok(listeners) => listeners,
        Err(error) => {
            tracing::error!(%error, "could not set a Listener fd nonblocking");
            return fail_ready(
                ready_fd,
                &format!("could not set a Listener fd nonblocking: {error}"),
            );
        }
    };

    // `from_std` registers the fd with this runtime's driver, so `startup`
    // runs inside the runtime's context.
    let _guard = runtime.enter();
    match startup(listeners, args) {
        Ok((listener, signal, tokens, server, state, build)) => {
            if let Err(error) = report_ready(ready_fd, "ready\n") {
                tracing::error!(%error, "could not report readiness");
                return ExitCode::FAILURE;
            }
            match runtime.block_on(daemon::run(Config {
                listener: Activated(listener),
                signal,
                idle_exit: timing.idle_exit,
                reconcile_wait: timing.reconcile_wait,
                load_bound: daemon::LOAD_BOUND,
                tokens,
                server,
                model_id: model_id(),
                state,
                build,
            })) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    tracing::error!(%error, "the daemon stopped on a fatal error");
                    ExitCode::FAILURE
                }
            }
        }
        Err((error, listeners)) => {
            tracing::error!(%error, "the daemon failed to start");
            let code = fail_ready(ready_fd, &error.to_string());
            runtime.block_on(join_all(
                listeners
                    .into_iter()
                    .map(|listener| daemon::refuse(listener, &error)),
            ));
            code
        }
    }
}

/// The `--listen` entry point: the daemon takes the lifetime lock, binds
/// its own socket, reports `ready` or `failed: <message>` on
/// `--ready-fd`, then runs — the Linux `daemon_main`'s shape, compiled
/// and exercised on macOS.
fn bound_main(
    runtime: tokio::runtime::Runtime,
    timing: Timing,
    ready_fd: Option<std::os::fd::RawFd>,
    args: DaemonArgs,
    path: PathBuf,
) -> ExitCode {
    use repo_search::daemon::listen::{Bound, Listener as _};
    use repo_search::daemon::{Config, LOAD_BOUND, StartupError};

    let bound = match runtime.block_on(Bound::acquire(&path)) {
        Ok(bound) => bound,
        Err(error) => {
            tracing::error!(%error, "could not bind the daemon socket");
            return fail_ready(
                ready_fd,
                &format!("could not bind the daemon socket: {error}"),
            );
        }
    };
    // `ready` is written only once the SIGTERM stream exists: a signal
    // landing before `run` polls it is buffered, never defaulted.
    let signal = {
        let _guard = runtime.enter();
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
    };
    let signal = match signal {
        Ok(signal) => signal,
        Err(error) => {
            let error = StartupError::Signal(error);
            tracing::error!(%error, "the daemon failed to start");
            let message = error.to_string();
            let _held = runtime.block_on(bound.refuse(&error));
            return fail_ready(ready_fd, &message);
        }
    };
    let (tokens, server, state, build) = match checks(args) {
        Ok(started) => started,
        Err(error) => {
            tracing::error!(%error, "the daemon failed to start");
            let message = error.to_string();
            let _held = runtime.block_on(bound.refuse(&error));
            return fail_ready(ready_fd, &message);
        }
    };
    if let Err(error) = report_ready(ready_fd, "ready\n") {
        tracing::error!(%error, "could not report readiness");
        let _held = runtime.block_on(bound.close());
        return ExitCode::FAILURE;
    }
    match runtime.block_on(repo_search::daemon::run(Config {
        listener: bound,
        signal,
        idle_exit: timing.idle_exit,
        reconcile_wait: timing.reconcile_wait,
        load_bound: LOAD_BOUND,
        tokens,
        server,
        model_id: model_id(),
        state,
        build,
    })) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "the daemon stopped on a fatal error");
            ExitCode::FAILURE
        }
    }
}

/// Writes `line` — `ready` on success, `failed: <message>` before
/// exit 1 — to the client's readiness pipe at `fd`, when one was
/// passed, then drops the daemon's end.
fn report_ready(ready_fd: Option<std::os::fd::RawFd>, line: &str) -> std::io::Result<()> {
    let Some(fd) = ready_fd else {
        return Ok(());
    };
    use std::os::fd::FromRawFd;
    // SAFETY: the spawn protocol hands the daemon ownership of this fd,
    // the pipe's write end, so this process may close it — and does, on
    // the drop below.
    let mut pipe = unsafe { std::fs::File::from_raw_fd(fd) };
    pipe.write_all(line.as_bytes())
}

/// Writes `failed: <message>` on the ready fd when one was passed, and
/// returns the failure exit code.
fn fail_ready(ready_fd: Option<std::os::fd::RawFd>, message: &str) -> ExitCode {
    let _ = report_ready(ready_fd, &format!("failed: {message}\n"));
    ExitCode::FAILURE
}

/// What a successful `startup` hands to `daemon::run`.
#[cfg(target_os = "macos")]
type Started = (
    tokio::net::UnixListener,
    tokio::signal::unix::Signal,
    repo_search::model_store::TokenizerStore,
    repo_search::daemon::server::Endpoint,
    repo_search_core::StateDir,
    repo_search_client::protocol::BuildId,
);

/// The reportable startup checks, in order. On failure the std listeners
/// come back so the caller can answer their queues.
#[cfg(target_os = "macos")]
fn startup(
    mut listeners: Vec<std::os::unix::net::UnixListener>,
    args: DaemonArgs,
) -> Result<
    Started,
    (
        repo_search::daemon::StartupError,
        Vec<std::os::unix::net::UnixListener>,
    ),
> {
    use repo_search::daemon::StartupError;

    if listeners.len() != 1 {
        return Err((StartupError::ListenerCount(listeners.len()), listeners));
    }
    // `ready` is written only once the SIGTERM stream exists: a signal
    // landing before `run` polls it is buffered, never defaulted.
    let signal = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(signal) => signal,
        Err(error) => return Err((StartupError::Signal(error), listeners)),
    };
    let (tokens, server, state, build) = match checks(args) {
        Ok(checked) => checked,
        Err(error) => return Err((error, listeners)),
    };
    let std_listener = listeners.pop().expect("the count check left one listener");
    // `try_clone` keeps a std handle for the refusal path if adoption fails.
    let spare = match std_listener.try_clone() {
        Ok(spare) => spare,
        Err(error) => return Err((StartupError::Adopt(error), vec![std_listener])),
    };
    match tokio::net::UnixListener::from_std(std_listener) {
        Ok(listener) => Ok((listener, signal, tokens, server, state, build)),
        Err(error) => Err((StartupError::Adopt(error), vec![spare])),
    }
}

/// The startup checks the two listener kinds share: the running binary's
/// id against `--build`, the tokenizer store, the state dir and the
/// server endpoint — including the `--server-token-file` read, so a
/// failure there reaches the ready line.
fn checks(
    args: DaemonArgs,
) -> Result<
    (
        repo_search::model_store::TokenizerStore,
        repo_search::daemon::server::Endpoint,
        repo_search_core::StateDir,
        repo_search_client::protocol::BuildId,
    ),
    repo_search::daemon::StartupError,
> {
    use repo_search::daemon::StartupError;
    use repo_search::model_store::TokenizerStore;
    use repo_search_client::protocol::BuildId;
    use repo_search_core::StateDir;

    let DaemonArgs {
        build: expected,
        model_cache,
        state_dir,
        server,
    } = args;
    let build = BuildId::of_current_exe().map_err(StartupError::Build)?;
    if let Some(expected) = expected
        && expected != build.to_string()
    {
        return Err(StartupError::BuildMismatch {
            expected,
            actual: build,
        });
    }
    let tokens = match &model_cache {
        Some(root) => TokenizerStore::new(root.clone(), repo_search_core::model::TOKENIZER),
        None => TokenizerStore::user_cache(),
    }
    .map_err(StartupError::ModelStore)?;
    let state = match &state_dir {
        Some(root) => StateDir::new(root.clone()),
        None => StateDir::new(
            repo_search_client::launcher::paths::UserPaths::user()
                .map_err(StartupError::StateDir)?
                .state,
        ),
    };
    let server = endpoint(server)?;
    Ok((tokens, server, state, build))
}

/// The flags of `search --server` off macOS.
#[cfg(not(target_os = "macos"))]
struct OrtServer {
    idle_exit: Option<Duration>,
    token_file: Option<PathBuf>,
    build: Option<String>,
    model_dir: Option<PathBuf>,
    model_cache: Option<PathBuf>,
    socket: Option<PathBuf>,
    addr: Vec<std::net::SocketAddr>,
    ready_fd: Option<std::os::fd::RawFd>,
}

/// The daemon flags its startup checks read.
struct DaemonArgs {
    /// `--build`, when given: the build the agent registered.
    build: Option<String>,
    /// `--model-cache`, when given.
    model_cache: Option<PathBuf>,
    /// `--state-dir`, when given.
    state_dir: Option<PathBuf>,
    /// The `--server-*` flags.
    server: ServerArgs,
}

/// The daemon's time settings from its flags.
struct Timing {
    /// `--idle-exit-secs`.
    idle_exit: Duration,
    /// `--reconcile-wait-ms`.
    reconcile_wait: Duration,
}

/// The `--server-*` flags: where the embedding server listens and the
/// daemon's credential for it.
struct ServerArgs {
    /// `--server-socket`, when given.
    socket: Option<PathBuf>,
    /// `--server-addr`, when given.
    addr: Option<std::net::SocketAddr>,
    /// `--server-token-file`, when given.
    token_file: Option<PathBuf>,
}

/// Where the daemon reaches the embedding server: `--server-addr` with
/// its token file, `--server-socket`, or the agent's default socket.
/// The token is read and trimmed exactly as `server_main` reads its own.
fn endpoint(
    server: ServerArgs,
) -> Result<repo_search::daemon::server::Endpoint, repo_search::daemon::StartupError> {
    use repo_search::daemon::StartupError;
    use repo_search::daemon::server::Endpoint;

    let token = match &server.token_file {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|source| StartupError::TokenFile {
                    path: path.clone(),
                    source,
                })?
                .trim_end_matches(['\n', '\r'])
                .to_owned(),
        ),
        None => None,
    };
    match (server.socket, server.addr) {
        (Some(path), None) => Ok(Endpoint::Unix(path)),
        (None, Some(addr)) => Ok(Endpoint::Tcp { addr, token }),
        (None, None) => Ok(Endpoint::Unix(
            repo_search_client::launcher::paths::server_socket_path()
                .map_err(StartupError::StateDir)?
                .into_path_buf(),
        )),
        (Some(_), Some(_)) => unreachable!("clap marks the flags conflicting"),
    }
}

/// Answers queued launchd connections with the 503 refusal, then exits 1.
#[cfg(target_os = "macos")]
fn fail_all(
    listeners: Vec<repo_search_server::Listener>,
    message: &str,
    runtime: &tokio::runtime::Runtime,
) -> ExitCode {
    tracing::error!(%message, "the server failed to start");
    runtime.block_on(repo_search_server::refuse_http(listeners, message));
    ExitCode::FAILURE
}

/// One listener adopted into the runtime.
#[cfg(target_os = "macos")]
enum Adopted {
    Unix(tokio::net::UnixListener),
    Tcp(tokio::net::TcpListener),
}

/// Adopts `listener`: `from_std` consumes the std listener either way,
/// so a `try_clone` spare rides along for the refusal path when the
/// adoption fails.
#[cfg(target_os = "macos")]
fn adopt(
    listener: repo_search_server::Listener,
) -> Result<Adopted, (std::io::Error, repo_search_server::Listener)> {
    use repo_search_server::Listener;
    match listener {
        Listener::Unix(listener) => match listener.try_clone() {
            Ok(spare) => tokio::net::UnixListener::from_std(listener)
                .map(Adopted::Unix)
                .map_err(|error| (error, Listener::Unix(spare))),
            Err(error) => Err((error, Listener::Unix(listener))),
        },
        Listener::Tcp(listener) => match listener.try_clone() {
            Ok(spare) => tokio::net::TcpListener::from_std(listener)
                .map(Adopted::Tcp)
                .map_err(|error| (error, Listener::Tcp(spare))),
            Err(error) => Err((error, Listener::Tcp(listener))),
        },
    }
}

/// The `search --server` entry point.
///
/// Mirrors [`daemon_main`]: launchd hands over `"Unix"` (exactly one) and
/// `"Tcp"` (zero or more) sockets; `--build` is checked against the running
/// binary before `repo_search_server::run` starts — a mismatch means the
/// agent wants another build, so the server fails fast after answering
/// every queued connection 503.
#[cfg(target_os = "macos")]
fn server_main(
    idle_exit: Option<Duration>,
    model_cache: Option<PathBuf>,
    token_file: Option<PathBuf>,
    build: Option<String>,
) -> ExitCode {
    use repo_search::launchd;
    use repo_search::model_store::ModelStore;
    use repo_search_client::protocol::BuildId;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(%error, "could not build the tokio runtime");
            return ExitCode::FAILURE;
        }
    };

    let mut listeners = Vec::new();
    for name in ["Unix", "Tcp"] {
        match launchd::activate_socket(name) {
            Ok(fds) => {
                // Every fd of the batch that accepts `set_nonblocking`
                // is refused on a failure — one bad fd must not drop
                // its siblings' queued connections.
                let mut failed = None;
                for fd in fds {
                    let listener = match name {
                        "Unix" => repo_search_server::Listener::Unix(
                            std::os::unix::net::UnixListener::from(fd),
                        ),
                        _ => repo_search_server::Listener::Tcp(std::net::TcpListener::from(fd)),
                    };
                    match match &listener {
                        repo_search_server::Listener::Unix(l) => l.set_nonblocking(true),
                        repo_search_server::Listener::Tcp(l) => l.set_nonblocking(true),
                    } {
                        Ok(()) => listeners.push((name, listener)),
                        Err(error) => {
                            tracing::error!(%error, %name, "could not set a socket nonblocking");
                            failed.get_or_insert(error);
                        }
                    }
                }
                if let Some(error) = failed {
                    return fail_all(
                        listeners.into_iter().map(|(_, l)| l).collect(),
                        &format!("could not set a {name} fd nonblocking: {error}"),
                        &runtime,
                    );
                }
            }
            // No socket of this name is registered: zero `Tcp` listeners
            // is a valid configuration; a missing `Unix` is caught by the
            // count check below. For a name the plist lacks, launchd
            // answers `ESRCH` rather than the documented `ENOENT` — once
            // `Unix` has activated, the process is launchd's, so `ESRCH`
            // for `Tcp` can only mean the agent has no TCP socket.
            Err(launchd::ActivateError::Launchd { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound
                    || (name == "Tcp" && source.raw_os_error() == Some(libc::ESRCH)) => {}
            Err(error) => {
                tracing::error!(%error, %name, "launchd socket activation failed");
                return fail_all(
                    listeners.into_iter().map(|(_, l)| l).collect(),
                    &format!("launchd socket activation failed for {name}: {error}"),
                    &runtime,
                );
            }
        }
    }
    let unix = listeners.iter().filter(|(n, _)| *n == "Unix").count();
    let mut refused = listeners.into_iter().map(|(_, l)| l).collect::<Vec<_>>();
    if unix != 1 {
        return fail_all(
            refused,
            &format!("launchd returned {unix} Unix fds, expected exactly 1"),
            &runtime,
        );
    }
    if let Some(expected) = &build {
        match BuildId::of_current_exe() {
            Ok(actual) if actual.to_string() == *expected => {}
            Ok(actual) => {
                return fail_all(
                    refused,
                    &format!("--build {expected} does not match this binary {actual}"),
                    &runtime,
                );
            }
            Err(error) => {
                return fail_all(
                    refused,
                    &format!("could not identify the running binary: {error}"),
                    &runtime,
                );
            }
        }
    }
    let model_store = match &model_cache {
        Some(root) => ModelStore::new(root.clone(), repo_search_core::model::ARCHIVE),
        None => ModelStore::user_cache(),
    };
    let model_store = match model_store {
        Ok(store) => store,
        Err(error) => {
            return fail_all(
                refused,
                &format!("could not open the model cache: {error}"),
                &runtime,
            );
        }
    };
    let chip = match repo_search_server::chip::host() {
        Ok(chip) => chip,
        Err(error) => {
            return fail_all(
                refused,
                &format!("could not determine the host's chip class: {error}"),
                &runtime,
            );
        }
    };
    let token = match &token_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(token) => Some(token.trim_end_matches(['\n', '\r']).to_owned()),
            Err(error) => {
                return fail_all(
                    refused,
                    &format!("could not read the token file {}: {error}", path.display()),
                    &runtime,
                );
            }
        },
        None => None,
    };
    // Adopt the listeners into the runtime after every check that can
    // fail: `from_std` needs the runtime's driver.
    let _guard = runtime.enter();
    let mut unix_listener = None;
    let mut tcp = Vec::new();
    while let Some(listener) = refused.pop() {
        match adopt(listener) {
            Ok(Adopted::Unix(l)) => unix_listener = Some(l),
            Ok(Adopted::Tcp(l)) => tcp.push(l),
            Err((error, failed)) => {
                refused.push(failed);
                // The listeners already adopted go back to std handles,
                // so their queues get the same 503 as the ones never
                // drained.
                if let Some(l) = unix_listener {
                    match l.into_std() {
                        Ok(l) => refused.push(repo_search_server::Listener::Unix(l)),
                        Err(error) => {
                            tracing::error!(%error, "could not un-adopt the unix listener");
                        }
                    }
                }
                for l in tcp {
                    match l.into_std() {
                        Ok(l) => refused.push(repo_search_server::Listener::Tcp(l)),
                        Err(error) => {
                            tracing::error!(%error, "could not un-adopt a tcp listener");
                        }
                    }
                }
                return fail_all(
                    refused,
                    &format!("could not adopt a listener fd: {error}"),
                    &runtime,
                );
            }
        }
    }
    match runtime.block_on(repo_search_server::run(repo_search_server::Config {
        unix: unix_listener.expect("the count check left one listener"),
        tcp,
        token,
        idle_exit: idle_exit.unwrap_or(repo_search_client::launcher::SERVER_IDLE_EXIT),
        load_bound: repo_search_core::embed_api::LOAD_BOUND,
        model: model_store,
        chip,
        backend: std::sync::Arc::new(repo_search_server::coreml::CoreMlBackend),
    })) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "the server stopped on a fatal error");
            ExitCode::FAILURE
        }
    }
}

/// The model id the daemon expects the embedding server to list: the
/// default fp16 build's, or the int8 build's when [`onnx_int8`] opts into
/// it. `REPO_SEARCH_MODEL_ID` overrides it for a daemon whose server runs
/// on another host with a different pinned build.
fn model_id() -> String {
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(id) = std::env::var("REPO_SEARCH_MODEL_ID") {
            return id;
        }
        if onnx_int8() {
            return repo_search_core::model::ONNX_INT8_MODEL_ID.to_owned();
        }
    }
    repo_search_core::model::MODEL_ID.to_owned()
}

/// Whether this host runs the int8 ONNX archive instead of the default
/// fp16 one: `REPO_SEARCH_ORT_ARCHIVE=int8`.
///
/// # Panics
///
/// When `REPO_SEARCH_ORT_ARCHIVE` is set to anything but `fp16` or `int8`.
#[cfg(not(target_os = "macos"))]
fn onnx_int8() -> bool {
    match std::env::var("REPO_SEARCH_ORT_ARCHIVE").as_deref() {
        Ok("int8") => true,
        Ok("fp16") | Err(_) => false,
        Ok(other) => panic!("REPO_SEARCH_ORT_ARCHIVE must be fp16 or int8, got {other:?}"),
    }
}

/// The archive pin matching [`onnx_int8`]'s choice.
#[cfg(not(target_os = "macos"))]
fn onnx_pin() -> ondevice::Pin<'static, ondevice::Archive> {
    if onnx_int8() {
        repo_search_core::model::ONNX_INT8_ARCHIVE
    } else {
        repo_search_core::model::ONNX_ARCHIVE
    }
}

/// The per-user [`ModelStore`] matching [`onnx_int8`]'s archive choice.
#[cfg(not(target_os = "macos"))]
fn onnx_store() -> Result<repo_search::model_store::ModelStore, repo_search::model_store::OpenError>
{
    if onnx_int8() {
        repo_search::model_store::ModelStore::user_cache_onnx_int8()
    } else {
        repo_search::model_store::ModelStore::user_cache_onnx()
    }
}

/// `search --server` off macOS: the ONNX embedding server — its own
/// listeners (a unix socket, plus optional TCP), the `--model-dir`
/// model and the same idle exit. The `--build` check matches the launchd
/// agent's: a mismatched registered build refuses to run.
///
/// Under the spawn launcher (`--ready-fd`) it reports `ready` once its
/// listeners are bound — the model loads after, as a launchd-activated
/// server's does behind its queued connections — and `failed: <message>`
/// for every startup failure before that.
#[cfg(not(target_os = "macos"))]
fn ort_server_main(args: OrtServer) -> ExitCode {
    let OrtServer {
        idle_exit,
        token_file,
        build,
        model_dir,
        model_cache,
        socket,
        addr,
        ready_fd,
    } = args;
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();
    let fail = |message: String| {
        tracing::error!("{message}");
        fail_ready(ready_fd, &message)
    };
    #[cfg(unix)]
    if let Some(expected) = &build {
        match repo_search_client::protocol::BuildId::of_current_exe() {
            Ok(actual) if actual.to_string() == *expected => {}
            Ok(actual) => {
                return fail(format!(
                    "--build {expected} does not match this binary {actual}"
                ));
            }
            Err(error) => {
                return fail(format!("could not identify the running binary: {error}"));
            }
        }
    }
    let token = match &token_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(token) => Some(token.trim_end_matches(['\n', '\r']).to_owned()),
            Err(error) => {
                return fail(format!(
                    "could not read the token file {}: {error}",
                    path.display()
                ));
            }
        },
        None => None,
    };
    // A non-loopback TCP listener without a token is open to the network.
    if token.is_none() && addr.iter().any(|addr| !addr.ip().is_loopback()) {
        return fail("a non-loopback --addr requires --token-file".to_owned());
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return fail(format!("could not build the tokio runtime: {error}")),
    };

    // The lifetime lock `server.lock` next to the socket proves its
    // owner and makes a replacement wait out the previous server; it is
    // held until this process exits.
    #[cfg(unix)]
    let (unix, _lifetime_lock) = {
        let path = match socket {
            Some(path) => path,
            None => match repo_search_client::launcher::paths::server_socket_path() {
                Ok(path) => path.into_path_buf(),
                Err(error) => {
                    return fail(format!("could not resolve the server socket path: {error}"));
                }
            },
        };
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            return fail(format!("could not create {}: {error}", parent.display()));
        }
        let (listener, lock) = match repo_search::daemon::listen::acquire_blocking(&path) {
            Ok(bound) => bound,
            Err(error) => return fail(format!("could not bind {}: {error}", path.display())),
        };
        // The socket's file permissions are the unix listener's auth.
        use std::os::unix::fs::PermissionsExt as _;
        if let Err(error) = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        {
            return fail(format!("could not permission {}: {error}", path.display()));
        }
        (listener, lock)
    };

    let mut tcp = Vec::new();
    for addr in addr {
        match std::net::TcpListener::bind(addr) {
            Ok(listener) => tcp.push(listener),
            Err(error) => {
                return fail(format!("could not bind the TCP listener {addr}: {error}"));
            }
        }
    }

    let _guard = runtime.enter();
    #[cfg(unix)]
    let unix = {
        if let Err(error) = unix.set_nonblocking(true) {
            return fail(format!(
                "could not set the unix listener nonblocking: {error}"
            ));
        }
        match tokio::net::UnixListener::from_std(unix) {
            Ok(listener) => Some(listener),
            Err(error) => return fail(format!("could not adopt the unix listener: {error}")),
        }
    };
    let mut adopted = Vec::with_capacity(tcp.len());
    for listener in tcp {
        if let Err(error) = listener.set_nonblocking(true) {
            return fail(format!("could not set a TCP listener nonblocking: {error}"));
        }
        match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => adopted.push(listener),
            Err(error) => return fail(format!("could not adopt a TCP listener: {error}")),
        }
    }
    // The listeners are bound: a connection made now queues until the
    // model has loaded, so the client may go on.
    if let Err(error) = report_ready(ready_fd, "ready\n") {
        tracing::error!(%error, "could not report readiness");
        return ExitCode::FAILURE;
    }
    let result = runtime.block_on(async {
        // `--model-dir` points at an unpacked model directory; without it
        // the pinned ONNX archive is fetched and unpacked under the model
        // cache (`--model-cache`, else the per-user cache).
        let model_dir = match model_dir {
            Some(dir) => dir,
            None => {
                use repo_search::model_store::ModelStore;
                let store = match model_cache {
                    Some(root) => ModelStore::new(root, onnx_pin()),
                    None => onnx_store(),
                }
                .map_err(|error| format!("model store: {error}"))?;
                let (progress, _watch) =
                    tokio::sync::watch::channel(repo_search_core::model::Fetch {
                        received: 0,
                        total: None,
                    });
                store
                    .ensure(&progress)
                    .await
                    .map_err(|error| format!("model download: {error}"))?
            }
        };
        repo_search_ort::server::run(repo_search_ort::server::Config {
            #[cfg(unix)]
            unix,
            tcp: adopted,
            token,
            idle_exit: idle_exit.unwrap_or(repo_search_client::launcher::SERVER_IDLE_EXIT),
            model_dir,
        })
        .await
        .map_err(|error| error.to_string())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "the server stopped on a fatal error");
            ExitCode::FAILURE
        }
    }
}
