//! The `search` binary: a thin dispatcher — `search --daemon` runs the
//! launchd daemon, everything else is the client over the daemon socket.

#[cfg(not(target_os = "macos"))]
compile_error!("the search binary runs on macOS only");

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
            model_cache,
            state_dir,
            server_socket,
        } => daemon_main(idle_exit, model_cache, state_dir, server_socket),
        Invocation::Server {
            idle_exit,
            model_cache,
            token_file,
            build,
        } => server_main(idle_exit, model_cache, token_file, build),
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
fn daemon_main(
    idle_exit: Duration,
    model_cache: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    server_socket: Option<PathBuf>,
) -> ExitCode {
    use futures_util::future::join_all;
    use repo_search::daemon::{self, Config};
    use repo_search::launchd;
    use repo_search_core::model::MODEL_ID;
    use std::os::unix::net::UnixListener;

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
            return ExitCode::FAILURE;
        }
    };

    let fds = match launchd::activate_socket("Listener") {
        Ok(fds) if fds.is_empty() => {
            tracing::error!("launchd returned no Listener fd");
            return ExitCode::FAILURE;
        }
        Ok(fds) => fds,
        Err(error) => {
            tracing::error!(%error, "launchd socket activation failed");
            return ExitCode::FAILURE;
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
            return ExitCode::FAILURE;
        }
    };

    // `from_std` registers the fd with this runtime's driver, so `startup`
    // runs inside the runtime's context.
    let _guard = runtime.enter();
    match startup(listeners, model_cache, state_dir, server_socket) {
        Ok((listener, tokens, server, state, build)) => {
            match runtime.block_on(daemon::run(Config {
                listener,
                idle_exit,
                load_bound: daemon::LOAD_BOUND,
                tokens,
                server,
                model_id: MODEL_ID.to_owned(),
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
            runtime.block_on(join_all(
                listeners
                    .into_iter()
                    .map(|listener| daemon::refuse(listener, &error)),
            ));
            ExitCode::FAILURE
        }
    }
}

/// What a successful `startup` hands to `daemon::run`.
type Started = (
    tokio::net::UnixListener,
    repo_search::model_store::TokenizerStore,
    repo_search::daemon::server::Endpoint,
    repo_search_core::StateDir,
    repo_search::protocol::BuildId,
);

/// The reportable startup checks, in order. On failure the std listeners
/// come back so the caller can answer their queues.
fn startup(
    mut listeners: Vec<std::os::unix::net::UnixListener>,
    model_cache: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    server_socket: Option<PathBuf>,
) -> Result<
    Started,
    (
        repo_search::daemon::StartupError,
        Vec<std::os::unix::net::UnixListener>,
    ),
> {
    use repo_search::daemon::StartupError;
    use repo_search::daemon::server::Endpoint;
    use repo_search::model_store::TokenizerStore;
    use repo_search::protocol::BuildId;
    use repo_search_core::StateDir;

    let checks = || {
        if listeners.len() != 1 {
            return Err(StartupError::ListenerCount(listeners.len()));
        }
        let build = BuildId::of_current_exe().map_err(StartupError::Build)?;
        let tokens = match &model_cache {
            Some(root) => TokenizerStore::new(
                root.clone(),
                repo_search::model_store::TokenizerSpec::embedding_gemma_2(),
            ),
            None => TokenizerStore::user_cache(),
        }
        .map_err(StartupError::ModelStore)?;
        let state = match &state_dir {
            Some(root) => StateDir::new(root.clone()),
            None => StateDir::user().map_err(StartupError::StateDir)?,
        };
        let server = match server_socket {
            Some(path) => Endpoint::Unix(path),
            None => Endpoint::Unix(
                repo_search::launchd::server_socket_path().map_err(StartupError::StateDir)?,
            ),
        };
        Ok((tokens, server, state, build))
    };
    let (tokens, server, state, build) = match checks() {
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
        Ok(listener) => Ok((listener, tokens, server, state, build)),
        Err(error) => Err((StartupError::Adopt(error), vec![spare])),
    }
}

/// Answers queued launchd connections with the 503 refusal, then exits 1.
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
enum Adopted {
    Unix(tokio::net::UnixListener),
    Tcp(tokio::net::TcpListener),
}

/// Adopts `listener`: `from_std` consumes the std listener either way,
/// so a `try_clone` spare rides along for the refusal path when the
/// adoption fails.
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
fn server_main(
    idle_exit: Duration,
    model_cache: Option<PathBuf>,
    token_file: Option<PathBuf>,
    build: Option<String>,
) -> ExitCode {
    use repo_search::launchd;
    use repo_search::model_store::{ModelSpec, ModelStore};
    use repo_search::protocol::BuildId;

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
            // count check below.
            Err(launchd::ActivateError::Launchd { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound => {}
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
        Some(root) => ModelStore::new(root.clone(), ModelSpec::embedding_gemma_2()),
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
        idle_exit,
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
