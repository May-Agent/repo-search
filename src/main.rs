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
        } => daemon_main(idle_exit, model_cache, state_dir),
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
) -> ExitCode {
    use futures_util::future::join_all;
    use repo_search::daemon::{self, Config};
    use repo_search::launchd;
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
    match startup(listeners, model_cache, state_dir) {
        Ok((listener, model_store, state, chip, build)) => {
            match runtime.block_on(daemon::run(Config {
                listener,
                idle_exit,
                load_bound: daemon::LOAD_BOUND,
                model_store,
                state,
                chip,
                backend: std::sync::Arc::new(daemon::coreml::CoreMlBackend),
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
    repo_search::model_store::ModelStore,
    repo_search_core::StateDir,
    repo_search::daemon::chip::ChipClass,
    repo_search::protocol::BuildId,
);

/// The reportable startup checks, in order. On failure the std listeners
/// come back so the caller can answer their queues.
fn startup(
    mut listeners: Vec<std::os::unix::net::UnixListener>,
    model_cache: Option<PathBuf>,
    state_dir: Option<PathBuf>,
) -> Result<
    Started,
    (
        repo_search::daemon::StartupError,
        Vec<std::os::unix::net::UnixListener>,
    ),
> {
    use repo_search::daemon::{StartupError, chip};
    use repo_search::model_store::{ModelSpec, ModelStore};
    use repo_search::protocol::BuildId;
    use repo_search_core::StateDir;

    let checks = || {
        if listeners.len() != 1 {
            return Err(StartupError::ListenerCount(listeners.len()));
        }
        let build = BuildId::of_current_exe().map_err(StartupError::Build)?;
        let model_store = match &model_cache {
            Some(root) => ModelStore::new(root.clone(), ModelSpec::embedding_gemma_2()),
            None => ModelStore::user_cache(),
        }
        .map_err(StartupError::ModelStore)?;
        let state = match &state_dir {
            Some(root) => StateDir::new(root.clone()),
            None => StateDir::user().map_err(StartupError::StateDir)?,
        };
        // The chip's class decides which indexing queue the GPU serves; a
        // brand the parse does not recognise is fatal — no fallback is
        // guessed.
        let chip = chip::host().map_err(StartupError::Chip)?;
        Ok((model_store, state, chip, build))
    };
    let (model_store, state, chip, build) = match checks() {
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
        Ok(listener) => Ok((listener, model_store, state, chip, build)),
        Err(error) => Err((StartupError::Adopt(error), vec![spare])),
    }
}
