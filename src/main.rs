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
        } => daemon_main(idle_exit, model_cache),
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
/// means the binary runs under a plist it was not built for. Exit is 0 on
/// idle exit or SIGTERM and 1 on a fatal error.
fn daemon_main(idle_exit: Duration, model_cache: Option<PathBuf>) -> ExitCode {
    use repo_search::daemon::coreml::CoreMlBackend;
    use repo_search::daemon::{self, Config};
    use repo_search::launchd;
    use repo_search::model_store::{ModelSpec, ModelStore};
    use repo_search::protocol::BuildId;
    use std::os::fd::OwnedFd;
    use std::sync::Arc;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    let build = match BuildId::of_current_exe() {
        Ok(build) => build,
        Err(error) => {
            tracing::error!(%error, "could not identify the running binary");
            return ExitCode::FAILURE;
        }
    };

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
        Ok(fds) => fds,
        Err(error) => {
            tracing::error!(%error, "launchd socket activation failed");
            return ExitCode::FAILURE;
        }
    };
    let [fd]: [OwnedFd; 1] = match fds.try_into() {
        Ok(fds) => fds,
        Err(fds) => {
            tracing::error!(
                count = fds.len(),
                "launchd returned a Listener fd count other than 1"
            );
            return ExitCode::FAILURE;
        }
    };

    let model_store = match model_cache {
        Some(root) => ModelStore::new(root, ModelSpec::embedding_gemma_2()),
        None => ModelStore::user_cache(),
    };
    let model_store = match model_store {
        Ok(store) => store,
        Err(error) => {
            tracing::error!(%error, "could not open the model cache");
            return ExitCode::FAILURE;
        }
    };

    // `from_std` registers the fd with this runtime's driver, so it runs
    // inside the runtime's context.
    let _guard = runtime.enter();
    let std_listener = std::os::unix::net::UnixListener::from(fd);
    let listener = std_listener
        .set_nonblocking(true)
        .and_then(|()| tokio::net::UnixListener::from_std(std_listener));
    let listener = match listener {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, "could not adopt the Listener fd");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(daemon::run(Config {
        listener,
        idle_exit,
        model_store,
        backend: Arc::new(CoreMlBackend),
        build,
    })) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "the daemon stopped on a fatal error");
            ExitCode::FAILURE
        }
    }
}
