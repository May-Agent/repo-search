//! The `search` command-line client.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::ExitCode;

/// Search a repository for code you can describe but cannot name.
///
/// Exit codes follow grep: 0 results, 1 none, 2 error.
#[derive(Parser)]
#[command(
    name = "search",
    version = repo_search::model_store::VERSION_INFO,
    subcommand_precedence_over_arg = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    /// Free-text query.
    #[arg(required_unless_present = "daemon")]
    query: Option<String>,

    #[command(flatten)]
    options: SearchOptions,

    #[command(subcommand)]
    command: Option<Command>,

    /// Run as the launchd socket-activated daemon.
    #[arg(long, hide = true)]
    daemon: bool,

    /// Exit after this many seconds with no connections and no work.
    #[arg(long, hide = true, requires = "daemon", value_name = "SECS")]
    idle_exit_secs: Option<u64>,

    /// The model cache root, defaulting to the per-user cache.
    #[arg(long, hide = true, requires = "daemon", value_name = "DIR")]
    model_cache: Option<PathBuf>,
}

/// Options for `search <query>`.
#[derive(Args)]
struct SearchOptions {
    /// Return results until the token budget is used.
    #[arg(long, default_value_t = 4000)]
    budget: u32,

    /// Ranked file list only.
    #[arg(short = 'l', long)]
    files: bool,

    /// Retrieval channels.
    #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
    mode: Mode,

    /// Restrict paths (repeatable).
    #[arg(long)]
    path: Vec<String>,

    /// Restrict language.
    #[arg(long)]
    lang: Option<String>,

    /// Restrict syntactic kinds (fn, struct, trait, ...).
    #[arg(long, value_delimiter = ',')]
    kind: Vec<String>,

    /// Restrict symbol path, e.g. `Environment::*`.
    #[arg(long)]
    symbol: Option<String>,

    /// Include files `.gitignore` excludes.
    #[arg(long)]
    include_ignored: bool,

    /// Structured output.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    /// FTS5 BM25 and vector KNN merged by reciprocal-rank fusion.
    Hybrid,
    /// FTS5 over chunk title and text.
    Lexical,
    /// Vector KNN only.
    Semantic,
}

#[derive(Subcommand)]
enum Command {
    /// Coverage, pending chunks, daemon state.
    Status,
    /// Reconcile now.
    Index {
        /// Block until every chunk has a vector.
        #[arg(long)]
        wait: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.daemon {
        return daemon_main(&cli);
    }
    eprintln!("error: the search daemon is not built yet");
    ExitCode::from(2)
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
#[cfg(target_os = "macos")]
fn daemon_main(cli: &Cli) -> ExitCode {
    use repo_search::daemon::coreml::CoreMlBackend;
    use repo_search::daemon::{self, Config};
    use repo_search::launchd;
    use repo_search::model_store::{ModelSpec, ModelStore};
    use std::os::fd::OwnedFd;
    use std::sync::Arc;
    use std::time::Duration;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    let build = match repo_search::protocol::BuildId::of_current_exe() {
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

    let model_store = match cli.model_cache.clone() {
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
        idle_exit: cli
            .idle_exit_secs
            .map_or(daemon::IDLE_EXIT, Duration::from_secs),
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

#[cfg(not(target_os = "macos"))]
fn daemon_main(_: &Cli) -> ExitCode {
    eprintln!("error: --daemon runs on macOS only");
    ExitCode::from(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::CommandFactory;
    use repo_search::model_store::{MODEL_LICENSE, MODEL_NAME, VERSION_INFO};

    #[test]
    fn version_flag_reports_binary_and_model() {
        assert_eq!(Cli::command().get_version(), Some(VERSION_INFO));
        let text = VERSION_INFO;
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
        assert!(text.contains(repo_search_core::CHUNKER_FINGERPRINT));
        assert!(text.contains(MODEL_NAME));
        assert!(text.contains(MODEL_LICENSE));
    }
}
