//! The `search` command line: `Cli` parses argv and [`Cli::invocation`]
//! turns the parse into what `main` dispatches on. `status` and `index`
//! are subcommands only as a bare first argument — every other first
//! argument is the free-text query.

use std::net::SocketAddr;
use std::os::fd::RawFd;
use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};

use crate::daemon;
use crate::protocol::WireMode;

/// `--version` output: the binary version, the chunker fingerprint and the
/// model's identity from [`repo_search_core::model`]'s card.
pub const VERSION_INFO: &str = const_format::concatcp!(
    env!("CARGO_PKG_VERSION"),
    "\nchunker: ",
    repo_search_core::CHUNKER_FINGERPRINT,
    "\nmodel: ",
    repo_search_core::model::MODEL_NAME,
    "\nlicense: ",
    repo_search_core::model::MODEL_LICENSE
);

/// Search a repository for code you can describe but cannot name.
///
/// Exit codes follow grep: 0 results, 1 none, 2 error.
#[derive(Debug, Parser)]
#[command(
    name = "search",
    version = VERSION_INFO,
    subcommand_precedence_over_arg = true,
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true,
    disable_help_subcommand = true,
    after_help = "To search for the single word `status` or `index`, write `search -- status`."
)]
struct RawCli {
    /// Free-text query.
    #[arg(required_unless_present_any = ["daemon", "server"],
          value_parser = clap::builder::NonEmptyStringValueParser::new())]
    query: Option<String>,

    #[command(flatten)]
    options: SearchOptions,

    #[command(subcommand)]
    command: Option<Command>,

    /// Run as the launchd socket-activated daemon.
    #[arg(long, hide = true, conflicts_with = "query", group = "agent")]
    daemon: bool,

    /// Run as the launchd socket-activated embedding server.
    #[arg(long, hide = true, conflicts_with = "query", group = "agent")]
    server: bool,

    /// The bearer token TCP listeners require, read from this file.
    #[arg(
        long,
        hide = true,
        requires = "server",
        conflicts_with = "query",
        value_name = "FILE"
    )]
    token_file: Option<PathBuf>,

    /// The build this agent registered; the server refuses to run any
    /// other.
    #[arg(
        long,
        hide = true,
        requires = "server",
        conflicts_with = "query",
        value_name = "ID"
    )]
    build: Option<String>,

    /// Exit after this many seconds with no connections and no work.
    #[arg(
        long,
        hide = true,
        requires = "agent",
        conflicts_with = "query",
        value_name = "SECS"
    )]
    idle_exit_secs: Option<u64>,

    /// The model cache root, defaulting to the per-user cache.
    #[arg(
        long,
        hide = true,
        requires = "agent",
        conflicts_with = "query",
        value_name = "DIR"
    )]
    model_cache: Option<PathBuf>,

    /// The daemon's state root, defaulting to the per-user state dir.
    #[arg(
        long,
        hide = true,
        requires = "daemon",
        conflicts_with = "query",
        value_name = "DIR"
    )]
    state_dir: Option<PathBuf>,

    /// The embedding server's unix socket, defaulting to the agent's.
    #[arg(
        long,
        hide = true,
        requires = "daemon",
        conflicts_with = "query",
        value_name = "PATH"
    )]
    server_socket: Option<PathBuf>,

    /// Bind this socket instead of taking launchd's.
    #[arg(
        long,
        hide = true,
        requires = "daemon",
        conflicts_with = "query",
        value_name = "PATH"
    )]
    listen: Option<PathBuf>,

    /// The fd the daemon reports `ready` or `failed: <message>` on.
    #[arg(
        long,
        hide = true,
        requires = "daemon",
        conflicts_with = "query",
        value_name = "FD"
    )]
    ready_fd: Option<RawFd>,

    /// The embedding server's TCP address, an IP literal.
    #[arg(
        long,
        hide = true,
        requires = "daemon",
        conflicts_with_all = ["query", "server_socket"],
        value_name = "IP:PORT"
    )]
    server_addr: Option<SocketAddr>,

    /// The bearer token `--server-addr` sends, read from this file.
    #[arg(
        long,
        hide = true,
        requires = "server_addr",
        conflicts_with = "query",
        value_name = "FILE"
    )]
    server_token_file: Option<PathBuf>,
}

/// The parsed `search` command line.
///
/// `Cli` wraps the clap derive to add the one check the attributes cannot
/// express. With `args_conflicts_with_subcommands`, clap tries subcommand
/// names only until the first valid argument, so `search --budget 10
/// status` reaches the parser as `Search` with the query `status` — the
/// option is applied, never dropped, but the subcommand's name is now a
/// query the user did not write. `search -- status` remains the escape,
/// and subcommand options go after the subcommand.
#[derive(Debug)]
pub struct Cli(RawCli);

impl Cli {
    /// Rejects a `status`/`index` query carrying arguments: `search
    /// --budget 10 status` is a usage error, `search -- status` is not.
    fn checked(raw: RawCli, matches: &ArgMatches) -> Result<Self, clap::Error> {
        let named_query = matches!(raw.query.as_deref(), Some("status" | "index"));
        let arguments = matches.ids().any(|id| {
            id != "query"
                && matches.value_source(id.as_str()) == Some(clap::parser::ValueSource::CommandLine)
        });
        if raw.command.is_none() && named_query && arguments {
            return Err(Self::command().error(
                clap::error::ErrorKind::ArgumentConflict,
                format!(
                    "options cannot precede a subcommand; to search for `{}`, write `search -- {}`",
                    raw.query.as_deref().expect("checked"),
                    raw.query.as_deref().expect("checked"),
                ),
            ));
        }
        Ok(Self(raw))
    }
}

impl clap::CommandFactory for Cli {
    fn command() -> clap::Command {
        RawCli::command()
    }

    fn command_for_update() -> clap::Command {
        RawCli::command_for_update()
    }
}

impl FromArgMatches for Cli {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        Self::checked(RawCli::from_arg_matches(matches)?, matches)
    }

    fn from_arg_matches_mut(matches: &mut ArgMatches) -> Result<Self, clap::Error> {
        Self::checked(RawCli::from_arg_matches_mut(matches)?, matches)
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        self.0.update_from_arg_matches(matches)
    }

    fn update_from_arg_matches_mut(&mut self, matches: &mut ArgMatches) -> Result<(), clap::Error> {
        self.0.update_from_arg_matches_mut(matches)
    }
}

impl Parser for Cli {}

/// Options for `search <query>`.
#[derive(Debug, Args)]
pub struct SearchOptions {
    /// Return results until the token budget is used.
    #[arg(long, default_value_t = 4000, value_parser = clap::value_parser!(u32).range(1..))]
    pub budget: u32,

    /// Ranked file list only.
    #[arg(short = 'l', long)]
    pub files: bool,

    /// Retrieval channels.
    #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
    pub mode: Mode,

    /// Restrict paths (repeatable).
    #[arg(long)]
    pub path: Vec<String>,

    /// Restrict languages (repeatable, comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub lang: Vec<String>,

    /// Restrict syntactic kinds (fn, struct, trait, ...).
    #[arg(long, value_delimiter = ',')]
    pub kind: Vec<String>,

    /// Restrict symbol path, e.g. `Environment::*`.
    #[arg(long)]
    pub symbol: Option<String>,

    /// Include files `.gitignore` excludes.
    #[arg(long)]
    pub include_ignored: bool,

    /// Structured output.
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// FTS5 BM25 and vector KNN merged by reciprocal-rank fusion.
    Hybrid,
    /// FTS5 over chunk title and text.
    Lexical,
    /// Vector KNN only.
    Semantic,
}

impl From<Mode> for WireMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Hybrid => Self::Hybrid,
            Mode::Lexical => Self::Lexical,
            Mode::Semantic => Self::Semantic,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Coverage, pending chunks, model, GPU and daemon state.
    Status {
        /// Structured output.
        #[arg(long)]
        json: bool,
    },
    /// Reconcile now.
    Index {
        /// Block until every chunk has a vector.
        #[arg(long)]
        wait: bool,
        /// Structured output.
        #[arg(long)]
        json: bool,
    },
}

/// What a parsed [`Cli`] dispatches on.
#[derive(Debug)]
pub enum Invocation {
    /// `search --server`: the embedding server's entry point.
    Server {
        /// `--idle-exit-secs`, when given; the server's own default
        /// applies otherwise.
        idle_exit: Option<Duration>,
        /// `--model-cache`, when given.
        model_cache: Option<PathBuf>,
        /// `--token-file`, when given.
        token_file: Option<PathBuf>,
        /// `--build`, when given.
        build: Option<String>,
    },
    /// `search --daemon`: the agent's entry point, socket-activated on
    /// macOS and self-bound under spawn on Linux.
    Daemon {
        /// Idle deadline; [`daemon::IDLE_EXIT`] unless `--idle-exit-secs`.
        idle_exit: Duration,
        /// `--model-cache`, when given.
        model_cache: Option<PathBuf>,
        /// `--state-dir`, when given.
        state_dir: Option<PathBuf>,
        /// `--server-socket`, when given.
        server_socket: Option<PathBuf>,
        /// `--listen`: bind this socket instead of activating.
        listen: Option<PathBuf>,
        /// `--ready-fd`, when given.
        ready_fd: Option<RawFd>,
        /// `--server-addr`, when given.
        server_addr: Option<SocketAddr>,
        /// `--server-token-file`, when given.
        server_token_file: Option<PathBuf>,
    },
    /// `search <query>`.
    Search {
        /// The free-text query.
        query: String,
        /// The search options.
        options: SearchOptions,
    },
    /// `search status`.
    Status {
        /// `--json`.
        json: bool,
    },
    /// `search index`.
    Index {
        /// `--wait`.
        wait: bool,
        /// `--json`.
        json: bool,
    },
}

impl Cli {
    /// The invocation this parse describes.
    pub fn invocation(self) -> Invocation {
        let raw = self.0;
        if raw.server {
            return Invocation::Server {
                idle_exit: raw.idle_exit_secs.map(Duration::from_secs),
                model_cache: raw.model_cache,
                token_file: raw.token_file,
                build: raw.build,
            };
        }
        if raw.daemon {
            return Invocation::Daemon {
                idle_exit: raw
                    .idle_exit_secs
                    .map_or(daemon::IDLE_EXIT, Duration::from_secs),
                model_cache: raw.model_cache,
                state_dir: raw.state_dir,
                server_socket: raw.server_socket,
                listen: raw.listen,
                ready_fd: raw.ready_fd,
                server_addr: raw.server_addr,
                server_token_file: raw.server_token_file,
            };
        }
        match raw.command {
            Some(Command::Status { json }) => Invocation::Status { json },
            Some(Command::Index { wait, json }) => Invocation::Index { wait, json },
            None => Invocation::Search {
                query: raw.query.expect("clap requires the query"),
                options: raw.options,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::CommandFactory;
    use repo_search_core::CHUNKER_FINGERPRINT;
    use repo_search_core::model::{MODEL_LICENSE, MODEL_NAME};

    fn parsed(args: &[&str]) -> Invocation {
        Cli::try_parse_from(args)
            .expect("the arguments parse")
            .invocation()
    }

    fn search(invocation: Invocation) -> (String, SearchOptions) {
        match invocation {
            Invocation::Search { query, options } => (query, options),
            other => panic!("expected a Search, got {other:?}"),
        }
    }

    #[test]
    fn version_flag_reports_binary_and_model() {
        assert_eq!(Cli::command().get_version(), Some(VERSION_INFO));
        let text = VERSION_INFO;
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
        assert!(text.contains(CHUNKER_FINGERPRINT));
        assert!(text.contains(MODEL_NAME));
        assert!(text.contains(MODEL_LICENSE));
    }

    #[test]
    fn first_argument_words_are_queries() {
        for arg in ["daemon", "help", "statu", "status page"] {
            let (query, _) = search(parsed(&["search", arg]));
            assert_eq!(query, arg, "{arg:?} is the query, not a subcommand");
        }
    }

    #[test]
    fn double_dash_makes_status_a_query() {
        let (query, _) = search(parsed(&["search", "--", "status"]));
        assert_eq!(query, "status");
    }

    #[test]
    fn bare_status_and_index_are_subcommands() {
        match parsed(&["search", "status"]) {
            Invocation::Status { json } => assert!(!json),
            other => panic!("expected Status, got {other:?}"),
        }
        match parsed(&["search", "status", "--json"]) {
            Invocation::Status { json } => assert!(json),
            other => panic!("expected Status --json, got {other:?}"),
        }
        match parsed(&["search", "index", "--wait", "--json"]) {
            Invocation::Index { wait, json } => {
                assert!(wait);
                assert!(json);
            }
            other => panic!("expected Index --wait --json, got {other:?}"),
        }
    }

    #[test]
    fn search_options_and_subcommand_are_a_usage_error() {
        assert!(Cli::try_parse_from(["search", "--budget", "10", "status"]).is_err());
    }

    #[test]
    fn a_missing_or_empty_query_is_a_usage_error() {
        assert!(Cli::try_parse_from(["search"]).is_err());
        assert!(Cli::try_parse_from(["search", ""]).is_err());
    }

    #[test]
    fn a_zero_budget_is_a_usage_error() {
        assert!(Cli::try_parse_from(["search", "q", "--budget", "0"]).is_err());
    }

    #[test]
    fn lang_and_kind_repeat_and_split_on_commas() {
        let (_, options) = search(parsed(&[
            "search",
            "q",
            "--lang",
            "rust,python",
            "--lang",
            "go",
            "--kind",
            "fn,struct",
        ]));
        assert_eq!(options.lang, ["rust", "python", "go"]);
        assert_eq!(options.kind, ["fn", "struct"]);
    }

    #[test]
    fn server_flags_form_the_server_invocation() {
        match parsed(&["search", "--server"]) {
            Invocation::Server {
                idle_exit,
                model_cache,
                token_file,
                build,
            } => {
                assert_eq!(idle_exit, None);
                assert_eq!(model_cache, None);
                assert_eq!(token_file, None);
                assert_eq!(build, None);
            }
            other => panic!("expected Server, got {other:?}"),
        }
        match parsed(&[
            "search",
            "--server",
            "--idle-exit-secs",
            "30",
            "--token-file",
            "/tmp/tok",
            "--build",
            "1:2:3:4",
        ]) {
            Invocation::Server {
                idle_exit,
                token_file,
                build,
                ..
            } => {
                assert_eq!(idle_exit, Some(Duration::from_secs(30)));
                assert_eq!(token_file, Some(PathBuf::from("/tmp/tok")));
                assert_eq!(build.as_deref(), Some("1:2:3:4"));
            }
            other => panic!("expected Server, got {other:?}"),
        }
    }

    #[test]
    fn server_flags_conflict_with_client_arguments() {
        assert!(Cli::try_parse_from(["search", "--server", "q"]).is_err());
        assert!(Cli::try_parse_from(["search", "q", "--token-file", "f"]).is_err());
        assert!(Cli::try_parse_from(["search", "--token-file", "f"]).is_err());
        assert!(Cli::try_parse_from(["search", "--build", "1:2:3:4"]).is_err());
    }

    #[test]
    fn daemon_flags_form_the_daemon_invocation() {
        match parsed(&["search", "--daemon"]) {
            Invocation::Daemon {
                idle_exit,
                model_cache,
                state_dir,
                server_socket,
                listen,
                ready_fd,
                server_addr,
                server_token_file,
            } => {
                assert_eq!(idle_exit, daemon::IDLE_EXIT);
                assert_eq!(model_cache, None);
                assert_eq!(state_dir, None);
                assert_eq!(server_socket, None);
                assert_eq!(listen, None);
                assert_eq!(ready_fd, None);
                assert_eq!(server_addr, None);
                assert_eq!(server_token_file, None);
            }
            other => panic!("expected Daemon, got {other:?}"),
        }
        match parsed(&["search", "--daemon", "--server-socket", "/tmp/s.sock"]) {
            Invocation::Daemon { server_socket, .. } => {
                assert_eq!(server_socket, Some(PathBuf::from("/tmp/s.sock")));
            }
            other => panic!("expected Daemon, got {other:?}"),
        }
        match parsed(&[
            "search",
            "--daemon",
            "--listen",
            "/tmp/d.sock",
            "--ready-fd",
            "3",
            "--server-addr",
            "127.0.0.1:41827",
            "--server-token-file",
            "/tmp/tok",
        ]) {
            Invocation::Daemon {
                listen,
                ready_fd,
                server_addr,
                server_token_file,
                ..
            } => {
                assert_eq!(listen, Some(PathBuf::from("/tmp/d.sock")));
                assert_eq!(ready_fd, Some(3));
                assert_eq!(server_addr, Some(SocketAddr::from(([127, 0, 0, 1], 41827))));
                assert_eq!(server_token_file, Some(PathBuf::from("/tmp/tok")));
            }
            other => panic!("expected Daemon, got {other:?}"),
        }
    }

    #[test]
    fn daemon_remote_flags_conflict() {
        // `--server-addr` and `--server-socket` exclude each other.
        assert!(
            Cli::try_parse_from([
                "search",
                "--daemon",
                "--server-addr",
                "127.0.0.1:1",
                "--server-socket",
                "/tmp/s.sock",
            ])
            .is_err()
        );
        // `--server-token-file` needs `--server-addr`.
        assert!(
            Cli::try_parse_from(["search", "--daemon", "--server-token-file", "/tmp/t"]).is_err()
        );
        // Every daemon flag needs `--daemon`.
        for args in [
            &["--listen", "/tmp/d.sock"][..],
            &["--ready-fd", "3"][..],
            &["--server-addr", "127.0.0.1:1"][..],
            &[
                "--server-addr",
                "127.0.0.1:1",
                "--server-token-file",
                "/tmp/t",
            ][..],
        ] {
            assert!(
                Cli::try_parse_from(["search"].iter().chain(args).copied().collect::<Vec<_>>())
                    .is_err(),
                "{args:?} without --daemon parses"
            );
        }
    }

    #[test]
    fn daemon_flags_conflict_with_client_arguments() {
        assert!(Cli::try_parse_from(["search", "--daemon", "q"]).is_err());
        assert!(Cli::try_parse_from(["search", "q", "--idle-exit-secs", "3"]).is_err());
    }
}
