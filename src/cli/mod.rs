//! The `search` client: resolve the worktree, ensure the agent, connect,
//! stream the reply, render the `Final`.
//!
//! Everything the client writes is program output through [`Terminal`]:
//! the rendered reply on stdout, progress and `search: <message>` lines
//! on stderr. The client installs no tracing subscriber and has no log.

mod args;
mod progress;
mod render;

pub use args::{Cli, Command, Invocation, Mode, SearchOptions};
pub use render::{IndexedOutput, StatusOutput};

use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use futures_util::StreamExt;
use progress::ProgressLine;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::client::{Client, ClientError};
use crate::install::{
    Agent, AgentConfig, Check, InstallError, Launchctl, SystemLaunchctl, ensure_agent,
};
use crate::protocol::{BuildId, Event, Final, Format, Hello, Request, SearchRequest};

/// The process's own environment: the working directory, the agent to
/// install, the `launchctl` that installs it and the handshake to send.
/// Production values come from [`Env::user`], test values from the tests.
pub struct Env<L: Launchctl> {
    /// The directory `search` ran in; the worktree is resolved under it.
    pub cwd: PathBuf,
    /// The embedding server's agent, ensured before the daemon's.
    pub server_agent: AgentConfig,
    /// The daemon agent the client keeps current.
    pub agent: AgentConfig,
    /// The `launchctl` implementation.
    pub launchctl: L,
    /// The handshake: this binary's protocol, version and file identity.
    pub hello: Hello,
}

/// A failure of [`Env::user`]: one of the process's own paths did not
/// resolve.
#[derive(Debug, thiserror::Error)]
#[error("{what}: {source}")]
pub struct EnvError {
    what: &'static str,
    source: io::Error,
}

impl EnvError {
    fn new(what: &'static str, source: io::Error) -> Self {
        Self { what, source }
    }
}

impl Env<SystemLaunchctl> {
    /// The real user: `cwd`, the canonical current executable, its
    /// [`BuildId`], and [`AgentConfig::user`]. Runs synchronously before
    /// any runtime exists; every step is a cheap filesystem call.
    pub fn user() -> Result<Self, EnvError> {
        let cwd = std::env::current_dir()
            .map_err(|e| EnvError::new("cannot read the working directory", e))?;
        let program = std::env::current_exe()
            .and_then(|exe| exe.canonicalize())
            .map_err(|e| EnvError::new("cannot identify the running executable", e))?;
        // Canonical, so that a symlink repointed at a new install (a
        // versioned directory, a package manager's shim) changes the plist
        // and reinstalls.
        let hello = Hello::new(
            BuildId::of_file(&program)
                .map_err(|e| EnvError::new("cannot stat the running executable", e))?,
        );
        let server_agent = AgentConfig::server(program.clone())
            .map_err(|e| EnvError::new("cannot resolve the server agent's paths", e))?;
        let agent = AgentConfig::user(program)
            .map_err(|e| EnvError::new("cannot resolve the agent's paths", e))?;
        Ok(Self {
            cwd,
            server_agent,
            agent,
            launchctl: SystemLaunchctl,
            hello,
        })
    }
}

/// The client's stdout and stderr. `stderr_tty` decides between the
/// redrawn progress line and the two one-time announcements.
pub struct Terminal<O, E> {
    /// Where the rendered reply goes.
    pub stdout: O,
    /// Where progress and `search: <message>` lines go.
    pub stderr: E,
    /// Whether `stderr` is a terminal.
    pub stderr_tty: bool,
}

impl Terminal<tokio::io::Stdout, tokio::io::Stderr> {
    /// The real standard streams.
    pub fn stdio() -> Self {
        use std::io::IsTerminal;
        Self {
            stdout: tokio::io::stdout(),
            stderr: tokio::io::stderr(),
            stderr_tty: std::io::stderr().is_terminal(),
        }
    }
}

/// The process's exit code; grep's: 0 results, 1 none, 2 error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The request succeeded.
    Ok = 0,
    /// The query matched nothing.
    NoMatches = 1,
    /// Every error and every usage error.
    Error = 2,
}

impl From<Exit> for ExitCode {
    fn from(exit: Exit) -> Self {
        ExitCode::from(exit as u8)
    }
}

/// `search: <message>\n` on stderr. Write errors are ignored — there is
/// nowhere left to report them.
async fn tell(stderr: &mut (impl AsyncWrite + Unpin), message: &str) {
    let _ = stderr
        .write_all(format!("search: {message}\n").as_bytes())
        .await;
}

/// Bytes as decimal MB with one decimal place.
fn mb(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / 1e6)
}

/// Announces the server agent's then the daemon agent's `ensure`
/// outcome; false, after telling the error, when either failed.
async fn announce_both(
    outcomes: [(Result<Agent, InstallError>, &AgentConfig); 2],
    stderr: &mut (impl AsyncWrite + Unpin),
) -> bool {
    for (outcome, config) in outcomes {
        match outcome {
            Ok(outcome) => announce(outcome, config, stderr).await,
            Err(error) => {
                tell(stderr, &error.to_string()).await;
                return false;
            }
        }
    }
    true
}

/// The `installed`/`replaced` stderr line an install outcome reports —
/// written on a TTY or not.
async fn announce(outcome: Agent, agent: &AgentConfig, stderr: &mut (impl AsyncWrite + Unpin)) {
    let line = match outcome {
        Agent::Current => return,
        Agent::Installed => format!("installed launch agent {}", agent.plist.display()),
        Agent::Replaced(reason) => {
            format!("replaced launch agent {} ({reason})", agent.plist.display())
        }
    };
    tell(stderr, &line).await;
}

/// The worktree `cwd` belongs to, as `git rev-parse` prints it — the
/// daemon canonicalizes and checks it.
async fn resolve_worktree(cwd: &Path) -> Result<PathBuf, String> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--path-format=absolute", "--show-toplevel"])
        .output()
        .await
        .map_err(|e| format!("cannot run git: {e}"))?;
    if output.status.success() {
        let root = String::from_utf8_lossy(&output.stdout);
        return Ok(PathBuf::from(root.trim_end()));
    }
    Err(format!(
        "{} is not inside a git worktree: {}",
        cwd.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

/// Runs `ensure_agent` off the executor (flock, launchctl, kqueue) and
/// hands back the config and handle it consumed.
async fn ensure<L: Launchctl + Send + 'static>(
    server_agent: AgentConfig,
    agent: AgentConfig,
    launchctl: L,
    check: Check,
) -> (
    AgentConfig,
    AgentConfig,
    L,
    Result<Agent, InstallError>,
    Result<Agent, InstallError>,
) {
    tokio::task::spawn_blocking(move || {
        let mut launchctl = launchctl;
        // The server agent first, then the daemon's — sequentially, so
        // the shared install lock serializes the two agents' steps.
        let server = ensure_agent(&server_agent, &mut launchctl, check);
        let daemon = ensure_agent(&agent, &mut launchctl, check);
        (server_agent, agent, launchctl, server, daemon)
    })
    .await
    .expect("the install thread does not panic")
}

/// One request's end to end: install check and worktree resolution
/// concurrently, connect, stream progress to stderr, render the `Final`
/// and return the grep exit code. `main` dispatches `Invocation::Daemon`
/// before `run`, so the variant is unreachable here.
pub async fn run<L, O, E>(invocation: Invocation, env: Env<L>, term: &mut Terminal<O, E>) -> Exit
where
    L: Launchctl + Send + 'static,
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    if matches!(invocation, Invocation::Daemon { .. }) {
        unreachable!("main dispatches --daemon before `run`");
    }
    let Env {
        cwd,
        server_agent,
        agent,
        launchctl,
        hello,
    } = env;

    let (worktree, ensured) = tokio::join!(
        resolve_worktree(&cwd),
        ensure(server_agent, agent, launchctl, Check::Fast)
    );
    let worktree = match worktree {
        Ok(worktree) => worktree,
        Err(message) => {
            tell(&mut term.stderr, &message).await;
            return Exit::Error;
        }
    };
    let (server_agent, mut agent, launchctl, server_ensured, ensured) = ensured;
    if !announce_both(
        [(server_ensured, &server_agent), (ensured, &agent)],
        &mut term.stderr,
    )
    .await
    {
        return Exit::Error;
    }

    // A refused or missing socket after `Check::Fast` means the trust was
    // wrong: `Check::Verify` asks launchd itself, then connect once more.
    let client = match Client::connect(agent.agent.socket(), &hello).await {
        Ok(client) => client,
        Err(ClientError::Connect(error))
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            let (checked_server, checked, _, server_ensured, ensured) =
                ensure(server_agent, agent, launchctl, Check::Verify).await;
            agent = checked;
            if !announce_both(
                [(server_ensured, &checked_server), (ensured, &agent)],
                &mut term.stderr,
            )
            .await
            {
                return Exit::Error;
            }
            match Client::connect(agent.agent.socket(), &hello).await {
                Ok(client) => client,
                Err(error) => {
                    tell(
                        &mut term.stderr,
                        &render::client_error(&error, agent.agent.socket(), agent.agent.stderr()),
                    )
                    .await;
                    return Exit::Error;
                }
            }
        }
        Err(error) => {
            tell(
                &mut term.stderr,
                &render::client_error(&error, agent.agent.socket(), agent.agent.stderr()),
            )
            .await;
            return Exit::Error;
        }
    };

    let request = match &invocation {
        Invocation::Search { query, options } => Request::Search(SearchRequest {
            worktree: worktree.clone(),
            query: query.clone(),
            mode: options.mode.into(),
            paths: options.path.clone(),
            langs: options.lang.clone(),
            kinds: options.kind.clone(),
            symbol: options.symbol.clone(),
            budget: options.budget,
            files_only: options.files,
            include_ignored: options.include_ignored,
            format: if options.json {
                Format::Json
            } else {
                Format::Text
            },
        }),
        Invocation::Status { .. } => Request::Status {
            worktree: worktree.clone(),
        },
        Invocation::Index { wait, .. } => Request::Index {
            worktree: worktree.clone(),
            wait: *wait,
        },
        Invocation::Daemon { .. } | Invocation::Server { .. } => unreachable!("checked above"),
    };
    let mut responses = match client.request(request).await {
        Ok(responses) => responses,
        Err(error) => {
            tell(
                &mut term.stderr,
                &render::client_error(&error, agent.agent.socket(), agent.agent.stderr()),
            )
            .await;
            return Exit::Error;
        }
    };

    let (kind, json) = match &invocation {
        Invocation::Search { options, .. } => ("search", options.json),
        Invocation::Status { json } => ("status", *json),
        Invocation::Index { json, .. } => ("index", *json),
        Invocation::Daemon { .. } | Invocation::Server { .. } => unreachable!("checked above"),
    };
    let mut progress = ProgressLine::new(term.stderr_tty);
    while let Some(item) = responses.next().await {
        match item {
            Ok(Event::Progress(update)) => progress.show(&update, &mut term.stderr).await,
            Ok(Event::Final(last)) => {
                progress.clear(&mut term.stderr).await;
                return match last {
                    Final::Error(report) => {
                        tell(
                            &mut term.stderr,
                            &render::daemon_error(&report, agent.agent.stderr()),
                        )
                        .await;
                        Exit::Error
                    }
                    Final::Answer(answer) if kind == "search" => {
                        render::answer(&answer, json, term).await
                    }
                    Final::Status(report) if kind == "status" => {
                        render::status(&report, json, term).await
                    }
                    Final::Indexed { with_vector, total } if kind == "index" => {
                        render::indexed(&worktree, with_vector, total, json, term).await
                    }
                    other => {
                        tell(
                            &mut term.stderr,
                            &format!(
                                "the daemon answered a {kind} request with a {} reply",
                                reply_kind(&other)
                            ),
                        )
                        .await;
                        Exit::Error
                    }
                };
            }
            Err(error) => {
                progress.clear(&mut term.stderr).await;
                tell(
                    &mut term.stderr,
                    &render::client_error(&error, agent.agent.socket(), agent.agent.stderr()),
                )
                .await;
                return Exit::Error;
            }
        }
    }
    unreachable!("the reply stream ends at its Final frame");
}

/// The `Final` variant's name, for the protocol-violation message.
fn reply_kind(last: &Final) -> &'static str {
    match last {
        Final::Answer(_) => "answer",
        Final::Status(_) => "status",
        Final::Indexed { .. } => "indexed",
        Final::Error(_) => "error",
    }
}
