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

#[cfg(any(target_os = "macos", target_os = "linux"))]
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use futures_util::StreamExt;
use progress::ProgressLine;
use tokio::io::{AsyncWrite, AsyncWriteExt};

#[cfg(target_os = "linux")]
use repo_search_client::launcher::spawn::OnDemand;
use repo_search_client::launcher::{ConnectError, Launcher};
#[cfg(target_os = "macos")]
use repo_search_client::launcher::{LaunchdAgents, install::SystemLaunchctl};
#[cfg(any(target_os = "macos", target_os = "linux"))]
use repo_search_client::protocol::BuildId;
use repo_search_client::protocol::{Event, Final, Format, Hello, Request, SearchRequest};

/// The process's own environment: the working directory, the launcher
/// that keeps the daemon reachable and the handshake to send. Production
/// values come from [`Env::user`], test values from the tests.
pub struct Env<A: Launcher> {
    /// The directory `search` ran in; the worktree is resolved under it.
    pub cwd: PathBuf,
    /// Brings the daemon up before the request and after a refused connect.
    pub launcher: A,
    /// The handshake: this binary's protocol, version and file identity.
    pub hello: Hello,
}

/// A failure of `Env::user`: one of the process's own paths did not
/// resolve.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[derive(Debug, thiserror::Error)]
#[error("{what}: {source}")]
pub struct EnvError {
    what: &'static str,
    source: io::Error,
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
impl EnvError {
    fn new(what: &'static str, source: io::Error) -> Self {
        Self { what, source }
    }

    /// The working directory, the canonical current executable and its
    /// handshake: the part of the real user's [`Env`] both platforms
    /// share. Runs synchronously before any runtime exists; every step
    /// is a cheap filesystem call.
    fn process() -> Result<(PathBuf, PathBuf, Hello), Self> {
        let cwd = std::env::current_dir()
            .map_err(|e| Self::new("cannot read the working directory", e))?;
        let program = std::env::current_exe()
            .and_then(|exe| exe.canonicalize())
            .map_err(|e| Self::new("cannot identify the running executable", e))?;
        // Canonical, so that a symlink repointed at a new install (a
        // versioned directory, a package manager's shim) changes the
        // launcher's command and replaces the daemon.
        let hello = Hello::new(
            BuildId::of_file(&program)
                .map_err(|e| Self::new("cannot stat the running executable", e))?,
        );
        Ok((cwd, program, hello))
    }
}

#[cfg(target_os = "macos")]
impl Env<LaunchdAgents<SystemLaunchctl>> {
    /// The real user: `cwd`, the canonical current executable, its
    /// [`BuildId`], and [`AgentConfig::user`].
    pub fn user() -> Result<Self, EnvError> {
        let (cwd, program, hello) = EnvError::process()?;
        let launcher = LaunchdAgents::user(program)
            .map_err(|e| EnvError::new("cannot resolve the launch agents", e))?;
        Ok(Self {
            cwd,
            launcher,
            hello,
        })
    }
}

#[cfg(target_os = "linux")]
impl Env<OnDemand> {
    /// The real user: `cwd`, the canonical current executable, its
    /// [`BuildId`], and the daemon [`OnDemand`] spawns from the XDG
    /// paths and `daemon.toml`.
    pub fn user() -> Result<Self, EnvError> {
        let (cwd, program, hello) = EnvError::process()?;
        let launcher = OnDemand::user(program)
            .map_err(|e| EnvError::new("cannot resolve the daemon's paths", e))?;
        Ok(Self {
            cwd,
            launcher,
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

/// One request's end to end: install check and worktree resolution
/// concurrently, connect, stream progress to stderr, render the `Final`
/// and return the grep exit code. `main` dispatches `Invocation::Daemon`
/// before `run`, so the variant is unreachable here.
pub async fn run<A, O, E>(invocation: Invocation, env: Env<A>, term: &mut Terminal<O, E>) -> Exit
where
    A: Launcher,
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    if matches!(invocation, Invocation::Daemon { .. }) {
        unreachable!("main dispatches --daemon before `run`");
    }
    let Env {
        cwd,
        mut launcher,
        hello,
    } = env;

    let (worktree, prepared) = tokio::join!(resolve_worktree(&cwd), launcher.prepare());
    let worktree = match worktree {
        Ok(worktree) => worktree,
        Err(message) => {
            tell(&mut term.stderr, &message).await;
            return Exit::Error;
        }
    };
    let client = match repo_search_client::launcher::connect(
        &mut launcher,
        &hello,
        prepared,
        async |note: &str| tell(&mut term.stderr, note).await,
    )
    .await
    {
        Ok(client) => client,
        Err(ConnectError::Launcher(message)) => {
            tell(&mut term.stderr, &message).await;
            return Exit::Error;
        }
        Err(ConnectError::Client(error)) => {
            tell(
                &mut term.stderr,
                &render::client_error(&error, launcher.socket(), launcher.log()),
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
                &render::client_error(&error, launcher.socket(), launcher.log()),
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
                            &render::daemon_error(&report, launcher.log()),
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
                    &render::client_error(&error, launcher.socket(), launcher.log()),
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
        Final::Collection(_) => "collection",
        Final::Error(_) => "error",
    }
}
