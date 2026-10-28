//! How the client gets a daemon to talk to: the [`Launcher`] trait and its
//! launchd implementation.

use std::future::Future;
use std::io;
use std::path::Path;

use crate::client::ClientError;
use crate::install::{Agent, AgentConfig, Check, InstallError, Launchctl, ensure_agent};

/// What one launcher step tells the user: `notes` are told in order, then
/// `error`, which also fails the step.
#[derive(Debug, Default)]
pub struct Report {
    /// Informational lines.
    pub notes: Vec<String>,
    /// The failure that ended the step.
    pub error: Option<String>,
}

/// Keeps a daemon reachable at [`Launcher::socket`] on behalf of the
/// client.
pub trait Launcher {
    /// The socket the daemon listens on.
    fn socket(&self) -> &Path;
    /// The file the daemon's stderr goes to.
    fn log(&self) -> &Path;
    /// The cheap check before the first connect: install or replace
    /// whatever is stale, touching nothing when all is current.
    fn prepare(&mut self) -> impl Future<Output = Report>;
    /// Whether a failed connect means the daemon can be brought back by
    /// [`Launcher::recover`].
    fn recoverable(&self, error: &ClientError) -> bool;
    /// The thorough step after a recoverable connect failure: make the
    /// daemon reachable again.
    fn recover(&mut self) -> impl Future<Output = Report>;
}

/// The launchd agents of a macOS user: the embedding server's and the
/// daemon's, kept current through `launchctl`.
pub struct LaunchdAgents<L: Launchctl> {
    /// Taken for the duration of an install step, which runs off the
    /// executor and hands the parts back.
    parts: Option<Parts<L>>,
}

struct Parts<L> {
    /// The embedding server's agent, ensured before the daemon's.
    server_agent: AgentConfig,
    /// The daemon agent the client keeps current.
    agent: AgentConfig,
    /// The `launchctl` implementation.
    launchctl: L,
}

impl<L: Launchctl> LaunchdAgents<L> {
    /// The agents, installed through `launchctl`.
    pub fn new(server_agent: AgentConfig, agent: AgentConfig, launchctl: L) -> Self {
        Self {
            parts: Some(Parts {
                server_agent,
                agent,
                launchctl,
            }),
        }
    }

    fn parts(&self) -> &Parts<L> {
        self.parts.as_ref().expect("no install step is in flight")
    }
}

impl<L: Launchctl + Send + 'static> LaunchdAgents<L> {
    /// Runs `ensure_agent` for both agents off the executor (flock,
    /// launchctl, kqueue) and reports what it did.
    async fn ensure(&mut self, check: Check) -> Report {
        let parts = self.parts.take().expect("no install step is in flight");
        let (parts, server, daemon) = tokio::task::spawn_blocking(move || {
            let Parts {
                server_agent,
                agent,
                mut launchctl,
            } = parts;
            // The server agent first, then the daemon's — sequentially, so
            // the shared install lock serializes the two agents' steps.
            let server = ensure_agent(&server_agent, &mut launchctl, check);
            let daemon = ensure_agent(&agent, &mut launchctl, check);
            (
                Parts {
                    server_agent,
                    agent,
                    launchctl,
                },
                server,
                daemon,
            )
        })
        .await
        .expect("the install thread does not panic");
        let report = report_both([(server, &parts.server_agent), (daemon, &parts.agent)]);
        self.parts = Some(parts);
        report
    }
}

impl<L: Launchctl + Send + 'static> Launcher for LaunchdAgents<L> {
    fn socket(&self) -> &Path {
        self.parts().agent.agent.socket()
    }

    fn log(&self) -> &Path {
        self.parts().agent.agent.stderr()
    }

    async fn prepare(&mut self) -> Report {
        self.ensure(Check::Fast).await
    }

    /// A refused or missing socket after `Check::Fast` means the trust was
    /// wrong: `Check::Verify` asks launchd itself.
    fn recoverable(&self, error: &ClientError) -> bool {
        matches!(
            error,
            ClientError::Connect(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                )
        )
    }

    async fn recover(&mut self) -> Report {
        self.ensure(Check::Verify).await
    }
}

/// The server agent's then the daemon agent's `ensure` outcome as one
/// report, ending at the first error.
fn report_both(outcomes: [(Result<Agent, InstallError>, &AgentConfig); 2]) -> Report {
    let mut report = Report::default();
    for (outcome, config) in outcomes {
        match outcome {
            Ok(outcome) => report.notes.extend(announcement(outcome, config)),
            Err(error) => {
                report.error = Some(error.to_string());
                break;
            }
        }
    }
    report
}

/// The `installed`/`replaced` stderr line an install outcome reports —
/// written on a TTY or not.
fn announcement(outcome: Agent, agent: &AgentConfig) -> Option<String> {
    match outcome {
        Agent::Current => None,
        Agent::Installed => Some(format!("installed launch agent {}", agent.plist.display())),
        Agent::Replaced(reason) => Some(format!(
            "replaced launch agent {} ({reason})",
            agent.plist.display()
        )),
    }
}
