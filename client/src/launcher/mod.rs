//! Starting the daemon on behalf of a host.
//!
//! A host hands a launcher the path of the daemon program and the
//! handshake whose [`BuildId`](crate::protocol::BuildId) is that file's
//! own ([`BuildId::of_file`](crate::protocol::BuildId::of_file)); nothing
//! here assumes the daemon is the running executable. [`connect`] then
//! performs the sequence every client runs: the cheap
//! [`Launcher::prepare`] check, a connect, and — after a recoverable
//! failure — [`Launcher::recover`] and one reconnect.
//!
//! On macOS the daemon and the embedding server run as launchd agents
//! ([`LaunchdAgents`]); on Linux the daemon is spawned on demand under
//! `spawn.lock` ([`spawn::OnDemand`]).

#[cfg(target_os = "macos")]
mod agents;
#[cfg(target_os = "macos")]
pub mod install;
#[cfg(target_os = "macos")]
pub mod kqueue;
#[cfg(target_os = "macos")]
pub mod launchd;
pub mod paths;
pub mod spawn;

use std::future::Future;
use std::path::Path;
use std::time::Duration;

#[cfg(target_os = "macos")]
pub use agents::LaunchdAgents;

use crate::client::{Client, ClientError};
use crate::protocol::Hello;

/// The daemon exits after this long with no connections and no work.
pub const DAEMON_IDLE_EXIT: Duration = Duration::from_secs(600);

/// The embedding server exits after this long with no request in flight,
/// nothing queued and the loader not busy.
pub const SERVER_IDLE_EXIT: Duration = Duration::from_secs(600);

/// `launchctl bootout` returns before launchd finishes the teardown; the
/// agent's process exit is awaited this long before `bootstrap`. On Linux
/// it is the old daemon's shutdown tail a replacement waits out.
pub const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// `launchctl bootstrap` returns once the job is loaded; launchd binds the
/// `SockPathName` a moment later, awaited this long. On Linux it is the
/// new daemon's startup after the old one is gone.
pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(10);

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

/// Why [`connect`] did not return a client.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// A launcher step failed; the message is the step's
    /// [`Report::error`].
    #[error("{0}")]
    Launcher(String),
    /// The connect failed, and either no recovery could cure it or the
    /// reconnect after the recovery failed too.
    #[error(transparent)]
    Client(#[from] ClientError),
}

/// Connects to the daemon the launcher keeps: after `prepared` — the
/// report of [`Launcher::prepare`], which the caller runs first so it can
/// overlap other work — it connects, and on a failure the launcher deems
/// recoverable it runs [`Launcher::recover`] and connects once more.
///
/// Every step's notes are handed to `note` in order, as each step ends.
///
/// # Errors
///
/// [`ConnectError::Launcher`] when `prepared` or the recovery failed,
/// [`ConnectError::Client`] when a connect failed for good.
pub async fn connect<L: Launcher>(
    launcher: &mut L,
    hello: &Hello,
    prepared: Report,
    mut note: impl AsyncFnMut(&str),
) -> Result<Client, ConnectError> {
    finish(prepared, &mut note).await?;
    match Client::connect(launcher.socket(), hello).await {
        Ok(client) => Ok(client),
        Err(error) if launcher.recoverable(&error) => {
            finish(launcher.recover().await, &mut note).await?;
            Ok(Client::connect(launcher.socket(), hello).await?)
        }
        Err(error) => Err(error.into()),
    }
}

/// A step's notes told in order, then its error as the failure.
async fn finish(report: Report, note: &mut impl AsyncFnMut(&str)) -> Result<(), ConnectError> {
    for line in &report.notes {
        note(line).await;
    }
    match report.error {
        Some(error) => Err(ConnectError::Launcher(error)),
        None => Ok(()),
    }
}
