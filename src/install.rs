//! Installing and replacing the per-user LaunchAgent.
//!
//! [`ensure_agent`] makes launchd's state match the [`AgentConfig`]: a byte
//! comparison of the rendered plist catches a moved binary or any other
//! changed input, and [`Check::Verify`] asks launchd itself whether the
//! service is loaded. The whole flow runs under an `flock(2)` file lock so
//! two `search` processes racing a first use bootstrap the agent once.
//!
//! It is synchronous — `launchctl` processes, the flock and the kqueue
//! waits are all short kernel waits — and callers run it inside
//! `spawn_blocking`.

use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::Duration;

use askama::Template;

use crate::daemon::IDLE_EXIT;
use crate::kqueue::{DirWatch, ExitWatch};
use crate::launchd::{LABEL, LaunchAgent, domain, lock_path, log_path, plist_path, socket_path};

/// `launchctl bootout` returns before launchd finishes the teardown; the
/// agent's process exit is awaited this long before `bootstrap`.
pub const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// `launchctl bootstrap` returns once the job is loaded; launchd binds the
/// `SockPathName` a moment later, awaited this long.
pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(10);

/// Everything the install flow touches; production values come from
/// [`AgentConfig::user`], test values from the tests.
pub struct AgentConfig {
    /// The launchd domain: `user/<uid>`.
    pub domain: String,
    /// Where the agent's plist is written.
    pub plist: PathBuf,
    /// The `flock(2)` file serializing installs.
    pub lock: PathBuf,
    /// The rendered agent: label, program, socket, idle exit, model cache,
    /// stderr log.
    pub agent: LaunchAgent,
    /// How long the bootout teardown and the socket wait may take.
    pub timeouts: Timeouts,
}

/// The two waits [`ensure_agent`] performs.
pub struct Timeouts {
    /// Waiting for the old daemon's exit after `bootout`.
    pub teardown: Duration,
    /// Waiting for launchd to bind the socket after `bootstrap`.
    pub socket: Duration,
}

impl AgentConfig {
    /// The real per-user agent: [`LABEL`], the `search` binary at
    /// `program`, and the production socket, plist, lock and log paths.
    pub fn user(program: PathBuf) -> io::Result<Self> {
        Ok(Self {
            domain: domain(),
            plist: plist_path(LABEL)?,
            lock: lock_path()?,
            agent: LaunchAgent {
                label: LABEL.to_owned(),
                program,
                socket: socket_path()?,
                idle_exit: IDLE_EXIT,
                model_cache: None,
                stderr: log_path()?,
            },
            timeouts: Timeouts {
                teardown: TEARDOWN_TIMEOUT,
                socket: SOCKET_TIMEOUT,
            },
        })
    }

    /// `<domain>/<label>`, the `launchctl` service target.
    pub fn service(&self) -> String {
        format!("{}/{}", self.domain, self.agent.label)
    }
}

/// The `launchctl` operations the install flow needs. Production runs
/// `/bin/launchctl` ([`SystemLaunchctl`]); tests pass a fake.
pub trait Launchctl {
    /// `launchctl print <service>`: `None` when launchd does not know the
    /// service.
    fn service(&mut self, service: &str) -> Result<Option<Service>, LaunchctlError>;
    /// `launchctl bootout <service>`.
    fn bootout(&mut self, service: &str) -> Result<(), LaunchctlError>;
    /// `launchctl bootstrap <domain> <plist>`.
    fn bootstrap(&mut self, domain: &str, plist: &Path) -> Result<(), LaunchctlError>;
}

/// What `launchctl print` says about a service.
pub struct Service {
    /// The running process's pid; `None` when the job is loaded but no
    /// process runs (a socket-activated agent between runs).
    pub pid: Option<u32>,
}

/// `/bin/launchctl` itself.
pub struct SystemLaunchctl;

impl SystemLaunchctl {
    /// Runs `launchctl <args>` and returns its output plus the argument
    /// list as it appears in error messages.
    fn run(args: &[&OsStr]) -> Result<(std::process::Output, String), LaunchctlError> {
        let display = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ");
        std::process::Command::new("/bin/launchctl")
            .args(args)
            .output()
            .map(|output| (output, display.clone()))
            .map_err(|source| LaunchctlError::Spawn {
                args: display,
                source,
            })
    }

    /// A nonzero exit keeps launchctl's own stderr verbatim, trimmed.
    fn failed(args: String, output: std::process::Output) -> LaunchctlError {
        LaunchctlError::Failed {
            args,
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        }
    }
}

impl Launchctl for SystemLaunchctl {
    fn service(&mut self, service: &str) -> Result<Option<Service>, LaunchctlError> {
        let (output, args) = Self::run(&[OsStr::new("print"), OsStr::new(service)])?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let pid = stdout
                .lines()
                .map(str::trim)
                .find_map(|line| line.strip_prefix("pid = "))
                .map(|value| {
                    value
                        .trim()
                        .parse::<u32>()
                        .expect("`launchctl print`'s pid parses")
                });
            return Ok(Some(Service { pid }));
        }
        // Exit 113 is "Could not find service".
        if output.status.code() == Some(113) {
            return Ok(None);
        }
        Err(Self::failed(args, output))
    }

    fn bootout(&mut self, service: &str) -> Result<(), LaunchctlError> {
        let (output, args) = Self::run(&[OsStr::new("bootout"), OsStr::new(service)])?;
        if output.status.success() {
            Ok(())
        } else {
            Err(Self::failed(args, output))
        }
    }

    fn bootstrap(&mut self, domain: &str, plist: &Path) -> Result<(), LaunchctlError> {
        let (output, args) = Self::run(&[
            OsStr::new("bootstrap"),
            OsStr::new(domain),
            plist.as_os_str(),
        ])?;
        if output.status.success() {
            Ok(())
        } else {
            Err(Self::failed(args, output))
        }
    }
}

/// A `launchctl` invocation's failure.
#[derive(Debug, thiserror::Error)]
pub enum LaunchctlError {
    /// `launchctl` could not be spawned at all.
    #[error("could not run `launchctl {args}`: {source}")]
    Spawn {
        /// The argument list, space-joined.
        args: String,
        /// The spawn error.
        source: io::Error,
    },
    /// `launchctl` ran and exited nonzero, stderr verbatim.
    #[error("`launchctl {args}` exited with {status}: {stderr}")]
    Failed {
        /// The argument list, space-joined.
        args: String,
        /// The exit status.
        status: ExitStatus,
        /// What launchd said, trimmed.
        stderr: String,
    },
}

/// How sure [`ensure_agent`] must be before doing nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// The plist matches and the socket file exists — no `launchctl` call.
    Fast,
    /// Ask launchd itself whether the service is loaded.
    Verify,
}

/// What [`ensure_agent`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agent {
    /// Already current; nothing was touched.
    Current,
    /// The agent was not installed; it is now.
    Installed,
    /// An existing agent was replaced.
    Replaced(Reason),
}

/// Why an existing agent was replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The rendered plist differs from the installed one — a moved binary
    /// or any other changed input.
    PlistChanged,
    /// The plist is current but launchd does not have the service loaded,
    /// or the socket file is missing.
    NotRunning,
}

/// A failure of [`ensure_agent`].
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    /// A filesystem step failed; `what` names the step and `path` names
    /// the file.
    #[error("{what} {path}: {source}")]
    Io {
        /// The failed step: `"cannot create"`, `"cannot write"`,
        /// `"cannot lock"`, `"cannot read"`, `"cannot remove"` or
        /// `"cannot watch"`.
        what: &'static str,
        /// The file the step was working on (the watched process's
        /// `pid <n>` stands in for a path on the exit watch).
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
    /// The plist template did not render.
    #[error("could not render the launch agent plist: {0}")]
    Render(#[from] askama::Error),
    /// A `launchctl` call failed.
    #[error(transparent)]
    Launchctl(#[from] LaunchctlError),
    /// The booted-out daemon did not exit in time.
    #[error("the old daemon (pid {pid}) did not exit within {after:?} of `launchctl bootout`")]
    Teardown {
        /// The daemon's pid.
        pid: u32,
        /// How long the exit was awaited.
        after: Duration,
    },
    /// `bootstrap` succeeded but launchd did not bind the socket in time.
    #[error("launchd did not create {path} within {after:?} of `launchctl bootstrap`")]
    Socket {
        /// The awaited socket path.
        path: PathBuf,
        /// How long it was awaited.
        after: Duration,
    },
}

/// The plist file's contents: `None` when absent, an [`InstallError::Io`]
/// on a real read failure.
fn read_plist(path: &Path) -> Result<Option<Vec<u8>>, InstallError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(InstallError::Io {
            what: "cannot read",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn io_err(what: &'static str, path: impl Into<PathBuf>, source: io::Error) -> InstallError {
    InstallError::Io {
        what,
        path: path.into(),
        source,
    }
}

fn create_dir(dir: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(dir).map_err(|e| io_err("cannot create", dir, e))
}

/// Makes launchd's state match `config`, installing or replacing the
/// agent as needed. Synchronous: every step is a short filesystem call, a
/// `launchctl` process or a kernel wait.
pub fn ensure_agent<L: Launchctl>(
    config: &AgentConfig,
    launchctl: &mut L,
    check: Check,
) -> Result<Agent, InstallError> {
    let desired = config.agent.render()?;

    // The fast path trusts "plist current and socket file present" so a
    // warm search spawns no `launchctl` at all.
    if matches!(check, Check::Fast)
        && read_plist(&config.plist)?.as_deref() == Some(desired.as_bytes())
        && config.agent.socket.exists()
    {
        return Ok(Agent::Current);
    }

    // The lock serializes the steps below; it is held until this function
    // returns, through bootout, bootstrap and the socket wait.
    let lock_dir = config.lock.parent().expect("the lock path has a parent");
    create_dir(lock_dir)?;
    let lock = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&config.lock)
        .map_err(|e| io_err("cannot create", &config.lock, e))?;
    lock.lock()
        .map_err(|e| io_err("cannot lock", &config.lock, e))?;

    // Re-read under the lock: another process may have installed while
    // this one waited for it.
    let state = launchctl.service(&config.service())?;
    let on_disk = read_plist(&config.plist)?;
    if on_disk.as_deref() == Some(desired.as_bytes())
        && state.is_some()
        && config.agent.socket.exists()
    {
        return Ok(Agent::Current);
    }

    let outcome = if on_disk.is_none() && state.is_none() {
        Agent::Installed
    } else if on_disk.as_deref() != Some(desired.as_bytes()) {
        Agent::Replaced(Reason::PlistChanged)
    } else {
        Agent::Replaced(Reason::NotRunning)
    };

    // Tear down a loaded agent. `bootout` returns before launchd finishes:
    // the slow part is the daemon's own SIGTERM shutdown, so when the
    // service has a process its exit is the event that ends it. With no
    // process there is nothing asynchronous left and `bootstrap` can
    // follow `bootout` directly.
    if let Some(Service { pid }) = state {
        let exit = pid
            .map(|pid| {
                ExitWatch::register(pid)
                    .map_err(|e| io_err("cannot watch", PathBuf::from(format!("pid {pid}")), e))
            })
            .transpose()?;
        launchctl.bootout(&config.service())?;
        if let (Some(exit), Some(pid)) = (exit, pid)
            && !exit.wait(config.timeouts.teardown)
        {
            return Err(InstallError::Teardown {
                pid,
                after: config.timeouts.teardown,
            });
        }
    }

    let plist_dir = config.plist.parent().expect("the plist path has a parent");
    let socket_dir = config
        .agent
        .socket
        .parent()
        .expect("the socket path has a parent");
    let log_dir = config
        .agent
        .stderr
        .parent()
        .expect("the log path has a parent");
    // `~/Library/LaunchAgents` may not exist on a fresh account, and
    // launchd does not create the `StandardErrorPath` directory — without
    // it the daemon starts with no log.
    create_dir(plist_dir)?;
    create_dir(socket_dir)?;
    create_dir(log_dir)?;

    // With the agent unloaded under the lock nothing owns the socket; a
    // stale file would satisfy the wait below before launchd binds.
    match std::fs::remove_file(&config.agent.socket) {
        Ok(()) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => return Err(io_err("cannot remove", &config.agent.socket, source)),
    }

    // Atomic: `NamedTempFile` lands the plist at its path only once fully
    // written. launchd ignores non-`.plist` names, so the temp file is
    // harmless mid-write.
    let mut tmp = tempfile::NamedTempFile::new_in(plist_dir)
        .map_err(|e| io_err("cannot create", plist_dir, e))?;
    tmp.write_all(desired.as_bytes())
        .map_err(|e| io_err("cannot write", &config.plist, e))?;
    tmp.persist(&config.plist)
        .map_err(|e| io_err("cannot write", &config.plist, e.error))?;

    // Registered before `bootstrap` so the socket's creation — which
    // launchd performs asynchronously — cannot be missed.
    let watch =
        DirWatch::register(socket_dir).map_err(|e| io_err("cannot watch", socket_dir, e))?;
    launchctl.bootstrap(&config.domain, &config.plist)?;
    if !watch.wait_for(&config.agent.socket, config.timeouts.socket) {
        return Err(InstallError::Socket {
            path: config.agent.socket.clone(),
            after: config.timeouts.socket,
        });
    }
    Ok(outcome)
}
