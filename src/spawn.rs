//! Spawn-on-demand: the client's [`Launcher`] for a machine without
//! launchd. A client that cannot reach the daemon takes `spawn.lock`,
//! starts a detached `search --daemon --listen` and reads its readiness
//! line off a pipe before it lets any other client spawn.
//!
//! Three files sit next to the daemon's socket, in the machine-local
//! runtime dir:
//!
//! - `daemon.lock`: the daemon's lifetime lock, held by the daemon until
//!   its process exits ([`Bound`](crate::daemon::listen::Bound)); its
//!   contents are the daemon's pid.
//! - `spawn.lock`: the clients' spawn lock, held from the liveness probe
//!   until the new daemon reported ready, so concurrent clients spawn
//!   exactly one daemon. Every fd the client opens is `CLOEXEC`, so the
//!   daemon never inherits it.
//! - `daemon.cmd`: the rendered [`DaemonCommand`] the running daemon was
//!   started with, the counterpart of the launchd plist: a client whose
//!   rendered command differs replaces the daemon.
//!
//! The daemon's configuration is `daemon.toml` ([`DaemonConfig`]): where
//! it reaches the embedding server.

use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::net::SocketAddr;
use std::os::fd::{AsRawFd as _, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::net::unix::pipe;

use crate::cli::{Launcher, Report};
use crate::client::ClientError;
use crate::daemon::{SOCKET_TIMEOUT, TEARDOWN_TIMEOUT};
use crate::paths::UserPaths;
use crate::protocol::ErrorKind;

/// How long a client waits for the spawned daemon's readiness line: the
/// previous daemon's shutdown tail plus the new daemon's startup — the
/// two waits the macOS install makes for its agent.
pub const READY_DEADLINE: Duration = TEARDOWN_TIMEOUT.saturating_add(SOCKET_TIMEOUT);

/// The fd the daemon finds the readiness pipe's write end at.
const READY_FD: RawFd = 3;

/// Where the spawned daemon reaches the embedding server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerTarget {
    /// A unix socket: the default forwarded `server.sock`, or the path
    /// `server_socket` names.
    Socket(PathBuf),
    /// A TCP address, an IP literal, with the file holding the bearer
    /// token when the server requires one.
    Tcp {
        /// `server_addr`.
        addr: SocketAddr,
        /// `token_file`.
        token_file: Option<PathBuf>,
    },
}

/// `daemon.toml`: where the daemon reaches the server. Every key is
/// optional; unknown keys, mistyped values and conflicting keys are
/// errors.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DaemonConfig {
    /// `server_socket` or `server_addr` with `token_file`; `None` when
    /// the file sets neither, which means the default socket.
    pub server: Option<ServerTarget>,
}

/// A `daemon.toml` failure.
#[derive(Debug, thiserror::Error)]
pub enum DaemonConfigError {
    /// The file could not be read (other than not existing).
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file read.
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
    /// The file is not valid TOML.
    #[error("{path}: {source}")]
    Parse {
        /// The file parsed.
        path: PathBuf,
        /// The TOML error.
        source: toml_edit::TomlError,
    },
    /// A key the file does not know.
    #[error(
        "{path}: unknown key {key:?} (server_socket, server_addr and token_file are supported)"
    )]
    UnknownKey {
        /// The file parsed.
        path: PathBuf,
        /// The offending key.
        key: String,
    },
    /// A key with the wrong value type.
    #[error("{path}: {key} must be {expected}")]
    Type {
        /// The file parsed.
        path: PathBuf,
        /// The mistyped key.
        key: &'static str,
        /// What the key's value must be.
        expected: &'static str,
    },
    /// `server_socket` and `server_addr` are both set.
    #[error("{path}: server_socket and server_addr are mutually exclusive")]
    Conflict {
        /// The file parsed.
        path: PathBuf,
    },
    /// `token_file` is set without `server_addr`.
    #[error("{path}: token_file needs server_addr")]
    TokenWithoutAddr {
        /// The file parsed.
        path: PathBuf,
    },
}

/// The string value of `item`, which `key` must hold.
fn string<'a>(
    path: &Path,
    key: &'static str,
    item: &'a toml_edit::Item,
    expected: &'static str,
) -> Result<&'a str, DaemonConfigError> {
    item.as_str().ok_or_else(|| DaemonConfigError::Type {
        path: path.to_path_buf(),
        key,
        expected,
    })
}

impl DaemonConfig {
    /// The config at `path`, or the defaults when the file does not
    /// exist.
    pub fn load(path: &Path) -> Result<Self, DaemonConfigError> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(source) => {
                return Err(DaemonConfigError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        Self::parse(path, &text)
    }

    /// The config `text` parses to, with `path` naming errors.
    pub fn parse(path: &Path, text: &str) -> Result<Self, DaemonConfigError> {
        let doc =
            text.parse::<toml_edit::DocumentMut>()
                .map_err(|source| DaemonConfigError::Parse {
                    path: path.to_path_buf(),
                    source,
                })?;
        let mut socket = None;
        let mut addr = None;
        let mut token_file = None;
        for (key, item) in doc.as_table().iter() {
            match key {
                "server_socket" => {
                    socket = Some(PathBuf::from(string(
                        path,
                        "server_socket",
                        item,
                        "a string path",
                    )?));
                }
                "server_addr" => {
                    addr = Some(
                        string(path, "server_addr", item, "a string IP:PORT")?
                            .parse::<SocketAddr>()
                            .map_err(|_| DaemonConfigError::Type {
                                path: path.to_path_buf(),
                                key: "server_addr",
                                expected: "a string IP:PORT",
                            })?,
                    );
                }
                "token_file" => {
                    token_file = Some(PathBuf::from(string(
                        path,
                        "token_file",
                        item,
                        "a string path",
                    )?));
                }
                other => {
                    return Err(DaemonConfigError::UnknownKey {
                        path: path.to_path_buf(),
                        key: other.to_owned(),
                    });
                }
            }
        }
        let server = match (socket, addr, token_file) {
            (Some(_), Some(_), _) => {
                return Err(DaemonConfigError::Conflict {
                    path: path.to_path_buf(),
                });
            }
            (None, None, Some(_)) | (Some(_), None, Some(_)) => {
                return Err(DaemonConfigError::TokenWithoutAddr {
                    path: path.to_path_buf(),
                });
            }
            (Some(socket), None, None) => Some(ServerTarget::Socket(socket)),
            (None, Some(addr), token_file) => Some(ServerTarget::Tcp { addr, token_file }),
            (None, None, None) => None,
        };
        Ok(Self { server })
    }

    /// The server the config names, or the unix socket at `default`.
    pub fn server_or(self, default: PathBuf) -> ServerTarget {
        self.server.unwrap_or(ServerTarget::Socket(default))
    }
}

/// The command a client starts the daemon with.
///
/// The token a TCP server needs never enters the command: only the file
/// holding it does, so `daemon.cmd` and `ps` never show it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonCommand {
    /// The executable.
    pub program: PathBuf,
    /// Its arguments.
    pub args: Vec<OsString>,
    /// Environment variables added to the client's own; not part of the
    /// rendered command. A host that wraps the daemon — the tests
    /// re-execute their own binary — configures it here.
    pub env: Vec<(OsString, OsString)>,
}

impl DaemonCommand {
    /// `program --daemon --listen <listen> --ready-fd 3` followed by the
    /// server flags `server` renders to.
    pub fn new(program: PathBuf, listen: &Path, server: &ServerTarget) -> Self {
        let mut args: Vec<OsString> = vec![
            "--daemon".into(),
            "--listen".into(),
            listen.into(),
            "--ready-fd".into(),
            READY_FD.to_string().into(),
        ];
        match server {
            ServerTarget::Socket(path) => {
                args.extend(["--server-socket".into(), path.into()]);
            }
            ServerTarget::Tcp { addr, token_file } => {
                args.extend(["--server-addr".into(), addr.to_string().into()]);
                if let Some(file) = token_file {
                    args.extend(["--server-token-file".into(), file.into()]);
                }
            }
        }
        Self {
            program,
            args,
            env: Vec::new(),
        }
    }

    /// The text written to `daemon.cmd`: the program, then each argument,
    /// one per line.
    fn render(&self) -> String {
        let mut text = self.program.to_string_lossy().into_owned();
        text.push('\n');
        for arg in &self.args {
            text.push_str(&arg.to_string_lossy());
            text.push('\n');
        }
        text
    }
}

/// A failed spawn step, told to the user.
#[derive(Debug, thiserror::Error)]
enum SpawnError {
    #[error("cannot {what}: {source}")]
    Io {
        what: &'static str,
        source: io::Error,
    },
    #[error("the daemon.lock names no pid to replace: {0:?}")]
    Pid(String),
    #[error("the daemon failed to start: {message}; see {log}")]
    Failed { message: String, log: String },
    #[error("the daemon exited before it was ready; see {log}")]
    Exited { log: String },
    #[error("the daemon was not ready within {after:?}; see {log}")]
    Timeout { after: Duration, log: String },
    #[error("the daemon said {line:?} instead of ready; see {log}")]
    Garbled { line: String, log: String },
}

impl SpawnError {
    fn io(what: &'static str) -> impl FnOnce(io::Error) -> Self {
        move |source| Self::Io { what, source }
    }
}

/// Keeps a daemon reachable by spawning it when a connect fails.
pub struct OnDemand {
    command: DaemonCommand,
    socket: PathBuf,
    log: PathBuf,
}

impl OnDemand {
    /// A launcher that starts `command` — which must bind `socket` — and
    /// whose stderr goes to `log`. The lock files and `daemon.cmd` sit in
    /// `socket`'s directory.
    pub fn new(command: DaemonCommand, socket: PathBuf, log: PathBuf) -> Self {
        Self {
            command,
            socket,
            log,
        }
    }

    /// The invoking user's launcher for the `search` at `program`: the
    /// socket and locks in the runtime dir, the log in the logs dir, the
    /// server from `daemon.toml` in the config dir.
    ///
    /// # Errors
    ///
    /// When the user's paths do not resolve or `daemon.toml` is invalid.
    pub fn user(program: PathBuf) -> io::Result<Self> {
        let paths = UserPaths::user()?;
        let config =
            DaemonConfig::load(&paths.config.join("daemon.toml")).map_err(io::Error::other)?;
        let socket = paths.runtime.join("daemon.sock");
        let server = config.server_or(paths.runtime.join("server.sock"));
        Ok(Self::new(
            DaemonCommand::new(program, &socket, &server),
            socket,
            paths.logs.join("daemon.log"),
        ))
    }

    /// A file next to the socket.
    fn sibling(&self, name: &str) -> PathBuf {
        self.socket.with_file_name(name)
    }

    /// The daemon's lifetime lock, next to its socket as
    /// [`Bound`](crate::daemon::listen::Bound) names it.
    fn daemon_lock(&self) -> PathBuf {
        self.socket.with_extension("lock")
    }

    /// Whether a daemon holds `daemon.lock`: a shared try-lock fails
    /// only against the daemon's exclusive one, so concurrent probes
    /// never mistake each other for a daemon.
    async fn alive(&self) -> Result<bool, SpawnError> {
        let path = self.daemon_lock();
        tokio::task::spawn_blocking(move || {
            let lock = match File::open(&path) {
                Ok(lock) => lock,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error),
            };
            match lock.try_lock_shared() {
                Ok(()) => Ok(false),
                Err(TryLockError::WouldBlock) => Ok(true),
                Err(TryLockError::Error(error)) => Err(error),
            }
        })
        .await
        .expect("the probe does not panic")
        .map_err(SpawnError::io("probe daemon.lock"))
    }

    /// Whether `daemon.cmd` records a command other than this launcher's.
    /// A missing file records none.
    async fn recorded_differs(&self) -> Result<bool, SpawnError> {
        match tokio::fs::read_to_string(self.sibling("daemon.cmd")).await {
            Ok(recorded) => Ok(recorded != self.command.render()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(SpawnError::io("read daemon.cmd")(error)),
        }
    }

    /// The pid `daemon.lock` names, sent SIGTERM.
    async fn terminate(&self) -> Result<(), SpawnError> {
        let text = tokio::fs::read_to_string(self.daemon_lock())
            .await
            .map_err(SpawnError::io("read daemon.lock"))?;
        let pid: libc::pid_t = text
            .trim()
            .parse()
            .map_err(|_| SpawnError::Pid(text.clone()))?;
        // SAFETY: `kill` takes plain integers.
        let rc = unsafe { libc::kill(pid, libc::SIGTERM) };
        if rc == -1 {
            let error = io::Error::last_os_error();
            // The daemon exited between the probe and now.
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(SpawnError::io("signal the daemon")(error));
            }
        }
        Ok(())
    }

    /// Takes `spawn.lock`, blocking off the executor until the holder
    /// lets go.
    async fn spawn_lock(&self) -> Result<File, SpawnError> {
        let path = self.sibling("spawn.lock");
        tokio::task::spawn_blocking(move || {
            let lock = OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(path)?;
            lock.lock()?;
            Ok(lock)
        })
        .await
        .expect("the lock wait does not panic")
        .map_err(SpawnError::io("take spawn.lock"))
    }

    /// Starts the daemon detached, records its command and waits for its
    /// readiness line.
    async fn start(&self) -> Result<(), SpawnError> {
        let log_text = self.log.display().to_string();
        let (reader, writer) = std::io::pipe().map_err(SpawnError::io("create the ready pipe"))?;
        let log = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)
            .await
            .map_err(SpawnError::io("open the daemon log"))?
            .into_std()
            .await;
        let mut command = tokio::process::Command::new(&self.command.program);
        command
            .args(&self.command.args)
            .envs(self.command.env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log);
        let write_fd = writer.as_raw_fd();
        // SAFETY: the closure calls only `setsid`, `dup2` and `fcntl`,
        // which are async-signal-safe, and touches no memory of the
        // parent's.
        unsafe {
            command.pre_exec(move || {
                // Leave the client's session and controlling terminal.
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                // The pipe's write end is the daemon's only inherited fd.
                // `dup2` clears `CLOEXEC` on the copy; an end that already
                // sits at the target keeps it, so clear it there.
                if write_fd == READY_FD {
                    let flags = libc::fcntl(READY_FD, libc::F_GETFD);
                    if flags == -1
                        || libc::fcntl(READY_FD, libc::F_SETFD, flags & !libc::FD_CLOEXEC) == -1
                    {
                        return Err(io::Error::last_os_error());
                    }
                } else if libc::dup2(write_fd, READY_FD) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command
            .spawn()
            .map_err(SpawnError::io("start the daemon"))?;
        // Only the daemon holds the write end now, so its exit is EOF.
        drop(command);
        drop(writer);

        // `spawn.lock` is held by the caller, so no other writer touches
        // the temp file; the rename keeps a reader from seeing half a
        // command.
        let record = self.sibling("daemon.cmd");
        let staging = self.sibling("daemon.cmd.tmp");
        tokio::fs::write(&staging, self.command.render())
            .await
            .map_err(SpawnError::io("write daemon.cmd"))?;
        tokio::fs::rename(&staging, &record)
            .await
            .map_err(SpawnError::io("write daemon.cmd"))?;

        let receiver = pipe::Receiver::from_owned_fd(OwnedFd::from(reader))
            .map_err(SpawnError::io("read the ready pipe"))?;
        let mut line = String::new();
        let read = tokio::time::timeout(
            READY_DEADLINE,
            BufReader::new(receiver).read_line(&mut line),
        )
        .await;
        // The child is detached: dropping the handle leaves it running,
        // and tokio reaps it if it exits first.
        drop(child);
        match read {
            Err(_) => Err(SpawnError::Timeout {
                after: READY_DEADLINE,
                log: log_text,
            }),
            Ok(Err(error)) => Err(SpawnError::io("read the ready pipe")(error)),
            Ok(Ok(0)) => Err(SpawnError::Exited { log: log_text }),
            Ok(Ok(_)) => match line.trim_end() {
                "ready" => Ok(()),
                other => match other.strip_prefix("failed: ") {
                    Some(message) => Err(SpawnError::Failed {
                        message: message.to_owned(),
                        log: log_text,
                    }),
                    None => Err(SpawnError::Garbled {
                        line: other.to_owned(),
                        log: log_text,
                    }),
                },
            },
        }
    }

    /// Makes a daemon reachable at the socket: under `spawn.lock`, a live
    /// daemon with a socket needs nothing, otherwise one is started —
    /// after signalling the live one when `replace` and the recorded
    /// command still differs. `spawn.lock` is released only once the new
    /// daemon is ready.
    async fn bring_up(&self, replace: bool) -> Result<(), SpawnError> {
        for dir in [self.socket.parent(), self.log.parent()]
            .into_iter()
            .flatten()
        {
            tokio::fs::create_dir_all(dir)
                .await
                .map_err(SpawnError::io("create the daemon's dirs"))?;
        }
        let _spawn_lock = self.spawn_lock().await?;
        let alive = self.alive().await?;
        // Another client may have replaced the daemon while this one
        // waited for the lock.
        let replace = replace && alive && self.recorded_differs().await?;
        if alive && !replace && self.socket.exists() {
            return Ok(());
        }
        if replace {
            self.terminate().await?;
        }
        self.start().await
    }

    /// `bring_up` as a launcher step's report.
    async fn report(&self, replace: bool) -> Report {
        match self.bring_up(replace).await {
            Ok(()) => Report::default(),
            Err(error) => Report {
                notes: Vec::new(),
                error: Some(error.to_string()),
            },
        }
    }
}

impl Launcher for OnDemand {
    fn socket(&self) -> &Path {
        &self.socket
    }

    fn log(&self) -> &Path {
        &self.log
    }

    /// Reads `daemon.cmd`; only a daemon that is alive and was started
    /// with another command is replaced. A warm search spawns nothing and
    /// takes no lock.
    async fn prepare(&mut self) -> Report {
        let replace = match self.recorded_differs().await {
            Ok(true) => self.alive().await,
            Ok(false) => Ok(false),
            Err(error) => Err(error),
        };
        match replace {
            Ok(true) => self.report(true).await,
            Ok(false) => Report::default(),
            Err(error) => Report {
                notes: Vec::new(),
                error: Some(error.to_string()),
            },
        }
    }

    /// Every connect failure a fresh daemon can cure: a missing or stale
    /// socket, a connection reset before the handshake, a daemon shutting
    /// down and a build mismatch. Only an oversized frame and a refusal of
    /// any other kind are final.
    fn recoverable(&self, error: &ClientError) -> bool {
        match error {
            ClientError::TooLarge => false,
            ClientError::Refused(report) => matches!(
                report.kind,
                ErrorKind::ShuttingDown | ErrorKind::ProtocolMismatch
            ),
            ClientError::Connect(_)
            | ClientError::Io(_)
            | ClientError::Closed
            | ClientError::Protocol(_)
            | ClientError::Frame(_) => true,
        }
    }

    async fn recover(&mut self) -> Report {
        self.report(false).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<DaemonConfig, DaemonConfigError> {
        DaemonConfig::parse(Path::new("daemon.toml"), text)
    }

    #[test]
    fn an_empty_file_means_the_default_socket() {
        let config = parse("").expect("an empty file parses");
        assert_eq!(config.server, None);
        assert_eq!(
            config.server_or(PathBuf::from("/run/server.sock")),
            ServerTarget::Socket(PathBuf::from("/run/server.sock"))
        );
    }

    #[test]
    fn a_missing_file_is_the_default() {
        let config = DaemonConfig::load(Path::new("/nonexistent/daemon.toml")).expect("defaults");
        assert_eq!(config, DaemonConfig::default());
    }

    #[test]
    fn the_server_keys_parse() {
        assert_eq!(
            parse("server_socket = \"/run/s.sock\"\n")
                .expect("socket")
                .server,
            Some(ServerTarget::Socket(PathBuf::from("/run/s.sock")))
        );
        assert_eq!(
            parse("server_addr = \"127.0.0.1:41827\"\n")
                .expect("addr")
                .server,
            Some(ServerTarget::Tcp {
                addr: "127.0.0.1:41827".parse().expect("an address"),
                token_file: None,
            })
        );
        assert_eq!(
            parse("server_addr = \"[::1]:9\"\ntoken_file = \"/t\"\n")
                .expect("addr and token")
                .server,
            Some(ServerTarget::Tcp {
                addr: "[::1]:9".parse().expect("an address"),
                token_file: Some(PathBuf::from("/t")),
            })
        );
    }

    #[test]
    fn strict_errors() {
        assert!(matches!(
            parse("bind = \"x\"\n"),
            Err(DaemonConfigError::UnknownKey { key, .. }) if key == "bind"
        ));
        assert!(matches!(
            parse("server_socket = 3\n"),
            Err(DaemonConfigError::Type {
                key: "server_socket",
                ..
            })
        ));
        assert!(matches!(
            parse("server_addr = \"localhost:9\"\n"),
            Err(DaemonConfigError::Type {
                key: "server_addr",
                ..
            })
        ));
        assert!(matches!(
            parse("token_file = 1\n"),
            Err(DaemonConfigError::Type {
                key: "token_file",
                ..
            })
        ));
        assert!(matches!(
            parse("server_socket = \"/s\"\nserver_addr = \"127.0.0.1:9\"\n"),
            Err(DaemonConfigError::Conflict { .. })
        ));
        assert!(matches!(
            parse("token_file = \"/t\"\n"),
            Err(DaemonConfigError::TokenWithoutAddr { .. })
        ));
        assert!(matches!(
            parse("server_socket = \"/s\"\ntoken_file = \"/t\"\n"),
            Err(DaemonConfigError::TokenWithoutAddr { .. })
        ));
        assert!(matches!(
            parse("server_socket = \n"),
            Err(DaemonConfigError::Parse { .. })
        ));
    }

    #[test]
    fn the_command_renders_the_daemon_flags_and_never_a_token() {
        let socket = Path::new("/run/daemon.sock");
        let unix = DaemonCommand::new(
            PathBuf::from("/bin/search"),
            socket,
            &ServerTarget::Socket(PathBuf::from("/run/server.sock")),
        );
        assert_eq!(
            unix.render(),
            "/bin/search\n--daemon\n--listen\n/run/daemon.sock\n--ready-fd\n3\n--server-socket\n/run/server.sock\n"
        );
        let tcp = DaemonCommand::new(
            PathBuf::from("/bin/search"),
            socket,
            &ServerTarget::Tcp {
                addr: "127.0.0.1:41827".parse().expect("an address"),
                token_file: Some(PathBuf::from("/etc/token")),
            },
        );
        assert_eq!(
            tcp.render(),
            "/bin/search\n--daemon\n--listen\n/run/daemon.sock\n--ready-fd\n3\n--server-addr\n127.0.0.1:41827\n--server-token-file\n/etc/token\n"
        );
        assert_ne!(unix.render(), tcp.render());
    }
}
