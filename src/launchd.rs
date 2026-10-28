//! launchd socket activation and the LaunchAgent document.
//!
//! `search --daemon` runs under launchd: the agent's plist declares a
//! `Listener` socket, and [`activate_socket`] trades its name for the
//! listening fds launchd already bound. [`LaunchAgent`] renders the plist
//! the client installs under `~/Library/LaunchAgents`.

use std::ffi::{CString, NulError, c_char, c_int};
use std::io;
use std::net::IpAddr;
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::time::Duration;

unsafe extern "C" {
    fn launch_activate_socket(name: *const c_char, fds: *mut *mut c_int, cnt: *mut usize) -> c_int;
}

/// The launchd label of the `search` daemon's agent.
pub const LABEL: &str = "com.github.may-agent.repo-search";

/// The launchd label of the embedding server's agent.
pub const SERVER_LABEL: &str = "com.github.may-agent.repo-search.server";

/// The launchd domain the agent lives in: `user/<uid>`, so `search` works
/// over SSH with no GUI login.
pub fn domain() -> String {
    // SAFETY: getuid cannot fail.
    format!("user/{}", unsafe { libc::getuid() })
}

/// The failure of [`activate_socket`].
#[derive(Debug, thiserror::Error)]
pub enum ActivateError {
    /// The socket name cannot be a C string.
    #[error("the socket name contains a NUL byte: {0}")]
    Name(#[from] NulError),
    /// `launch_activate_socket` returned a nonzero errno.
    #[error("launch_activate_socket failed: {source}; {hint}")]
    Launchd {
        /// The errno `launch_activate_socket` returned.
        source: io::Error,
        /// The named meaning of a common errno.
        hint: &'static str,
    },
}

/// The named meaning of a common `launch_activate_socket` errno.
fn hint(error: &io::Error) -> &'static str {
    match error.raw_os_error() {
        Some(libc::ENOENT) => "no socket of that name in the plist",
        Some(libc::ESRCH) => "the process was not started by launchd",
        Some(libc::EALREADY) => "the socket was already activated",
        _ => "an unusual errno",
    }
}

/// Trade the plist's socket `name` for the fds launchd bound for it.
///
/// # Errors
///
/// [`ActivateError::Name`] when `name` is not a C string, and
/// [`ActivateError::Launchd`] when launchd refuses the activation.
pub fn activate_socket(name: &str) -> Result<Vec<OwnedFd>, ActivateError> {
    let name = CString::new(name)?;
    let mut fds: *mut c_int = std::ptr::null_mut();
    let mut cnt = 0usize;
    // SAFETY: `name` is a live C string; `fds` and `cnt` are writable
    // out-parameters the call fills on success.
    let rc = unsafe { launch_activate_socket(name.as_ptr(), &mut fds, &mut cnt) };
    if rc != 0 {
        let source = io::Error::from_raw_os_error(rc);
        return Err(ActivateError::Launchd {
            hint: hint(&source),
            source,
        });
    }
    // The returned array is malloc'd by libSystem: take each fd into an
    // `OwnedFd`, then free the array itself.
    let owned = (0..cnt)
        // SAFETY: fds[i] is a live descriptor the call transferred to us.
        .map(|i| unsafe { OwnedFd::from_raw_fd(*fds.add(i)) })
        .collect();
    // SAFETY: `fds` is the malloc'd array the call returned, or null, which
    // `free` accepts.
    unsafe { libc::free(fds.cast()) };
    Ok(owned)
}

/// The LaunchAgent document, rendered through `templates/launch-agent.plist`.
///
/// The plist sets no `KeepAlive`, `RunAtLoad` or `ProcessType`. With no
/// `KeepAlive`, launchd holds the listening socket while no daemon runs and
/// starts the daemon when a connection arrives; a connection made after the
/// daemon stopped accepting stays in the socket backlog and launchd
/// relaunches the daemon for it. `ProcessType` stays the default `Standard`:
/// `Background` or `Adaptive` would throttle the whole process — unix-socket
/// traffic is not XPC — and defeat the user-interactive query QoS.
///
/// Note that launchd's default `ThrottleInterval` (10 s) can delay a
/// relaunch that follows an exit less than 10 s after launch; only the
/// protocol-mismatch exit hits this.
///
/// A daemon that fails after socket activation answers every queued
/// connection with `Startup` before exiting, so no connection is left to
/// drive a relaunch loop.
#[derive(askama::Template)]
#[template(path = "launch-agent.plist", escape = "html")]
pub struct LaunchAgent {
    /// The job's launchd label.
    pub label: String,
    /// The `search` binary to run.
    pub program: PathBuf,
    /// The `Listener` socket's path.
    pub socket: PathBuf,
    /// The `--idle-exit-secs` argument.
    pub idle_exit: Duration,
    /// The `--model-cache` argument, when a non-default cache is used.
    pub model_cache: Option<PathBuf>,
    /// The `--state-dir` argument, when a non-default state dir is used.
    pub state_dir: Option<PathBuf>,
    /// The `--server-socket` argument: where the embedding server listens.
    pub server_socket: PathBuf,
    /// The `StandardErrorPath` log file.
    pub stderr: PathBuf,
}

/// The TCP socket's default bind address and service name when
/// `server.toml` does not set them: loopback, and a port below macOS's
/// ephemeral range.
pub const DEFAULT_BIND: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
/// The default `SockServiceName`.
pub const DEFAULT_PORT: u16 = 41827;

/// The server agent's `server.toml`: the TCP listener's bind and port and
/// the bearer token's file. Every key is optional; unknown keys are an
/// error.
#[derive(Debug, PartialEq, Eq)]
pub struct ServerConfig {
    /// `SockNodeName`: the address the TCP socket binds.
    pub bind: IpAddr,
    /// `SockServiceName`: the port.
    pub port: u16,
    /// The bearer token's file, passed as `--token-file`; the token
    /// itself never enters the plist.
    pub token_file: Option<PathBuf>,
}

/// A `server.toml` failure.
#[derive(Debug, thiserror::Error)]
pub enum ServerConfigError {
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
    #[error("{path}: unknown key {key:?} (bind, port and token_file are supported)")]
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
    /// `bind` names a non-loopback address and no `token_file` is set.
    #[error("{path}: bind {bind} is not loopback and no token_file is set")]
    NonLoopback {
        /// The file parsed.
        path: PathBuf,
        /// The offending address.
        bind: IpAddr,
    },
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND,
            port: DEFAULT_PORT,
            token_file: None,
        }
    }
}

impl ServerConfig {
    /// The config at `path`, or the defaults when the file does not
    /// exist.
    pub fn load(path: &Path) -> Result<Self, ServerConfigError> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(source) => {
                return Err(ServerConfigError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        Self::parse(path, &text)
    }

    /// The config `text` parses to, with `path` naming errors.
    fn parse(path: &Path, text: &str) -> Result<Self, ServerConfigError> {
        let doc =
            text.parse::<toml_edit::DocumentMut>()
                .map_err(|source| ServerConfigError::Parse {
                    path: path.to_path_buf(),
                    source,
                })?;
        let mut config = Self::default();
        for (key, item) in doc.as_table().iter() {
            match key {
                "bind" => {
                    let value = item.as_str().ok_or_else(|| ServerConfigError::Type {
                        path: path.to_path_buf(),
                        key: "bind",
                        expected: "a string address",
                    })?;
                    config.bind = value.parse().map_err(|_| ServerConfigError::Type {
                        path: path.to_path_buf(),
                        key: "bind",
                        expected: "a string address",
                    })?;
                }
                "port" => {
                    let value = item.as_integer().ok_or_else(|| ServerConfigError::Type {
                        path: path.to_path_buf(),
                        key: "port",
                        expected: "a port number",
                    })?;
                    config.port = u16::try_from(value).map_err(|_| ServerConfigError::Type {
                        path: path.to_path_buf(),
                        key: "port",
                        expected: "a port number",
                    })?;
                }
                "token_file" => {
                    let value = item.as_str().ok_or_else(|| ServerConfigError::Type {
                        path: path.to_path_buf(),
                        key: "token_file",
                        expected: "a string path",
                    })?;
                    config.token_file = Some(PathBuf::from(value));
                }
                other => {
                    return Err(ServerConfigError::UnknownKey {
                        path: path.to_path_buf(),
                        key: other.to_owned(),
                    });
                }
            }
        }
        if !config.bind.is_loopback() && config.token_file.is_none() {
            return Err(ServerConfigError::NonLoopback {
                path: path.to_path_buf(),
                bind: config.bind,
            });
        }
        Ok(config)
    }
}

/// The server's LaunchAgent document, rendered through
/// `templates/server-agent.plist`.
///
/// Like [`LaunchAgent`] it sets no `KeepAlive`, `RunAtLoad` or
/// `ProcessType`: launchd holds the `Unix` and `Tcp` sockets while no
/// server runs and starts it when a connection arrives. `--build` is the
/// self-heal: a binary rebuilt at the same path changes the plist, so
/// `ensure_agent` replaces the agent, and the server itself refuses to
/// run when the argument does not match its own build.
#[derive(askama::Template)]
#[template(path = "server-agent.plist", escape = "html")]
pub struct ServerAgent {
    /// The launchd label.
    pub label: String,
    /// The `search` binary to run.
    pub program: PathBuf,
    /// The `Unix` socket's path.
    pub socket: PathBuf,
    /// The `--idle-exit-secs` argument.
    pub idle_exit: Duration,
    /// The `--build` argument: the binary's identity at install time.
    pub build: String,
    /// The `--model-cache` argument, when a non-default cache is used.
    pub model_cache: Option<PathBuf>,
    /// The `--token-file` argument, when `server.toml` names one.
    pub token_file: Option<PathBuf>,
    /// `SockNodeName`: the TCP socket's bind address.
    pub bind: IpAddr,
    /// `SockServiceName`: the TCP socket's port.
    pub port: u16,
    /// The `StandardErrorPath` log file.
    pub stderr: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> LaunchAgent {
        LaunchAgent {
            label: LABEL.to_owned(),
            program: PathBuf::from("/usr/local/bin/search"),
            socket: PathBuf::from("/tmp/daemon.sock"),
            idle_exit: Duration::from_secs(600),
            model_cache: None,
            state_dir: None,
            server_socket: PathBuf::from("/tmp/server.sock"),
            stderr: PathBuf::from("/tmp/daemon.log"),
        }
    }

    #[test]
    fn the_plist_renders_the_agent() {
        let plist = askama::Template::render(&agent()).expect("the template renders");
        for fragment in [
            "<key>Label</key>\n\t<string>com.github.may-agent.repo-search</string>",
            "<key>LimitLoadToSessionType</key>\n\t<string>Background</string>",
            "<string>/usr/local/bin/search</string>",
            "<string>--idle-exit-secs</string>\n\t\t<string>600</string>",
            "<string>--server-socket</string>\n\t\t<string>/tmp/server.sock</string>",
            "<key>SockPathName</key>\n\t\t\t<string>/tmp/daemon.sock</string>",
            "<key>SockPathMode</key>\n\t\t\t<integer>384</integer>",
            "<key>StandardErrorPath</key>\n\t<string>/tmp/daemon.log</string>",
        ] {
            assert!(
                plist.contains(fragment),
                "missing {fragment:?} in:\n{plist}"
            );
        }
        assert!(!plist.contains("--model-cache"));
        assert!(!plist.contains("--state-dir"));
    }

    #[test]
    fn the_plist_renders_a_model_cache() {
        let mut agent = agent();
        agent.model_cache = Some(PathBuf::from("/tmp/models"));
        let plist = askama::Template::render(&agent).expect("the template renders");
        assert!(plist.contains("<string>--model-cache</string>\n\t\t<string>/tmp/models</string>"));
    }

    #[test]
    fn the_plist_renders_a_state_dir() {
        let mut agent = agent();
        agent.state_dir = Some(PathBuf::from("/tmp/state"));
        let plist = askama::Template::render(&agent).expect("the template renders");
        assert!(plist.contains("<string>--state-dir</string>\n\t\t<string>/tmp/state</string>"));
    }

    fn server_agent() -> ServerAgent {
        ServerAgent {
            label: SERVER_LABEL.to_owned(),
            program: PathBuf::from("/usr/local/bin/search"),
            socket: PathBuf::from("/tmp/server.sock"),
            idle_exit: Duration::from_secs(600),
            build: "123@456".to_owned(),
            model_cache: None,
            token_file: None,
            bind: DEFAULT_BIND,
            port: DEFAULT_PORT,
            stderr: PathBuf::from("/tmp/server.log"),
        }
    }

    #[test]
    fn the_server_plist_renders_the_agent() {
        let plist = askama::Template::render(&server_agent()).expect("the template renders");
        for fragment in [
            "<key>Label</key>\n\t<string>com.github.may-agent.repo-search.server</string>",
            "<key>LimitLoadToSessionType</key>\n\t<string>Background</string>",
            "<string>/usr/local/bin/search</string>",
            "<string>--server</string>",
            "<string>--idle-exit-secs</string>\n\t\t<string>600</string>",
            "<string>--build</string>\n\t\t<string>123@456</string>",
            "<key>Unix</key>",
            "<key>SockPathName</key>\n\t\t\t<string>/tmp/server.sock</string>",
            "<key>SockPathMode</key>\n\t\t\t<integer>384</integer>",
            "<key>Tcp</key>",
            "<key>SockNodeName</key>\n\t\t\t<string>127.0.0.1</string>",
            "<key>SockServiceName</key>\n\t\t\t<string>41827</string>",
            "<key>SockFamily</key>\n\t\t\t<string>IPv4</string>",
            "<key>StandardErrorPath</key>\n\t<string>/tmp/server.log</string>",
        ] {
            assert!(
                plist.contains(fragment),
                "missing {fragment:?} in:\n{plist}"
            );
        }
        assert!(!plist.contains("--model-cache"));
        assert!(!plist.contains("--token-file"));
    }

    #[test]
    fn the_server_plist_renders_the_options() {
        let mut agent = server_agent();
        agent.model_cache = Some(PathBuf::from("/tmp/models"));
        agent.token_file = Some(PathBuf::from("/tmp/token"));
        agent.bind = "0.0.0.0".parse().unwrap();
        agent.port = 443;
        let plist = askama::Template::render(&agent).expect("the template renders");
        for fragment in [
            "<string>--model-cache</string>\n\t\t<string>/tmp/models</string>",
            "<string>--token-file</string>\n\t\t<string>/tmp/token</string>",
            "<key>SockNodeName</key>\n\t\t\t<string>0.0.0.0</string>",
            "<key>SockServiceName</key>\n\t\t\t<string>443</string>",
        ] {
            assert!(
                plist.contains(fragment),
                "missing {fragment:?} in:\n{plist}"
            );
        }
    }

    fn config_file(text: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.toml");
        std::fs::write(&path, text).unwrap();
        (dir, path)
    }

    #[test]
    fn a_missing_server_toml_is_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig::load(&dir.path().join("server.toml")).unwrap();
        assert_eq!(config, ServerConfig::default());
        assert_eq!(config.bind, IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        assert_eq!(config.port, DEFAULT_PORT);
    }

    #[test]
    fn a_server_toml_overrides_the_defaults() {
        let (_dir, path) =
            config_file("bind = \"10.0.0.5\"\nport = 8080\ntoken_file = \"/tmp/token\"\n");
        let config = ServerConfig::load(&path).unwrap();
        assert_eq!(config.bind, "10.0.0.5".parse::<IpAddr>().unwrap());
        assert_eq!(config.port, 8080);
        assert_eq!(config.token_file, Some(PathBuf::from("/tmp/token")));
    }

    #[test]
    fn an_unknown_key_is_an_error() {
        let (_dir, path) = config_file("bind = \"127.0.0.1\"\nidle = 3\n");
        let Err(ServerConfigError::UnknownKey { key, .. }) = ServerConfig::load(&path) else {
            panic!("an unknown key is rejected");
        };
        assert_eq!(key, "idle");
    }

    #[test]
    fn a_non_loopback_bind_needs_a_token() {
        let (_dir, path) = config_file("bind = \"0.0.0.0\"\n");
        let Err(ServerConfigError::NonLoopback { bind, .. }) = ServerConfig::load(&path) else {
            panic!("a non-loopback bind without a token is rejected");
        };
        assert_eq!(bind, "0.0.0.0".parse::<IpAddr>().unwrap());

        let (_dir, path) = config_file("bind = \"0.0.0.0\"\ntoken_file = \"/t\"\n");
        ServerConfig::load(&path).expect("a token file allows a non-loopback bind");
    }

    #[test]
    fn the_server_plist_takes_its_socket_family_from_the_bind() {
        let mut agent = server_agent();
        agent.bind = IpAddr::V6(std::net::Ipv6Addr::LOCALHOST);
        let plist = askama::Template::render(&agent).expect("the template renders");
        assert!(plist.contains("<key>SockNodeName</key>\n\t\t\t<string>::1</string>"));
        assert!(plist.contains("<key>SockFamily</key>\n\t\t\t<string>IPv6</string>"));
    }
}
