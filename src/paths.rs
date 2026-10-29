//! The per-user paths the client and the daemon look up: state, runtime,
//! cache, logs and config — `~/Library` on macOS, the XDG base
//! directories on Linux.
//!
//! [`UserPaths::user`] is the one lookup every per-user path comes from;
//! the functions below derive the files the daemon and the client use.
//! On macOS `runtime` and `config` coincide with `state`, so every path
//! the launchd agents already use is unchanged; on Linux `runtime` is
//! `$XDG_RUNTIME_DIR`, whose machine-local tmpfs can never hand two
//! machines sharing a home one lock or socket.

use std::io;
use std::path::PathBuf;

/// The per-user roots [`UserPaths::user`] resolves.
#[derive(Debug)]
pub struct UserPaths {
    /// The state root: the per-worktree indexes, the per-repository
    /// vector caches and — on Linux, where XDG puts logs in
    /// `STATE_HOME` — the daemon's log.
    pub state: PathBuf,
    /// The machine-local runtime dir: `daemon.sock`, `daemon.lock`,
    /// `spawn.lock`, `daemon.cmd` and `server.sock`. On macOS these
    /// live in the state root, so this field equals `state` there.
    pub runtime: PathBuf,
    /// The cache root; the tokenizer and model archives live under its
    /// `models/`.
    pub cache: PathBuf,
    /// The daemon's and the server's log directory.
    pub logs: PathBuf,
    /// The configuration root: `daemon.toml` on Linux, `server.toml`
    /// where the launchd agents live.
    pub config: PathBuf,
}

/// `~`, when it can be known.
#[cfg(target_os = "macos")]
fn home() -> io::Result<PathBuf> {
    std::env::home_dir().ok_or_else(|| io::Error::other("the home directory is unknown"))
}

impl UserPaths {
    /// The invoking user's paths, under `~/Library`.
    #[cfg(target_os = "macos")]
    pub fn user() -> io::Result<Self> {
        let home = home()?;
        let state = home.join("Library/Application Support/repo-search");
        Ok(Self {
            runtime: state.clone(),
            config: state.clone(),
            state,
            cache: home.join("Library/Caches/repo-search"),
            logs: home.join("Library/Logs/repo-search"),
        })
    }

    /// The invoking user's paths, under the XDG base directories.
    ///
    /// A missing `XDG_RUNTIME_DIR` is an error: the sockets and locks
    /// have no machine-local home without it, and a fallback under the
    /// user's home could hand a shared or NFS home's lock and socket to
    /// two machines at once. The runtime dir is created `0700`.
    ///
    /// # Errors
    ///
    /// When the home directory or `XDG_RUNTIME_DIR` is unknown, or the
    /// runtime dir cannot be created.
    #[cfg(target_os = "linux")]
    pub fn user() -> io::Result<Self> {
        use etcetera::BaseStrategy as _;
        use std::os::unix::fs::DirBuilderExt;

        let xdg = etcetera::base_strategy::Xdg::new().map_err(io::Error::other)?;
        let runtime = xdg
            .runtime_dir()
            .ok_or_else(|| io::Error::other("XDG_RUNTIME_DIR is not set"))?
            .join("repo-search");
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&runtime)?;
        let state = xdg
            .state_dir()
            .ok_or_else(|| io::Error::other("the XDG state directory is unknown"))?
            .join("repo-search");
        Ok(Self {
            logs: state.clone(),
            state,
            runtime,
            cache: xdg.cache_dir().join("repo-search"),
            config: xdg.config_dir().join("repo-search"),
        })
    }
}

/// The daemon's socket: under the runtime dir, so on Linux it is
/// machine-local.
pub fn socket_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.runtime.join("daemon.sock"))
}

/// The file the daemon's stderr — its tracing output — goes to.
pub fn log_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.logs.join("daemon.log"))
}

/// The embedding server's unix socket, next to the daemon's.
pub fn server_socket_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.runtime.join("server.sock"))
}

/// The file the server agent's `StandardErrorPath` writes to.
#[cfg(target_os = "macos")]
pub fn server_log_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.logs.join("server.log"))
}

/// The server agent's configuration file.
#[cfg(target_os = "macos")]
pub fn server_config_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.config.join("server.toml"))
}

/// The plist the agent is installed as:
/// `~/Library/LaunchAgents/<label>.plist`.
#[cfg(target_os = "macos")]
pub fn plist_path(label: &str) -> io::Result<PathBuf> {
    Ok(home()?.join(format!("Library/LaunchAgents/{label}.plist")))
}

/// The file whose `flock(2)` serializes installs. It sits next to the
/// socket, in a directory this program owns, rather than in
/// `~/Library/LaunchAgents`, which belongs to launchd.
#[cfg(target_os = "macos")]
pub fn lock_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.state.join("install.lock"))
}
