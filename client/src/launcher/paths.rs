//! The per-user paths the client and the daemon look up: state, runtime,
//! cache, logs and config — `~/Library` on macOS, the XDG base
//! directories on Linux.
//!
//! [`UserPaths::user`] is the one lookup every per-user path comes from;
//! the functions below derive the files the daemon and the client use.
//! On macOS `config` coincides with `state`; on Linux `runtime` is
//! `$XDG_RUNTIME_DIR`, whose machine-local tmpfs can never hand two
//! machines sharing a home one lock or socket.
//!
//! A unix socket's path must fit `sockaddr_un.sun_path` (104 bytes on
//! macOS, 108 on Linux, NUL included), so the sockets never live under
//! `$HOME`, whose length nothing bounds: on macOS the runtime dir is a
//! per-home directory under the short per-user `DARWIN_USER_DIR`, and
//! every socket path is a [`SocketPath`], which cannot hold a path the
//! platform cannot bind.

use std::io;
use std::path::{Path, PathBuf};

/// The size of `sockaddr_un.sun_path` on this platform. A socket path
/// must be shorter, leaving room for its NUL.
// SAFETY: an all-zero `sockaddr_un` is a valid value of the plain C struct.
pub const SUN_PATH_LEN: usize = unsafe { std::mem::zeroed::<libc::sockaddr_un>() }
    .sun_path
    .len();

/// A unix socket path that fits this platform's `sun_path`: one a
/// socket can be bound or connected at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketPath(PathBuf);

/// A socket path [`SocketPath::new`] refused.
#[derive(Debug, thiserror::Error)]
#[error(
    "the unix socket path {} is {len} bytes; this platform's limit is {} bytes",
    .path.display(),
    SUN_PATH_LEN - 1
)]
pub struct SocketPathTooLong {
    /// The refused path.
    pub path: PathBuf,
    /// Its length in bytes.
    pub len: usize,
}

impl From<SocketPathTooLong> for io::Error {
    fn from(error: SocketPathTooLong) -> Self {
        Self::new(io::ErrorKind::InvalidInput, error)
    }
}

impl SocketPath {
    /// `path`, when it is shorter than [`SUN_PATH_LEN`].
    ///
    /// # Errors
    ///
    /// [`SocketPathTooLong`], naming the path and the limit, when it is
    /// not.
    pub fn new(path: PathBuf) -> Result<Self, SocketPathTooLong> {
        let len = path.as_os_str().len();
        if len < SUN_PATH_LEN {
            Ok(Self(path))
        } else {
            Err(SocketPathTooLong { path, len })
        }
    }

    /// The path.
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    /// The path, for display.
    pub fn display(&self) -> std::path::Display<'_> {
        self.0.display()
    }

    /// The path, owned.
    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }
}

impl AsRef<Path> for SocketPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

/// The per-user roots [`UserPaths::user`] resolves.
#[derive(Debug)]
pub struct UserPaths {
    /// The state root: the per-worktree indexes, the per-repository
    /// vector caches and — on Linux, where XDG puts logs in
    /// `STATE_HOME` — the daemon's log.
    pub state: PathBuf,
    /// The machine-local runtime dir: `daemon.sock`, `daemon.lock`,
    /// `spawn.lock`, `daemon.cmd` and `server.sock`. On macOS it is
    /// `<DARWIN_USER_DIR>/repo-search/<home key>`, short whatever the
    /// home, and one per home, so a relocated `$HOME` never shares the
    /// login home's sockets.
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

/// `~`, when it can be known. Follows `$HOME`.
#[cfg(target_os = "macos")]
pub(crate) fn home() -> io::Result<PathBuf> {
    std::env::home_dir().ok_or_else(|| io::Error::other("the home directory is unknown"))
}

/// The login directory from the passwd database, independent of `$HOME`.
#[cfg(target_os = "macos")]
pub(crate) fn login_home() -> io::Result<PathBuf> {
    // SAFETY: getpwuid returns a pointer into libc storage; `pw_dir` is
    // copied before any later libc call.
    let pw = unsafe { libc::getpwuid(libc::getuid()) };
    if pw.is_null() {
        return Err(io::Error::other("the login directory is unknown"));
    }
    let dir = unsafe { std::ffi::CStr::from_ptr((*pw).pw_dir) };
    let dir = dir
        .to_str()
        .map_err(|_| io::Error::other("the login directory is not UTF-8"))?;
    Ok(PathBuf::from(dir))
}

/// A stable 64-bit FNV-1a hex of `home`'s bytes: the key that gives each
/// home its own launchd labels and runtime dir.
#[cfg(target_os = "macos")]
pub(crate) fn home_key(home: &Path) -> String {
    use std::os::unix::ffi::OsStrExt as _;
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for byte in home.as_os_str().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

/// The invoking user's `DARWIN_USER_DIR` (`/var/folders/…/0/`): per
/// user, writable by that user alone, machine-local and of a fixed short
/// length whatever `$HOME` is.
#[cfg(target_os = "macos")]
fn darwin_user_dir() -> io::Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt as _;
    let mut buf = vec![0u8; libc::PATH_MAX as usize];
    // SAFETY: `buf` is writable for its whole length, which is passed.
    let len = unsafe {
        libc::confstr(
            libc::_CS_DARWIN_USER_DIR,
            buf.as_mut_ptr().cast(),
            buf.len(),
        )
    };
    if len == 0 {
        return Err(io::Error::other(format!(
            "confstr(_CS_DARWIN_USER_DIR) failed: {}",
            io::Error::last_os_error()
        )));
    }
    assert!(len <= buf.len(), "DARWIN_USER_DIR is longer than PATH_MAX");
    // `len` counts the terminating NUL.
    buf.truncate(len - 1);
    Ok(PathBuf::from(std::ffi::OsString::from_vec(buf)))
}

impl UserPaths {
    /// The invoking user's paths, under `~/Library`, with the runtime dir
    /// under `DARWIN_USER_DIR`.
    #[cfg(target_os = "macos")]
    pub fn user() -> io::Result<Self> {
        Self::for_home(&home()?)
    }

    /// The paths of the invoking user when `$HOME` is `home`.
    #[cfg(target_os = "macos")]
    pub fn for_home(home: &Path) -> io::Result<Self> {
        let state = home.join("Library/Application Support/repo-search");
        Ok(Self {
            runtime: darwin_user_dir()?.join("repo-search").join(home_key(home)),
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
///
/// # Errors
///
/// When the user's paths do not resolve, or the path does not fit the
/// platform's socket path limit.
pub fn socket_path() -> io::Result<SocketPath> {
    Ok(SocketPath::new(
        UserPaths::user()?.runtime.join("daemon.sock"),
    )?)
}

/// The file the daemon's stderr — its tracing output — goes to.
pub fn log_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.logs.join("daemon.log"))
}

/// The embedding server's unix socket, next to the daemon's.
///
/// # Errors
///
/// As [`socket_path`].
pub fn server_socket_path() -> io::Result<SocketPath> {
    Ok(SocketPath::new(
        UserPaths::user()?.runtime.join("server.sock"),
    )?)
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

/// The file whose `flock(2)` serializes installs. It sits in the state
/// dir, which this program owns, rather than in
/// `~/Library/LaunchAgents`, which belongs to launchd.
#[cfg(target_os = "macos")]
pub fn lock_path() -> io::Result<PathBuf> {
    Ok(UserPaths::user()?.state.join("install.lock"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_socket_path_must_leave_room_for_its_nul() {
        let fits = PathBuf::from(format!("/{}", "a".repeat(SUN_PATH_LEN - 2)));
        assert_eq!(fits.as_os_str().len(), SUN_PATH_LEN - 1);
        SocketPath::new(fits).expect("one byte short of sun_path fits");

        let long = PathBuf::from(format!("/{}", "a".repeat(SUN_PATH_LEN - 1)));
        let error = SocketPath::new(long.clone()).expect_err("a path filling sun_path does not");
        assert_eq!(error.len, SUN_PATH_LEN);
        let message = error.to_string();
        assert!(message.contains(&long.display().to_string()), "{message}");
        assert!(
            message.contains(&format!("limit is {} bytes", SUN_PATH_LEN - 1)),
            "{message}"
        );
    }

    #[test]
    fn a_too_long_socket_path_cannot_be_bound() {
        // The limit is the platform's own: the longest path `SocketPath`
        // admits binds, one byte more does not.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("s");
        let pad = SUN_PATH_LEN - 1 - base.as_os_str().len();
        let fits = PathBuf::from(format!("{}{}", base.display(), "a".repeat(pad)));
        let socket = SocketPath::new(fits).expect("fits");
        std::os::unix::net::UnixListener::bind(socket.as_path()).expect("the kernel binds it");
        let over = PathBuf::from(format!("{}{}", base.display(), "a".repeat(pad + 1)));
        assert!(SocketPath::new(over.clone()).is_err());
        std::os::unix::net::UnixListener::bind(&over).expect_err("nor can it be bound");
    }

    /// A home of any length — a temp home under `$TMPDIR`, or one far
    /// longer — has sockets that fit, its own and stable.
    #[cfg(target_os = "macos")]
    #[test]
    fn every_home_gets_sockets_that_fit() {
        let long = PathBuf::from(format!(
            "/private/var/folders/xx/T/.tmpUf07X1/{}",
            "h".repeat(300)
        ));
        let paths = UserPaths::for_home(&long).expect("the paths resolve");
        for name in ["daemon.sock", "server.sock"] {
            SocketPath::new(paths.runtime.join(name)).expect("the socket fits");
        }
        assert!(
            paths.state.starts_with(&long),
            "the state stays in the home"
        );
        assert_eq!(
            paths.runtime,
            UserPaths::for_home(&long).unwrap().runtime,
            "stable"
        );
        assert_ne!(
            paths.runtime,
            UserPaths::for_home(Path::new("/Users/example"))
                .unwrap()
                .runtime,
            "per home"
        );
    }
}
