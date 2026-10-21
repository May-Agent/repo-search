//! launchd socket activation and the LaunchAgent document.
//!
//! `search --daemon` runs under launchd: the agent's plist declares a
//! `Listener` socket, and [`activate_socket`] trades its name for the
//! listening fds launchd already bound. [`LaunchAgent`] renders the plist
//! the client installs under `~/Library/LaunchAgents`.

use std::ffi::{CString, NulError, c_char, c_int};
use std::io;
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::PathBuf;
use std::time::Duration;

unsafe extern "C" {
    fn launch_activate_socket(name: *const c_char, fds: *mut *mut c_int, cnt: *mut usize) -> c_int;
}

/// The launchd label of the `search` daemon's agent.
pub const LABEL: &str = "com.github.may-agent.repo-search";

/// The socket the LaunchAgent binds: launchd holds it while no daemon runs
/// and launches the daemon when a connection arrives. `Application Support`
/// rather than `Caches`, which the system may purge while the agent is
/// loaded.
pub fn socket_path() -> io::Result<PathBuf> {
    Ok(home()?.join("Library/Application Support/repo-search/daemon.sock"))
}

/// The file the agent's `StandardErrorPath` writes the daemon's tracing
/// output to.
pub fn log_path() -> io::Result<PathBuf> {
    Ok(home()?.join("Library/Logs/repo-search/daemon.log"))
}

/// The plist the agent is installed as:
/// `~/Library/LaunchAgents/<label>.plist`.
pub fn plist_path(label: &str) -> io::Result<PathBuf> {
    Ok(home()?.join(format!("Library/LaunchAgents/{label}.plist")))
}

/// The file whose `flock(2)` serializes installs. It sits next to the
/// socket, in a directory this program owns, rather than in
/// `~/Library/LaunchAgents`, which belongs to launchd.
pub fn lock_path() -> io::Result<PathBuf> {
    Ok(home()?.join("Library/Application Support/repo-search/install.lock"))
}

/// The launchd domain the agent lives in: `user/<uid>`, so `search` works
/// over SSH with no GUI login.
pub fn domain() -> String {
    // SAFETY: getuid cannot fail.
    format!("user/{}", unsafe { libc::getuid() })
}

fn home() -> io::Result<PathBuf> {
    std::env::home_dir().ok_or_else(|| io::Error::other("the home directory is unknown"))
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
    }

    #[test]
    fn the_plist_renders_a_model_cache() {
        let mut agent = agent();
        agent.model_cache = Some(PathBuf::from("/tmp/models"));
        let plist = askama::Template::render(&agent).expect("the template renders");
        assert!(plist.contains("<string>--model-cache</string>\n\t\t<string>/tmp/models</string>"));
    }
}
