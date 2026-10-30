//! launchd socket activation.
//!
//! `search --daemon` runs under launchd: the agent's plist (installed by
//! `repo_search_client::launcher`) declares a `Listener` socket, and
//! [`activate_socket`] trades its name for the listening fds launchd
//! already bound.

use std::ffi::{CString, NulError, c_char, c_int};
use std::io;
use std::os::fd::{FromRawFd, OwnedFd};

unsafe extern "C" {
    fn launch_activate_socket(name: *const c_char, fds: *mut *mut c_int, cnt: *mut usize) -> c_int;
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
