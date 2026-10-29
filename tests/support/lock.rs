//! The daemon's lifetime lock as the tests see it: whether a daemon holds
//! `daemon.lock`, the pid it names, and a guard that kills only that
//! daemon when a test ends.

use std::fs::{File, TryLockError};
use std::path::{Path, PathBuf};

/// The pid `daemon.lock` in `run_dir` names, when a daemon holds the
/// lock. A shared try-lock fails only against the daemon's exclusive
/// one, so a stale pid in a free lock file is never returned.
pub fn alive(run_dir: &Path) -> Option<u32> {
    let path = run_dir.join("daemon.lock");
    let lock = File::open(&path).ok()?;
    match lock.try_lock_shared() {
        Err(TryLockError::WouldBlock) => std::fs::read_to_string(&path).ok()?.trim().parse().ok(),
        _ => None,
    }
}

/// Sends `signal` to the daemon `pid` names; callers pass a pid read from
/// a held `daemon.lock`, a daemon their test spawned.
pub fn signal(pid: u32, signal: i32) {
    // SAFETY: `kill` takes plain integers.
    let rc = unsafe { libc::kill(pid as i32, signal) };
    assert_eq!(rc, 0, "signal {signal} to {pid}");
}

/// Kills the daemon holding `run_dir`'s lock when dropped, so a failing
/// test leaves nothing running.
pub struct Reaper(pub PathBuf);

impl Drop for Reaper {
    fn drop(&mut self) {
        if let Some(pid) = alive(&self.0) {
            // SAFETY: as in `signal`; a failed kill means the daemon is
            // already gone.
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
    }
}
