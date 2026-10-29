//! Event-driven waits on kqueue: a process's exit and a directory entry's
//! creation.
//!
//! Each watcher is built in two steps — `register`, then `wait`/`wait_for` —
//! so the registration can precede the action that causes the event: the
//! kernel queue holds the event even when the action lands between them.
//! The install flow uses them to wait out a `launchctl bootout` teardown
//! and the socket file a `bootstrap` materializes asynchronously.

use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use kqueue::{EventData, EventFilter, FilterFlag, Proc, Watcher};

/// A `NOTE_EXIT` registration for one pid.
pub struct ExitWatch {
    /// `None` when the process was already gone at registration: that is
    /// the exit, already delivered.
    watcher: Option<Watcher>,
}

impl ExitWatch {
    /// Registers `pid`'s `NOTE_EXIT`. A `watch()` error of `ESRCH` means
    /// the process is already gone, which is the event: the `Watcher` is
    /// stored as `None`.
    pub fn register(pid: u32) -> io::Result<Self> {
        let mut watcher = Watcher::new()?;
        watcher.add_pid(
            pid as libc::pid_t,
            EventFilter::EVFILT_PROC,
            FilterFlag::NOTE_EXIT,
        )?;
        let watcher = match watcher.watch() {
            Ok(()) => Some(watcher),
            Err(error) if error.raw_os_error() == Some(libc::ESRCH) => None,
            Err(error) => return Err(error),
        };
        Ok(Self { watcher })
    }

    /// `true` once the process has exited, `false` on `timeout`. One
    /// blocking `kevent` wait at a time, repeated only for events that are
    /// not this registration's `NOTE_EXIT`, against a single deadline.
    pub fn wait(self, timeout: Duration) -> bool {
        let Some(watcher) = self.watcher else {
            return true;
        };
        let deadline = Instant::now() + timeout;
        loop {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            match watcher.poll(Some(remaining)) {
                None => return false,
                Some(event) => match event.data {
                    EventData::Proc(Proc::Exit(_)) => return true,
                    EventData::Error(error) => {
                        panic!("kevent failed: {}", error.raw_os_error().unwrap_or(-1))
                    }
                    _ => {}
                },
            }
        }
    }
}

/// A `NOTE_WRITE` registration on a directory: an entry was added or
/// removed.
pub struct DirWatch {
    watcher: Watcher,
}

impl DirWatch {
    /// Registers a `NOTE_WRITE` watch on `dir`. Run before the action that
    /// creates the watched entry so the creation cannot be missed.
    pub fn register(dir: &Path) -> io::Result<Self> {
        let mut watcher = Watcher::new()?;
        watcher.add_filename(dir, EventFilter::EVFILT_VNODE, FilterFlag::NOTE_WRITE)?;
        watcher.watch()?;
        Ok(Self { watcher })
    }

    /// `true` once `path` (an entry of the watched dir) exists, `false` on
    /// `timeout`. `path` is checked first, then after each event, against
    /// a single deadline.
    pub fn wait_for(self, path: &Path, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if path.exists() {
                return true;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            match self.watcher.poll(Some(remaining)) {
                None => return false,
                Some(event) => {
                    if let EventData::Error(error) = event.data {
                        panic!("kevent failed: {}", error.raw_os_error().unwrap_or(-1))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::process::{Command, Stdio};

    /// A `cat` that lives until its piped stdin is dropped.
    fn cat() -> std::process::Child {
        Command::new("/bin/cat")
            .stdin(Stdio::piped())
            .spawn()
            .expect("/bin/cat spawns")
    }

    #[test]
    fn exit_watch_sees_a_child_exit() {
        let mut child = cat();
        let watch = ExitWatch::register(child.id()).expect("the exit registers");
        drop(child.stdin.take());
        assert!(
            watch.wait(Duration::from_secs(30)),
            "the exit event arrives"
        );
        child.wait().expect("the cat is reaped");
    }

    #[test]
    fn exit_watch_of_a_reaped_child_reports_exit() {
        let mut child = cat();
        drop(child.stdin.take());
        child.wait().expect("the cat is reaped");
        let watch = ExitWatch::register(child.id()).expect("the exit registers");
        assert!(
            watch.wait(Duration::from_secs(30)),
            "the gone process reads as exited"
        );
    }

    #[test]
    fn exit_watch_times_out_on_a_live_child() {
        let mut child = cat();
        let watch = ExitWatch::register(child.id()).expect("the exit registers");
        assert!(
            !watch.wait(Duration::from_secs(1)),
            "a live process does not exit"
        );
        drop(child.stdin.take());
        child.wait().expect("the cat is reaped");
    }

    #[test]
    fn dir_watch_sees_a_created_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.sock");
        let watch = DirWatch::register(dir.path()).expect("the dir registers");
        let created = std::thread::spawn({
            let path = path.clone();
            move || std::fs::write(&path, b"x").expect("the entry is created")
        });
        assert!(
            watch.wait_for(&path, Duration::from_secs(30)),
            "the creation arrives"
        );
        created.join().expect("the creator joins");
    }

    #[test]
    fn dir_watch_times_out_without_a_creation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watch = DirWatch::register(dir.path()).expect("the dir registers");
        assert!(
            !watch.wait_for(&dir.path().join("never"), Duration::from_secs(1)),
            "nothing was created"
        );
    }
}
