//! The FSEvents watcher: one `notify` watcher per worktree plus a
//! debounce task.
//!
//! The notify handler runs on the FSEvents thread and only filters and
//! signals — it never blocks. The debounce task waits for a quiet spell,
//! capped so a busy tree still reports, then sends one `Changed`-mark to
//! the owning worktree task. An event that names the worktree root itself
//! with the path gone is a `Gone`: the tree was removed.

use std::path::Path;
use std::time::Duration;

use notify::{Event, RecursiveMode, Watcher as _};
use tokio::sync::mpsc;

/// A change lands after this long without a further event.
const QUIET: Duration = Duration::from_millis(500);

/// ...but no later than this long after the first event of a burst.
const MAX_DELAY: Duration = Duration::from_secs(5);

/// What the debounce hands the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchMsg {
    /// A debounced burst of filesystem changes.
    Changed,
    /// The worktree root itself is gone.
    Gone,
}

/// The handler's raw signal; `Gone` skips the debounce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    Changed,
    Gone,
}

/// A running watch. `changed` receives one message per debounced burst; the
/// FSEvents stream stops when the value is dropped.
pub struct Watcher {
    /// Debounced change marks for the owner.
    pub changed: mpsc::Receiver<WatchMsg>,
    _watcher: notify::FsEventWatcher,
}

/// Watches `root` recursively, and `git_dir` too when it lies outside
/// `root` (a linked worktree). Events under `.git/` directories are
/// dropped — git's own writes are not worktree changes — except `HEAD`
/// directly in the git dir.
pub fn start(root: &Path, git_dir: &Path) -> Result<Watcher, notify::Error> {
    let (signal_tx, signal_rx) = mpsc::channel::<Signal>(1);
    let root = root.to_path_buf();
    let git_dir = git_dir.to_path_buf();
    let handler = {
        let git_dir = git_dir.clone();
        let head = git_dir.join("HEAD");
        let root = root.clone();
        move |event: Result<Event, notify::Error>| {
            let event = match event {
                Ok(event) => event,
                // A dropped or garbled event is treated as a change: a
                // rescan barrier is cheap and never loses an edit.
                Err(error) => {
                    tracing::warn!(%error, "FSEvents error");
                    let _ = signal_tx.try_send(Signal::Changed);
                    return;
                }
            };
            let mut gone = false;
            let mut changed = false;
            for path in &event.paths {
                if path == &root {
                    // The root's own event is a removal or a rename away.
                    gone |= !root.exists();
                } else if path.starts_with(&git_dir) && path == &head
                    || !path.starts_with(&git_dir)
                        && !path.components().any(|part| part.as_os_str() == ".git")
                {
                    changed = true;
                }
            }
            let signal = if gone {
                Some(Signal::Gone)
            } else if changed {
                Some(Signal::Changed)
            } else {
                None
            };
            if let Some(signal) = signal {
                // A full channel means a signal is already pending, which
                // coalesces the burst.
                let _ = signal_tx.try_send(signal);
            }
        }
    };
    let mut watcher = notify::FsEventWatcher::new(handler, notify::Config::default())?;
    watcher.watch(&root, RecursiveMode::Recursive)?;
    if !git_dir.starts_with(&root) {
        watcher.watch(&git_dir, RecursiveMode::Recursive)?;
    }

    let (changed_tx, changed) = mpsc::channel(1);
    tokio::spawn(debounce(signal_rx, changed_tx));
    Ok(Watcher {
        changed,
        _watcher: watcher,
    })
}

/// Folds the raw signal stream into debounced `Changed` marks and `Gone`.
/// A burst ends after `QUIET` without a signal, but a `Changed` always
/// lands within `MAX_DELAY` of the burst's first signal.
async fn debounce(mut signal_rx: mpsc::Receiver<Signal>, changed_tx: mpsc::Sender<WatchMsg>) {
    while let Some(signal) = signal_rx.recv().await {
        if signal == Signal::Gone {
            if changed_tx.send(WatchMsg::Gone).await.is_err() {
                return;
            }
            continue;
        }
        let deadline = tokio::time::Instant::now() + MAX_DELAY;
        let flush = loop {
            // The burst ends on QUIET of silence or at the deadline,
            // whichever is first.
            let wait = QUIET.min(deadline.saturating_duration_since(tokio::time::Instant::now()));
            match tokio::time::timeout(wait, signal_rx.recv()).await {
                Ok(Some(Signal::Gone)) => break WatchMsg::Gone,
                Ok(Some(Signal::Changed)) => {}
                // The signal side went away; flush what is pending.
                Ok(None) | Err(_) => break WatchMsg::Changed,
            }
        };
        if changed_tx.send(flush).await.is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A burst of signals longer than `MAX_DELAY` still produces its
    /// `Changed` exactly `MAX_DELAY` after the first signal. Time is paused
    /// and advances only when every task waits, so the bound is exact.
    #[tokio::test(start_paused = true)]
    async fn a_burst_signals_within_max_delay() {
        let (signal_tx, signal_rx) = mpsc::channel::<Signal>(1);
        let (changed_tx, mut changed) = mpsc::channel(4);
        tokio::spawn(debounce(signal_rx, changed_tx));
        let start = tokio::time::Instant::now();
        // A signal every 400 ms never lets `QUIET` elapse.
        let burst = tokio::spawn(async move {
            for _ in 0..20 {
                signal_tx.send(Signal::Changed).await.expect("signal send");
                tokio::time::sleep(Duration::from_millis(400)).await;
            }
        });
        let mark = changed.recv().await.expect("a Changed mark");
        assert_eq!(mark, WatchMsg::Changed);
        assert_eq!(start.elapsed(), MAX_DELAY);
        burst.abort();
    }

    /// A single signal lands after exactly `QUIET` of silence.
    #[tokio::test(start_paused = true)]
    async fn a_quiet_burst_signals_after_quiet() {
        let (signal_tx, signal_rx) = mpsc::channel::<Signal>(1);
        let (changed_tx, mut changed) = mpsc::channel(4);
        tokio::spawn(debounce(signal_rx, changed_tx));
        let start = tokio::time::Instant::now();
        signal_tx.send(Signal::Changed).await.expect("signal send");
        let mark = changed.recv().await.expect("a Changed mark");
        assert_eq!(mark, WatchMsg::Changed);
        assert_eq!(start.elapsed(), QUIET);
    }

    /// The worktree root's removal skips the debounce.
    #[tokio::test(start_paused = true)]
    async fn gone_is_not_debounced() {
        let (signal_tx, signal_rx) = mpsc::channel::<Signal>(1);
        let (changed_tx, mut changed) = mpsc::channel(4);
        tokio::spawn(debounce(signal_rx, changed_tx));
        let start = tokio::time::Instant::now();
        signal_tx.send(Signal::Gone).await.expect("signal send");
        let mark = changed.recv().await.expect("a Gone mark");
        assert_eq!(mark, WatchMsg::Gone);
        assert_eq!(start.elapsed(), Duration::ZERO);
    }
}
