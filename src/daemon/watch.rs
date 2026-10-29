//! The filesystem watcher: one `notify` watcher per worktree plus a
//! debounce task.
//!
//! The notify handler runs on the watcher's event thread and only filters
//! and signals — it never blocks. `Access` events are dropped outright:
//! inotify reports every open and close, so the reconciler's own reads
//! would retrigger indexing forever, while real writes still arrive as
//! `Modify`/`Create`/`Remove`. The debounce task waits for a quiet spell,
//! capped so a busy tree still reports, then sends one `Changed`-mark to
//! the owning worktree task. An event that names the worktree root itself
//! with the path gone is a `Gone`: the tree was removed. `Gone` is
//! terminal and never folds into a burst, so it crosses its own one-shot
//! channel — no queued change signal can swallow it.

use std::path::Path;
use std::time::Duration;

use notify::{Event, EventKind, RecursiveMode, Watcher as _};
use tokio::sync::{mpsc, oneshot};

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

/// The handler's change signal, the channel's whole vocabulary. `Gone`
/// is left out on purpose: a full channel is how a burst coalesces, and
/// a queued signal would swallow the terminal mark.
struct Signal;

/// A running watch. `changed` receives one message per debounced burst
/// and a terminal `Gone`; the event stream stops when the value is
/// dropped.
pub struct Watcher {
    /// Debounced change marks for the owner.
    pub changed: mpsc::Receiver<WatchMsg>,
    _watcher: notify::RecommendedWatcher,
}

/// Watches `root` recursively, and `git_dir` too when it lies outside
/// `root` (a linked worktree). Events under `.git/` directories are
/// dropped — git's own writes are not worktree changes — except `HEAD`
/// directly in the git dir.
pub fn start(root: &Path, git_dir: &Path) -> Result<Watcher, notify::Error> {
    let (signal_tx, signal_rx) = mpsc::channel::<Signal>(1);
    let (gone_tx, gone_rx) = oneshot::channel::<()>();
    let root = root.to_path_buf();
    let git_dir = git_dir.to_path_buf();
    let handler = {
        let git_dir = git_dir.clone();
        let head = git_dir.join("HEAD");
        let root = root.clone();
        let mut gone_tx = Some(gone_tx);
        move |event: Result<Event, notify::Error>| {
            let event = match event {
                Ok(event) => event,
                // A dropped or garbled event is treated as a change: a
                // rescan barrier is cheap and never loses an edit.
                Err(error) => {
                    tracing::warn!(%error, "watcher error");
                    let _ = signal_tx.try_send(Signal);
                    return;
                }
            };
            match classify(&event, &root, &git_dir, &head) {
                // A full channel means a signal is already pending,
                // which coalesces the burst.
                Some(WatchMsg::Changed) => {
                    let _ = signal_tx.try_send(Signal);
                }
                // `Gone` cannot ride the signal channel — a pending
                // change would swallow the terminal mark — so the first
                // `Gone` fires a one-shot the debounce selects on.
                Some(WatchMsg::Gone) => {
                    if let Some(gone_tx) = gone_tx.take() {
                        let _ = gone_tx.send(());
                    }
                }
                None => {}
            }
        }
    };
    let mut watcher = notify::RecommendedWatcher::new(handler, notify::Config::default())?;
    watcher.watch(&root, RecursiveMode::Recursive)?;
    if !git_dir.starts_with(&root) {
        watcher.watch(&git_dir, RecursiveMode::Recursive)?;
    }

    let (changed_tx, changed) = mpsc::channel(1);
    tokio::spawn(debounce(signal_rx, gone_rx, changed_tx));
    Ok(Watcher {
        changed,
        _watcher: watcher,
    })
}

/// What one notify event means for the worktree; `None` is dropped.
/// `Access` events are dropped before the path check: FSEvents never
/// emits them, and real writes arrive under other kinds.
fn classify(event: &Event, root: &Path, git_dir: &Path, head: &Path) -> Option<WatchMsg> {
    if matches!(event.kind, EventKind::Access(_)) {
        return None;
    }
    let mut gone = false;
    let mut changed = false;
    for path in &event.paths {
        if path.as_path() == root {
            // The root's own event is a removal or a rename away.
            gone |= !root.exists();
        } else if path.starts_with(git_dir) && path.as_path() == head
            || !path.starts_with(git_dir)
                && !path.components().any(|part| part.as_os_str() == ".git")
        {
            changed = true;
        }
    }
    if gone {
        Some(WatchMsg::Gone)
    } else if changed {
        Some(WatchMsg::Changed)
    } else {
        None
    }
}

/// Folds the change signals into debounced `Changed` marks; `Gone`
/// arrives on its own one-shot — where no pending signal can swallow
/// it — skips the debounce and ends the task. A burst ends after
/// `QUIET` without a signal, but a `Changed` always lands within
/// `MAX_DELAY` of the burst's first signal.
async fn debounce(
    mut signal_rx: mpsc::Receiver<Signal>,
    mut gone_rx: oneshot::Receiver<()>,
    changed_tx: mpsc::Sender<WatchMsg>,
) {
    loop {
        tokio::select! {
            biased;
            // A dropped sender is the handler going away with the
            // watcher, not a removal.
            gone = &mut gone_rx => {
                if gone.is_ok() {
                    let _ = changed_tx.send(WatchMsg::Gone).await;
                }
                return;
            }
            signal = signal_rx.recv() => {
                // The handler went away with the watcher.
                if signal.is_none() {
                    return;
                }
            }
        }
        let deadline = tokio::time::Instant::now() + MAX_DELAY;
        let flush = loop {
            // The burst ends on QUIET of silence or at the deadline,
            // whichever is first.
            let wait = QUIET.min(deadline.saturating_duration_since(tokio::time::Instant::now()));
            tokio::select! {
                biased;
                gone = &mut gone_rx => match gone {
                    Ok(()) => break WatchMsg::Gone,
                    // The handler went away: flush what is pending and
                    // stop, since a completed one-shot is never polled
                    // again.
                    Err(_) => {
                        let _ = changed_tx.send(WatchMsg::Changed).await;
                        return;
                    }
                },
                result = tokio::time::timeout(wait, signal_rx.recv()) => match result {
                    // The burst grows.
                    Ok(Some(_)) => {}
                    // The signal side went away, or the quiet spell ran
                    // out; flush what is pending.
                    Ok(None) | Err(_) => break WatchMsg::Changed,
                }
            }
        };
        // A `Gone` is the last mark the owner will read.
        if changed_tx.send(flush).await.is_err() || flush == WatchMsg::Gone {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use notify::event::{AccessKind, ModifyKind};

    /// A spawned `debounce` and the ends its driver holds.
    fn rig() -> (
        mpsc::Sender<Signal>,
        oneshot::Sender<()>,
        mpsc::Receiver<WatchMsg>,
    ) {
        let (signal_tx, signal_rx) = mpsc::channel(1);
        let (gone_tx, gone_rx) = oneshot::channel();
        let (changed_tx, changed) = mpsc::channel(4);
        tokio::spawn(debounce(signal_rx, gone_rx, changed_tx));
        (signal_tx, gone_tx, changed)
    }

    /// `Access` events produce no signal — inotify's reports of the
    /// reconciler's own reads cannot start a reindex loop — while a write
    /// still classifies as `Changed`.
    #[test]
    fn access_events_are_dropped() {
        let root = Path::new("/worktree");
        let git_dir = root.join(".git");
        let head = git_dir.join("HEAD");
        let file = root.join("src/a.rs");
        let read = Event::new(EventKind::Access(AccessKind::Read)).add_path(file.clone());
        assert_eq!(classify(&read, root, &git_dir, &head), None);
        let write = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(file);
        assert_eq!(
            classify(&write, root, &git_dir, &head),
            Some(WatchMsg::Changed)
        );
    }

    /// A burst of signals longer than `MAX_DELAY` still produces its
    /// `Changed` exactly `MAX_DELAY` after the first signal. Time is paused
    /// and advances only when every task waits, so the bound is exact.
    #[tokio::test(start_paused = true)]
    async fn a_burst_signals_within_max_delay() {
        let (signal_tx, _gone_tx, mut changed) = rig();
        let start = tokio::time::Instant::now();
        // A signal every 400 ms never lets `QUIET` elapse.
        let burst = tokio::spawn(async move {
            for _ in 0..20 {
                signal_tx.send(Signal).await.expect("signal send");
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
        let (signal_tx, _gone_tx, mut changed) = rig();
        let start = tokio::time::Instant::now();
        signal_tx.send(Signal).await.expect("signal send");
        let mark = changed.recv().await.expect("a Changed mark");
        assert_eq!(mark, WatchMsg::Changed);
        assert_eq!(start.elapsed(), QUIET);
    }

    /// The worktree root's removal skips the debounce.
    #[tokio::test(start_paused = true)]
    async fn gone_is_not_debounced() {
        let (_signal_tx, gone_tx, mut changed) = rig();
        let start = tokio::time::Instant::now();
        gone_tx.send(()).expect("gone send");
        let mark = changed.recv().await.expect("a Gone mark");
        assert_eq!(mark, WatchMsg::Gone);
        assert_eq!(start.elapsed(), Duration::ZERO);
    }

    /// A `Gone` fired while a change signal is still queued is delivered
    /// anyway: the terminal mark never waits behind a pending burst.
    #[tokio::test(start_paused = true)]
    async fn gone_arrives_through_a_pending_change() {
        let (signal_tx, gone_tx, mut changed) = rig();
        let start = tokio::time::Instant::now();
        signal_tx.send(Signal).await.expect("signal send");
        gone_tx.send(()).expect("gone send");
        let mark = changed.recv().await.expect("a Gone mark");
        assert_eq!(mark, WatchMsg::Gone);
        assert_eq!(start.elapsed(), Duration::ZERO);
    }
}
