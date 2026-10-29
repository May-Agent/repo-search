//! `StubTokens` and `StubCounter`: the test-side [`Tokens`].
//!
//! `StubTokens::load` returns at once — no download — with a
//! [`StubCounter`] carrying the holds the test installed. The counter
//! keeps the arithmetic the server-side stub derives its `tokens` from
//! (`len / 4 + 1`), so the server's `StubEncoder` reports the same token
//! counts for the same document prompts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};

use repo_search::daemon::counter::Tokens;
use repo_search::daemon::scheduler::QueryInput;
use repo_search_core::TokenCounter;
use repo_search_core::model::{Fetch, QueryTask};
use tokio::sync::watch;

/// A test-installed hold on a stub hook: the first call into the hook
/// signals `entered` and blocks until [`HoldHandle::open`]; every call
/// after it passes.
pub struct Hold {
    armed: AtomicBool,
    entered: mpsc::Sender<()>,
    open: Arc<Barrier>,
    fired: AtomicBool,
}

/// The test's half of a [`Hold`].
pub struct HoldHandle {
    /// Resolves once the held call is inside the hook.
    pub entered: mpsc::Receiver<()>,
    open: Arc<Barrier>,
    hold: Arc<Hold>,
}

impl HoldHandle {
    /// Releases the held call.
    pub fn open(&self) {
        self.open.wait();
    }

    /// Arms a hold installed disarmed: the next call into the hook is
    /// held.
    pub fn arm(&self) {
        self.hold.armed.store(true, Ordering::Release);
    }
}

impl Hold {
    fn install(armed: bool) -> (Arc<Self>, HoldHandle) {
        let (entered_tx, entered) = mpsc::channel();
        let open = Arc::new(Barrier::new(2));
        let hold = Arc::new(Self {
            armed: AtomicBool::new(armed),
            entered: entered_tx,
            open: open.clone(),
            fired: AtomicBool::new(false),
        });
        let handle = HoldHandle {
            entered,
            open,
            hold: hold.clone(),
        };
        (hold, handle)
    }

    /// The first call reports on `entered` and blocks until `open`; calls
    /// after it — and every call once nobody is listening — pass.
    fn enter(&self) {
        if !self.armed.load(Ordering::Acquire) {
            return;
        }
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.entered.send(()).is_err() {
            return;
        }
        self.open.wait();
    }
}

/// A [`TokenCounter`] counting `len / 4 + 1` tokens; `hold` parks the
/// first call, which in a barrier lands inside `reconcile`'s walk;
/// `document_hold` parks the first `document_tokens` call once armed,
/// which only the feeder's token counting makes.
#[derive(Clone, Default)]
pub struct StubCounter {
    hold: Option<Arc<Hold>>,
    document_hold: Option<Arc<Hold>>,
}

impl StubCounter {
    fn enter_hold(&self) {
        if let Some(hold) = &self.hold {
            hold.enter();
        }
    }

    fn tokens_of(title: &str, text: &str) -> usize {
        (title.len() + text.len()) / 4 + 1
    }
}

impl TokenCounter for StubCounter {
    fn count(&self, text: &str) -> usize {
        self.enter_hold();
        text.len() / 4 + 1
    }

    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool {
        self.enter_hold();
        Self::tokens_of(title, &text.concat()) <= max
    }

    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize> {
        self.enter_hold();
        if self.fits(title, text, max) {
            return None;
        }
        let joined = text.concat();
        Some(
            max.saturating_mul(4)
                .saturating_sub(title.len())
                .min(joined.len()),
        )
    }

    fn document_tokens(&self, title: &str, text: &str) -> usize {
        self.enter_hold();
        if let Some(hold) = &self.document_hold {
            hold.enter();
        }
        Self::tokens_of(title, text)
    }
}

impl QueryInput for StubCounter {
    /// The query `input` the scheduler sends: the task's prompt —
    /// `task.prompt(query)` is what the old stub embedded.
    fn input(&self, task: QueryTask, query: &str) -> String {
        task.prompt(query)
    }
}

/// The test-side [`Tokens`]: `load` returns a [`StubCounter`] at once,
/// carrying the holds the test installed before handing the tokens over.
#[derive(Default)]
pub struct StubTokens {
    counter_hold: Option<Arc<Hold>>,
    document_hold: Option<Arc<Hold>>,
}

impl StubTokens {
    /// Holds the first token count of any kind: `count`, `fits`,
    /// `fit_end` or `document_tokens` signals `entered` and blocks until
    /// [`HoldHandle::open`].
    pub fn hold_counter(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(true);
        self.counter_hold = Some(hold);
        handle
    }

    /// Installs a disarmed hold on the counter's `document_tokens`: once
    /// the handle is armed, the next call (the feeder counting a
    /// non-empty feed, after its cache read) signals `entered` and blocks
    /// until [`HoldHandle::open`].
    pub fn hold_document_tokens(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(false);
        self.document_hold = Some(hold);
        handle
    }
}

impl Tokens for StubTokens {
    type Counter = StubCounter;

    async fn load(&self, _progress: &watch::Sender<Fetch>) -> Result<Self::Counter, String> {
        Ok(StubCounter {
            hold: self.counter_hold.clone(),
            document_hold: self.document_hold.clone(),
        })
    }
}
