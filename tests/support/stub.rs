//! `StubBackend` and `StubEmbedder`: the test-side [`Backend`].
//!
//! Every embed call sleeps the configured per-bucket duration on the
//! embedder's own thread and reports a [`Record`] on the `std::sync::mpsc`
//! the test holds, so tests wait on `recv_timeout` and never poll.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier, OnceLock};
use std::time::{Duration, Instant};

use repo_search::daemon::{Backend, GpuOpen, Headroom, LoadStage};
use repo_search_core::qos::{self, QosClass};
use repo_search_core::store::ContentHash;
use repo_search_core::{BatchEmbedder, Embedder, QueryTask, TokenCounter, UnitVector};
use repo_search_coreml::GpuBlocked;

/// Which embedder instance made a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Instance {
    /// The single-chunk ANE instance.
    Ane,
    /// The batch GPU instance.
    Gpu,
}

/// What a [`Record`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A query embedding.
    Query,
    /// A document embedded as query work (on-the-spot): `embed_document` at
    /// `UserInteractive`, which only query jobs run at.
    QueryDocument,
    /// A document embedded as indexing work.
    Index,
    /// The GPU embedder was dropped.
    Dropped,
    /// An embed call began, recorded before its sleep (and its gate).
    Started,
    /// An armed [`IndexGate`] holds an Index-kind call.
    Held,
}

/// One embedded document. A batch call sends one record per document, all
/// sharing `instance`, `kind`, `batch_len`, `qos`, `start` and `end`.
#[derive(Debug)]
pub struct Record {
    /// The instance that embedded the document.
    pub instance: Instance,
    /// What the call was, derived from the method and the thread's QoS.
    pub kind: Kind,
    /// The document's content hash; `None` for queries.
    pub hash: Option<ContentHash>,
    /// The document's token count under [`StubCounter`].
    pub tokens: usize,
    /// The documents in the call: 1 for single embeds, up to 8 for batches.
    pub batch_len: usize,
    /// The thread's QoS when the call ran.
    pub qos: QosClass,
    /// The thread's effective scheduling priority, overrides included.
    #[allow(dead_code)] // asserted by the scheduler-level tests
    pub pri: i32,
    /// When the call started.
    pub start: Instant,
    /// When the call ended.
    #[allow(dead_code)] // asserted by the scheduler-level tests
    pub end: Instant,
}

/// The bucket lengths `StubCounter` reports and `StubEmbedder` sleeps by.
pub const BUCKETS: [usize; 3] = [128, 256, 512];

/// A stub's failure; nothing in the stub fails today.
#[derive(Debug, thiserror::Error)]
#[error("stub backend failure: {0}")]
pub struct StubError(&'static str);

/// The callback `watch_headroom` stores and `push_headroom` drives.
type OnChange = Box<dyn Fn(Headroom) + Send + Sync>;

/// The headroom monitor's guard; dropping it stops nothing in the stub.
pub struct StubMonitor;

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

    /// The bucket a document of `tokens` lands in.
    fn bucket(tokens: usize) -> usize {
        BUCKETS
            .into_iter()
            .find(|&b| tokens <= b)
            .unwrap_or_else(|| panic!("{tokens} tokens is over the {}-token cap", BUCKETS[2]))
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

/// A gate on an instance's Index-kind calls. Disarmed it is inert; once
/// [`IndexGate::arm`]ed, each such call registers its release channel,
/// sends a [`Kind::Held`] record and blocks until [`IndexGate::open`],
/// which releases every held and future call for good. A `Held` record
/// lands only after the release channel is registered, so a test that sees
/// it can `open` without racing the wait. The embedder thread is the gate's
/// only caller, so at most one call is ever held.
pub struct IndexGate {
    armed: AtomicBool,
    released: AtomicBool,
    release: OnceLock<mpsc::Sender<()>>,
}

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
    #[allow(dead_code)] // not every test target reads it
    pub entered: mpsc::Receiver<()>,
    open: Arc<Barrier>,
    hold: Arc<Hold>,
}

impl HoldHandle {
    /// Releases the held call.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn open(&self) {
        self.open.wait();
    }

    /// Arms a hold installed disarmed: the next call into the hook is
    /// held.
    #[allow(dead_code)] // only the socket-level tests use it
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

impl IndexGate {
    /// Holds the Index-kind calls that start from now on.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    /// Releases every held and future call.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn open(&self) {
        self.released.store(true, Ordering::Release);
        if let Some(release) = self.release.get() {
            let _ = release.send(());
        }
    }
}

fn held_record(instance: Instance, qos: QosClass, now: Instant) -> Record {
    Record {
        instance,
        kind: Kind::Held,
        hash: None,
        tokens: 0,
        batch_len: 0,
        qos,
        pri: 0,
        start: now,
        end: now,
    }
}

/// The test-side [`Backend`]. `open_gpu` counts its calls and is `Blocked`
/// while the pushed headroom says so; `watch_headroom` stores the callback
/// `push_headroom` drives.
pub struct StubBackend {
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    /// Published by `open_ane`, which runs on the ANE thread itself: how a
    /// daemon-level test reaches the thread `loaded` does not surface.
    ane_thread: OnceLock<qos::ThreadId>,
    ane_opens: AtomicUsize,
    ane_open_fails: AtomicBool,
    open_ane_hold: Option<Arc<Hold>>,
    counter_hold: Option<Arc<Hold>>,
    document_hold: Option<Arc<Hold>>,
    gpu_opens: AtomicUsize,
    gpu_open_fails: AtomicBool,
    headroom: tokio::sync::watch::Sender<Headroom>,
    /// Keeps the watch alive so `send_replace` stores; never read.
    _headroom_rx: tokio::sync::watch::Receiver<Headroom>,
    on_change: OnceLock<OnChange>,
    gates: BTreeMap<Instance, BTreeMap<usize, Arc<Barrier>>>,
    index_gates: BTreeMap<Instance, Arc<IndexGate>>,
    dim: usize,
}

impl StubBackend {
    /// Routes the record log at `records`.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn records_to(&mut self, records: mpsc::Sender<Record>) {
        self.records = records;
    }

    /// A stub whose embed calls return at once; `set_delay` adds sleeps.
    pub fn new(records: mpsc::Sender<Record>) -> Self {
        let (headroom, _headroom_rx) =
            tokio::sync::watch::channel(Err(GpuBlocked::PowerSourceUnknown));
        Self {
            records,
            delays: BTreeMap::new(),
            ane_thread: OnceLock::new(),
            ane_opens: AtomicUsize::new(0),
            ane_open_fails: AtomicBool::new(false),
            open_ane_hold: None,
            counter_hold: None,
            document_hold: None,
            gpu_opens: AtomicUsize::new(0),
            gpu_open_fails: AtomicBool::new(false),
            headroom,
            _headroom_rx,
            on_change: OnceLock::new(),
            gates: BTreeMap::new(),
            index_gates: BTreeMap::new(),
            dim: 64,
        }
    }

    /// The sleep for calls of `instance` in every bucket.
    pub fn set_instance_delay(&mut self, instance: Instance, delay: Duration) {
        for bucket in BUCKETS {
            self.delays.insert((instance, bucket), delay);
        }
    }

    /// The sleep for calls of `instance` in one token `bucket`.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn set_delay(&mut self, instance: Instance, bucket: usize, delay: Duration) {
        self.delays.insert((instance, bucket), delay);
    }

    /// Gates `instance`'s first embed call on a barrier the test opens.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn gate_first(&mut self, instance: Instance) -> Arc<Barrier> {
        self.gate_call(instance, 1)
    }

    /// Gates `instance`'s `call`-th embed call — `call` is 1-based over the
    /// embedder's `embed_*` calls — on a barrier the test opens.
    pub fn gate_call(&mut self, instance: Instance, call: usize) -> Arc<Barrier> {
        let gate = Arc::new(Barrier::new(2));
        self.gates
            .entry(instance)
            .or_default()
            .insert(call, gate.clone());
        gate
    }

    /// Gates `instance`'s Index-kind calls on an [`IndexGate`], disarmed
    /// until the test arms it.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn gate_first_index(&mut self, instance: Instance) -> Arc<IndexGate> {
        let gate = Arc::new(IndexGate {
            armed: AtomicBool::new(false),
            released: AtomicBool::new(false),
            release: OnceLock::new(),
        });
        self.index_gates.insert(instance, gate.clone());
        gate
    }

    /// Holds the first `open_ane` call: it signals `entered` and blocks
    /// until the handle's `open`.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn hold_open_ane(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(true);
        self.open_ane_hold = Some(hold);
        handle
    }

    /// Holds the first [`TokenCounter`] call the model's counter serves:
    /// inside a barrier that is `reconcile`'s walk.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn hold_counter(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(true);
        self.counter_hold = Some(hold);
        handle
    }

    /// Installs a disarmed hold on the model counter's
    /// `document_tokens`: once the handle is armed, the next call (the
    /// feeder counting a non-empty feed, after its cache read) signals
    /// `entered` and blocks until the handle's `open`.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn hold_document_tokens(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(false);
        self.document_hold = Some(hold);
        handle
    }

    /// The ANE thread's id, published by `open_ane` from that thread.
    ///
    /// # Panics
    ///
    /// When `open_ane` has not run yet; any stub record proves it has.
    pub fn ane_thread(&self) -> qos::ThreadId {
        *self
            .ane_thread
            .get()
            .expect("open_ane publishes the ANE thread")
    }

    /// Makes `open_ane` fail, for the fatal-path test.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn fail_ane_open(&self) {
        self.ane_open_fails.store(true, Ordering::Relaxed);
    }

    /// How often `open_ane` was called.
    #[allow(dead_code)] // only the socket-level tests use it
    pub fn ane_opens(&self) -> usize {
        self.ane_opens.load(Ordering::Relaxed)
    }

    /// Makes `open_gpu` fail, for the fatal-path test.
    #[allow(dead_code)] // used by the scheduler-level tests
    pub fn fail_gpu_open(&self) {
        self.gpu_open_fails.store(true, Ordering::Relaxed);
    }

    /// How often `open_gpu` was called.
    pub fn gpu_opens(&self) -> usize {
        self.gpu_opens.load(Ordering::Relaxed)
    }

    /// Pushes a headroom value: stored for `open_gpu` and delivered to the
    /// callback `watch_headroom` registered.
    ///
    /// # Panics
    ///
    /// When `watch_headroom` has not been called yet.
    pub fn push_headroom(&self, headroom: Headroom) {
        let _ = self.headroom.send_replace(headroom.clone());
        self.on_change.get().expect("watch_headroom was not called")(headroom);
    }

    fn embedder(&self, instance: Instance) -> StubEmbedder {
        StubEmbedder {
            instance,
            counter: StubCounter::default(),
            records: self.records.clone(),
            delays: self.delays.clone(),
            gates: self.gates.get(&instance).cloned().unwrap_or_default(),
            calls: 0,
            index_gate: self.index_gates.get(&instance).cloned(),
            dim: self.dim,
        }
    }
}

impl Backend for StubBackend {
    type Ane = StubEmbedder;
    type Gpu = StubEmbedder;
    type Counter = StubCounter;
    type LoadError = StubError;
    type Monitor = StubMonitor;

    fn load_counter(&self, _: &Path) -> Result<(Self::Counter, Box<[usize]>), Self::LoadError> {
        Ok((
            StubCounter {
                hold: self.counter_hold.clone(),
                document_hold: self.document_hold.clone(),
            },
            BUCKETS.into(),
        ))
    }

    fn open_ane(
        &self,
        _: &Path,
        _: &str,
        _: &tokio::sync::watch::Sender<LoadStage>,
    ) -> Result<Self::Ane, Self::LoadError> {
        let _ = self.ane_thread.set(qos::ThreadId::current());
        self.ane_opens.fetch_add(1, Ordering::Relaxed);
        if let Some(hold) = &self.open_ane_hold {
            hold.enter();
        }
        if self.ane_open_fails.load(Ordering::Relaxed) {
            return Err(StubError("injected ANE open failure"));
        }
        Ok(self.embedder(Instance::Ane))
    }

    fn open_gpu(&self, _: &Path, _: &str) -> Result<Self::Gpu, GpuOpen<Self::LoadError>> {
        self.gpu_opens.fetch_add(1, Ordering::Relaxed);
        if self.gpu_open_fails.load(Ordering::Relaxed) {
            return Err(GpuOpen::Failed(StubError("injected GPU open failure")));
        }
        if let Err(blocked) = self.headroom.borrow().clone() {
            return Err(GpuOpen::Blocked(blocked));
        }
        Ok(self.embedder(Instance::Gpu))
    }

    fn watch_headroom(&self, on_change: OnChange) -> Self::Monitor {
        self.on_change
            .set(on_change)
            .ok()
            .expect("watch_headroom called once");
        let headroom = self.headroom.borrow().clone();
        self.push_headroom(headroom);
        StubMonitor
    }
}

/// The stub embedder, for either instance. Every call reports a
/// [`Kind::Started`] record, waits out the gate installed for its 1-based
/// ordinal if any, sleeps its bucket's configured duration, then reports a
/// [`Record`] per document.
pub struct StubEmbedder {
    instance: Instance,
    counter: StubCounter,
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    gates: BTreeMap<usize, Arc<Barrier>>,
    /// The embed calls this embedder served, for picking a call's gate.
    calls: usize,
    index_gate: Option<Arc<IndexGate>>,
    dim: usize,
}

impl StubEmbedder {
    fn sleep(&self, tokens: usize) {
        let delay = self
            .delays
            .get(&(self.instance, StubCounter::bucket(tokens)))
            .copied()
            .unwrap_or_default();
        std::thread::sleep(delay);
    }

    /// Blocks this call while the barrier installed for its 1-based
    /// ordinal is closed; calls with no gate pass.
    fn wait_gate(&mut self) {
        self.calls += 1;
        if let Some(gate) = self.gates.get(&self.calls) {
            gate.wait();
        }
    }

    /// Holds an Index-kind call while the gate is armed and unreleased,
    /// sending its [`Kind::Held`] record once the release channel is
    /// registered. The call blocks on that channel until
    /// [`IndexGate::open`].
    fn hold_index_call(&self) {
        let Some(gate) = &self.index_gate else {
            return;
        };
        if !gate.armed.load(Ordering::Acquire) || gate.released.load(Ordering::Acquire) {
            return;
        }
        let (release, wait) = mpsc::channel::<()>();
        gate.release
            .set(release)
            .expect("the embedder thread holds one call per gate");
        let _ = self.records.send(held_record(
            self.instance,
            qos::current_thread(),
            Instant::now(),
        ));
        // `open` may have landed between the armed check and registering:
        // then this call passes without waiting.
        if gate.released.load(Ordering::Acquire) {
            return;
        }
        wait.recv().expect("open releases every held call");
    }

    /// Reports the call's start, before the gate and the sleep.
    fn started(&self, hash: Option<ContentHash>, tokens: usize, batch_len: usize) {
        let now = Instant::now();
        self.record(Kind::Started, hash, tokens, batch_len, now, now);
    }

    fn kind(&self) -> Kind {
        match qos::current_thread() {
            QosClass::UserInteractive => Kind::QueryDocument,
            _ => Kind::Index,
        }
    }

    fn record(
        &self,
        kind: Kind,
        hash: Option<ContentHash>,
        tokens: usize,
        batch_len: usize,
        start: Instant,
        end: Instant,
    ) {
        let _ = self.records.send(Record {
            instance: self.instance,
            kind,
            hash,
            tokens,
            batch_len,
            qos: qos::current_thread(),
            pri: qos::scheduling_priority(qos::ThreadId::current()),
            start,
            end,
        });
    }
}

impl Drop for StubEmbedder {
    fn drop(&mut self) {
        let now = Instant::now();
        self.record(Kind::Dropped, None, 0, 0, now, now);
    }
}

impl Embedder for StubEmbedder {
    type Error = StubError;

    fn model_id(&self) -> &str {
        "stub-model"
    }

    fn embed_document(&mut self, title: &str, text: &str) -> Result<UnitVector, Self::Error> {
        let tokens = self.counter.document_tokens(title, text);
        let kind = self.kind();
        self.started(Some(ContentHash::of_chunk(title, text)), tokens, 1);
        self.wait_gate();
        if kind == Kind::Index {
            self.hold_index_call();
        }
        let start = Instant::now();
        self.sleep(tokens);
        let end = Instant::now();
        self.record(
            kind,
            Some(ContentHash::of_chunk(title, text)),
            tokens,
            1,
            start,
            end,
        );
        Ok(bag_of_words(&format!("{title} {text}"), self.dim))
    }

    fn embed_query(&mut self, task: QueryTask, query: &str) -> Result<UnitVector, Self::Error> {
        let tokens = self.counter.count(&task.prompt(query));
        self.started(None, tokens, 1);
        self.wait_gate();
        let start = Instant::now();
        self.sleep(tokens);
        let end = Instant::now();
        self.record(Kind::Query, None, tokens, 1, start, end);
        Ok(bag_of_words(query, self.dim))
    }
}

impl BatchEmbedder for StubEmbedder {
    fn embed_documents(&mut self, docs: &[(&str, &str)]) -> Result<Vec<UnitVector>, Self::Error> {
        let tokens: Vec<usize> = docs
            .iter()
            .map(|(title, text)| self.counter.document_tokens(title, text))
            .collect();
        let mut buckets = tokens.iter().map(|&t| StubCounter::bucket(t));
        if let Some(first) = buckets.next() {
            assert!(
                buckets.all(|b| b == first),
                "a batch mixes token buckets: {tokens:?}"
            );
        }
        for (&(title, text), tokens) in docs.iter().zip(&tokens) {
            self.started(
                Some(ContentHash::of_chunk(title, text)),
                *tokens,
                docs.len(),
            );
        }
        self.wait_gate();
        self.hold_index_call();
        let start = Instant::now();
        if let Some(&first) = tokens.first() {
            self.sleep(first);
        }
        let end = Instant::now();
        // The batch method is GPU-only indexing work: the kind is the
        // method's, never a query's.
        for ((title, text), tokens) in docs.iter().zip(tokens) {
            self.record(
                Kind::Index,
                Some(ContentHash::of_chunk(title, text)),
                tokens,
                docs.len(),
                start,
                end,
            );
        }
        Ok(docs
            .iter()
            .map(|(title, text)| bag_of_words(&format!("{title} {text}"), self.dim))
            .collect())
    }
}

/// Holds `class` over the backend's ANE thread for the caller's scope: the
/// thread's effective priority lifts while its requested class — what the
/// stub's records report — stays unchanged. `open_ane` must already have
/// run; any stub record proves it did.
#[must_use]
pub fn lift_ane(backend: &StubBackend, class: QosClass) -> qos::Override {
    qos::Override::start(backend.ane_thread(), class)
}

/// A deterministic unit vector: FNV-1a per whitespace word, modulo `dim`.
fn bag_of_words(text: &str, dim: usize) -> UnitVector {
    let mut v = vec![0.0f32; dim];
    for word in text.split_whitespace() {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in word.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        v[hash as usize % dim] += 1.0;
    }
    // A tiny dense jitter seeded by the whole text, so two different texts
    // differ in every dimension: the knn calibration's per-dimension bounds
    // are never flat and `query_weights` never degenerates.
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        d ^= u64::from(*byte);
        d = d.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for x in &mut v {
        d ^= d >> 33;
        d = d.wrapping_mul(0xff51_afd7_ed55_8ccd);
        *x += (d & 0xff) as f32 * 1e-3;
    }
    UnitVector::normalize(v)
}
