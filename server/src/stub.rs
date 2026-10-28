//! `StubBackend` and `StubEmbedder`: the test-side [`Backend`].
//!
//! Every embed call sleeps the configured per-bucket duration on the
//! embedder's own thread and reports a [`Record`] on the `std::sync::mpsc`
//! the test holds, so tests wait on `recv_timeout` and never poll.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier, OnceLock, Weak};
use std::time::{Duration, Instant};

use std::collections::HashMap;
use std::sync::Mutex;

use repo_search_core::UnitVector;
use repo_search_core::qos::{self, QosClass};
use repo_search_core::store::ContentHash;
use repo_search_coreml::GpuBlocked;
use tokio::sync::watch;

use crate::backend::{
    Backend, EncodeError, EncoderLoad, GpuOpen, Headroom, LoadStage, TextEncoder,
    TokenBatchEmbedder, TokenEmbedder,
};

/// Which embedder instance made a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Instance {
    /// The single-input ANE instance.
    Ane,
    /// The batch GPU instance.
    Gpu,
}

/// What a [`Record`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An embed call at `UserInteractive`: u=0 work.
    Interactive,
    /// An embed call below `UserInteractive`: normal or background work.
    Index,
    /// The GPU embedder was dropped.
    Dropped,
    /// An embed call began, recorded before its sleep (and its gate).
    Started,
    /// An armed [`IndexGate`] holds an Index-kind call.
    Held,
}

/// One embedded input. A batch call sends one record per input, all
/// sharing `instance`, `kind`, `batch_len`, `qos`, `start` and `end`.
#[derive(Debug)]
pub struct Record {
    /// The instance that embedded the input.
    pub instance: Instance,
    /// What the call was, derived from the thread's QoS.
    pub kind: Kind,
    /// The input's token ids; empty for `Started`/`Held`/`Dropped`.
    pub ids: Vec<u32>,
    /// The document's content hash, recovered by the encoder from the
    /// `DOCUMENT` prompt; `None` for queries and non-embedding records.
    pub hash: Option<ContentHash>,
    /// The input's token count: `ids.len()`.
    pub tokens: usize,
    /// The inputs in the call: 1 for single embeds, up to 8 for batches.
    pub batch_len: usize,
    /// The thread's QoS when the call ran.
    pub qos: QosClass,
    /// The thread's effective scheduling priority, overrides included.
    pub pri: i32,
    /// When the call started.
    pub start: Instant,
    /// When the call ended.
    pub end: Instant,
}

/// The bucket lengths the stub reports and sleeps by.
pub const BUCKETS: [usize; 3] = [128, 256, 512];

/// The bucket `tokens` lands in.
pub fn bucket(tokens: usize) -> usize {
    BUCKETS
        .into_iter()
        .find(|&b| tokens <= b)
        .unwrap_or_else(|| panic!("{tokens} tokens is over the {}-token cap", BUCKETS[2]))
}

/// A stub's failure; nothing in the stub fails today.
#[derive(Debug, thiserror::Error)]
#[error("stub backend failure: {0}")]
pub struct StubError(&'static str);

/// The callback `watch_headroom` stores and `push_headroom` drives.
type OnChange = Box<dyn Fn(Headroom) + Send + Sync>;

/// The headroom monitor's guard. Like `HeadroomMonitor`, it owns the
/// callback: dropping it drops the callback and whatever the callback
/// holds, and `push_headroom` reaches it only while the guard lives.
pub struct StubMonitor {
    /// The strong handle; the backend keeps only a weak one.
    _on_change: Arc<OnChange>,
}

/// The ids→document map the encoder and embedders share: `encode`
/// registers a `DOCUMENT` input's hash under the ids it produced, so the
/// embedder's [`Record`] names the chunk it embedded.
type Hashes = Arc<Mutex<HashMap<Vec<u32>, ContentHash>>>;

/// The stub's text encoder: `len / 4 + 1` ids, today's `StubCounter`
/// arithmetic, capped at the largest bucket. The ids are seeded by the
/// text, so two different inputs embed to different vectors; a `DOCUMENT`
/// prompt additionally registers its content hash.
#[derive(Debug)]
pub struct StubEncoder {
    hashes: Hashes,
}

/// Recovers a `DOCUMENT` prompt's content hash: `title: t | text: b`
/// parses back into the chunk `of_chunk` hashed.
fn document_hash(text: &str) -> Option<ContentHash> {
    let rest = text.strip_prefix("title: ")?;
    let (title, body) = rest.split_once(" | text: ")?;
    Some(ContentHash::of_chunk(title, body))
}

/// The ids [`StubEncoder`] gives `text`, cap aside: `len / 4 + 1` of them,
/// counting up from the text's FNV-1a, so two different inputs embed to
/// different vectors. Tests name an input's records by them.
pub fn encode_ids(text: &str) -> Vec<u32> {
    let mut seed = 0x811c_9dc5u32;
    for byte in text.as_bytes() {
        seed ^= u32::from(*byte);
        seed = seed.wrapping_mul(0x0100_0193);
    }
    (0..text.len() / 4 + 1)
        .map(|i| seed.wrapping_add(i as u32))
        .collect()
}

impl TextEncoder for StubEncoder {
    fn encode(&self, text: &str) -> Result<Vec<u32>, EncodeError> {
        let ids = encode_ids(text);
        if ids.len() > BUCKETS[2] {
            return Err(EncodeError::TooLong(format!(
                "{} tokens exceeds the {}-token cap",
                ids.len(),
                BUCKETS[2]
            )));
        }
        if let Some(hash) = document_hash(text) {
            self.hashes
                .lock()
                .expect("the hashes map")
                .insert(ids.clone(), hash);
        }
        Ok(ids)
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

impl IndexGate {
    /// Holds the Index-kind calls that start from now on.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    /// Releases every held and future call.
    pub fn open(&self) {
        self.released.store(true, Ordering::Release);
        if let Some(release) = self.release.get() {
            let _ = release.send(());
        }
    }
}

fn held_record(instance: Instance, now: Instant) -> Record {
    Record {
        instance,
        kind: Kind::Held,
        ids: Vec::new(),
        hash: None,
        tokens: 0,
        batch_len: 0,
        qos: qos::current_thread(),
        pri: 0,
        start: now,
        end: now,
    }
}

/// The test-side [`Backend`]. `open_gpu` counts its calls and is `Blocked`
/// while the pushed headroom says so; `watch_headroom` keeps a weak handle
/// on the callback `push_headroom` drives.
pub struct StubBackend {
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    /// Published by `open_ane`, which runs on the ANE thread itself: how a
    /// test reaches the thread `loaded` does not surface.
    ane_thread: OnceLock<qos::ThreadId>,
    ane_opens: AtomicUsize,
    ane_open_fails: AtomicBool,
    open_ane_hold: Option<Arc<Hold>>,
    gpu_opens: AtomicUsize,
    gpu_open_fails: AtomicBool,
    headroom: watch::Sender<Headroom>,
    /// Keeps the watch alive so `send_replace` stores; never read.
    _headroom_rx: watch::Receiver<Headroom>,
    on_change: OnceLock<Weak<OnChange>>,
    gates: BTreeMap<Instance, BTreeMap<usize, Arc<Barrier>>>,
    index_gates: BTreeMap<Instance, Arc<IndexGate>>,
    dim: usize,
    hashes: Hashes,
}

impl StubBackend {
    /// Routes the record log at `records`.
    pub fn records_to(&mut self, records: mpsc::Sender<Record>) {
        self.records = records;
    }

    /// A stub whose embed calls return at once; `set_delay` adds sleeps.
    pub fn new(records: mpsc::Sender<Record>) -> Self {
        let (headroom, _headroom_rx) = watch::channel(Err(GpuBlocked::PowerSourceUnknown));
        Self {
            records,
            delays: BTreeMap::new(),
            ane_thread: OnceLock::new(),
            ane_opens: AtomicUsize::new(0),
            ane_open_fails: AtomicBool::new(false),
            open_ane_hold: None,
            gpu_opens: AtomicUsize::new(0),
            gpu_open_fails: AtomicBool::new(false),
            headroom,
            _headroom_rx,
            on_change: OnceLock::new(),
            gates: BTreeMap::new(),
            index_gates: BTreeMap::new(),
            dim: 64,
            hashes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The stub vector's width; 64 by default.
    pub fn set_dim(&mut self, dim: usize) {
        self.dim = dim;
    }

    /// The sleep for calls of `instance` in every bucket.
    pub fn set_instance_delay(&mut self, instance: Instance, delay: Duration) {
        for bucket in BUCKETS {
            self.delays.insert((instance, bucket), delay);
        }
    }

    /// The sleep for calls of `instance` in one token `bucket`.
    pub fn set_delay(&mut self, instance: Instance, bucket: usize, delay: Duration) {
        self.delays.insert((instance, bucket), delay);
    }

    /// Gates `instance`'s first embed call on a barrier the test opens.
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
    pub fn hold_open_ane(&mut self) -> HoldHandle {
        let (hold, handle) = Hold::install(true);
        self.open_ane_hold = Some(hold);
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
    pub fn fail_ane_open(&self) {
        self.ane_open_fails.store(true, Ordering::Relaxed);
    }

    /// How often `open_ane` was called.
    pub fn ane_opens(&self) -> usize {
        self.ane_opens.load(Ordering::Relaxed)
    }

    /// Makes `open_gpu` fail, for the fatal-path test.
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
    /// When `watch_headroom` has not been called yet, or its monitor was
    /// dropped.
    pub fn push_headroom(&self, headroom: Headroom) {
        let _ = self.headroom.send_replace(headroom.clone());
        let on_change = self
            .on_change
            .get()
            .expect("watch_headroom was not called")
            .upgrade()
            .expect("the headroom monitor was dropped");
        on_change(headroom);
    }

    fn embedder(&self, instance: Instance) -> StubEmbedder {
        StubEmbedder {
            instance,
            records: self.records.clone(),
            delays: self.delays.clone(),
            gates: self.gates.get(&instance).cloned().unwrap_or_default(),
            calls: 0,
            index_gate: self.index_gates.get(&instance).cloned(),
            dim: self.dim,
            hashes: self.hashes.clone(),
        }
    }
}

impl Backend for StubBackend {
    type Ane = StubEmbedder;
    type Gpu = StubEmbedder;
    type Encoder = StubEncoder;
    type LoadError = StubError;
    type Monitor = StubMonitor;

    fn load_encoder(&self, _: &Path) -> EncoderLoad<Self::Encoder, Self::LoadError> {
        Ok((
            StubEncoder {
                hashes: self.hashes.clone(),
            },
            BUCKETS.into(),
        ))
    }

    fn open_ane(
        &self,
        _: &Path,
        _: &watch::Sender<LoadStage>,
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

    fn open_gpu(&self, _: &Path) -> Result<Self::Gpu, GpuOpen<Self::LoadError>> {
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
        let on_change = Arc::new(on_change);
        self.on_change
            .set(Arc::downgrade(&on_change))
            .expect("watch_headroom called once");
        let headroom = self.headroom.borrow().clone();
        self.push_headroom(headroom);
        StubMonitor {
            _on_change: on_change,
        }
    }
}

/// The stub embedder, for either instance. Every call reports a
/// [`Kind::Started`] record, waits out the gate installed for its 1-based
/// ordinal if any, sleeps its bucket's configured duration, then reports a
/// [`Record`] per input.
pub struct StubEmbedder {
    instance: Instance,
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    gates: BTreeMap<usize, Arc<Barrier>>,
    /// The embed calls this embedder served, for picking a call's gate.
    calls: usize,
    index_gate: Option<Arc<IndexGate>>,
    dim: usize,
    hashes: Hashes,
}

impl StubEmbedder {
    fn sleep(&self, tokens: usize) {
        let delay = self
            .delays
            .get(&(self.instance, bucket(tokens)))
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
        let _ = self
            .records
            .send(held_record(self.instance, Instant::now()));
        // `open` may have landed between the armed check and registering:
        // then this call passes without waiting.
        if gate.released.load(Ordering::Acquire) {
            return;
        }
        wait.recv().expect("open releases every held call");
    }

    /// Reports the call's start, before the gate and the sleep.
    fn started(&self, tokens: usize, batch_len: usize) {
        let now = Instant::now();
        self.record(Kind::Started, Vec::new(), tokens, batch_len, now, now);
    }

    fn kind(&self) -> Kind {
        match qos::current_thread() {
            QosClass::UserInteractive => Kind::Interactive,
            _ => Kind::Index,
        }
    }

    fn record(
        &self,
        kind: Kind,
        ids: Vec<u32>,
        tokens: usize,
        batch_len: usize,
        start: Instant,
        end: Instant,
    ) {
        let hash = self
            .hashes
            .lock()
            .expect("the hashes map")
            .get(&ids)
            .copied();
        let _ = self.records.send(Record {
            instance: self.instance,
            kind,
            ids,
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
        self.record(Kind::Dropped, Vec::new(), 0, 0, now, now);
    }
}

impl TokenEmbedder for StubEmbedder {
    type Error = StubError;

    fn embed(&mut self, ids: &[u32]) -> Result<UnitVector, Self::Error> {
        let kind = self.kind();
        self.started(ids.len(), 1);
        self.wait_gate();
        if kind == Kind::Index {
            self.hold_index_call();
        }
        let start = Instant::now();
        self.sleep(ids.len());
        let end = Instant::now();
        self.record(kind, ids.to_vec(), ids.len(), 1, start, end);
        Ok(bag_of_ids(ids, self.dim))
    }
}

impl TokenBatchEmbedder for StubEmbedder {
    fn embed_batch(&mut self, ids: &[Vec<u32>]) -> Result<Vec<UnitVector>, Self::Error> {
        let mut buckets = ids.iter().map(|input| bucket(input.len()));
        if let Some(first) = buckets.next() {
            assert!(buckets.all(|b| b == first), "a batch mixes token buckets");
        }
        for input in ids {
            self.started(input.len(), ids.len());
        }
        self.wait_gate();
        self.hold_index_call();
        let start = Instant::now();
        if let Some(first) = ids.first() {
            self.sleep(first.len());
        }
        let end = Instant::now();
        // The batch method is GPU-only work: the kind is the method's,
        // never interactive.
        for input in ids {
            self.record(
                Kind::Index,
                input.clone(),
                input.len(),
                ids.len(),
                start,
                end,
            );
        }
        Ok(ids
            .iter()
            .map(|input| bag_of_ids(input, self.dim))
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

/// A deterministic unit vector: FNV-1a over the ids, modulo `dim`.
pub fn bag_of_ids(ids: &[u32], dim: usize) -> UnitVector {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &id in ids {
        hash ^= u64::from(id);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut v = vec![0.0f32; dim];
    v[hash as usize % dim] = 1.0;
    UnitVector::normalize(v)
}
