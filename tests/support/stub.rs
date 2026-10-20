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

use repo_search::daemon::{Backend, GpuOpen, Headroom};
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
    pub pri: i32,
    /// When the call started.
    pub start: Instant,
    /// When the call ended.
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

/// A [`TokenCounter`] counting `len / 4 + 1` tokens.
#[derive(Clone, Default)]
pub struct StubCounter;

impl StubCounter {
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
        text.len() / 4 + 1
    }

    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool {
        self.document_tokens(title, &text.concat()) <= max
    }

    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize> {
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
        (title.len() + text.len()) / 4 + 1
    }
}

/// The test-side [`Backend`]. `open_gpu` counts its calls and is `Blocked`
/// while the pushed headroom says so; `watch_headroom` stores the callback
/// `push_headroom` drives.
pub struct StubBackend {
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    gpu_opens: AtomicUsize,
    gpu_open_fails: AtomicBool,
    headroom: tokio::sync::watch::Sender<Headroom>,
    /// Keeps the watch alive so `send_replace` stores; never read.
    _headroom_rx: tokio::sync::watch::Receiver<Headroom>,
    on_change: OnceLock<OnChange>,
    gates: BTreeMap<Instance, Arc<Barrier>>,
    dim: usize,
}

impl StubBackend {
    /// A stub whose embed calls return at once; `set_delay` adds sleeps.
    pub fn new(records: mpsc::Sender<Record>) -> Self {
        let (headroom, _headroom_rx) =
            tokio::sync::watch::channel(Err(GpuBlocked::PowerSourceUnknown));
        Self {
            records,
            delays: BTreeMap::new(),
            gpu_opens: AtomicUsize::new(0),
            gpu_open_fails: AtomicBool::new(false),
            headroom,
            _headroom_rx,
            on_change: OnceLock::new(),
            gates: BTreeMap::new(),
            dim: 64,
        }
    }

    /// The sleep for calls of `instance` in every bucket.
    pub fn set_instance_delay(&mut self, instance: Instance, delay: Duration) {
        for bucket in BUCKETS {
            self.delays.insert((instance, bucket), delay);
        }
    }

    /// Gates `instance`'s first embed call on a barrier the test opens.
    pub fn gate_first(&mut self, instance: Instance) -> Arc<Barrier> {
        let gate = Arc::new(Barrier::new(2));
        self.gates.insert(instance, gate.clone());
        gate
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
    /// When `watch_headroom` has not been called yet.
    pub fn push_headroom(&self, headroom: Headroom) {
        let _ = self.headroom.send_replace(headroom.clone());
        self.on_change.get().expect("watch_headroom was not called")(headroom);
    }

    fn embedder(&self, instance: Instance) -> StubEmbedder {
        StubEmbedder {
            instance,
            counter: StubCounter,
            records: self.records.clone(),
            delays: self.delays.clone(),
            gate: self.gates.get(&instance).cloned(),
            gated: false,
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
        Ok((StubCounter, BUCKETS.into()))
    }

    fn open_ane(&self, _: &Path, _: &str) -> Result<Self::Ane, Self::LoadError> {
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
/// [`Kind::Started`] record, waits out its gate on the first call, sleeps its
/// bucket's configured duration, then reports a [`Record`] per document.
pub struct StubEmbedder {
    instance: Instance,
    counter: StubCounter,
    records: mpsc::Sender<Record>,
    delays: BTreeMap<(Instance, usize), Duration>,
    gate: Option<Arc<Barrier>>,
    /// The first call's gate was already waited on.
    gated: bool,
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

    /// Blocks the first embed call until the test opens the gate.
    fn wait_gate(&mut self) {
        if self.gated {
            return;
        }
        self.gated = true;
        if let Some(gate) = &self.gate {
            gate.wait();
        }
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
        if self.instance == Instance::Gpu {
            let now = Instant::now();
            self.record(Kind::Dropped, None, 0, 0, now, now);
        }
    }
}

impl Embedder for StubEmbedder {
    type Error = StubError;

    fn model_id(&self) -> &str {
        "stub-model"
    }

    fn embed_document(&mut self, title: &str, text: &str) -> Result<UnitVector, Self::Error> {
        let tokens = self.counter.document_tokens(title, text);
        self.started(Some(ContentHash::of_chunk(title, text)), tokens, 1);
        self.wait_gate();
        let start = Instant::now();
        self.sleep(tokens);
        let end = Instant::now();
        self.record(
            self.kind(),
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
        let start = Instant::now();
        if let Some(&first) = tokens.first() {
            self.sleep(first);
        }
        let end = Instant::now();
        let kind = self.kind();
        for ((title, text), tokens) in docs.iter().zip(tokens) {
            self.record(
                kind,
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
    if v.iter().all(|x| *x == 0.0) {
        v[0] = 1.0;
    }
    UnitVector::normalize(v)
}
