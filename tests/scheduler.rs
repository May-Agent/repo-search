//! `Enqueue`s, `QueryJob`s and `DocumentsJob`s are fed to the scheduler
//! directly, with no daemon and no sockets. The stub's `Record` channel
//! carries what the embedding threads did; waits use `recv_timeout` on it.

#![cfg(target_os = "macos")]

#[path = "support/stub.rs"]
mod stub;

use std::collections::{BTreeMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use repo_search::daemon::chip::ChipClass;
use repo_search::daemon::scheduler::{
    Counts, DocumentsJob, EmbedError, Enqueue, IndexClass, Interest, PendingItem, QueryJob,
    SchedulerHandle, Stage1Done, WorktreeId,
};
use repo_search::daemon::{Backend, Error, ane, scheduler, writer};
use repo_search_core::QueryTask;
use repo_search_core::UnitVector;
use repo_search_core::qos::{self, QosClass};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::store::{ContentHash, VectorCache};
use repo_search_coreml::GpuBlocked;
use stub::{Instance, Kind, Record, StubBackend, StubMonitor};
use tokio::sync::oneshot;

const TIMEOUT: Duration = Duration::from_secs(60);

const WORKTREE: WorktreeId = WorktreeId(1);
const REPO: scheduler::RepoId = scheduler::RepoId(1);

/// The test rig: scheduler, ANE thread, one repo writer and the record log.
struct Rig {
    sched: SchedulerHandle,
    backend: Arc<StubBackend>,
    records: mpsc::Receiver<Record>,
    writer: writer::WriterHandle,
    /// The repository's vector cache, read back after a flush.
    cache: VectorCache,
    fatal: tokio::sync::mpsc::Receiver<Error>,
    gates: BTreeMap<(Instance, usize), Arc<Barrier>>,
    /// The ANE thread, for reading its effective priority.
    ane_thread: qos::ThreadId,
    _monitor: StubMonitor,
    _tmp: tempfile::TempDir,
    _ane: tokio::sync::watch::Receiver<bool>,
}

impl Rig {
    /// The barrier installed for `instance`'s first call.
    fn gate(&self, instance: Instance) -> &Barrier {
        self.gate_call(instance, 1)
    }

    /// The barrier `gate_call` installed for `instance`'s `call`-th call.
    fn gate_call(&self, instance: Instance, call: usize) -> &Barrier {
        &self.gates[&(instance, call)]
    }
}

async fn rig(chip: ChipClass, ane_delay: Duration, gpu_delay: Duration, gated: &[Instance]) -> Rig {
    rig_gated(
        chip,
        ane_delay,
        gpu_delay,
        &gated
            .iter()
            .map(|&instance| (instance, 1))
            .collect::<Vec<_>>(),
    )
    .await
}

/// `rig` with a barrier on chosen calls: each `(instance, call)` gates
/// `instance`'s `call`-th embed call (1-based over its `embed_*` calls).
async fn rig_gated(
    chip: ChipClass,
    ane_delay: Duration,
    gpu_delay: Duration,
    gated: &[(Instance, usize)],
) -> Rig {
    let (record_tx, records) = mpsc::channel();
    let mut backend = StubBackend::new(record_tx);
    backend.set_instance_delay(Instance::Ane, ane_delay);
    backend.set_instance_delay(Instance::Gpu, gpu_delay);
    let gates: BTreeMap<(Instance, usize), Arc<Barrier>> = gated
        .iter()
        .map(|&(instance, call)| ((instance, call), backend.gate_call(instance, call)))
        .collect();
    let backend = Arc::new(backend);

    let tmp = tempfile::tempdir().expect("tempdir");
    let git = tmp.path().join("git");
    std::fs::create_dir_all(&git).expect("git dir");

    let (headroom_tx, headroom_rx) =
        tokio::sync::watch::channel(Err(GpuBlocked::PowerSourceUnknown));
    let monitor = backend.watch_headroom(Box::new(move |h| {
        let _ = headroom_tx.send_replace(h);
    }));

    let (fatal_tx, fatal) = tokio::sync::mpsc::channel(8);
    let (busy_tx, _busy_rx) = tokio::sync::watch::channel(false);
    let ane::AneThread {
        counted,
        sched: ane_sched,
        stage: _,
        loaded,
        exited,
    } = ane::spawn(
        backend.clone(),
        tmp.path().to_path_buf(),
        "stub-model".to_owned(),
    );
    let (_, buckets) = counted
        .await
        .expect("ane thread reports")
        .expect("stub load_counter does not fail");
    let (sched, _sched_exited) = scheduler::spawn(scheduler::SchedulerConfig {
        backend: backend.clone(),
        dir: tmp.path().to_path_buf(),
        model_id: "stub-model".to_owned(),
        buckets,
        headroom: headroom_rx,
        chip,
        fatal: fatal_tx.clone(),
        busy: busy_tx.clone(),
        ane_exited: exited.clone(),
    });
    ane_sched.send(sched.tx()).expect("scheduler sender");
    let ane_thread = loaded
        .await
        .expect("ane thread reports")
        .expect("stub open_ane does not fail");

    let cache = VectorCache::open(&git).await.expect("vector cache opens");
    let (writer_busy_tx, _writer_busy_rx) = tokio::sync::watch::channel(false);
    let (writer, _writer_task) = writer::spawn(
        cache.clone(),
        "stub-model".to_owned(),
        fatal_tx,
        writer_busy_tx,
    );

    Rig {
        sched,
        backend,
        records,
        writer,
        cache,
        fatal,
        gates,
        ane_thread,
        _monitor: monitor,
        _tmp: tmp,
        _ane: exited,
    }
}

/// A pending chunk whose `StubCounter` token count is exactly `tokens`.
fn pending(seq: u64, tokens: u32) -> scheduler::PendingItem {
    let title = format!("t{seq}");
    let width = 4 * (tokens as usize - 1) - title.len();
    let text = format!("{seq:0>width$}");
    PendingItem {
        chunk: PendingChunk {
            path: format!("src/f{seq}.rs"),
            content_hash: ContentHash::of_chunk(&title, &text),
            title,
            text,
            lines: 1..=1,
        },
        tokens,
    }
}

/// Sends `items` to the scheduler for `worktree` as `class`, replacing its
/// interest. The caller holds a barrier of `REPO` open around it.
async fn send_replace(
    rig: &Rig,
    worktree: WorktreeId,
    items: Vec<scheduler::PendingItem>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    let (done_tx, done_rx) = tokio::sync::mpsc::channel(8);
    rig.sched
        .enqueue(Enqueue {
            worktree,
            repo: REPO,
            class,
            interest: Interest::Replace(items),
            writer: rig.writer.clone(),
            stage1_done: done_tx,
        })
        .await
        .expect("scheduler is up");
    done_rx
}

/// Sends `items` to the scheduler for `worktree` as `class`, replacing its
/// interest inside a barrier of its own, as the feeder sends it.
async fn enqueue_for(
    rig: &Rig,
    worktree: WorktreeId,
    items: Vec<scheduler::PendingItem>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    rig.sched.barrier(REPO, true).await;
    let done = send_replace(rig, worktree, items, class).await;
    rig.sched.barrier(REPO, false).await;
    done
}

/// Sends `items` to the scheduler as `class`, replacing its interest.
async fn enqueue(
    rig: &Rig,
    items: Vec<scheduler::PendingItem>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    enqueue_for(rig, WORKTREE, items, class).await
}

/// Sends one pass's changes for `worktree` as `class`. The caller holds a
/// barrier of `REPO` open around it.
async fn send_change(
    rig: &Rig,
    worktree: WorktreeId,
    added: Vec<scheduler::PendingItem>,
    dropped: Vec<ContentHash>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    let (done_tx, done_rx) = tokio::sync::mpsc::channel(8);
    rig.sched
        .enqueue(Enqueue {
            worktree,
            repo: REPO,
            class,
            interest: Interest::Change { added, dropped },
            writer: rig.writer.clone(),
            stage1_done: done_tx,
        })
        .await
        .expect("scheduler is up");
    done_rx
}

/// Sends one pass's changes for `worktree` as `class`, inside a barrier of
/// its own, as the feeder sends it.
async fn change_for(
    rig: &Rig,
    worktree: WorktreeId,
    added: Vec<scheduler::PendingItem>,
    dropped: Vec<ContentHash>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    rig.sched.barrier(REPO, true).await;
    let done = send_change(rig, worktree, added, dropped, class).await;
    rig.sched.barrier(REPO, false).await;
    done
}

/// The expected `Counts` for the scheduler-level assertions.
fn counts(stage1: u64, background: u64, in_flight: u64, stage1_done: bool) -> Counts {
    Counts {
        queued_stage1: stage1,
        queued_background: background,
        in_flight,
        stage1_done,
    }
}

/// `worktree`'s counts, answered by the scheduler.
async fn counts_of(rig: &Rig, worktree: WorktreeId) -> Counts {
    rig.sched
        .counts_for(worktree)
        .await
        .expect("counts answered")
}

/// Collects records until `n` `Started` records are in `log`.
fn wait_started(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>, n: usize) {
    while log.iter().filter(|r| r.kind == Kind::Started).count() < n {
        recv(log, records);
    }
}

/// Receives the next record within `TIMEOUT`, off the async executor, and
/// appends it to `log`.
fn recv(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) -> Record {
    tokio::task::block_in_place(|| recv_blocking(log, records))
}

fn recv_blocking(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) -> Record {
    let record = records
        .recv_timeout(TIMEOUT)
        .expect("a record must arrive within the timeout");
    log.push(Record {
        instance: record.instance,
        kind: record.kind,
        hash: record.hash,
        tokens: record.tokens,
        batch_len: record.batch_len,
        qos: record.qos,
        pri: record.pri,
        start: record.start,
        end: record.end,
    });
    record
}

/// Collects records until `pred` holds for one, off the async executor;
/// returns its index in `log`.
fn wait_for(
    log: &mut Vec<Record>,
    records: &mpsc::Receiver<Record>,
    pred: impl Fn(&Record) -> bool,
) -> usize {
    tokio::task::block_in_place(|| {
        loop {
            if let Some(i) = log.iter().position(&pred) {
                return i;
            }
            recv_blocking(log, records);
        }
    })
}

/// Opens a `gate_first` barrier, off the async executor.
fn open(gate: &Barrier) {
    tokio::task::block_in_place(|| gate.wait());
}

/// Sends a `Query` job for `WORKTREE`; the receiver resolves with its vector
/// or its error.
async fn send_query(rig: &Rig, text: &str) -> oneshot::Receiver<Result<UnitVector, EmbedError>> {
    let (reply, rx) = oneshot::channel();
    rig.sched
        .query(QueryJob {
            worktree: WORKTREE,
            task: QueryTask::CodeRetrieval,
            text: text.to_owned(),
            enqueued: Instant::now(),
            stamp: 0,
            reply,
        })
        .await
        .expect("scheduler is up");
    rx
}

/// Sends a `Query` job for `WORKTREE` and awaits its vector.
async fn query(rig: &Rig, text: &str) {
    send_query(rig, text)
        .await
        .await
        .expect("query answered")
        .expect("embeds fine");
}

/// Sends a `Documents` job of `chunks` for `WORKTREE`; the receiver
/// resolves once every chunk's vector is with the rig's writer. The caller
/// holds a search window of `REPO` open around it.
async fn send_spot(
    rig: &Rig,
    chunks: Vec<PendingChunk>,
) -> oneshot::Receiver<Result<(), EmbedError>> {
    let (reply, rx) = oneshot::channel();
    rig.sched
        .documents(DocumentsJob {
            worktree: WORKTREE,
            repo: REPO,
            chunks,
            writer: rig.writer.clone(),
            enqueued: Instant::now(),
            reply,
        })
        .await
        .expect("scheduler is up");
    rx
}

/// Flushes the rig's writer and asserts the cache holds a vector for each
/// of `hashes`: what the retriever reads back after a `Documents` job.
async fn assert_cached(rig: &Rig, hashes: &[ContentHash]) {
    rig.writer.flush().await.expect("the writer commits");
    let cached = rig
        .cache
        .get_many("stub-model", hashes)
        .await
        .expect("the cache reads");
    assert_eq!(cached.len(), hashes.len(), "every chunk has a vector");
}

/// Drains the record channel into `log`; every embedding of `hash` in
/// `log`, by instance and kind, in order.
fn embeds_of(log: &mut Vec<Record>, rig: &Rig, hash: ContentHash) -> Vec<(Instance, Kind)> {
    log.extend(rig.records.try_iter());
    log.iter()
        .filter(|r| matches!(r.kind, Kind::Index | Kind::QueryDocument) && r.hash == Some(hash))
        .map(|r| (r.instance, r.kind))
        .collect()
}

/// Sends a `Documents` job of `n` chunks for `WORKTREE` inside a search
/// window of its own, awaits its answer, checks every vector is cached and
/// returns the chunks' content hashes.
async fn query_documents(rig: &Rig, first_seq: u64, n: usize) -> HashSet<ContentHash> {
    let chunks: Vec<PendingChunk> = (first_seq..first_seq + n as u64)
        .map(|seq| pending(seq, 100).chunk)
        .collect();
    let hashes: Vec<ContentHash> = chunks.iter().map(|chunk| chunk.content_hash).collect();
    rig.sched.barrier(REPO, true).await;
    send_spot(rig, chunks)
        .await
        .await
        .expect("documents answered")
        .expect("embeds fine");
    rig.sched.barrier(REPO, false).await;
    assert_cached(rig, &hashes).await;
    hashes.into_iter().collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_preempts_indexing() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(20),
        Duration::ZERO,
        &[],
    )
    .await;
    // Battery keeps the GPU out of this test entirely.
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let _done = enqueue(
        &rig,
        (0..600).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    // Mid stage 1: the ANE is embedding.
    wait_for(&mut log, &rig.records, |r| r.kind == Kind::Index);

    let rx = send_query(&rig, "find the scheduler").await;
    let sent = Instant::now();
    rx.await.expect("query answered").expect("embeds fine");
    let doc_hashes = query_documents(&rig, 1_000, 30).await;

    // The window between the send and the query's first ANE record holds at
    // most one started `Index` record: the chunk already in flight.
    let first = wait_for(&mut log, &rig.records, |r| r.kind == Kind::Query);
    let first_start = log[first].start;
    let started = log
        .iter()
        .filter(|r| r.kind == Kind::Index && r.start >= sent && r.start < first_start)
        .count();
    assert!(
        started <= 1,
        "{started} index records started inside the query's window"
    );

    let stats = rig
        .sched
        .take_stats(WORKTREE)
        .await
        .expect("stats answered");
    assert_eq!(stats.overtaken, 0, "no indexing item overtakes a query");

    // The Documents job embedded each chunk on the ANE at user-interactive.
    let is_doc = |r: &&Record| {
        r.kind == Kind::QueryDocument && r.hash.is_some_and(|h| doc_hashes.contains(&h))
    };
    while log.iter().filter(|r| is_doc(r)).count() < 30 {
        recv(&mut log, &rig.records);
    }
    let docs: Vec<&Record> = log.iter().filter(|r| is_doc(r)).collect();
    assert_eq!(docs.len(), 30);
    assert!(
        docs.iter()
            .all(|r| r.instance == Instance::Ane && r.qos == QosClass::UserInteractive),
        "query document work is ANE-only at user-interactive QoS"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_takes_the_long_end_in_same_bucket_batches() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(20),
        Duration::from_millis(10),
        &[],
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut done = enqueue(
        &rig,
        (0..326)
            .map(|seq| {
                let tokens = match seq {
                    0..=5 => 500,
                    6..=35 => 200,
                    _ => 100,
                };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    // A query mid-stage-1 must never reach the GPU.
    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Index
    });
    query(&rig, "find the scheduler").await;

    // Collect until the GPU drops (stage 1 over) or it has no more batches.
    while !log.iter().any(|r| r.kind == Kind::Dropped) {
        recv(&mut log, &rig.records);
    }

    let gpu: Vec<&Record> = log.iter().filter(|r| r.instance == Instance::Gpu).collect();
    assert!(gpu.iter().any(|r| r.kind == Kind::Index));
    for record in &gpu {
        if record.kind == Kind::Dropped {
            continue;
        }
        assert!(
            matches!(record.kind, Kind::Index | Kind::Started),
            "the GPU only embeds index work"
        );
        assert!(record.batch_len <= 8, "batches are at most 8 documents");
    }

    // One bucket per batch: group by the shared start instant.
    let mut batches: Vec<Vec<&Record>> = Vec::new();
    for record in gpu.iter().filter(|r| r.kind == Kind::Index) {
        match batches.last_mut() {
            Some(batch) if batch[0].start == record.start => batch.push(record),
            _ => batches.push(vec![record]),
        }
    }
    for batch in &batches {
        let bucket = |t: usize| stub::BUCKETS.iter().find(|&&b| t <= b).unwrap();
        assert!(
            batch
                .iter()
                .all(|r| bucket(r.tokens) == bucket(batch[0].tokens)),
            "a GPU batch mixes token buckets"
        );
    }
    let first = &batches[0];
    assert!(
        first.iter().all(|r| r.tokens == 500),
        "the first GPU batch holds the longest items"
    );

    // The ANE walks the short end upward.
    let ane: Vec<usize> = log
        .iter()
        .filter(|r| r.instance == Instance::Ane && r.kind == Kind::Index)
        .map(|r| r.tokens)
        .collect();
    assert!(
        ane.windows(2).all(|w| w[0] <= w[1]),
        "ANE stage-1 tokens are non-decreasing"
    );

    assert!(
        !gpu.iter()
            .any(|r| matches!(r.kind, Kind::Query | Kind::QueryDocument)),
        "no query record ever comes from the GPU"
    );
    let _ = done.recv().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_base_chip_keeps_the_gpu_off_stage_1() {
    // Every item starts stage-1 classed; the 30 small ones count for
    // completion. On a base chip the GPU opens only once the stage-1 queue
    // has drained and the >256-token leftovers are background work.
    let rig = rig(
        ChipClass::Base,
        Duration::from_millis(10),
        Duration::ZERO,
        &[],
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut done = enqueue(
        &rig,
        (0..100)
            .map(|seq| {
                let tokens = if seq < 70 { 500 } else { 100 };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    // The leftovers are background work now: the GPU batches them until
    // the queue drains and its pull is answered `Stop`.
    let dropped = wait_for(&mut log, &rig.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert_eq!(rig.backend.gpu_opens(), 1, "the GPU opened once");

    let gpu: Vec<&Record> = log.iter().filter(|r| r.instance == Instance::Gpu).collect();
    assert!(
        gpu.iter().any(|r| r.kind == Kind::Index),
        "the GPU takes background work once stage 1 is done"
    );
    assert!(
        gpu.iter()
            .filter(|r| r.kind == Kind::Index)
            .all(|r| r.tokens == 500),
        "the GPU took only stage-1 leftovers"
    );

    // By order, not by the clock: every stage-1 record precedes the GPU's
    // first, and each small item was embedded on the ANE exactly once.
    let last_stage1 = log
        .iter()
        .rposition(|r| r.kind == Kind::Index && r.tokens == 100)
        .expect("the stage-1 items embedded");
    let first_gpu = log
        .iter()
        .position(|r| r.instance == Instance::Gpu)
        .expect("the GPU ran");
    assert!(
        first_gpu > last_stage1,
        "a GPU record precedes the last stage-1 item"
    );
    let embedded = index_hashes(&mut log, &rig);
    assert_each_once(&embedded);
    for seq in 70..100 {
        let hash = pending(seq, 100).chunk.content_hash;
        let embeds: Vec<&Record> = log
            .iter()
            .filter(|r| r.kind == Kind::Index && r.hash == Some(hash))
            .collect();
        assert_eq!(
            embeds.len(),
            1,
            "a stage-1 item embedded {} times",
            embeds.len()
        );
        assert_eq!(
            embeds[0].instance,
            Instance::Ane,
            "a stage-1 item reached the GPU"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_gpu_on_battery() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(5),
        Duration::from_millis(5),
        &[],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut done = enqueue(
        &rig,
        (0..100).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(
        rig.backend.gpu_opens(),
        0,
        "the GPU is never opened on battery"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_released_on_power_loss() {
    // Both first calls are gated: a GPU batch and an ANE chunk are provably
    // in flight, so the power loss — not an empty stage 1 — drops the GPU.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(25),
        &[Instance::Ane, Instance::Gpu],
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut done = enqueue(
        &rig,
        (0..124)
            .map(|seq| {
                let tokens = if seq < 24 { 500 } else { 100 };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    // Stage 1 holds 124 - 8 - 1 = 115 items with both calls in flight.
    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));

    // The batch finishes and the next pull is answered `Stop`.
    open(rig.gate(Instance::Gpu));
    let dropped = wait_for(&mut log, &rig.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert_eq!(
        log[..dropped]
            .iter()
            .filter(|r| r.instance == Instance::Gpu && r.kind == Kind::Index)
            .count(),
        8,
        "the GPU embedded exactly its gated batch"
    );

    // Indexing continues on the ANE through stage-1 completion. The 16 long
    // items the GPU left behind are stage-2 Background work, which the test
    // does not wait for: only the 100 short items are stage 1, and their
    // records precede `Stage1Done`.
    open(rig.gate(Instance::Ane));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes on the ANE")
        .expect("stage-1 channel open");
    while log
        .iter()
        .filter(|r| r.instance == Instance::Ane && r.kind == Kind::Index && r.tokens == 100)
        .count()
        < 100
    {
        recv(&mut log, &rig.records);
    }
    assert!(
        !log[dropped + 1..]
            .iter()
            .any(|r| r.instance == Instance::Gpu),
        "no GPU record starts after the embedder was dropped"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stage_2_runs_at_background_qos() {
    // The GPU's first call is gated: it provably takes exactly one batch (8
    // of the 70 long items) and its next pull finds stage 1 empty, leaving
    // 62 long items for the ANE however loaded the host is.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Gpu],
    )
    .await;
    // Those leftovers are Background work, which a loaded host can starve:
    // lift the thread's effective priority for the run — its requested
    // class stays Background.
    let _lift = stub::lift_ane(&rig.backend, QosClass::UserInitiated);
    rig.backend.push_headroom(Ok(()));
    let mut done = enqueue(
        &rig,
        (0..100)
            .map(|seq| {
                let tokens = if seq < 70 { 500 } else { 100 };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    // The GPU holds its batch on the gate while the ANE works the 30 small
    // items through stage-1 completion.
    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    open(rig.gate(Instance::Gpu));

    let dropped = wait_for(&mut log, &rig.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert_eq!(
        log[..dropped]
            .iter()
            .filter(|r| r.instance == Instance::Gpu && r.kind == Kind::Index)
            .count(),
        8,
        "the GPU took exactly one batch"
    );
    assert!(
        !log[dropped + 1..]
            .iter()
            .any(|r| r.instance == Instance::Gpu),
        "no GPU record follows the drop"
    );

    // The chunks stage 1 left — every >256-token item — now embed on the ANE
    // at background QoS.
    let first_long = wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index && r.tokens > 256
    });
    assert_eq!(
        log[first_long].qos,
        QosClass::Background,
        "stage-2 indexing runs at background QoS"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_lifts_an_in_flight_background_chunk() {
    // The ANE's first call is gated, so a background chunk is provably in
    // flight when the query lands. The test's own Utility override gives
    // the chunk CPU on a loaded host and marks this host's lifted band.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(50),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    let lift = stub::lift_ane(&rig.backend, QosClass::Utility);
    let _done = enqueue(
        &rig,
        (0..10).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Background,
    )
    .await;

    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });

    let band = qos::scheduling_priority(rig.ane_thread);
    let rx = send_query(&rig, "boost the chunk ahead").await;
    // The inbox is FIFO: once `counts_for` is answered, the Query ahead of it
    // was handled and the scheduler's override on the ANE thread is in place.
    rig.sched
        .counts_for(WORKTREE)
        .await
        .expect("counts answered");
    // `pth_curpri` sits at the same lifted band under any override while
    // the thread waits, so the lift cannot be told apart by reading up:
    // end the test's override instead — what still holds the band is
    // provably the scheduler's UserInteractive override.
    drop(lift);
    assert_eq!(
        qos::scheduling_priority(rig.ane_thread),
        band,
        "the scheduler's override still holds the chunk at the lifted band"
    );
    let _lift = stub::lift_ane(&rig.backend, QosClass::Utility);
    open(rig.gate(Instance::Ane));
    rx.await.expect("query answered").expect("embeds fine");

    // The gated chunk and the next one ran at the same requested class.
    let lifted = wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index
    });
    assert_eq!(log[lifted].qos, QosClass::Background);
    let lifted_end = log[lifted].end;
    let released = wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index && r.start > lifted_end
    });
    assert_eq!(log[released].qos, QosClass::Background);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_after_a_fatal_error_is_answered_and_the_scheduler_exits() {
    // The ANE's first call is gated so it cannot pull its `Stop`: the actor
    // is stopping but cannot exit, and it must still answer the query.
    let mut rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    rig.backend.fail_gpu_open();
    let _done = enqueue(
        &rig,
        (0..20).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });
    // The open failure is fatal; `run` is told over the fatal channel.
    let error = match rig.fatal.recv().await.expect("the fatal channel is open") {
        Error::Embedder(message) => message,
        other => panic!("expected an embedder failure, got {other}"),
    };

    // A query that arrives while the scheduler is stopping is answered with
    // the fatal error, not left hanging.
    let rx = send_query(&rig, "too late").await;
    let message = match rx.await.expect("the query is answered") {
        Err(EmbedError::Model(message)) => message,
        Err(EmbedError::SchedulerGone) => panic!("expected the fatal error, got SchedulerGone"),
        Err(EmbedError::Writer(error)) => {
            panic!("expected the fatal error, got a writer failure: {error}")
        }
        Ok(_) => panic!("expected a model error, got a vector"),
    };
    assert_eq!(message, error);

    // Releasing the gate lets the ANE pull its `Stop`; the actor exits and
    // its inbox closes.
    open(rig.gate(Instance::Ane));
    tokio::time::timeout(TIMEOUT, rig.sched.tx().closed())
        .await
        .expect("the scheduler task ends");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_track_an_ane_pull_and_done() {
    // The ANE's first call — a query — is gated, so the enqueue's counts are
    // exact before any index item is dispatched. Every index call is gated
    // too: the item its `Started` announces is provably in flight while the
    // counts are read.
    let rig = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[
            (Instance::Ane, 1),
            (Instance::Ane, 2),
            (Instance::Ane, 3),
            (Instance::Ane, 4),
        ],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..3).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(3, 0, 0, false));

    // Item N's `Started` proves item N-1's `AneDone` was handled: each pull
    // moves one queued item to in_flight and each done removes it.
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    wait_started(&mut log, &rig.records, 2);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 1, false));
    open(rig.gate_call(Instance::Ane, 2));
    wait_started(&mut log, &rig.records, 3);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 1, false));
    open(rig.gate_call(Instance::Ane, 3));
    wait_started(&mut log, &rig.records, 4);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    open(rig.gate_call(Instance::Ane, 4));

    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_track_a_gpu_batch_pull_and_done() {
    // Both first calls are gated: a provably in-flight ANE chunk and GPU
    // batch make the counts exact.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(10),
        &[Instance::Ane, Instance::Gpu],
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut done = enqueue(
        &rig,
        (0..9)
            .map(|seq| {
                let tokens = if seq < 8 { 500 } else { 100 };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    // The GPU took its batch of eight 500-token items; the ANE holds the
    // one small item. All nine are in flight.
    let mut log = Vec::new();
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 9, false));

    // `GpuDone` removes the batch; the pull after it finds stage 1 empty and
    // the embedder drops.
    open(rig.gate(Instance::Gpu));
    let dropped = wait_for(&mut log, &rig.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    open(rig.gate(Instance::Ane));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_drop_a_re_enqueued_in_flight_item() {
    // The ANE's first call is gated, so a chunk is provably in flight when
    // the re-`Enqueue` drops it: the tally falls at once and the late
    // `AneDone` changes nothing for this worktree. The second call is gated
    // as well, holding its item in flight while its counts are read.
    let rig = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[(Instance::Ane, 1), (Instance::Ane, 2)],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let _old_done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 1, false));

    // The replacement lists only item 2: item 0's in-flight count and item
    // 1's queued count drop.
    let mut done = enqueue_for(&rig, WORKTREE, vec![pending(2, 100)], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    // The gated item's `AneDone` arrives after the drop; the next `Started`
    // proves it changed nothing for this worktree.
    open(rig.gate(Instance::Ane));
    wait_started(&mut log, &rig.records, 2);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    open(rig.gate_call(Instance::Ane, 2));

    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_are_per_worktree_on_a_shared_item() {
    const OTHER: WorktreeId = WorktreeId(2);
    // A gated query holds the ANE's first call, so both enqueues are counted
    // before any dispatch; the second call's gate keeps the shared item in
    // flight while the worktrees' counts are read.
    let rig = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[(Instance::Ane, 1), (Instance::Ane, 2)],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let _other_done = enqueue_for(
        &rig,
        OTHER,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    // One `(repo, hash)` item, two interested worktrees: both count it.
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(2, 0, 0, false));

    // The other worktree's re-`Enqueue` drops the shared items; only its
    // tally moves.
    let mut other_done = enqueue_for(&rig, OTHER, vec![pending(2, 100)], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(1, 0, 0, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));

    // The shared item goes in flight for the worktree that kept it.
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    wait_started(&mut log, &rig.records, 2);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 1, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(1, 0, 0, false));
    open(rig.gate_call(Instance::Ane, 2));

    for done_rx in [&mut done, &mut other_done] {
        tokio::time::timeout(TIMEOUT, done_rx.recv())
            .await
            .expect("stage 1 completes")
            .expect("stage-1 channel open");
    }
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_move_to_background_at_stage1_completion() {
    // `AneDone` is handled before the pull that follows it on the wire, so
    // the next `Started` already shows the post-completion queues. Both
    // calls are gated: each counts read sees a provably in-flight item.
    let rig = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[(Instance::Ane, 1), (Instance::Ane, 2)],
    )
    .await;
    // The completed items embed at background QoS, which a loaded host can
    // starve: lift the thread's effective priority for the run — its
    // requested class stays Background.
    let _lift = stub::lift_ane(&rig.backend, QosClass::UserInitiated);
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut done = enqueue(
        &rig,
        (0..3)
            .map(|seq| {
                let tokens = if seq == 0 { 100 } else { 500 };
                pending(seq, tokens)
            })
            .collect(),
        IndexClass::Stage1,
    )
    .await;

    // The ANE takes the short item first; the two large ones wait queued.
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 1, false));

    // Completion moves the large items to background, and the next pull
    // already draws from there: the gate holds it in flight while the
    // counts are read.
    open(rig.gate(Instance::Ane));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    wait_started(&mut log, &rig.records, 2);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 1, 1, true));
    open(rig.gate_call(Instance::Ane, 2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_documents_job_takes_its_queued_twin_out_of_the_queue() {
    // The ANE's first call is gated, so the twin is provably still queued
    // when the `Documents` job lands.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..3).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;

    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(3, 0, 0, false));

    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![pending(1, 100).chunk]).await;
    // The twin is held and counted in flight, so the queue lost it.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 1, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    docs.await
        .expect("documents answered")
        .expect("embeds fine");
    rig.sched.barrier(REPO, false).await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    let twin = pending(1, 100).chunk.content_hash;
    let embedded = index_hashes(&mut log, &rig);
    assert!(!embedded.contains(&twin), "no Index record for the twin");
    assert_eq!(
        log.iter()
            .filter(|r| r.kind == Kind::QueryDocument && r.hash == Some(twin))
            .count(),
        1,
        "one QueryDocument record for the twin"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_counts_seeds_and_pushes_counts() {
    // Both ANE calls are gated, so every pushed count is one the scheduler
    // provably holds.
    let rig = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[(Instance::Ane, 1), (Instance::Ane, 2)],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let _old_done = enqueue(&rig, vec![pending(0, 100)], IndexClass::Stage1).await;

    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    let mut watch = rig
        .sched
        .watch_counts(WORKTREE)
        .await
        .expect("watch answered");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 1, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    // A re-`Enqueue` pushes the drop of the in-flight item.
    let mut done = enqueue_for(&rig, WORKTREE, vec![pending(1, 100)], IndexClass::Stage1).await;
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(1, 0, 0, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    // The late `AneDone` and the next dispatch push the same value
    // `counts_for` answers.
    open(rig.gate(Instance::Ane));
    wait_started(&mut log, &rig.records, 2);
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 1, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());
    open(rig.gate_call(Instance::Ane, 2));

    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 0, true));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());
}

/// Collects every `Kind::Index` record hash the run produced; all records
/// precede `Stage1Done`, which follows the writer flush, so `try_iter`
/// sees them all once stage 1 is done.
fn index_records(rig: &Rig) -> Vec<ContentHash> {
    rig.records
        .try_iter()
        .filter(|r| r.kind == Kind::Index)
        .map(|r| r.hash.expect("an Index record carries a hash"))
        .collect()
}

/// Drains the record channel into `log`; every `Kind::Index` hash in
/// `log`, in order.
fn index_hashes(log: &mut Vec<Record>, rig: &Rig) -> Vec<ContentHash> {
    log.extend(rig.records.try_iter());
    log.iter()
        .filter(|r| r.kind == Kind::Index)
        .map(|r| r.hash.expect("an Index record carries a hash"))
        .collect()
}

/// Asserts that no hash repeats in `hashes`.
fn assert_each_once(hashes: &[ContentHash]) {
    let distinct: HashSet<&ContentHash> = hashes.iter().collect();
    assert_eq!(
        distinct.len(),
        hashes.len(),
        "a hash was embedded twice: {hashes:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_adds_and_drops_on_top_of_a_replace() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..3).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(3, 0, 0, false));

    let dropped = pending(1, 100).chunk.content_hash;
    let _change = change_for(
        &rig,
        WORKTREE,
        vec![pending(3, 100)],
        vec![dropped],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(3, 0, 0, false));

    let _empty = change_for(&rig, WORKTREE, vec![], vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(3, 0, 0, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let embedded = index_records(&rig);
    let want: HashSet<ContentHash> = [0, 2, 3]
        .into_iter()
        .map(|seq| pending(seq, 100).chunk.content_hash)
        .collect();
    assert_eq!(embedded.len(), 3, "{embedded:?}");
    assert!(embedded.iter().all(|h| want.contains(h)));
    assert!(!embedded.contains(&dropped));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_re_adding_a_wanted_item_is_a_no_op() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));

    let _change = change_for(
        &rig,
        WORKTREE,
        vec![pending(0, 100), pending(1, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let embedded = index_records(&rig);
    let want: HashSet<ContentHash> = [0, 1]
        .into_iter()
        .map(|seq| pending(seq, 100).chunk.content_hash)
        .collect();
    assert_eq!(embedded.len(), 2, "each hash embeds exactly once");
    assert!(embedded.iter().all(|h| want.contains(h)));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_dropping_a_finished_or_unknown_hash_is_a_no_op() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![pending(0, 100)], IndexClass::Stage1).await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    let unknown = pending(9, 100).chunk.content_hash;
    let _change = change_for(&rig, WORKTREE, vec![], vec![unknown], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    // Hash 0 is finished; dropping it now changes nothing and does not
    // panic.
    let finished = pending(0, 100).chunk.content_hash;
    let _late = change_for(&rig, WORKTREE, vec![], vec![finished], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_on_one_worktree_keeps_a_shared_item_for_the_other() {
    const OTHER: WorktreeId = WorktreeId(2);
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let _other_done = enqueue_for(
        &rig,
        OTHER,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(2, 0, 0, false));

    let drops: Vec<ContentHash> = [0, 1]
        .into_iter()
        .map(|seq| pending(seq, 100).chunk.content_hash)
        .collect();
    let _change = change_for(&rig, OTHER, vec![], drops, IndexClass::Stage1).await;
    let other = counts_of(&rig, OTHER).await;
    assert_eq!(
        (
            other.queued_stage1,
            other.queued_background,
            other.in_flight
        ),
        (0, 0, 0)
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(2, 0, 0, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let embedded = index_records(&rig);
    let want: HashSet<ContentHash> = [0, 1]
        .into_iter()
        .map(|seq| pending(seq, 100).chunk.content_hash)
        .collect();
    assert_eq!(embedded.len(), 2, "each shared hash embeds exactly once");
    assert!(embedded.iter().all(|h| want.contains(h)));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_change_after_stage1_done_moves_nothing_back() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        vec![pending(0, 100), pending(1, 500)],
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    // A background pass's additions stay background: nothing moves back to
    // stage 1 once it completed.
    let _change = change_for(
        &rig,
        WORKTREE,
        vec![pending(2, 100)],
        vec![],
        IndexClass::Background,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    // A reopen flips `stage1_done` inside the same `on_enqueue`, before
    // `counts_of` is served: this is exact, not timing.
    assert!(now.stage1_done, "{now:?}");
    assert_eq!(now.queued_stage1, 0, "{now:?}");
    assert!(matches!(
        done.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stage1_change_after_completion_reopens_stage1_once() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![pending(0, 100)], IndexClass::Stage1).await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    // A late stage-1 change reopens completion; the second `Stage1Done` is
    // the documented idempotent double delivery.
    let _change = change_for(
        &rig,
        WORKTREE,
        vec![pending(1, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    assert!(!now.stage1_done, "{now:?}");
    assert_eq!(now.queued_stage1 + now.in_flight, 1, "{now:?}");
    wait_for(&mut log, &rig.records, |r| {
        r.kind == Kind::Index && r.hash == Some(pending(1, 100).chunk.content_hash)
    });
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes again")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resync_skips_chunks_finished_inside_its_barrier() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..3).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);

    // The resync's barrier, whose read saw 0, 1 and 2 uncached.
    rig.sched.barrier(REPO, true).await;
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    // The resync's `Replace`, built from that read.
    let mut again = send_replace(
        &rig,
        WORKTREE,
        (0..4).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    assert!(!now.stage1_done, "{now:?}");
    assert_eq!(now.queued_stage1 + now.in_flight, 1, "{now:?}");
    tokio::time::timeout(TIMEOUT, again.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &rig);
    assert_each_once(&embedded);
    let want: HashSet<ContentHash> = [
        pending(0, 100).chunk.content_hash,
        pending(1, 100).chunk.content_hash,
        pending(2, 100).chunk.content_hash,
        pending(3, 100).chunk.content_hash,
    ]
    .into_iter()
    .collect();
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        want,
        "{embedded:?}"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_skips_chunks_another_worktree_finished_inside_its_barrier() {
    const OTHER: WorktreeId = WorktreeId(2);
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let _other = enqueue_for(&rig, OTHER, vec![], IndexClass::Background).await;
    let mut done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    // OTHER's work runs at background QoS, which a loaded host can starve;
    // the `Started` record proves `open_ane` ran.
    let _lift = stub::lift_ane(&rig.backend, QosClass::UserInitiated);

    // OTHER's barrier, whose reconcile read saw 0 and 1 uncached.
    rig.sched.barrier(REPO, true).await;
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let _change = send_change(
        &rig,
        OTHER,
        (0..3).map(|seq| pending(seq, 100)).collect(),
        vec![],
        IndexClass::Background,
    )
    .await;
    let now = counts_of(&rig, OTHER).await;
    assert_eq!(now.queued_background + now.in_flight, 1, "{now:?}");
    wait_for(&mut log, &rig.records, |r| {
        r.kind == Kind::Index && r.hash == Some(pending(2, 100).chunk.content_hash)
    });
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &rig);
    assert_each_once(&embedded);
    let want: HashSet<ContentHash> = [
        pending(0, 100).chunk.content_hash,
        pending(1, 100).chunk.content_hash,
        pending(2, 100).chunk.content_hash,
    ]
    .into_iter()
    .collect();
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        want,
        "{embedded:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_skips_chunks_a_search_embedded_inside_its_barrier() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let _w = enqueue(&rig, vec![], IndexClass::Background).await;

    // The edit's barrier, whose reconcile read saw 5, 6 and 7 uncached.
    rig.sched.barrier(REPO, true).await;
    // A search embeds 5 and 6 on the spot; the scheduler holds neither
    // hash, and `DocumentsDone` notes both committed.
    query_documents(&rig, 5, 2).await;
    let mut log = Vec::new();
    // The `Documents` job's records prove `open_ane` ran.
    let _lift = stub::lift_ane(&rig.backend, QosClass::UserInitiated);
    let _change = send_change(
        &rig,
        WORKTREE,
        (5..8).map(|seq| pending(seq, 100)).collect(),
        vec![],
        IndexClass::Background,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    assert_eq!(now.queued_background + now.in_flight, 1, "{now:?}");
    wait_for(&mut log, &rig.records, |r| {
        r.kind == Kind::Index && r.hash == Some(pending(7, 100).chunk.content_hash)
    });
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &rig);
    for hash in [
        pending(5, 100).chunk.content_hash,
        pending(6, 100).chunk.content_hash,
    ] {
        assert!(
            !embedded.contains(&hash),
            "a stored hash was embedded again: {embedded:?}"
        );
        assert_eq!(
            log.iter()
                .filter(|r| r.kind == Kind::QueryDocument && r.hash == Some(hash))
                .count(),
            1,
            "one QueryDocument record for {hash:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_worktrees_resync_of_finished_chunks_completes_stage1_at_once() {
    const OTHER: WorktreeId = WorktreeId(2);
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(
        &rig,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);

    // OTHER's first barrier; its `missing` read saw 0 and 1 uncached.
    rig.sched.barrier(REPO, true).await;
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let mut other_done = send_replace(
        &rig,
        OTHER,
        (0..2).map(|seq| pending(seq, 100)).collect(),
        IndexClass::Stage1,
    )
    .await;
    // FIFO-exact: stage-1 completion runs at the end of the same
    // `on_enqueue`, before `CountsFor` is served.
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 0, true));
    tokio::time::timeout(TIMEOUT, other_done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &rig);
    assert_each_once(&embedded);
    assert_eq!(embedded.len(), 2, "{embedded:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn committed_hashes_are_kept_only_while_a_barrier_runs() {
    // Steps 2 and 4 send a `Change` naming a hash that already has a
    // vector. No production feed can do that — a read inside the barrier
    // sees the vector — so the steps exist only to observe that the set is
    // not kept.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut done = enqueue(&rig, vec![pending(0, 100)], IndexClass::Stage1).await;
    // Item 0's embed waits at the gate until the bracketed `Enqueue`'s
    // barrier has ended.
    let mut log = Vec::new();
    wait_started(&mut log, &rig.records, 1);
    open(rig.gate(Instance::Ane));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    // Not kept without a barrier.
    rig.sched.barrier(REPO, true).await;
    let _a = send_change(
        &rig,
        WORKTREE,
        vec![pending(0, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    assert!(!now.stage1_done, "{now:?}");
    assert_eq!(now.queued_stage1 + now.in_flight, 1, "{now:?}");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;

    // Kept until the last of two overlapping barriers ends.
    rig.sched.barrier(REPO, true).await;
    rig.sched.barrier(REPO, true).await;
    let _b = send_change(
        &rig,
        WORKTREE,
        vec![pending(1, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;
    let _c = send_change(
        &rig,
        WORKTREE,
        vec![pending(1, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    rig.sched.barrier(REPO, false).await;

    // Dropped once none runs.
    rig.sched.barrier(REPO, true).await;
    let _d = send_change(
        &rig,
        WORKTREE,
        vec![pending(1, 100)],
        vec![],
        IndexClass::Stage1,
    )
    .await;
    let now = counts_of(&rig, WORKTREE).await;
    assert!(!now.stage1_done, "{now:?}");
    assert_eq!(now.queued_stage1 + now.in_flight, 1, "{now:?}");
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &rig);
    for seq in [0, 1] {
        let hash = pending(seq, 100).chunk.content_hash;
        assert_eq!(
            embedded.iter().filter(|h| **h == hash).count(),
            2,
            "hash {seq} embedded exactly twice: {embedded:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_embeds_its_gpu_chunk_again_without_waiting() {
    // Both first calls are gated: the GPU provably holds `C` in its batch
    // when the search's job lands, and the job answers without it.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(10),
        &[Instance::Ane, Instance::Gpu],
    )
    .await;
    let mut log = Vec::new();
    let c = pending(0, 200);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);

    rig.backend.push_headroom(Ok(()));
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started && r.hash == Some(hash)
    });
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    rig.sched.barrier(REPO, true).await;
    let mut docs = send_spot(&rig, vec![chunk]).await;
    // The job twins `C`; the item stays in flight on the GPU, neither held
    // nor counted again.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    // The GPU's gate is still closed, so the search was answered without
    // waiting for the GPU batch.
    tokio::time::timeout(TIMEOUT, &mut docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::QueryDocument)],
        "the job's part embedded C; the GPU batch has not finished"
    );
    assert_eq!(
        counts_of(&rig, WORKTREE).await,
        counts(0, 0, 1, false),
        "DocumentsDone does not finish the GPU's item"
    );
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;

    open(rig.gate(Instance::Gpu));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    // `GpuDone` finished the item once.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Dropped
    });
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![
            (Instance::Ane, Kind::QueryDocument),
            (Instance::Gpu, Kind::Index)
        ]
    );
    // The GPU's second vector for `C` committed without error, and the
    // cache still answers one vector for it.
    assert_cached(&rig, &[hash]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gpu_batch_skips_a_chunk_a_search_holds() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(10),
        &[Instance::Ane],
    )
    .await;
    let mut log = Vec::new();
    let c = pending(0, 200);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut items = vec![c];
    items.extend((1..9).map(|seq| pending(seq, 100)));
    let mut done = enqueue(&rig, items, IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(9, 0, 0, false));

    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![chunk]).await;
    // `C` is held and counted in flight.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 1, false));

    rig.backend.push_headroom(Ok(()));
    // The GPU's first pull is FIFO-behind the job, so `C` is already held.
    // The GPU embeds the eight fillers as one batch, and its next pull
    // finds stage 1's queue empty.
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Dropped
    });
    // `GpuDone` is handled before the pull that got `Stop`.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::QueryDocument)]
    );
    let indexed = index_hashes(&mut log, &rig);
    assert_each_once(&indexed);
    let want: HashSet<ContentHash> = (1..9)
        .map(|seq| pending(seq, 100).chunk.content_hash)
        .collect();
    assert_eq!(indexed.iter().copied().collect::<HashSet<_>>(), want);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_borrows_its_chunk_from_the_ane() {
    // The ANE's first call is gated: `C` is provably in flight on the ANE
    // when the job lands. `D` is never enqueued, so the job's own part
    // embeds it behind the borrowed item.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(0, 100);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let d = pending(1, 100);
    let other_hash = d.chunk.content_hash;
    let other = d.chunk.clone();
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    rig.sched.barrier(REPO, true).await;
    let mut docs = send_spot(&rig, vec![chunk, other]).await;
    // `C` is borrowed, and `D` has no item.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    // The `counts_of` reply proves the job was handled, and the gate holds
    // `AneDone`: the job waits for the borrowed item.
    assert!(matches!(
        docs.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    open(rig.gate(Instance::Ane));
    // The borrow did not stall the ANE thread. It finished `C`, then
    // pulled and ran the job's part.
    tokio::time::timeout(TIMEOUT, &mut docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_cached(&rig, &[hash, other_hash]).await;
    rig.sched.barrier(REPO, false).await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::Index)]
    );
    assert_eq!(
        embeds_of(&mut log, &rig, other_hash),
        vec![(Instance::Ane, Kind::QueryDocument)]
    );
    let index_pos = log
        .iter()
        .position(|r| r.kind == Kind::Index && r.hash == Some(hash))
        .expect("C's Index record");
    let part_pos = log
        .iter()
        .position(|r| r.kind == Kind::QueryDocument && r.hash == Some(other_hash))
        .expect("D's QueryDocument record");
    assert!(
        part_pos > index_pos,
        "the part ran after the borrowed item, on the same thread"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_skips_a_chunk_finished_inside_its_window() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(0, 100);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);

    // The search's window; its vector sync, inside it, saw `C` uncached.
    rig.sched.barrier(REPO, true).await;
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    // `C` was embedded and `item_done` ran inside the window.
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");

    let docs = send_spot(&rig, vec![chunk]).await;
    // Answered at once, since `C` is in the committed set.
    tokio::time::timeout(TIMEOUT, docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::Index)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_search_hands_its_held_chunk_back_to_indexing() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(1, 100);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    // The search hung up before its part ran.
    drop(docs);

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    // The dispatch skipped the job and put `C` back in stage 1, and the
    // ANE embedded it as indexing work.
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    rig.sched.barrier(REPO, false).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::Index)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunk_two_searches_hold_stays_held_when_one_is_dropped() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(1, 100);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    // Two searches' windows.
    rig.sched.barrier(REPO, true).await;
    rig.sched.barrier(REPO, true).await;
    let first = send_spot(&rig, vec![chunk.clone()]).await;
    let second = send_spot(&rig, vec![chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    drop(first);

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, second)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;
    rig.sched.barrier(REPO, false).await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    // Without the count, releasing `first` would requeue `C` while
    // `second` still embeds it, and `on_documents_done`'s `Held` assertion
    // would panic the scheduler.
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::QueryDocument)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_whose_only_chunk_is_on_the_ane_is_answered_at_its_ane_done() {
    // The ANE's first call is gated: `C` is provably in flight on the ANE
    // when the job lands, and the job has no part of its own.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(0, 100);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);

    rig.sched.barrier(REPO, true).await;
    let mut docs = send_spot(&rig, vec![chunk]).await;
    // `C` is borrowed: neither held nor counted again.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    // The `counts_of` reply proves the job was handled, and the gate holds
    // `AneDone`: a job with only a borrow is not answered on arrival.
    assert!(matches!(
        docs.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    open(rig.gate(Instance::Ane));
    // `AneDone` settles the borrow, which was the job's only part.
    tokio::time::timeout(TIMEOUT, &mut docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![(Instance::Ane, Kind::Index)]
    );
    assert!(
        !log.iter().any(|r| r.kind == Kind::QueryDocument),
        "the job ran no part: {log:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gpu_twin_finished_before_the_search_part_is_counted_once() {
    // As in `a_search_embeds_its_gpu_chunk_again_without_waiting`, but the
    // GPU's gate opens first: `GpuDone` finishes the twin's item while the
    // job's part still waits behind the gated query.
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(10),
        &[Instance::Ane, Instance::Gpu],
    )
    .await;
    let mut log = Vec::new();
    let c = pending(0, 200);
    let hash = c.chunk.content_hash;
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let mut done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);

    rig.backend.push_headroom(Ok(()));
    wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started && r.hash == Some(hash)
    });
    rig.sched.barrier(REPO, true).await;
    let mut docs = send_spot(&rig, vec![chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    open(rig.gate(Instance::Gpu));
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
    // `GpuDone` finished the item; the ANE is still at its gate, so the
    // job's part has not run.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert!(matches!(
        docs.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
    tokio::time::timeout(TIMEOUT, &mut docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
        .expect("embeds fine");
    // `DocumentsDone` found the twin's item gone and only noted its hash.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &rig, hash),
        vec![
            (Instance::Gpu, Kind::Index),
            (Instance::Ane, Kind::QueryDocument)
        ]
    );
    assert_cached(&rig, &[hash]).await;
    rig.sched.barrier(REPO, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_fails_a_queued_search_part_and_requeues_its_held_chunk() {
    let rig = rig(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Ane],
    )
    .await;
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    let mut log = Vec::new();
    let c = pending(1, 100);
    let chunk = c.chunk.clone();
    let query = send_query(&rig, "hold the pull").await;
    let _done = enqueue(&rig, vec![c], IndexClass::Stage1).await;
    wait_started(&mut log, &rig.records, 1);

    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    // The part is queued behind the gated query when `Stop` lands: the job
    // fails, and `C` goes back to stage 1's queue.
    rig.sched.stop().await;
    let answer = tokio::time::timeout(TIMEOUT, docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered");
    assert!(
        matches!(answer, Err(EmbedError::SchedulerGone)),
        "{answer:?}"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(1, 0, 0, false));

    // The running query still finishes before the ANE pulls `Stop`.
    open(rig.gate(Instance::Ane));
    query.await.expect("query answered").expect("embeds fine");
}
