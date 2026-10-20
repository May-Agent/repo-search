//! Scheduler-level tests: `Enqueue` and `QueryJob`s are fed to the scheduler
//! directly, with no daemon and no sockets. The stub's `Record` channel
//! carries what the embedding threads did; waits use `recv_timeout` on it.

#![cfg(target_os = "macos")]

#[path = "support/stub.rs"]
mod stub;

use std::collections::{BTreeMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use repo_search::daemon::scheduler::{
    EmbedError, Enqueue, IndexClass, PendingItem, QueryJob, SchedulerHandle, Stage1Done, WorktreeId,
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
    fatal: tokio::sync::mpsc::Receiver<Error>,
    gates: BTreeMap<Instance, Arc<Barrier>>,
    /// The ANE thread, for reading its effective priority.
    ane_thread: qos::ThreadId,
    _monitor: StubMonitor,
    _tmp: tempfile::TempDir,
    _ane: std::thread::JoinHandle<()>,
}

impl Rig {
    /// The barrier `gate_first` installed for `instance`.
    fn gate(&self, instance: Instance) -> &Barrier {
        &self.gates[&instance]
    }
}

async fn rig(ane_delay: Duration, gpu_delay: Duration, gated: &[Instance]) -> Rig {
    let (record_tx, records) = mpsc::channel();
    let mut backend = StubBackend::new(record_tx);
    backend.set_instance_delay(Instance::Ane, ane_delay);
    backend.set_instance_delay(Instance::Gpu, gpu_delay);
    let gates: BTreeMap<Instance, Arc<Barrier>> = gated
        .iter()
        .map(|&instance| (instance, backend.gate_first(instance)))
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
    let (_, buckets) = backend
        .load_counter(tmp.path())
        .expect("stub counter loads");
    let sched = scheduler::spawn(
        backend.clone(),
        tmp.path().to_path_buf(),
        "stub-model".to_owned(),
        buckets,
        headroom_rx,
        fatal_tx.clone(),
    );
    let ane::AneThread { loaded, join } = ane::spawn(
        backend.clone(),
        tmp.path().to_path_buf(),
        "stub-model".to_owned(),
        sched.tx(),
    );
    let ane_thread = loaded
        .await
        .expect("ane thread reports")
        .expect("stub open_ane does not fail");

    let cache = VectorCache::open(&git).await.expect("vector cache opens");
    let writer = writer::spawn(cache, "stub-model".to_owned(), fatal_tx);

    Rig {
        sched,
        backend,
        records,
        writer,
        fatal,
        gates,
        ane_thread,
        _monitor: monitor,
        _tmp: tmp,
        _ane: join,
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

/// Sends `items` to the scheduler as `class`, generation 1.
async fn enqueue(
    rig: &Rig,
    items: Vec<scheduler::PendingItem>,
    class: IndexClass,
) -> tokio::sync::mpsc::Receiver<Stage1Done> {
    let (done_tx, done_rx) = tokio::sync::mpsc::channel(8);
    rig.sched
        .enqueue(Enqueue {
            worktree: WORKTREE,
            repo: REPO,
            class,
            generation: 1,
            items,
            writer: rig.writer.clone(),
            stage1_done: done_tx,
        })
        .await
        .expect("scheduler is up");
    done_rx
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
        .query(QueryJob::Query {
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

/// Sends a `Documents` job of `n` chunks for `WORKTREE`, awaits the vectors
/// and returns the chunks' content hashes.
async fn query_documents(rig: &Rig, first_seq: u64, n: usize) -> HashSet<ContentHash> {
    let (reply, rx) = oneshot::channel();
    let chunks: Vec<PendingChunk> = (first_seq..first_seq + n as u64)
        .map(|seq| pending(seq, 100).chunk)
        .collect();
    let hashes: HashSet<ContentHash> = chunks.iter().map(|chunk| chunk.content_hash).collect();
    rig.sched
        .query(QueryJob::Documents {
            worktree: WORKTREE,
            repo: REPO,
            chunks,
            enqueued: Instant::now(),
            stamp: 0,
            reply,
        })
        .await
        .expect("scheduler is up");
    let vectors = rx.await.expect("documents answered").expect("embeds fine");
    assert_eq!(vectors.len(), n);
    hashes
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_preempts_indexing() {
    let rig = rig(Duration::from_millis(20), Duration::ZERO, &[]).await;
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
    let rig = rig(Duration::from_millis(20), Duration::from_millis(10), &[]).await;
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
async fn no_gpu_on_battery() {
    let rig = rig(Duration::from_millis(5), Duration::from_millis(5), &[]).await;
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
    let rig = rig(Duration::from_millis(10), Duration::ZERO, &[Instance::Gpu]).await;
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
    // flight when the query lands.
    let rig = rig(Duration::from_millis(50), Duration::ZERO, &[Instance::Ane]).await;
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

    let base = qos::scheduling_priority(rig.ane_thread);
    let rx = send_query(&rig, "boost the chunk ahead").await;
    // The inbox is FIFO: once `counts_for` is answered, the Query ahead of it
    // was handled and the override on the ANE thread is in place.
    rig.sched
        .counts_for(WORKTREE)
        .await
        .expect("counts answered");
    assert!(
        qos::scheduling_priority(rig.ane_thread) > base,
        "the override lifts the in-flight chunk's effective priority"
    );
    open(rig.gate(Instance::Ane));
    rx.await.expect("query answered").expect("embeds fine");

    // The gated chunk and the next one ran at the same requested class; only
    // the override separates their effective priorities.
    let lifted = wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index
    });
    assert_eq!(log[lifted].qos, QosClass::Background);
    let lifted_end = log[lifted].end;
    let released = wait_for(&mut log, &rig.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index && r.start > lifted_end
    });
    assert_eq!(log[released].qos, QosClass::Background);
    assert!(
        log[lifted].pri > log[released].pri,
        "the override ended with the pull that followed the lifted chunk"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_after_a_fatal_error_is_answered_and_the_scheduler_exits() {
    // The ANE's first call is gated so it cannot pull its `Stop`: the actor
    // is stopping but cannot exit, and it must still answer the query.
    let mut rig = rig(Duration::from_millis(10), Duration::ZERO, &[Instance::Ane]).await;
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
