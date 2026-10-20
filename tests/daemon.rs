//! Daemon-level tests over sockets: `Client` requests against a daemon run
//! in process, plus the `daemon_host` re-execution the real-kill test uses.

#![cfg(target_os = "macos")]

mod support;

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use repo_search::client::Client;
use repo_search::daemon::{self, Config, Error};
use repo_search::model_store::{ModelSpec, ModelStore, Sha256Digest};
use repo_search::protocol::{
    self, ErrorKind, Event, Final, Format, Request, SearchRequest, WireMode, WirePhase,
};
use repo_search_core::qos::QosClass;
use repo_search_core::store::{ContentHash, Index, VectorCache};
use repo_search_coreml::GpuBlocked;
use sha2::Digest;
use support::repo::{self, TestRepo};
use support::stub::{IndexGate, Instance, Kind, Record, StubBackend};
use tokio::net::UnixListener;
use tokio::task::JoinHandle;

const TIMEOUT: Duration = Duration::from_secs(60);

const MARKER: &str = "xylophonemarker";

/// A `rust_fn` text size that keeps every function above the chunker's
/// 400-byte pack threshold, so each becomes a chunk of its own at ~131
/// `StubCounter` tokens: the stage-1 smalls.
const SMALL_BYTES: usize = 500;
/// And these land above the 256-token stage-1 bound: the longs.
const LONG_BYTES: usize = 1900;

/// A repo of `n` functions of `bytes` text each, `per_file` per file, plus
/// one extra function carrying `MARKER` in its name and body — `n + 1`
/// chunks in total.
fn fn_repo(n: usize, bytes: usize, per_file: usize) -> TestRepo {
    let mut files = Vec::new();
    for f in 0..n.div_ceil(per_file) {
        let mut body = String::new();
        for j in 0..per_file {
            let i = f * per_file + j;
            if i >= n {
                break;
            }
            body.push_str(&repo::rust_fn(&format!("fn_{i:04}"), bytes));
        }
        if f == 0 {
            body.push_str(&repo::rust_fn(&format!("{MARKER}_fn"), bytes));
        }
        files.push((format!("src/f{f}.rs"), body));
    }
    repo::repo(&files)
}

/// The test daemon: a socket in a temp dir, the stub backend's record log
/// and the run task's handle.
struct Daemon {
    socket: PathBuf,
    /// The store's archive dir: `<models>/<sha>` appears only on a
    /// completed download.
    model_dir: PathBuf,
    records: mpsc::Receiver<Record>,
    backend: Arc<StubBackend>,
    join: JoinHandle<Result<(), Error>>,
    _tmp: tempfile::TempDir,
}

/// How the model cache is laid out for the store.
enum Model {
    /// `models/<sha>/` exists, so `ensure` returns at once.
    Cached,
    /// The dir is absent and the spec points at `url` serving `body`.
    Served { url: String, sha: Sha256Digest },
}

fn digest_of(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::from_hex(&hex::encode(sha2::Sha256::digest(bytes))).expect("hex digest")
}

/// A fixed model id for the cached-model tests.
fn sha_7() -> Sha256Digest {
    Sha256Digest::from_hex(&"07".repeat(32)).expect("hex digest")
}

/// Starts a daemon with `backend`; the record log drains into `records`.
async fn daemon(
    tmp: tempfile::TempDir,
    mut backend: StubBackend,
    model: Model,
    idle_exit: Duration,
) -> Daemon {
    let root = tmp.path().join("models");
    let spec = match &model {
        Model::Cached => ModelSpec {
            url: "http://127.0.0.1:1/unused".into(),
            sha256: sha_7(),
        },
        Model::Served { url, sha } => ModelSpec {
            url: url.clone(),
            sha256: *sha,
        },
    };
    let store = ModelStore::new(root, spec).expect("model store");
    if matches!(model, Model::Cached) {
        std::fs::create_dir_all(store.dir()).expect("cached model dir");
    }
    let model_dir = store.dir();
    let socket = tmp.path().join("daemon.sock");
    let listener = UnixListener::bind(&socket).expect("socket binds");
    let (records_tx, records) = mpsc::channel();
    backend.records_to(records_tx);
    let backend = Arc::new(backend);
    let join = tokio::spawn(daemon::run(Config {
        listener,
        idle_exit,
        model_store: store,
        backend: backend.clone(),
    }));
    Daemon {
        socket,
        model_dir,
        records,
        backend,
        join,
        _tmp: tmp,
    }
}

/// Receives the next record within `TIMEOUT`, off the async executor.
fn recv(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) {
    tokio::task::block_in_place(|| {
        let record = records
            .recv_timeout(TIMEOUT)
            .expect("a record must arrive within the timeout");
        log.push(record);
    })
}

/// Collects records until `pred` holds for one; returns its index in `log`.
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
            let record = records
                .recv_timeout(TIMEOUT)
                .expect("a record must arrive within the timeout");
            log.push(record);
        }
    })
}

fn open(gate: &Barrier) {
    tokio::task::block_in_place(|| gate.wait());
}

/// A `search` request for `query` on `worktree`.
fn search(worktree: &Path, query: &str) -> Request {
    Request::Search(SearchRequest {
        worktree: worktree.to_path_buf(),
        query: query.to_owned(),
        mode: WireMode::Hybrid,
        paths: vec![],
        langs: vec![],
        kinds: vec![],
        symbol: None,
        budget: 4000,
        files_only: false,
        include_ignored: false,
        format: Format::Json,
    })
}

/// Sends `request` and collects events to the `Final` frame.
async fn request(socket: &Path, request: Request) -> Vec<Event> {
    let client = Client::connect(socket)
        .await
        .expect("connect to the daemon");
    let responses = client.request(request).await.expect("request sends");
    tokio::time::timeout(TIMEOUT, responses.collect::<Vec<_>>())
        .await
        .expect("the reply stream does not stall")
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("every frame decodes")
}

/// The events' `Final`.
fn final_of(events: &[Event]) -> &Final {
    match events.last() {
        Some(Event::Final(f)) => f,
        other => panic!("the last event is the Final frame, got {other:?}"),
    }
}

/// The daemon's stub-backed start: cached model, ANE and GPU delays.
async fn start(
    _repo: &TestRepo,
    ane: Duration,
    gpu: Duration,
    gated: &[Instance],
) -> (Daemon, Arc<Barrier>, Arc<Barrier>) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    backend.set_instance_delay(Instance::Ane, ane);
    backend.set_instance_delay(Instance::Gpu, gpu);
    let ane_gate = if gated.contains(&Instance::Ane) {
        backend.gate_first(Instance::Ane)
    } else {
        Arc::new(Barrier::new(1))
    };
    let gpu_gate = if gated.contains(&Instance::Gpu) {
        backend.gate_first(Instance::Gpu)
    } else {
        Arc::new(Barrier::new(1))
    };
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(600)).await;
    (d, ane_gate, gpu_gate)
}

/// A daemon whose request/answer drive indexing, with records collected in
/// `log`.
async fn answer_search(daemon: &Daemon, worktree: &Path, query: &str) -> Vec<Event> {
    request(&daemon.socket, search(worktree, query)).await
}

/// A daemon whose ANE holds index calls at an [`IndexGate`] once the test
/// arms it.
async fn start_index_gated(ane: Duration) -> (Daemon, Arc<IndexGate>) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    backend.set_instance_delay(Instance::Ane, ane);
    let gate = backend.gate_first_index(Instance::Ane);
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(600)).await;
    (d, gate)
}

/// The next `Held` record: an armed gate holds an index call.
fn wait_held(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) {
    wait_for(log, records, |r| r.kind == Kind::Held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_search_answers_and_starts_stage_1() {
    let repo = fn_repo(200, SMALL_BYTES, 10);
    let (d, gate) = start_index_gated(Duration::from_millis(20)).await;

    let events = answer_search(&d, &repo.root, MARKER).await;
    let answer = match final_of(&events) {
        Final::Answer(answer) => answer,
        other => panic!("expected an Answer, got {other:?}"),
    };
    assert!(answer.has_results, "the first search answers from FTS");

    // The next index call is held at the gate: stage 1 is queued behind it
    // and the phase is still first-index, by construction rather than by
    // timing.
    gate.arm();
    let mut log = Vec::new();
    wait_held(&mut log, &d.records);
    let events = request(
        &d.socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    let index = status.index.as_ref().expect("the worktree is indexed");
    assert_eq!(index.phase, WirePhase::FirstIndex);
    assert!(index.queued_stage1 > 0, "stage-1 items are queued");
    gate.open();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_preempts_indexing() {
    let repo = fn_repo(100, SMALL_BYTES, 20);
    let (d, gate) = start_index_gated(Duration::from_millis(20)).await;

    // Stage 1 completes first: every later chunk is stage-2 work, which the
    // ANE runs at background QoS, the class a queued query lifts.
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));

    // New functions need vectors: the gate holds the first background call
    // of their embedding, in flight, when the query arrives.
    gate.arm();
    let mut body = String::new();
    for i in 0..40 {
        body.push_str(&repo::rust_fn(&format!("edited_item_{i}"), SMALL_BYTES));
    }
    repo::append(&repo.root, "src/f0.rs", &body);
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: false,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    let mut log = Vec::new();
    wait_held(&mut log, &d.records);

    // The query's lexical candidates are the edited, unembedded functions:
    // the search embeds them on the spot.
    let sent = Instant::now();
    let events = answer_search(&d, &repo.root, "edited_item_3").await;
    let answer = match final_of(&events) {
        Final::Answer(answer) => answer,
        other => panic!("expected an Answer, got {other:?}"),
    };
    assert_eq!(
        answer.stats.indexing_overtaken, 0,
        "no indexing item overtakes a query"
    );

    // Every record of the search precedes its answer.
    log.extend(d.records.try_iter());
    let query = log
        .iter()
        .find(|r| r.instance == Instance::Ane && r.kind == Kind::Query && r.start >= sent)
        .expect("the search's query embedding ran on the ANE");
    // The held call is the only index job in flight when the query arrives,
    // and the lift releases it ahead of every later one: exactly that call
    // starts between the query's arrival and its own start.
    let overtaking = log
        .iter()
        .filter(|r| r.kind == Kind::Index && r.start >= sent && r.start < query.start)
        .count();
    assert_eq!(
        overtaking, 1,
        "only the held index call starts inside the query's window"
    );

    // On-the-spot documents are ANE-only at user-interactive QoS.
    let docs: Vec<&Record> = log
        .iter()
        .filter(|r| r.kind == Kind::QueryDocument && r.start >= sent)
        .collect();
    assert!(
        !docs.is_empty() && docs.len() <= 100,
        "0 < {} <= 100 on-the-spot documents",
        docs.len()
    );
    assert!(
        docs.iter()
            .all(|r| r.instance == Instance::Ane && r.qos == QosClass::UserInteractive),
        "query document work is ANE-only at user-interactive QoS"
    );
    gate.open();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_takes_the_long_end_in_same_bucket_batches() {
    let mut files = Vec::new();
    for i in 0..6 {
        files.push((
            format!("src/long{i}.rs"),
            repo::rust_fn(&format!("long_{i}"), LONG_BYTES),
        ));
    }
    for i in 0..30 {
        files.push((
            format!("src/mid{i}.rs"),
            repo::rust_fn(&format!("mid_{i}"), 800),
        ));
    }
    files.extend((0..290).map(|i| {
        (
            format!("src/s{i}.rs"),
            repo::rust_fn(&format!("small_{i}"), SMALL_BYTES),
        )
    }));
    let repo = repo::repo(&files);
    // The ANE gate holds stage 1 open while the headroom lands.
    let (d, ane_gate, _) = start(
        &repo,
        Duration::from_millis(20),
        Duration::from_millis(10),
        &[Instance::Ane],
    )
    .await;
    let socket = d.socket.clone();
    let root = repo.root.clone();
    let query = tokio::spawn(async move { request(&socket, search(&root, "long_0")).await });
    // A Started record proves the daemon is up — its headroom callback is
    // registered — so the push below lands.
    let mut log: Vec<Record> = Vec::new();
    wait_for(&mut log, &d.records, |r| r.kind == Kind::Started);
    d.backend.push_headroom(Ok(()));
    open(&ane_gate);
    query.await.expect("the search task answers");

    // The GPU drops once stage-1 items are drained.
    let mut log = Vec::new();
    wait_for(&mut log, &d.records, |r| r.kind == Kind::Dropped);

    let gpu: Vec<&Record> = log.iter().filter(|r| r.instance == Instance::Gpu).collect();
    assert!(gpu.iter().any(|r| r.kind == Kind::Index));
    for record in &gpu {
        assert!(
            matches!(record.kind, Kind::Index | Kind::Started | Kind::Dropped),
            "the GPU only embeds index work"
        );
        if record.kind == Kind::Index {
            assert!(record.batch_len <= 8, "batches are at most 8 documents");
        }
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
        let bucket = |t: usize| support::stub::BUCKETS.iter().find(|&&b| t <= b).unwrap();
        assert!(
            batch
                .iter()
                .all(|r| bucket(r.tokens) == bucket(batch[0].tokens)),
            "a GPU batch mixes token buckets"
        );
    }
    let first = &batches[0];
    assert!(
        first.iter().all(|r| r.tokens > 400),
        "the first GPU batch holds the longest items"
    );

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
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_gpu_on_battery() {
    let repo = fn_repo(100, SMALL_BYTES, 10);
    let (d, _, _) = start(
        &repo,
        Duration::from_millis(5),
        Duration::from_millis(5),
        &[],
    )
    .await;
    // A completed request proves the daemon's headroom callback is
    // registered before the push.
    answer_search(&d, &repo.root, MARKER).await;
    d.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));

    // Every embedded chunk lands on the ANE — Index records for queued
    // work, QueryDocument records for on-the-spot result chunks.
    let mut log: Vec<Record> = Vec::new();
    while log
        .iter()
        .filter(|r| {
            r.instance == Instance::Ane && matches!(r.kind, Kind::Index | Kind::QueryDocument)
        })
        .count()
        < 101
    {
        recv(&mut log, &d.records);
    }
    assert_eq!(
        d.backend.gpu_opens(),
        0,
        "the GPU is never opened on battery"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_released_on_power_loss() {
    let mut files = Vec::new();
    for i in 0..24 {
        files.push((
            format!("src/long{i}.rs"),
            repo::rust_fn(&format!("long_{i}"), LONG_BYTES),
        ));
    }
    files.extend((0..100).map(|i| {
        (
            format!("src/s{i}.rs"),
            repo::rust_fn(&format!("s_{i}"), SMALL_BYTES),
        )
    }));
    let repo = repo::repo(&files);
    let (d, ane_gate, gpu_gate) = start(
        &repo,
        Duration::from_millis(10),
        Duration::from_millis(25),
        &[Instance::Ane, Instance::Gpu],
    )
    .await;
    // Drive the first barrier; the ANE Started record proves the daemon's
    // headroom callback is registered before we push.
    let socket = d.socket.clone();
    let root = repo.root.clone();
    let query = tokio::spawn(async move { request(&socket, search(&root, "long_0")).await });
    let mut log = Vec::new();
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });
    d.backend.push_headroom(Ok(()));
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    // Both instances hold a call in flight; the power loss drops the GPU.
    d.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    open(&gpu_gate);
    let dropped = wait_for(&mut log, &d.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert_eq!(
        log[..dropped]
            .iter()
            .filter(|r| r.instance == Instance::Gpu && r.kind == Kind::Index)
            .count(),
        8,
        "the GPU embedded exactly its gated batch"
    );

    // Indexing continues on the ANE.
    open(&ane_gate);
    query.await.expect("the search task answers");
    let dropped_start = log[dropped].start;
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index && r.start > dropped_start
    });
    assert!(
        !log[dropped + 1..]
            .iter()
            .any(|r| r.instance == Instance::Gpu),
        "no GPU record starts after the embedder was dropped"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stage_2_runs_at_background_qos() {
    // Disjoint identifiers: the query below names exactly one long, so the
    // search embeds no stage-1 small on the spot.
    let mut files = Vec::new();
    for i in 0..8 {
        files.push((
            format!("src/long{i}.rs"),
            repo::rust_fn(&format!("longitem{i}"), LONG_BYTES),
        ));
    }
    files.extend((0..30).map(|i| {
        (
            format!("src/s{i}.rs"),
            repo::rust_fn(&format!("smallitem{i}"), SMALL_BYTES),
        )
    }));
    let repo = repo::repo(&files);
    // The GPU's first call is gated: it takes exactly one batch while the
    // ANE works the 30 small items through stage-1 completion.
    let (d, _, gpu_gate) = start(
        &repo,
        Duration::from_millis(10),
        Duration::ZERO,
        &[Instance::Gpu],
    )
    .await;
    let socket = d.socket.clone();
    let root = repo.root.clone();
    // The query matches nothing, so no chunk is embedded on the spot.
    let query = tokio::spawn(async move { request(&socket, search(&root, "nomatchterm")).await });
    let mut log: Vec<Record> = Vec::new();
    // The daemon is up once the ANE's first record lands; the headroom push
    // after it reaches a registered callback.
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Started
    });
    d.backend.push_headroom(Ok(()));
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    query.await.expect("the search task answers");
    open(&gpu_gate);

    // The GPU drops once stage 1 is empty.
    let dropped = wait_for(&mut log, &d.records, |r| r.kind == Kind::Dropped);
    assert_eq!(log[dropped].instance, Instance::Gpu);
    assert!(
        !log[dropped + 1..]
            .iter()
            .any(|r| r.instance == Instance::Gpu),
        "no GPU record follows the drop"
    );

    // The GPU's one batch took every long, so what is left is stage 1's
    // smalls: `Indexed` comes after the last of them and after the phase flip.
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    let events = request(
        &d.socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::Maintained,
        "stage 1 completed"
    );

    // Stage 2: a new long chunk is embedded by the ANE, the one record this
    // test waits for, at background QoS.
    repo::append(
        &repo.root,
        "src/s0.rs",
        &repo::rust_fn("longitem_edit", LONG_BYTES),
    );
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    log.extend(d.records.try_iter());
    let long = log
        .iter()
        .find(|r| r.instance == Instance::Ane && r.kind == Kind::Index && r.tokens > 256)
        .expect("the ANE embedded the stage-2 chunk");
    assert_eq!(
        long.qos,
        QosClass::Background,
        "stage-2 indexing runs at background QoS"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edit_is_reflected_in_next_query() {
    let repo = fn_repo(40, SMALL_BYTES, 10);
    let (d, _, _) = start(&repo, Duration::ZERO, Duration::ZERO, &[]).await;
    answer_search(&d, &repo.root, MARKER).await;

    let ident = "zygomorphic_reflected";
    repo::append(
        &repo.root,
        "src/f0.rs",
        &format!("\npub fn {ident}() -> usize {{ \"{ident} {ident}\".len() }}\n"),
    );

    let events = answer_search(&d, &repo.root, ident).await;
    let answer = match final_of(&events) {
        Final::Answer(answer) => answer,
        other => panic!("expected an Answer, got {other:?}"),
    };
    assert!(answer.has_results, "the new function is found");
    let json: serde_json::Value = serde_json::from_str(&answer.output).expect("json output");
    let chunks: Vec<&serde_json::Value> = json["code"]
        .as_array()
        .map(|files| {
            files
                .iter()
                .flat_map(|f| f["chunks"].as_array().into_iter().flatten())
                .collect()
        })
        .unwrap_or_default();
    let hit = chunks
        .iter()
        .find(|c| c["text"].as_str().is_some_and(|t| t.contains(ident)))
        .unwrap_or_else(|| panic!("the answer contains the new function: {json}"));
    let channels: Vec<&str> = hit["channels"]
        .as_array()
        .map(|c| c.iter().filter_map(|c| c.as_str()).collect())
        .unwrap_or_default();
    assert!(
        channels.contains(&"bm25"),
        "the new chunk is found by bm25: {channels:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fsevents_change_is_embedded_in_background() {
    let repo = fn_repo(40, SMALL_BYTES, 10);
    let (d, _, _) = start(&repo, Duration::ZERO, Duration::ZERO, &[]).await;
    answer_search(&d, &repo.root, MARKER).await;
    // Every pending chunk is stage-1 class while the phase is first-index,
    // so `index --wait` waits only on UserInitiated work — never starved —
    // and the phase flips to maintained before it answers.
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    let events = request(
        &d.socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::Maintained,
        "the first index completed"
    );
    // `Indexed` came after every record of the pass was sent, so the channel
    // already holds all of them.
    let known: HashSet<ContentHash> = d.records.try_iter().filter_map(|r| r.hash).collect();

    // No request follows: the watcher's barrier enqueues the edit's chunks.
    repo::append(
        &repo.root,
        "src/f0.rs",
        &format!("\npub fn background_edit() -> usize {{ \"{MARKER} background_edit\".len() }}\n"),
    );
    tokio::task::block_in_place(|| {
        let record = loop {
            let record = d
                .records
                .recv_timeout(TIMEOUT)
                .expect("an Index record must arrive after the edit");
            if record.kind == Kind::Index && record.hash.is_some_and(|h| !known.contains(&h)) {
                break record;
            }
        };
        assert_eq!(
            record.qos,
            QosClass::Background,
            "the edit's chunks embed at background QoS"
        );
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_exit() {
    // No work left: run returns shortly after the last connection closes.
    let repo = fn_repo(30, SMALL_BYTES, 10);
    let tmp = tempfile::tempdir().expect("tempdir");
    let backend = StubBackend::new(mpsc::channel().0);
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(2)).await;
    answer_search(&d, &repo.root, MARKER).await;
    // Nothing is queued, buffered or in flight once the index is complete,
    // so only the idle deadline remains.
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    let closed = Instant::now();
    tokio::time::timeout(Duration::from_secs(5), d.join)
        .await
        .unwrap_or_else(|_| panic!("run must return within 5 s of the last connection"))
        .expect("run task does not panic")
        .expect("idle exit is not an error");
    assert!(closed.elapsed() >= Duration::from_secs(1));

    // Work queued keeps the daemon alive; it exits once the queue drains.
    let repo = fn_repo(30, SMALL_BYTES, 10);
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    backend.set_instance_delay(Instance::Ane, Duration::from_millis(100));
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(2)).await;
    let events = request(
        &d.socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: false,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    let mut log: Vec<Record> = Vec::new();
    while log.iter().filter(|r| r.kind == Kind::Index).count() < 30 {
        recv(&mut log, &d.records);
        assert!(!d.join.is_finished(), "run must not exit mid-queue");
    }
    tokio::time::timeout(Duration::from_secs(10), d.join)
        .await
        .expect("run returns once the queue drains")
        .expect("run task does not panic")
        .expect("idle exit is not an error");
}

/// A raw client whose `Hello` mismatches: the daemon answers
/// `ProtocolMismatch` and begins shutdown. Returns the parsed frame.
fn mismatched_hello(socket: &Path) -> Event {
    let line = tokio::task::block_in_place(|| {
        use std::io::{BufRead, BufReader};
        let socket = socket.to_path_buf();
        let mut stream = std::os::unix::net::UnixStream::connect(&socket).expect("connect");
        let hello = serde_json::to_string(&protocol::Hello {
            protocol: protocol::PROTOCOL + 1,
            version: "0.0.0".into(),
        })
        .unwrap();
        writeln!(stream, "{hello}").unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        line
    });
    serde_json::from_str(&line).expect("a protocol frame")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn protocol_mismatch_shuts_down() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Cached,
        Duration::from_secs(600),
    )
    .await;

    match mismatched_hello(&d.socket) {
        Event::Final(Final::Error(report)) => {
            assert_eq!(report.kind, ErrorKind::ProtocolMismatch)
        }
        other => panic!("expected a ProtocolMismatch, got {other:?}"),
    }
    tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run returns after the mismatch")
        .expect("run task does not panic")
        .expect("a protocol mismatch is a clean shutdown");
}

/// A shutdown during a gated `open_ane` must not abandon the ANE thread:
/// the loader awaits the load attempt and stops the half-built pipeline,
/// so the thread's drop is recorded before `run` returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_during_open_ane_joins_the_thread() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    let hold = backend.hold_open_ane();
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(600)).await;

    // The ANE thread is inside `open_ane` when the shutdown begins.
    tokio::task::block_in_place(|| {
        hold.entered
            .recv_timeout(TIMEOUT)
            .expect("open_ane is entered")
    });
    mismatched_hello(&d.socket);
    tokio::task::block_in_place(|| hold.open());

    // `open_ane` returns a live embedder; the pipeline stop pulls `Stop`
    // on it, so its `Drop` record precedes the loader's shutdown reply,
    // which precedes `run` returning.
    let mut log: Vec<Record> = Vec::new();
    wait_for(&mut log, &d.records, |r| {
        r.instance == Instance::Ane && r.kind == Kind::Dropped
    });
    tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run returns once the ANE thread is joined")
        .expect("run task does not panic")
        .expect("a mismatch shutdown is clean");
}

/// A shutdown during a gated download cancels `ensure`: `run` returns and
/// no `<sha>/` model directory is left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_during_download_leaves_no_model_dir() {
    let body = support::http::archive(&[("manifest.json", b"{}")]);
    let (release_tx, release) = mpsc::channel();
    let (url, server) = support::http::serve_gated(body.len() as u64, body.clone(), release);
    let tmp = tempfile::tempdir().expect("tempdir");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Served {
            url,
            sha: digest_of(&body),
        },
        Duration::from_secs(600),
    )
    .await;

    // A request waits out the download; `ModelDownload` progress proves
    // `ensure` is mid-body, held by the server.
    let repo = fn_repo(5, SMALL_BYTES, 10);
    let mut stream = stream_request(&d.socket, search(&repo.root, MARKER)).await;
    loop {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::ModelDownload { received, .. }) if received > 0 => {
                break;
            }
            Event::Final(f) => panic!("the request ended before the download: {f:?}"),
            _ => {}
        }
    }

    mismatched_hello(&d.socket);
    // The waiting request is answered `ShuttingDown`, not left hanging.
    loop {
        match next_event(&mut stream).await {
            Event::Final(Final::Error(report)) => {
                assert_eq!(report.kind, ErrorKind::ShuttingDown);
                break;
            }
            Event::Final(other) => panic!("expected ShuttingDown, got {other:?}"),
            _ => {}
        }
    }
    drop(release_tx);
    server.join().expect("the gated server exits");
    tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run returns after cancelling the download")
        .expect("run task does not panic")
        .expect("a mismatch shutdown is clean");
    assert!(
        !d.model_dir.exists(),
        "a cancelled download leaves no model directory"
    );
}

/// A client that hangs up mid-barrier does not leak it: the barrier still
/// completes — its enqueue reaches the scheduler — and the daemon
/// idle-exits once the marker and the work are gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_client_still_completes_the_barrier() {
    let repo = fn_repo(20, SMALL_BYTES, 10);
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    // The counter is called inside `reconcile`, which the barrier awaits.
    let hold = backend.hold_counter();
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(1)).await;

    let stream = stream_request(&d.socket, search(&repo.root, MARKER)).await;
    // The search is inside its barrier when the counter is entered.
    tokio::task::block_in_place(|| {
        hold.entered
            .recv_timeout(TIMEOUT)
            .expect("the barrier's reconcile is counting")
    });
    // The client hangs up inside the barrier.
    drop(stream);
    tokio::task::block_in_place(|| hold.open());

    // The barrier ran to completion: its `Enqueue` reached the scheduler,
    // whose ANE dispatches produce index records.
    let mut log: Vec<Record> = Vec::new();
    wait_for(&mut log, &d.records, |r| r.kind == Kind::Index);
    // The barrier's busy marker was cleared by its own completion, so the
    // idle deadline fires.
    tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run idle-exits after the dropped search's barrier")
        .expect("run task does not panic")
        .expect("idle exit is not an error");
}

/// A client that never sends `Hello` is closed at the 5 s deadline; the
/// daemon then idle-exits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn silent_client_dropped_at_hello_deadline() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Cached,
        Duration::from_secs(1),
    )
    .await;

    use tokio::io::AsyncReadExt as _;
    let mut raw = tokio::net::UnixStream::connect(&d.socket)
        .await
        .expect("connect");
    let mut buf = [0u8; 8];
    let read = tokio::time::timeout(TIMEOUT, raw.read(&mut buf))
        .await
        .expect("the daemon closes a silent client")
        .expect("the read succeeds");
    assert_eq!(read, 0, "the daemon closed the connection");
    tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run idle-exits after the closed connection")
        .expect("run task does not panic")
        .expect("idle exit is not an error");
}

/// An `open_ane` failure on a verified archive is fatal: the request is
/// answered `Model`, `run` returns the error and nothing is retried.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fatal_open_ane_fails_the_request() {
    let repo = fn_repo(5, SMALL_BYTES, 10);
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut backend = StubBackend::new(mpsc::channel().0);
    // The failure lands only once the request is provably parked in
    // `await_model`, so `Error { kind: Model }` is deterministic rather
    // than racing the daemon's close.
    let hold = backend.hold_open_ane();
    backend.fail_ane_open();
    let d = daemon(tmp, backend, Model::Cached, Duration::from_secs(600)).await;
    let backend = d.backend.clone();

    tokio::task::block_in_place(|| {
        hold.entered
            .recv_timeout(TIMEOUT)
            .expect("open_ane is entered")
    });
    let mut stream = stream_request(&d.socket, search(&repo.root, MARKER)).await;
    loop {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::ModelLoading) => break,
            Event::Final(f) => panic!("the request ended while the model loaded: {f:?}"),
            _ => {}
        }
    }
    tokio::task::block_in_place(|| hold.open());

    loop {
        match next_event(&mut stream).await {
            Event::Final(Final::Error(report)) => {
                assert_eq!(report.kind, ErrorKind::Model);
                break;
            }
            Event::Final(other) => panic!("expected a Model error, got {other:?}"),
            _ => {}
        }
    }
    let outcome = tokio::time::timeout(TIMEOUT, d.join)
        .await
        .expect("run returns after the fatal load failure")
        .expect("run task does not panic");
    assert!(
        matches!(outcome, Err(Error::Embedder(_))),
        "the fatal failure is run's error: {outcome:?}"
    );
    assert_eq!(backend.ane_opens(), 1, "a fatal open_ane is never retried");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_download_streams_progress() {
    let body = support::http::archive(&[("manifest.json", b"{}")]);
    // The server stalls mid-body until released, so the download state
    // exists long enough to be observed on the wire.
    let (release_tx, release) = mpsc::channel();
    let (url, server) = support::http::serve_gated(body.len() as u64, body.clone(), release);
    let tmp = tempfile::tempdir().expect("tempdir");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Served {
            url,
            sha: digest_of(&body),
        },
        Duration::from_secs(600),
    )
    .await;

    let repo = fn_repo(20, SMALL_BYTES, 10);
    let client = Client::connect(&d.socket).await.expect("connect");
    let mut responses = client
        .request(search(&repo.root, MARKER))
        .await
        .expect("request sends");
    loop {
        match next_event(&mut responses).await {
            Event::Progress(protocol::Progress::ModelDownload { .. }) => break,
            Event::Final(Final::Error(report)) => {
                panic!("the request failed before the download: {report:?}")
            }
            _ => {}
        }
    }
    let _ = release_tx.send(());
    let last = loop {
        let event = next_event(&mut responses).await;
        if matches!(event, Event::Final(_)) {
            break event;
        }
    };
    assert!(
        matches!(last, Event::Final(Final::Answer(_))),
        "the query is answered after the download"
    );
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_model_download_is_rejected() {
    let good = support::http::archive(&[("manifest.json", b"{}")]);
    let corrupt = support::http::archive(&[("manifest.json", b"corrupted")]);
    let (url, server) = support::http::serve_sequence(vec![
        ("200 OK", corrupt.len() as u64, corrupt.clone()),
        ("200 OK", corrupt.len() as u64, corrupt.clone()),
        ("200 OK", good.len() as u64, good.clone()),
    ]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("models");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Served {
            url,
            sha: digest_of(&good),
        },
        Duration::from_secs(600),
    )
    .await;

    let repo = fn_repo(20, SMALL_BYTES, 10);
    let events = answer_search(&d, &repo.root, MARKER).await;
    match final_of(&events) {
        Final::Error(report) => {
            assert_eq!(report.kind, ErrorKind::Model);
            assert!(
                report.message.contains("sha256"),
                "the error names the hash mismatch: {}",
                report.message
            );
        }
        other => panic!("expected a Model error, got {other:?}"),
    }
    assert!(
        !root.join(digest_of(&good).hex()).exists(),
        "no model directory remains"
    );

    // The next request retries and succeeds on the corrected server.
    let events = answer_search(&d, &repo.root, MARKER).await;
    assert!(
        matches!(final_of(&events), Final::Answer(_)),
        "the retry is answered"
    );
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_recreated_worktree_gets_a_fresh_index() {
    let repo = fn_repo(10, SMALL_BYTES, 10);
    let (d, _, _) = start(&repo, Duration::ZERO, Duration::ZERO, &[]).await;
    let worktree = repo::add_worktree(&repo, "linked", "first");
    let events = answer_search(&d, &worktree, MARKER).await;
    assert!(matches!(final_of(&events), Final::Answer(_)));
    assert!(worktree.join(".repo-search/index.db").exists());

    // The same path again, a different tree: the daemon must not keep
    // writing to the first tree's deleted index.
    repo::remove_worktree(&repo, &worktree);
    assert!(!worktree.exists());
    let again = repo::add_worktree(&repo, "linked", "second");
    assert_eq!(again, worktree);
    let events = answer_search(&d, &worktree, MARKER).await;
    assert!(matches!(final_of(&events), Final::Answer(_)));
    assert!(
        worktree.join(".repo-search/index.db").exists(),
        "the search indexes the new tree"
    );
    let events = request(&d.socket, Request::Status { worktree }).await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert!(status.index.is_some(), "Status reports the new index");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_does_not_create_an_index() {
    let repo = fn_repo(10, SMALL_BYTES, 10);
    let tmp = tempfile::tempdir().expect("tempdir");
    let d = daemon(
        tmp,
        StubBackend::new(mpsc::channel().0),
        Model::Cached,
        Duration::from_secs(600),
    )
    .await;

    let events = request(
        &d.socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert!(status.index.is_none(), "no index exists yet");
    assert!(
        !repo.root.join(".repo-search").exists(),
        "a Status never creates .repo-search/"
    );
}

/// Configuration handed to a re-executed `daemon_host` through
/// `REPO_SEARCH_TEST_HOST`.
#[derive(serde::Serialize, serde::Deserialize)]
struct HostConfig {
    socket: PathBuf,
    model_cache: PathBuf,
    ane_delay_ms: u64,
    gpu_delay_ms: u64,
}

/// One `RECORD <json>` line the host prints per stub record.
#[derive(Debug, serde::Deserialize)]
struct HostRecord {
    #[expect(dead_code)]
    instance: String,
    kind: String,
    hash: Option<String>,
    #[expect(dead_code)]
    tokens: usize,
    #[expect(dead_code)]
    batch_len: usize,
    #[expect(dead_code)]
    qos: String,
}

/// The ignored test re-executed as the kill target; prints `READY` once its
/// socket is bound and `RECORD <json>` per stub record on stdout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "re-executed by kill_mid_index_resumes"]
async fn daemon_host() {
    let raw = std::env::var("REPO_SEARCH_TEST_HOST")
        .expect("REPO_SEARCH_TEST_HOST is set by kill_mid_index_resumes");
    let config: HostConfig = serde_json::from_str(&raw).expect("HostConfig json");
    support::host::watch_stdin();
    let (records_tx, records) = mpsc::channel::<Record>();
    let mut backend = StubBackend::new(records_tx);
    backend.set_instance_delay(Instance::Ane, Duration::from_millis(config.ane_delay_ms));
    backend.set_instance_delay(Instance::Gpu, Duration::from_millis(config.gpu_delay_ms));
    std::thread::spawn(move || {
        for record in records {
            let json = serde_json::json!({
                "instance": format!("{:?}", record.instance),
                "kind": format!("{:?}", record.kind),
                "hash": record.hash.map(|h| h.to_string()),
                "tokens": record.tokens,
                "batch_len": record.batch_len,
                "qos": format!("{:?}", record.qos),
            });
            println!("RECORD {json}");
        }
    });
    let spec = ModelSpec {
        url: "http://127.0.0.1:1/unused".into(),
        sha256: sha_7(),
    };
    let store = ModelStore::new(config.model_cache, spec).expect("model store");
    let listener = UnixListener::bind(&config.socket).expect("socket binds");
    // The harness left its `test ... ` prefix unterminated on this line.
    println!("\nREADY");
    daemon::run(Config {
        listener,
        idle_exit: Duration::from_secs(3600),
        model_store: store,
        backend: Arc::new(backend),
    })
    .await
    .expect("the host runs");
}

/// A spawned `daemon_host`: the child, the write end of the lifeline its
/// stdin watch reads, and its records.
struct Host {
    child: std::process::Child,
    /// Held open and never written: the host exits when this end closes,
    /// however the holding process dies.
    _stdin: std::process::ChildStdin,
    records: mpsc::Receiver<HostRecord>,
}

impl Host {
    /// Spawns a host and returns once it printed `READY`: its socket is
    /// bound, so a connect cannot race the bind.
    fn spawn(config: &HostConfig) -> Self {
        let exe = std::env::current_exe().expect("the test binary");
        let mut child = std::process::Command::new(exe)
            .args([
                "--exact",
                "daemon_host",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                "REPO_SEARCH_TEST_HOST",
                serde_json::to_string(config).unwrap(),
            )
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("the daemon host spawns");
        let stdin = child.stdin.take().expect("piped");
        let (ready_tx, ready) = mpsc::channel();
        let (tx, records) = mpsc::channel();
        let stdout = child.stdout.take().expect("piped");
        let mut stdout = std::io::BufReader::new(stdout);
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut line = String::new();
            while let Ok(n) = stdout.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                let text = line.trim();
                if text == "READY" {
                    let _ = ready_tx.send(());
                } else if let Some(json) = text.strip_prefix("RECORD ") {
                    let record = serde_json::from_str::<HostRecord>(json)
                        .expect("the host prints well-formed records");
                    let _ = tx.send(record);
                }
                line.clear();
            }
        });
        tokio::task::block_in_place(|| {
            ready
                .recv_timeout(TIMEOUT)
                .expect("the host prints READY after binding");
        });
        Self {
            child,
            _stdin: stdin,
            records,
        }
    }

    fn kill(&mut self) {
        self.child.kill().expect("SIGKILL");
        self.child.wait().expect("wait");
    }

    /// Sends SIGTERM and waits for the exit status.
    fn term(&mut self) -> std::process::ExitStatus {
        let rc = unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        assert_eq!(rc, 0, "SIGTERM to the host");
        self.child.wait().expect("wait")
    }
}

impl Drop for Host {
    /// A panicking test must not leave the host running.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A re-executed helper test: the intermediate process between the test
/// and a `daemon_host`. Its stdout lines stream into `lines`, it watches
/// its own stdin lifeline through the helper it runs, and the child is
/// killed and reaped on drop.
struct Intermediate {
    child: std::process::Child,
    lines: mpsc::Receiver<String>,
}

impl Intermediate {
    /// Re-executes the test binary on the ignored `host_parent` helper
    /// with `config` in `REPO_SEARCH_TEST_HOST`: stdin piped for the
    /// helper's own stdin watch, stdout piped for its `TAG value` lines.
    fn spawn(config: &HostConfig) -> Self {
        let exe = std::env::current_exe().expect("the test binary");
        let mut child = std::process::Command::new(exe)
            .args([
                "--exact",
                "host_parent",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                "REPO_SEARCH_TEST_HOST",
                serde_json::to_string(config).unwrap(),
            )
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("the intermediate spawns");
        let (tx, lines) = mpsc::channel();
        let stdout = child.stdout.take().expect("piped");
        let mut stdout = std::io::BufReader::new(stdout);
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut line = String::new();
            while let Ok(n) = stdout.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                let _ = tx.send(line.trim_end().to_owned());
                line.clear();
            }
        });
        Self { child, lines }
    }

    /// The value after `tag` on the next stdout line carrying it.
    fn tagged(&self, tag: &str) -> String {
        loop {
            let line = self
                .lines
                .recv_timeout(TIMEOUT)
                .expect("the intermediate prints its line within the timeout");
            if let Some(value) = line.strip_prefix(tag) {
                return value.trim().to_owned();
            }
        }
    }
}

impl Drop for Intermediate {
    /// A panicking test must not leave the intermediate parked.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The ignored test re-executed as `host_exits_when_its_parent_is_killed`'s
/// intermediate: spawns a `daemon_host`, prints `PID <pid>` for the outer
/// test, then waits to be killed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "re-executed by host_exits_when_its_parent_is_killed"]
async fn host_parent() {
    support::host::watch_stdin();
    let raw = std::env::var("REPO_SEARCH_TEST_HOST")
        .expect("REPO_SEARCH_TEST_HOST is set by host_exits_when_its_parent_is_killed");
    let config: HostConfig = serde_json::from_str(&raw).expect("HostConfig json");
    let host = Host::spawn(&config);
    // The harness left its `test ... ` prefix unterminated on this line.
    println!("\nPID {}", host.child.id());
    std::future::pending::<()>().await;
}

/// Sends `request` and returns the live event stream, uncollected.
async fn stream_request(socket: &Path, request: Request) -> repo_search::client::Responses {
    let client = Client::connect(socket).await.expect("connect");
    client.request(request).await.expect("request sends")
}

/// The next frame on a live stream within `TIMEOUT`.
async fn next_event(stream: &mut repo_search::client::Responses) -> Event {
    tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("a frame arrives")
        .expect("the stream stays open")
        .expect("the frame decodes")
}

/// SIGTERM during an `index --wait`: the in-flight request is answered
/// `ShuttingDown`, the host exits 0, and every vector a `Coverage` frame
/// reported is committed in the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sigterm_answers_in_flight_and_flushes() {
    let repo = fn_repo(400, SMALL_BYTES, 20);
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let spec = ModelSpec {
        url: "http://127.0.0.1:1/unused".into(),
        sha256: sha_7(),
    };
    let store = ModelStore::new(model_cache.clone(), spec).expect("model store");
    std::fs::create_dir_all(store.dir()).expect("cached model dir");
    let model_id = store.model_id();

    let mut host = Host::spawn(&HostConfig {
        socket: tmp.path().join("daemon.sock"),
        model_cache,
        ane_delay_ms: 5,
        gpu_delay_ms: 0,
    });
    let socket = tmp.path().join("daemon.sock");
    let mut stream = stream_request(
        &socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    // `Coverage` reports only committed vectors: what the host says is
    // cached is on disk.
    let mut committed = 0;
    while committed == 0 {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::Coverage { with_vector, .. }) => {
                committed = with_vector;
            }
            Event::Final(f) => panic!("the index wait ended early: {f:?}"),
            _ => {}
        }
    }

    let status = host.term();
    // Drain what the host wrote before exiting: more coverage may have
    // landed, and the request ends in `ShuttingDown`.
    let mut last = None;
    let report = loop {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::Coverage { with_vector, .. }) => {
                last = Some(with_vector)
            }
            Event::Final(Final::Error(report)) => break report,
            Event::Final(other) => panic!("expected ShuttingDown, got {other:?}"),
            _ => {}
        }
    };
    assert_eq!(report.kind, ErrorKind::ShuttingDown);
    assert!(status.success(), "the host exits 0 on SIGTERM: {status}");

    let committed = last.unwrap_or(committed);
    let index = Index::open(&repo.root).await.expect("index opens");
    let hashes = index.chunk_hashes(true).await.expect("chunk hashes");
    let cache = VectorCache::open(&repo.common_dir)
        .await
        .expect("cache opens");
    let cached = cache
        .cached(&model_id, &hashes)
        .await
        .expect("cached reads");
    assert!(
        cached.len() >= committed as usize,
        "every reported vector is committed: {} < {committed}",
        cached.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill_mid_index_resumes() {
    let repo = fn_repo(1200, SMALL_BYTES, 20);
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let spec = ModelSpec {
        url: "http://127.0.0.1:1/unused".into(),
        sha256: sha_7(),
    };
    let store = ModelStore::new(model_cache.clone(), spec).expect("model store");
    std::fs::create_dir_all(store.dir()).expect("cached model dir");
    let model_id = store.model_id();
    let host_config = |sock: &str| HostConfig {
        socket: tmp.path().join(sock),
        model_cache: model_cache.clone(),
        ane_delay_ms: 5,
        gpu_delay_ms: 0,
    };

    // First host: `index --wait` streams coverage, which is measured from
    // the committed cache; kill it once 500 chunks are committed.
    let mut host = Host::spawn(&host_config("daemon1.sock"));
    let socket = tmp.path().join("daemon1.sock");
    let mut stream = stream_request(
        &socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    let mut coverage = 0;
    while coverage < 500 {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::Coverage { with_vector, .. }) => {
                coverage = with_vector;
            }
            Event::Final(f) => panic!("the index wait ended early: {f:?}"),
            _ => {}
        }
    }
    drop(stream);
    host.kill();

    // Whatever the writer's close-flush could not reach is simply absent;
    // what coverage reported as committed is not.
    let index = Index::open(&repo.root).await.expect("index opens");
    let hashes = index.chunk_hashes(true).await.expect("chunk hashes");
    let cache = VectorCache::open(&repo.common_dir)
        .await
        .expect("cache opens");
    let cached = cache
        .cached(&model_id, &hashes)
        .await
        .expect("cached reads");
    assert!(cached.len() >= 500, "{} committed vectors", cached.len());
    let cached_names: HashSet<String> = cached.iter().map(|h| h.to_string()).collect();
    let remaining = hashes.len() - cached.len();

    // Second host on a fresh socket path.
    let host = Host::spawn(&host_config("daemon2.sock"));
    let socket = tmp.path().join("daemon2.sock");
    let events = request(
        &socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::FirstIndex,
        "the kill left the phase at first-index"
    );

    // The resumed pass starts at the committed count and runs to the end.
    let mut stream = stream_request(
        &socket,
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    let mut first: Option<u64> = None;
    let indexed = loop {
        match next_event(&mut stream).await {
            Event::Progress(protocol::Progress::Coverage { with_vector, .. }) => {
                if first.is_none() {
                    assert!(
                        with_vector >= cached.len() as u64,
                        "the resumed coverage starts at the committed count: {with_vector} < {}",
                        cached.len()
                    );
                    first = Some(with_vector);
                }
            }
            Event::Final(Final::Indexed { with_vector, total }) => break (with_vector, total),
            Event::Final(other) => panic!("the index wait failed: {other:?}"),
            _ => {}
        }
    };
    assert_eq!(indexed.0, indexed.1, "the resumed pass completes coverage");
    let events = request(
        &socket,
        Request::Status {
            worktree: repo.root.clone(),
        },
    )
    .await;
    let status = match final_of(&events) {
        Final::Status(status) => status,
        other => panic!("expected a Status, got {other:?}"),
    };
    assert_eq!(
        status.index.as_ref().expect("indexed").phase,
        WirePhase::Maintained,
        "the resumed stage 1 reached Stage1Done"
    );

    // Exactly the uncommitted chunks were embedded: the host prints its
    // records in order, so reading until all of them arrived sees every
    // earlier record, and none of them is a committed one.
    let mut embedded = 0;
    while embedded < remaining {
        let record = tokio::task::block_in_place(|| {
            host.records
                .recv_timeout(TIMEOUT)
                .expect("a record for every uncommitted chunk")
        });
        if record.kind == "Index" {
            let hash = record.hash.expect("an index record names its chunk");
            assert!(
                !cached_names.contains(&hash),
                "a committed vector was re-embedded"
            );
            embedded += 1;
        }
    }
}

/// The stdin watch ends a host whose parent dies without unwinding: the
/// intermediate is SIGKILLed and its `daemon_host` — which the lifeline
/// alone can now stop — exits on its own.
#[test]
fn host_exits_when_its_parent_is_killed() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let spec = ModelSpec {
        url: "http://127.0.0.1:1/unused".into(),
        sha256: sha_7(),
    };
    let store = ModelStore::new(model_cache.clone(), spec).expect("model store");
    std::fs::create_dir_all(store.dir()).expect("cached model dir");
    let mut parent = Intermediate::spawn(&HostConfig {
        socket: tmp.path().join("daemon.sock"),
        model_cache,
        ane_delay_ms: 0,
        gpu_delay_ms: 0,
    });
    let pid: u32 = parent
        .tagged("PID")
        .parse()
        .expect("the intermediate prints the host's pid");

    parent.child.kill().expect("SIGKILL");
    parent.child.wait().expect("wait");
    assert!(
        support::host::wait_for_exit(pid, Duration::from_secs(30)),
        "the host outlived a SIGKILLed parent"
    );
}
