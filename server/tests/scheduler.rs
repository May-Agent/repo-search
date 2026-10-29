//! The hardware scheduler's tests, driven by `Submit`.

use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use repo_search_core::embed_api::Urgency;
use repo_search_core::qos::{self, QosClass};
use repo_search_coreml::GpuBlocked;
use repo_search_server::ane;
use repo_search_server::backend::Backend;
use repo_search_server::chip::ChipClass;
use repo_search_server::scheduler::{
    Answered, GpuState, RequestId, SchedulerConfig, SchedulerHandle, Submit, spawn,
};
use repo_search_server::stub::{
    Instance, Kind, Record, StubBackend, StubMonitor, bag_of_ids, bucket, lift_ane,
};
use tokio::sync::{mpsc as tokio_mpsc, oneshot, watch};

/// A started pipeline: stub backend, ANE thread and scheduler.
struct Rig {
    backend: Arc<StubBackend>,
    handle: SchedulerHandle,
    records: mpsc::Receiver<Record>,
    gpu_state: watch::Receiver<GpuState>,
    fatal: tokio_mpsc::Receiver<String>,
    next_request: u64,
    _gpu_exited: oneshot::Receiver<Vec<oneshot::Receiver<()>>>,
    _monitor: StubMonitor,
}

const TIMEOUT: Duration = Duration::from_secs(30);

fn recv(rig: &Rig, what: &str) -> Record {
    rig.records
        .recv_timeout(TIMEOUT)
        .unwrap_or_else(|e| panic!("a record for {what}: {e}"))
}

/// Drains records until `matches` fires; panics on timeout.
fn wait_for(rig: &Rig, what: &str, mut matches: impl FnMut(&Record) -> bool) -> Vec<Record> {
    let mut log = Vec::new();
    loop {
        let record = recv(rig, what);
        let done = matches(&record);
        log.push(record);
        if done {
            return log;
        }
        if log.len() > 1000 {
            panic!("timed out waiting for {what}");
        }
    }
}

async fn rig(chip: ChipClass, ane_delay: Duration, gpu_delay: Duration) -> Rig {
    rig_gated(chip, ane_delay, gpu_delay, |_| ()).await.0
}

async fn rig_gated<G>(
    chip: ChipClass,
    ane_delay: Duration,
    gpu_delay: Duration,
    gates: impl FnOnce(&mut StubBackend) -> G,
) -> (Rig, G) {
    let (records_tx, records) = mpsc::channel();
    let mut backend = StubBackend::new(records_tx);
    backend.set_instance_delay(Instance::Ane, ane_delay);
    backend.set_instance_delay(Instance::Gpu, gpu_delay);
    let g = gates(&mut backend);
    let backend = Arc::new(backend);
    let ane = ane::spawn(backend.clone(), std::env::temp_dir());
    let (_, buckets) = ane
        .encoded
        .await
        .expect("encoder oneshot")
        .expect("load_encoder");
    ane.loaded.await.expect("loaded oneshot").expect("open_ane");
    let (headroom_tx, headroom_rx) = watch::channel(Err(GpuBlocked::PowerSourceUnknown));
    let monitor = backend.watch_headroom(Box::new(move |h| {
        let _ = headroom_tx.send_replace(h);
    }));
    let (fatal_tx, fatal) = tokio_mpsc::channel(4);
    let (busy, _busy_rx) = watch::channel(false);
    let (gpu_state_tx, gpu_state) = watch::channel(GpuState::Absent { blocked: None });
    let (handle, gpu_exited) = spawn(SchedulerConfig::<StubBackend> {
        backend: backend.clone(),
        dir: std::env::temp_dir(),
        buckets,
        headroom: headroom_rx,
        chip,
        fatal: fatal_tx,
        busy,
        ane_exited: ane.exited.clone(),
        gpu_state: gpu_state_tx,
    });
    let _ = ane.sched.send(handle.tx());
    (
        Rig {
            backend,
            handle,
            records,
            gpu_state,
            fatal,
            next_request: 0,
            _gpu_exited: gpu_exited,
            _monitor: monitor,
        },
        g,
    )
}

/// Submits `inputs` at `urgency` as a fresh request.
async fn submit(
    rig: &mut Rig,
    urgency: u8,
    inputs: Vec<Vec<u32>>,
) -> (RequestId, oneshot::Receiver<Result<Answered, String>>) {
    rig.next_request += 1;
    let request = RequestId(rig.next_request);
    let (reply, rx) = oneshot::channel();
    rig.handle
        .submit(Submit {
            request,
            urgency: Urgency::new(urgency),
            inputs,
            reply,
        })
        .await
        .expect("submit");
    (request, rx)
}

/// `seed` fills `tokens` ids — a distinguishable input.
fn ids(seed: u32, tokens: usize) -> Vec<u32> {
    vec![seed; tokens]
}

/// Opens a barrier gate without blocking the current-thread runtime.
fn open(gate: &Arc<Barrier>) {
    let gate = gate.clone();
    std::thread::spawn(move || gate.wait());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn query_preempts_indexing() {
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first(Instance::Ane)
    })
    .await;
    let (_, first) = submit(&mut rig, 7, vec![ids(1, 100)]).await;
    // The background item is inside the gated call; the next pull is queued.
    wait_for(&rig, "the gated call", |r| r.kind == Kind::Started);
    let (_, urgent) = submit(&mut rig, 0, vec![ids(2, 100)]).await;
    let (_, later) = submit(&mut rig, 7, vec![ids(3, 100)]).await;
    open(&gate);
    first.await.expect("answered").expect("ok");
    // The u=0 item runs before the second background one.
    let log = wait_for(&rig, "the interactive embed", |r| {
        r.kind == Kind::Interactive
    });
    let interactive = log.iter().position(|r| r.ids == ids(2, 100));
    assert!(interactive.is_some(), "{log:?}");
    let later_index = log
        .iter()
        .position(|r| r.ids == ids(3, 100) && r.kind == Kind::Index);
    assert!(
        later_index.is_none_or(|i| i > interactive.unwrap()),
        "{log:?}"
    );
    later.await.expect("answered").expect("ok");
    urgent.await.expect("answered").expect("ok");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gpu_takes_the_long_end_in_same_bucket_batches() {
    // Original's mix: 6 long, 30 middle, 290 short items in the GPU's
    // queue; the ANE works the short end at 20ms, the GPU at 10ms.
    let mut rig = rig(
        ChipClass::Wide,
        Duration::from_millis(20),
        Duration::from_millis(10),
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut inputs = Vec::new();
    for i in 0..326u32 {
        let tokens = match i {
            0..=5 => 500,
            6..=35 => 200,
            _ => 100,
        };
        inputs.push(ids(i, tokens));
    }
    let (_, rx) = submit(&mut rig, 3, inputs.clone()).await;
    // Collect until the GPU drops — its queue drained — and the request
    // answers, then drain whatever records remain.
    let mut log = wait_for(&rig, "the GPU drop", |r| r.kind == Kind::Dropped);
    rx.await.expect("answered").expect("ok");
    while let Ok(record) = rig.records.try_recv() {
        log.push(record);
    }

    let gpu: Vec<&Record> = log.iter().filter(|r| r.instance == Instance::Gpu).collect();
    assert!(gpu.iter().any(|r| r.kind == Kind::Index));
    for record in &gpu {
        assert!(record.batch_len <= 8, "batches are at most 8 inputs");
        if record.kind == Kind::Index {
            assert_eq!(record.qos, QosClass::UserInitiated);
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
        assert!(
            batch
                .iter()
                .all(|r| bucket(r.tokens) == bucket(batch[0].tokens)),
            "a GPU batch mixes token buckets"
        );
    }
    assert!(
        batches[0].iter().all(|r| r.tokens == 500),
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
        "ANE tokens are non-decreasing"
    );
    // No input is ever embedded twice.
    let mut counts: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
    for record in log.iter().filter(|r| r.kind == Kind::Index) {
        *counts.entry(record.ids.clone()).or_default() += 1;
    }
    assert!(
        counts.values().all(|&c| c == 1),
        "every input embedded exactly once"
    );
    assert_eq!(counts.len(), 326);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_base_chip_keeps_the_gpu_off_stage_1() {
    // Normal and background submitted together on a base chip: the GPU
    // opens only once the normal queue drains — it never processes normal
    // work.
    let mut rig = rig(ChipClass::Base, Duration::from_millis(10), Duration::ZERO).await;
    rig.backend.push_headroom(Ok(()));
    let normal_ids: Vec<Vec<u32>> = (0..30u32).map(|i| ids(i, 100)).collect();
    let background_ids: Vec<Vec<u32>> = (30..100u32).map(|i| ids(i, 500)).collect();
    let (_, normal) = submit(&mut rig, 3, normal_ids.clone()).await;
    let (_, background) = submit(&mut rig, 7, background_ids.clone()).await;
    let mut log = wait_for(&rig, "the GPU embed", |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Index
    });
    normal.await.expect("answered").expect("ok");
    background.await.expect("answered").expect("ok");
    while let Ok(record) = rig.records.try_recv() {
        log.push(record);
    }
    assert_eq!(rig.backend.gpu_opens(), 1, "the GPU opened once");
    // Every normal input embedded on the ANE, exactly once, at
    // user-initiated QoS.
    for input in &normal_ids {
        let hits: Vec<&Record> = log
            .iter()
            .filter(|r| r.kind == Kind::Index && r.ids == *input)
            .collect();
        assert_eq!(hits.len(), 1, "a normal item embedded {} times", hits.len());
        assert_eq!(hits[0].instance, Instance::Ane);
        assert_eq!(hits[0].qos, QosClass::UserInitiated);
    }
    // The GPU only ever embedded background inputs.
    for record in log
        .iter()
        .filter(|r| r.instance == Instance::Gpu && r.kind == Kind::Index)
    {
        assert!(
            background_ids.contains(&record.ids),
            "the GPU embedded a normal input: {:?}",
            record.ids
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_gpu_on_battery() {
    let mut rig = rig(ChipClass::Wide, Duration::ZERO, Duration::ZERO).await;
    let (_, rx) = submit(&mut rig, 3, vec![ids(1, 100), ids(2, 100)]).await;
    rx.await.expect("answered").expect("ok");
    assert_eq!(rig.backend.gpu_opens(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gpu_released_on_power_loss() {
    // Both first calls are gated: a GPU batch and an ANE item are provably
    // in flight, so the power loss — not an empty queue — drops the GPU.
    let (mut rig, (ane_gate, gpu_gate)) = rig_gated(
        ChipClass::Wide,
        Duration::from_millis(10),
        Duration::from_millis(25),
        |b| (b.gate_first(Instance::Ane), b.gate_first(Instance::Gpu)),
    )
    .await;
    rig.backend.push_headroom(Ok(()));
    let mut inputs: Vec<Vec<u32>> = (0..124u32)
        .map(|i| ids(i, if i < 24 { 500 } else { 100 }))
        .collect();
    let (_, rx) = submit(&mut rig, 3, inputs.split_off(0)).await;
    // The queue holds 124 - 8 - 1 items with both calls in flight; either
    // side's `Started` may arrive first, so collect until both are in.
    let mut log: Vec<Record> = Vec::new();
    while !(log
        .iter()
        .any(|r| r.instance == Instance::Gpu && r.kind == Kind::Started)
        && log
            .iter()
            .any(|r| r.instance == Instance::Ane && r.kind == Kind::Started))
    {
        log.push(recv(&rig, "both gated calls"));
    }
    rig.backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    // The batch finishes and the next pull is answered `Stop`.
    open(&gpu_gate);
    let gpu_index = |log: &[Record]| {
        log.iter()
            .filter(|r| r.instance == Instance::Gpu && r.kind == Kind::Index)
            .count()
    };
    // The A3 watch reports Loading then Running on the way here.
    let mut gpu_state = rig.gpu_state.clone();
    let drop_log = wait_for(&rig, "the GPU drop", |r| r.kind == Kind::Dropped);
    let drop_pos = log.len() + drop_log.len() - 1;
    log.extend(drop_log);
    assert_eq!(
        gpu_index(&log),
        8,
        "the GPU embedded exactly its gated batch"
    );
    // The `GpuGone` report turns the A3 watch Absent.
    tokio::time::timeout(Duration::from_secs(10), async {
        while !matches!(*gpu_state.borrow(), GpuState::Absent { .. }) {
            gpu_state.changed().await.expect("gpu state watch");
        }
    })
    .await
    .expect("the GPU state went Absent");
    // Indexing continues on the ANE until the request answers.
    open(&ane_gate);
    rx.await.expect("answered").expect("ok");
    while let Ok(record) = rig.records.try_recv() {
        log.push(record);
    }
    assert!(
        !log[drop_pos + 1..]
            .iter()
            .any(|r| r.instance == Instance::Gpu && r.kind == Kind::Index),
        "no GPU embed after the drop"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stage_2_runs_at_background_qos() {
    let mut rig = rig(ChipClass::Wide, Duration::ZERO, Duration::ZERO).await;
    let (_, rx) = submit(&mut rig, 7, vec![ids(1, 100)]).await;
    let log = wait_for(&rig, "the background embed", |r| {
        r.instance == Instance::Ane && r.kind == Kind::Index
    });
    assert_eq!(log.last().unwrap().qos, QosClass::Background);
    rx.await.expect("answered").expect("ok");
    // Normal work runs at user-initiated.
    let (_, rx) = submit(&mut rig, 3, vec![ids(2, 100)]).await;
    let log = wait_for(&rig, "the normal embed", |r| {
        r.instance == Instance::Ane && r.ids == ids(2, 100) && r.kind == Kind::Index
    });
    assert_eq!(log.last().unwrap().qos, QosClass::UserInitiated);
    rx.await.expect("answered").expect("ok");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn query_lifts_an_in_flight_background_chunk() {
    // The ANE's Index calls are gated on an armed IndexGate, so a
    // background item is provably in flight when the u=0 submit lands. The
    // test's own Utility override marks this host's lifted band.
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first_index(Instance::Ane)
    })
    .await;
    gate.arm();
    let lift = lift_ane(&rig.backend, QosClass::Utility);
    let (_, first) = submit(&mut rig, 7, vec![ids(1, 100), ids(4, 100)]).await;
    wait_for(&rig, "the held call", |r| r.kind == Kind::Held);
    let band = qos::scheduling_priority(rig.backend.ane_thread());
    let (_, urgent) = submit(&mut rig, 0, vec![ids(2, 100)]).await;
    // The inbox is FIFO: a dummy request cancelled after the submit is
    // resolved only once the submit's on_submit — and with it the override
    // — has been handled.
    let (dummy, dummy_rx) = submit(&mut rig, 7, vec![ids(9, 100)]).await;
    rig.handle.cancel(dummy).await.expect("cancel");
    dummy_rx
        .await
        .expect_err("a cancelled request's reply is dropped");
    // `pth_curpri` sits at the same lifted band under any override while
    // the thread waits, so the lift cannot be told apart by reading up:
    // end the test's override instead — what still holds the band is
    // provably the scheduler's UserInteractive override.
    drop(lift);
    assert_eq!(
        qos::scheduling_priority(rig.backend.ane_thread()),
        band,
        "the scheduler's override still holds the thread at the lifted band"
    );
    let _lift = lift_ane(&rig.backend, QosClass::Utility);
    gate.open();
    first.await.expect("answered").expect("ok");
    urgent.await.expect("answered").expect("ok");
    // The held item and the next one ran at the same requested class: the
    // override released with the pull between them.
    let log = wait_for(&rig, "the second background embed", |r| {
        r.kind == Kind::Index && r.ids == ids(4, 100)
    });
    assert_eq!(
        log.iter()
            .filter(|r| r.kind == Kind::Index)
            .map(|r| r.qos)
            .collect::<Vec<_>>(),
        vec![QosClass::Background; 2],
        "background items run at background QoS, override or not"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn query_after_a_fatal_error_is_answered_and_the_scheduler_exits() {
    // The ANE's first call is gated so work stays pending when the GPU's
    // open fails: item one sits on the thread, item two in the normal
    // queue, and the GPU wants that queue.
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first(Instance::Ane)
    })
    .await;
    rig.backend.fail_gpu_open();
    // The GPU stays blocked until the ANE's pull is provably holding the
    // first item: only then does the push give the GPU its queue, so the
    // failed open races nothing.
    let (_, rx) = submit(&mut rig, 3, vec![ids(1, 100), ids(1, 101)]).await;
    wait_for(&rig, "the gated call", |r| r.kind == Kind::Started);
    let (_, rx2) = submit(&mut rig, 0, vec![ids(2, 100)]).await;
    rig.backend.push_headroom(Ok(()));
    // The fatal signal lands once the GPU's open failure is processed:
    // every pending request is already answered `Err`.
    let fatal = rig.fatal.recv().await.expect("fatal report");
    assert!(fatal.contains("injected GPU open failure"), "{fatal}");
    for rx in [rx, rx2] {
        let err = rx.await.expect("answered").expect_err("fatal");
        assert!(err.contains("injected GPU open failure"), "{err}");
    }
    // A late submit never embeds: either the scheduler answers it with the
    // fatal error, or it is already gone and the reply channel drops.
    rig.next_request += 1;
    let (reply, rx3) = oneshot::channel();
    let _ = rig
        .handle
        .submit(Submit {
            request: RequestId(rig.next_request),
            urgency: Urgency::new(0),
            inputs: vec![ids(3, 100)],
            reply,
        })
        .await;
    let late = rx3.await;
    assert!(
        !matches!(late, Ok(Ok(_))),
        "a late submit is never embedded"
    );
    // The gated item's vector is dropped; the scheduler then exits.
    open(&gate);
    rig.handle.tx().closed().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancelled_request_leaves_the_queue_unembedded() {
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first(Instance::Ane)
    })
    .await;
    let (_, first) = submit(&mut rig, 7, vec![ids(1, 100)]).await;
    wait_for(&rig, "the gated call", |r| r.kind == Kind::Started);
    let (request, second) = submit(&mut rig, 7, vec![ids(2, 100), ids(3, 100)]).await;
    let (_, third) = submit(&mut rig, 7, vec![ids(4, 100)]).await;
    rig.handle.cancel(request).await.expect("cancel");
    // Cancelling drops the request's reply; the receiver resolves `Err`
    // once the cancel is processed, and none of its items ever embed.
    second
        .await
        .expect_err("a cancelled request's reply is dropped");
    open(&gate);
    first.await.expect("answered").expect("ok");
    third.await.expect("answered").expect("ok");
    let (_, fourth) = submit(&mut rig, 7, vec![ids(5, 100)]).await;
    fourth.await.expect("answered").expect("ok");
    let log = wait_for(&rig, "the last embed", |r| {
        r.ids == ids(5, 100) && r.kind == Kind::Index
    });
    assert!(
        !log.iter()
            .any(|r| r.ids == ids(2, 100) || r.ids == ids(3, 100)),
        "cancelled items never embed: {log:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_closed_reply_is_purged_at_pop() {
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first(Instance::Ane)
    })
    .await;
    let (_, first) = submit(&mut rig, 7, vec![ids(1, 100)]).await;
    wait_for(&rig, "the gated call", |r| r.kind == Kind::Started);
    let (_, closed) = submit(&mut rig, 7, vec![ids(2, 100)]).await;
    drop(closed);
    let (_, third) = submit(&mut rig, 7, vec![ids(3, 100)]).await;
    open(&gate);
    first.await.expect("answered").expect("ok");
    third.await.expect("answered").expect("ok");
    let log = wait_for(&rig, "the last embed", |r| {
        r.ids == ids(3, 100) && r.kind == Kind::Index
    });
    assert!(
        !log.iter().any(|r| r.ids == ids(2, 100)),
        "a closed reply's item is purged: {log:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_with_mixed_buckets_answers_in_input_order() {
    let mut rig = rig(ChipClass::Wide, Duration::ZERO, Duration::ZERO).await;
    rig.backend.push_headroom(Ok(()));
    let inputs = vec![ids(1, 100), ids(2, 400), ids(3, 200)];
    let (_, rx) = submit(&mut rig, 3, inputs.clone()).await;
    let answered = rx.await.expect("answered").expect("ok");
    assert_eq!(answered.vectors.len(), 3);
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(
            answered.vectors[i],
            bag_of_ids(input, 64),
            "vector {i} is not the embedding of its own input"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn u0_items_run_fifo_and_never_on_the_gpu() {
    // The ANE's first call is gated: the GPU runs on the normal queue while
    // the u=0 items queue up on the ANE in submission order.
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first(Instance::Ane)
    })
    .await;
    rig.backend.push_headroom(Ok(()));
    let (_, normal) = submit(&mut rig, 3, vec![ids(9, 400); 12]).await;
    let (_, first) = submit(&mut rig, 7, vec![ids(8, 100)]).await;
    wait_for(&rig, "the gated call", |r| r.kind == Kind::Started);
    wait_for(&rig, "the GPU running", |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Index
    });
    let (_, r1) = submit(&mut rig, 0, vec![ids(1, 100)]).await;
    let (_, r2) = submit(&mut rig, 0, vec![ids(2, 100)]).await;
    let (_, r3) = submit(&mut rig, 0, vec![ids(3, 100)]).await;
    open(&gate);
    r1.await.expect("answered").expect("ok");
    r2.await.expect("answered").expect("ok");
    r3.await.expect("answered").expect("ok");
    first.await.expect("answered").expect("ok");
    normal.await.expect("answered").expect("ok");
    let mut log = wait_for(&rig, "the third interactive embed", |r| {
        r.kind == Kind::Interactive && r.ids == ids(3, 100)
    });
    while let Ok(record) = rig.records.try_recv() {
        log.push(record);
    }
    let order: Vec<Vec<u32>> = log
        .iter()
        .filter(|r| r.kind == Kind::Interactive)
        .map(|r| r.ids.clone())
        .collect();
    assert_eq!(order, vec![ids(1, 100), ids(2, 100), ids(3, 100)]);
    // The GPU ran — and never once took a u=0 item.
    assert!(
        log.iter()
            .any(|r| r.instance == Instance::Gpu && r.kind == Kind::Index),
        "the GPU ran"
    );
    assert!(!log.iter().any(|r| {
        r.instance == Instance::Gpu
            && (r.ids == ids(1, 100) || r.ids == ids(2, 100) || r.ids == ids(3, 100))
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wait_and_overtaken_are_reported_per_request() {
    let (mut rig, gate) = rig_gated(ChipClass::Wide, Duration::ZERO, Duration::ZERO, |b| {
        b.gate_first_index(Instance::Ane)
    })
    .await;
    gate.arm();
    let (_, first) = submit(&mut rig, 7, vec![ids(1, 100)]).await;
    wait_for(&rig, "the held call", |r| r.kind == Kind::Held);
    let (_, urgent) = submit(&mut rig, 0, vec![ids(2, 100)]).await;
    // The held item is on the ANE, so the urgent one is the only queued item.
    let mut queued = rig.handle.queued();
    tokio::time::timeout(TIMEOUT, queued.wait_for(|&n| n == 1))
        .await
        .expect("the urgent item queues")
        .expect("the scheduler runs");
    gate.open();
    first.await.expect("answered").expect("ok");
    let answered = urgent.await.expect("answered").expect("ok");
    assert!(
        answered.timing.wait > Duration::ZERO,
        "the u=0 item waited on the held background item: {:?}",
        answered.timing
    );
    assert_eq!(answered.timing.overtaken, 0);
}
