//! Scheduler-level tests of `src/daemon/scheduler_http.rs` against the
//! gated, recording `FakeEmbeddings`, plus `ServerClient` unit tests
//! against a hyper test server.
//!
//! These are the `tests/scheduler.rs` cases ported to the HTTP boundary.
//! The in-flight window — 8 short-end and 24 long-end flights — replaces
//! the ANE/GPU pulls, so every queued item dispatches while the enqueue
//! is handled until the window fills: an item the originals saw queued
//! behind a gated query is in flight here, held at its gate. A search's
//! `Documents` job embeds its chunks in one `u=0` request, aborting the
//! flight of a chunk it takes, and a `Background`→`Stage1` flip of an
//! in-flight item aborts and requeues it; a stale completion is ignored.
//! Counts read right after a message are exact: the scheduler handles its
//! inbox in order and refills the window before the next message. Every
//! wait runs on gates, records, scheduler state or the busy report —
//! never a sleep.

#![cfg(target_os = "macos")]

#[path = "support/fake_embeddings.rs"]
mod fake_embeddings;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use fake_embeddings::{FakeEmbeddings, Gate, Kind, Record, StubQueries};
use repo_search::daemon::scheduler::{
    self, Counts, DocumentsJob, EmbedError, Enqueue, IndexClass, Interest, QueryJob, RepoId,
    SchedulerHandle, SchedulerMsg, Stage1Done, WorktreeId,
};
use repo_search::daemon::server::{Embeddings, Endpoint, ServerClient};
use repo_search::daemon::{Error, writer};
use repo_search_core::embed_api::{Urgency, encode_vector};
use repo_search_core::reconcile::PendingChunk;
use repo_search_core::store::{ContentHash, VectorCache};
use repo_search_core::{QueryTask, UnitVector};
use tokio::sync::oneshot;

const WORKTREE: WorktreeId = WorktreeId(0);
const OTHER: WorktreeId = WorktreeId(1);
const REPO: RepoId = RepoId(0);
const TIMEOUT: Duration = Duration::from_secs(30);

struct Rig {
    sched: SchedulerHandle,
    writer: writer::WriterHandle,
    cache: VectorCache,
    fatal: tokio::sync::mpsc::Receiver<Error>,
    /// The scheduler's busy report to the idle-exit tracker.
    busy: tokio::sync::watch::Receiver<bool>,
    _writer_task: tokio::task::JoinHandle<()>,
    _tmp: tempfile::TempDir,
}

/// A scheduler and its repository writer against `embeddings`.
async fn rig<E: Embeddings>(embeddings: Arc<E>) -> Rig {
    let tmp = tempfile::tempdir().expect("tempdir");
    let git = tmp.path().join("git");
    std::fs::create_dir_all(&git).expect("git dir");
    let (fatal_tx, fatal) = tokio::sync::mpsc::channel(8);
    let (busy_tx, busy) = tokio::sync::watch::channel(false);
    let sched = scheduler::spawn(scheduler::SchedulerConfig {
        embeddings,
        query_input: Arc::new(StubQueries),
        fatal: fatal_tx.clone(),
        busy: busy_tx,
    });
    let state = repo_search_core::StateDir::new(tmp.path().join("state"));
    let cache = VectorCache::open(&state, &git)
        .await
        .expect("vector cache opens");
    let (writer_busy_tx, _writer_busy_rx) = tokio::sync::watch::channel(false);
    let (writer, writer_task) = writer::spawn(
        cache.clone(),
        "stub-model".to_owned(),
        fatal_tx,
        writer_busy_tx,
    );
    Rig {
        sched,
        writer,
        cache,
        fatal,
        busy,
        _writer_task: writer_task,
        _tmp: tmp,
    }
}

/// A rig whose scheduler drives `fake`; the test keeps the receiver of
/// the channel the fake was built on.
async fn fake_rig(fake: FakeEmbeddings) -> Rig {
    rig(Arc::new(fake)).await
}

/// A pending chunk whose stub token count is exactly `tokens`.
fn pending(seq: u64, tokens: u32) -> scheduler::PendingItem {
    let title = format!("t{seq}");
    let width = 4 * (tokens as usize - 1) - title.len();
    let text = format!("{seq:0>width$}");
    scheduler::PendingItem {
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

/// The content hash of `pending(seq, tokens)`.
fn hash(seq: u64, tokens: u32) -> ContentHash {
    pending(seq, tokens).chunk.content_hash
}

/// `seqs` as 100-token pending chunks.
fn small(seqs: impl IntoIterator<Item = u64>) -> Vec<scheduler::PendingItem> {
    seqs.into_iter().map(|seq| pending(seq, 100)).collect()
}

/// A gate on each of `seqs` as 100-token chunks.
fn gates(fake: &mut FakeEmbeddings, seqs: impl IntoIterator<Item = u64>) -> Vec<Gate> {
    seqs.into_iter()
        .map(|seq| fake.gate(hash(seq, 100)))
        .collect()
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
/// barrier of `REPO` open around it. A `Change` keeps the stage-1 channel
/// of the worktree's last `Replace`.
async fn send_change(
    rig: &Rig,
    worktree: WorktreeId,
    added: Vec<scheduler::PendingItem>,
    dropped: Vec<ContentHash>,
    class: IndexClass,
) {
    let (done_tx, _done_rx) = tokio::sync::mpsc::channel(8);
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
}

/// Sends one pass's changes for `worktree` as `class`, inside a barrier of
/// its own, as the feeder sends it.
async fn change_for(
    rig: &Rig,
    worktree: WorktreeId,
    added: Vec<scheduler::PendingItem>,
    dropped: Vec<ContentHash>,
    class: IndexClass,
) {
    rig.sched.barrier(REPO, true).await;
    send_change(rig, worktree, added, dropped, class).await;
    rig.sched.barrier(REPO, false).await;
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

/// Waits until `worktree`'s counts are `want`, polling them: holding a
/// `watch_counts` would attach an `index --wait` to the worktree and
/// lift its background work to stage-1 urgency.
async fn await_counts(rig: &Rig, worktree: WorktreeId, want: Counts) {
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if counts_of(rig, worktree).await == want {
                return;
            }
        }
    })
    .await
    .expect("counts reach the expected value within the timeout");
}

/// Waits for the next `Stage1Done` on `done`.
async fn stage1(done: &mut tokio::sync::mpsc::Receiver<Stage1Done>) {
    tokio::time::timeout(TIMEOUT, done.recv())
        .await
        .expect("stage 1 completes")
        .expect("stage-1 channel open");
}

/// Waits until the scheduler reports no work: every request it sent has
/// landed and been handled.
async fn await_idle(rig: &Rig) {
    let mut busy = rig.busy.clone();
    tokio::time::timeout(TIMEOUT, busy.wait_for(|busy| !*busy))
        .await
        .expect("the scheduler goes idle")
        .expect("the busy report stays open");
}

/// Receives the next record within `TIMEOUT`, off the async executor, and
/// appends it to `log`.
fn recv(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) {
    tokio::task::block_in_place(|| {
        log.push(
            records
                .recv_timeout(TIMEOUT)
                .expect("a record must arrive within the timeout"),
        );
    });
}

/// Collects records until `pred` holds for one; returns its index in
/// `log`.
fn wait_for(
    log: &mut Vec<Record>,
    records: &mpsc::Receiver<Record>,
    pred: impl Fn(&Record) -> bool,
) -> usize {
    loop {
        if let Some(i) = log.iter().position(&pred) {
            return i;
        }
        recv(log, records);
    }
}

/// Collects records until a request of `hash` at `urgency` was dropped
/// unanswered: its aborted task is gone and can no longer pass a gate.
fn wait_cancelled(
    log: &mut Vec<Record>,
    records: &mpsc::Receiver<Record>,
    hash: ContentHash,
    urgency: u8,
) {
    wait_for(log, records, |r| {
        r.kind == Kind::Cancelled && r.hash == Some(hash) && r.urgency.get() == urgency
    });
}

/// Collects records until `n` `Started` records are in `log`.
fn wait_started(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>, n: usize) {
    while log.iter().filter(|r| r.kind == Kind::Started).count() < n {
        recv(log, records);
    }
}

/// Collects records until `hash` was embedded as indexing work.
fn wait_indexed(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>, hash: ContentHash) {
    wait_for(log, records, |r| {
        r.kind == Kind::Index && r.hash == Some(hash)
    });
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
    tokio::time::timeout(TIMEOUT, send_query(rig, text).await)
        .await
        .expect("the query is answered")
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

/// Awaits a `Documents` job's answer within the timeout.
async fn answered(docs: oneshot::Receiver<Result<(), EmbedError>>) -> Result<(), EmbedError> {
    tokio::time::timeout(TIMEOUT, docs)
        .await
        .expect("the documents job is answered")
        .expect("documents answered")
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
/// `log`, as `(kind, urgency)`, in order.
fn embeds_of(
    log: &mut Vec<Record>,
    records: &mpsc::Receiver<Record>,
    hash: ContentHash,
) -> Vec<(Kind, u8)> {
    log.extend(records.try_iter());
    log.iter()
        .filter(|r| matches!(r.kind, Kind::Index | Kind::QueryDocument) && r.hash == Some(hash))
        .map(|r| (r.kind, r.urgency.get()))
        .collect()
}

/// Drains the record channel into `log`; the urgency of every request
/// started for `hash`, in order.
fn starts_of(
    log: &mut Vec<Record>,
    records: &mpsc::Receiver<Record>,
    hash: ContentHash,
) -> Vec<u8> {
    log.extend(records.try_iter());
    log.iter()
        .filter(|r| r.kind == Kind::Started && r.hash == Some(hash))
        .map(|r| r.urgency.get())
        .collect()
}

/// Drains the record channel into `log`; every hash embedded as indexing
/// work, in order.
fn index_hashes(log: &mut Vec<Record>, records: &mpsc::Receiver<Record>) -> Vec<ContentHash> {
    log.extend(records.try_iter());
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

/// `seqs` as the hashes of 100-token chunks.
fn hash_set(seqs: impl IntoIterator<Item = u64>) -> HashSet<ContentHash> {
    seqs.into_iter().map(|seq| hash(seq, 100)).collect()
}

/// Sends a `Documents` job of `n` chunks for `WORKTREE` inside a search
/// window of its own, awaits its answer, checks every vector is cached and
/// returns the chunks' content hashes.
async fn query_documents(rig: &Rig, first_seq: u64, n: usize) -> HashSet<ContentHash> {
    let chunks: Vec<PendingChunk> = small(first_seq..first_seq + n as u64)
        .into_iter()
        .map(|item| item.chunk)
        .collect();
    let hashes: Vec<ContentHash> = chunks.iter().map(|chunk| chunk.content_hash).collect();
    rig.sched.barrier(REPO, true).await;
    answered(send_spot(rig, chunks).await)
        .await
        .expect("embeds fine");
    rig.sched.barrier(REPO, false).await;
    assert_cached(rig, &hashes).await;
    hashes.into_iter().collect()
}

// ---------------------------------------------------------------------
// counts: the ported `counts_*` cases, against the in-flight window.
// ---------------------------------------------------------------------

/// Every item flies once its enqueue is handled; each completion drops
/// the in-flight count, and the last small item completes stage 1. (Port
/// of `counts_track_an_ane_pull_and_done`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_track_a_flight_and_done() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..3);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..3), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));
    wait_started(&mut log, &records, 3);

    gates[0].open();
    await_counts(&rig, WORKTREE, counts(0, 0, 2, false)).await;
    gates[1].open();
    await_counts(&rig, WORKTREE, counts(0, 0, 1, false)).await;
    gates[2].open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// Eight large items and one small one fly together; the large ones'
/// completions leave the small one in flight, whose completion closes
/// stage 1. (Port of `counts_track_a_gpu_batch_pull_and_done`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_track_a_window_of_flights() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let large: Vec<Gate> = (0..8).map(|seq| fake.gate(hash(seq, 500))).collect();
    let small_gate = fake.gate(hash(8, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let items = (0..9)
        .map(|seq| pending(seq, if seq < 8 { 500 } else { 100 }))
        .collect();
    let mut done = enqueue(&rig, items, IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 9, false));
    wait_started(&mut log, &records, 9);

    for gate in &large {
        gate.open();
    }
    await_counts(&rig, WORKTREE, counts(0, 0, 1, false)).await;

    small_gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// A `Replace` that drops in-flight items uncounts them at once and
/// aborts their requests — most of the window waits queued on the
/// server, where the closed request cancels it — so nothing embeds them
/// and the worktree's counts never see them again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_drop_a_re_enqueued_in_flight_item() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..3);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let _old_done = enqueue(&rig, small(0..2), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));
    // Both requests are parked at their gates.
    wait_started(&mut log, &records, 2);

    // The replacement lists only item 2: both in-flight items drop and
    // their requests are aborted.
    let mut done = enqueue(&rig, small([2]), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    wait_cancelled(&mut log, &records, hash(0, 100), 3);
    wait_cancelled(&mut log, &records, hash(1, 100), 3);
    wait_started(&mut log, &records, 3);

    // Item 2 closes stage 1.
    gates[2].open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    // Opening the dropped items' gates reaches no request: the scheduler
    // is idle and the worktree's counts never moved.
    gates[0].open();
    gates[1].open();
    await_idle(&rig).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &records, hash(2, 100)),
        vec![(Kind::Index, 3)]
    );
    for seq in 0..2 {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![],
            "the dropped item {seq} is never embedded"
        );
    }
}

/// Two worktrees count the same item under their own tallies; one losing
/// interest moves only its own tally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_are_per_worktree_on_a_shared_item() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..3);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue_for(&rig, WORKTREE, small(0..2), IndexClass::Stage1).await;
    let _other_first = enqueue_for(&rig, OTHER, small(0..2), IndexClass::Stage1).await;

    // One `(repo, hash)` item, two interested worktrees: both count it.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 2, false));
    wait_started(&mut log, &records, 2);

    // The other worktree's re-`Enqueue` drops the shared items; only its
    // tally moves.
    let mut other_done = enqueue_for(&rig, OTHER, small([2]), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 1, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));

    // The shared items finish for the worktree that kept them.
    gates[0].open();
    gates[1].open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 1, false));

    gates[2].open();
    stage1(&mut other_done).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 0, true));
    for seq in 0..3 {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![(Kind::Index, 3)],
            "each item embeds once"
        );
    }
}

/// The small item's completion closes stage 1 and flips the large items
/// to background. Their flights were sent at stage-1 urgency and keep
/// flying: HTTP/1.1 cannot re-prioritize a request, and the
/// `Stage1`→`Background` flip leaves a flight alone. (Port of
/// `counts_move_to_background_at_stage1_completion`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counts_move_to_background_at_stage1_completion() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let small_gate = fake.gate(hash(0, 100));
    let large: Vec<Gate> = (1..3).map(|seq| fake.gate(hash(seq, 500))).collect();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(
        &rig,
        vec![pending(0, 100), pending(1, 500), pending(2, 500)],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));

    small_gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, true));

    for gate in &large {
        gate.open();
    }
    await_counts(&rig, WORKTREE, counts(0, 0, 0, true)).await;
    for seq in 1..3 {
        assert_eq!(
            starts_of(&mut log, &records, hash(seq, 500)),
            vec![3],
            "the large item flew once, at stage-1 urgency"
        );
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 500)),
            vec![(Kind::Index, 3)]
        );
    }
}

// ---------------------------------------------------------------------
// Interest edits: the ported `a_change_*` and `a_resync_skips_*` cases.
// ---------------------------------------------------------------------

/// A `Change` edits on top of a `Replace`: the added item flies and the
/// dropped one leaves the worktree's interest. The dropped item was in
/// flight, so its request is aborted and nothing embeds it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_adds_and_drops_on_top_of_a_replace() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..4);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..3), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));
    // Every request is parked at its gate.
    wait_started(&mut log, &records, 3);

    let dropped = hash(1, 100);
    change_for(
        &rig,
        WORKTREE,
        small([3]),
        vec![dropped],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));
    wait_cancelled(&mut log, &records, dropped, 3);

    change_for(&rig, WORKTREE, vec![], vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));

    for gate in &gates {
        gate.open();
    }
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    await_idle(&rig).await;
    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        hash_set([0, 2, 3]),
        "the wanted items embed once each and the dropped in-flight one never"
    );
}

/// Re-adding a wanted in-flight item is a no-op: no second request goes
/// out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_re_adding_a_wanted_item_is_a_no_op() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..2), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));

    change_for(&rig, WORKTREE, small(0..2), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));

    gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    let embedded = index_hashes(&mut log, &records);
    assert_eq!(embedded.len(), 2, "each hash embeds exactly once");
    assert_each_once(&embedded);
    assert!(embedded.iter().all(|h| hash_set(0..2).contains(h)));
    assert_eq!(
        log.iter().filter(|r| r.kind == Kind::Started).count(),
        2,
        "the re-added items sent no new requests"
    );
}

/// Dropping a hash that never was an item — or already finished — is a
/// no-op.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_dropping_a_finished_or_unknown_hash_is_a_no_op() {
    let (tx, _records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(0, 100));
    let rig = fake_rig(fake).await;
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    let unknown = hash(9, 100);
    change_for(&rig, WORKTREE, vec![], vec![unknown], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));

    // Hash 0 is finished; dropping it now changes nothing and does not
    // panic.
    change_for(
        &rig,
        WORKTREE,
        vec![],
        vec![hash(0, 100)],
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// One worktree's drop leaves the shared items to the other worktree's
/// interest; they still embed once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_on_one_worktree_keeps_a_shared_item_for_the_other() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue_for(&rig, WORKTREE, small(0..2), IndexClass::Stage1).await;
    let _other_done = enqueue_for(&rig, OTHER, small(0..2), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 2, false));

    let drops = vec![hash(0, 100), hash(1, 100)];
    change_for(&rig, OTHER, vec![], drops, IndexClass::Stage1).await;
    // OTHER wants nothing left: its stage 1 completes in the same
    // `Enqueue`.
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 0, true));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, false));

    gate.open();
    stage1(&mut done).await;
    let embedded = index_hashes(&mut log, &records);
    assert_eq!(embedded.len(), 2, "each shared hash embeds exactly once");
    assert_each_once(&embedded);
    assert!(embedded.iter().all(|h| hash_set(0..2).contains(h)));
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// A `Background` pass's additions after `Stage1Done` stay background:
/// nothing moves back to stage 1 and no second completion runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_change_after_stage1_done_moves_nothing_back() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let large_gate = fake.gate(hash(1, 500));
    let added_gate = fake.gate(hash(2, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(
        &rig,
        vec![pending(0, 100), pending(1, 500)],
        IndexClass::Stage1,
    )
    .await;
    stage1(&mut done).await;
    // The large item still flies, sent at stage-1 urgency before the
    // completion.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, true));

    change_for(&rig, WORKTREE, small([2]), vec![], IndexClass::Background).await;
    // A reopen would flip `stage1_done` inside the same `on_enqueue`,
    // before `CountsFor` is served: this is exact, not timing.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 2, true));
    assert!(matches!(
        done.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));

    large_gate.open();
    added_gate.open();
    await_counts(&rig, WORKTREE, counts(0, 0, 0, true)).await;
    assert_eq!(
        embeds_of(&mut log, &records, hash(1, 500)),
        vec![(Kind::Index, 3)]
    );
    assert_eq!(
        embeds_of(&mut log, &records, hash(2, 100)),
        vec![(Kind::Index, 7)],
        "the background addition flies at u=7"
    );
}

/// A late `Stage1` `Change` after completion reopens stage 1; the second
/// `Stage1Done` is the documented idempotent double delivery.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stage1_change_after_completion_reopens_stage1_once() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(1, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;
    stage1(&mut done).await;

    change_for(&rig, WORKTREE, small([1]), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    wait_for(&mut log, &records, |r| {
        r.kind == Kind::Started && r.hash == Some(hash(1, 100))
    });
    assert_eq!(starts_of(&mut log, &records, hash(1, 100)), vec![3]);

    gate.open();
    wait_indexed(&mut log, &records, hash(1, 100));
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// A resync does not re-queue chunks that got a vector inside its
/// barrier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resync_skips_chunks_finished_inside_its_barrier() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..4);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..3), IndexClass::Stage1).await;
    wait_started(&mut log, &records, 3);

    // The resync's barrier, whose read saw 0, 1 and 2 uncached.
    rig.sched.barrier(REPO, true).await;
    for gate in &gates[..3] {
        gate.open();
    }
    stage1(&mut done).await;

    // The resync's `Replace`, built from that read.
    let mut again = send_replace(&rig, WORKTREE, small(0..4), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    gates[3].open();
    stage1(&mut again).await;
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        hash_set(0..4),
        "{embedded:?}"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// A `Change` inside a barrier skips chunks another worktree finished
/// inside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_skips_chunks_another_worktree_finished_inside_its_barrier() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..3);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let _other = enqueue_for(&rig, OTHER, vec![], IndexClass::Background).await;
    let mut done = enqueue(&rig, small(0..2), IndexClass::Stage1).await;
    wait_started(&mut log, &records, 2);

    // OTHER's barrier, whose reconcile read saw 0 and 1 uncached.
    rig.sched.barrier(REPO, true).await;
    gates[0].open();
    gates[1].open();
    stage1(&mut done).await;

    send_change(&rig, OTHER, small(0..3), vec![], IndexClass::Background).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 1, false));
    gates[2].open();
    wait_indexed(&mut log, &records, hash(2, 100));
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        hash_set(0..3),
        "{embedded:?}"
    );
}

/// A `Change` inside a barrier skips chunks a search's `Documents` job
/// embedded inside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_skips_chunks_a_search_embedded_inside_its_barrier() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(7, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let _w = enqueue(&rig, vec![], IndexClass::Background).await;

    // The edit's barrier, whose reconcile read saw 5, 6 and 7 uncached.
    rig.sched.barrier(REPO, true).await;
    // A search embeds 5 and 6 on the spot; the scheduler holds neither
    // hash, and `DocumentsDone` notes both committed.
    let searched = query_documents(&rig, 5, 2).await;
    send_change(&rig, WORKTREE, small(5..8), vec![], IndexClass::Background).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    gate.open();
    wait_indexed(&mut log, &records, hash(7, 100));
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &records);
    for hash in searched {
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

/// A second worktree's resync of chunks finished inside its barrier
/// completes its stage 1 at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_worktrees_resync_of_finished_chunks_completes_stage1_at_once() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..2), IndexClass::Stage1).await;
    wait_started(&mut log, &records, 2);

    // OTHER's first barrier; its `missing` read saw 0 and 1 uncached.
    rig.sched.barrier(REPO, true).await;
    gate.open();
    stage1(&mut done).await;

    let mut other_done = send_replace(&rig, OTHER, small(0..2), IndexClass::Stage1).await;
    // FIFO-exact: stage-1 completion runs at the end of the same
    // `on_enqueue`, before `CountsFor` is served.
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 0, 0, true));
    stage1(&mut other_done).await;
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(embedded.len(), 2, "{embedded:?}");
}

/// The committed set that skips re-queueing lives only while a barrier of
/// the repository runs. Steps 2 and 4 send a `Change` naming a hash that
/// already has a vector. No production feed can do that — a read inside
/// the barrier sees the vector — so the steps exist only to observe that
/// the set is not kept.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn committed_hashes_are_kept_only_while_a_barrier_runs() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate0 = fake.gate(hash(0, 100));
    let gate1 = fake.gate(hash(1, 100));
    let rig = fake_rig(fake).await;
    let mut log: Vec<Record> = Vec::new();
    // Item 0's embed waits at its gate until the bracketed `Enqueue`'s
    // barrier has ended.
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;
    gate0.open();
    stage1(&mut done).await;
    gate0.close();

    // Not kept without a barrier.
    rig.sched.barrier(REPO, true).await;
    send_change(&rig, WORKTREE, small([0]), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    gate0.open();
    stage1(&mut done).await;
    rig.sched.barrier(REPO, false).await;

    // Kept until the last of two overlapping barriers ends.
    rig.sched.barrier(REPO, true).await;
    rig.sched.barrier(REPO, true).await;
    send_change(&rig, WORKTREE, small([1]), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    gate1.open();
    stage1(&mut done).await;
    gate1.close();
    rig.sched.barrier(REPO, false).await;
    send_change(&rig, WORKTREE, small([1]), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    rig.sched.barrier(REPO, false).await;

    // Dropped once none runs.
    rig.sched.barrier(REPO, true).await;
    send_change(&rig, WORKTREE, small([1]), vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    gate1.open();
    stage1(&mut done).await;
    rig.sched.barrier(REPO, false).await;

    let embedded = index_hashes(&mut log, &records);
    for seq in [0, 1] {
        let hash = hash(seq, 100);
        assert_eq!(
            embedded.iter().filter(|h| **h == hash).count(),
            2,
            "hash {seq} embedded exactly twice: {embedded:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Watch and `Documents` jobs.
// ---------------------------------------------------------------------

/// `watch_counts` seeds with the live counts and pushes each change —
/// and nothing when a dropped item's late completion changes no count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_counts_seeds_and_pushes_counts() {
    let (tx, _records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..3);
    let rig = fake_rig(fake).await;
    let _old_done = enqueue(&rig, small([0]), IndexClass::Stage1).await;

    let mut watch = rig
        .sched
        .watch_counts(WORKTREE)
        .await
        .expect("watch answered");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 1, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    // A re-`Enqueue` pushes the drop of the in-flight item and the
    // dispatch of the new ones.
    let mut done = enqueue(&rig, small(1..3), IndexClass::Stage1).await;
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 2, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    gates[1].open();
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 1, false));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    gates[2].open();
    stage1(&mut done).await;
    tokio::time::timeout(TIMEOUT, watch.changed())
        .await
        .expect("counts pushed")
        .expect("watch open");
    assert_eq!(*watch.borrow_and_update(), counts(0, 0, 0, true));
    assert_eq!(counts_of(&rig, WORKTREE).await, *watch.borrow());

    // The dropped item's completion lands last and pushes nothing: the
    // push runs before the idle report.
    gates[0].open();
    await_idle(&rig).await;
    assert!(
        !watch.has_changed().expect("watch open"),
        "no push for an unchanged count"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// A `Documents` job takes a chunk waiting in a queue out of it, embeds
/// it in its own `u=0` request and finishes it at `DocumentsDone`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_documents_job_takes_its_queued_twin_out_of_the_queue() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let index_gate = fake.gate_index();
    let interactive_gate = fake.gate_interactive();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    // 40 same-token items: the window takes items 0..=7 off the short
    // end and 16..=39 off the long end, leaving 8..=15 queued.
    let mut done = enqueue(&rig, small(0..40), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));

    let twin = hash(8, 100);
    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![pending(8, 100).chunk]).await;
    // The twin is held and counted in flight, so the queue lost it.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 33, false));

    interactive_gate.open();
    answered(docs).await.expect("embeds fine");
    // `DocumentsDone` finished the twin before answering.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 32, false));
    assert_cached(&rig, &[twin]).await;
    rig.sched.barrier(REPO, false).await;

    index_gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &records, twin),
        vec![(Kind::QueryDocument, 0)],
        "the twin embeds once, at interactive urgency, and never as index work"
    );
}

/// A dropped search releases the chunk its `Documents` job held; the
/// chunk goes back to its queue and embeds as index work.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_search_hands_its_held_chunk_back_to_indexing() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let index_gate = fake.gate_index();
    let interactive_gate = fake.gate_interactive();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..40), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));

    let held = hash(8, 100);
    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![pending(8, 100).chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 33, false));
    wait_started(&mut log, &records, 33);
    // The search hangs up before its request is answered. The reap runs
    // after the next message is handled: the second `CountsFor` sees it.
    drop(docs);
    counts_of(&rig, WORKTREE).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));
    wait_cancelled(&mut log, &records, held, 0);
    rig.sched.barrier(REPO, false).await;

    // The job's request was aborted at its gate; the chunk embeds as
    // indexing work once the window reaches it.
    interactive_gate.open();
    index_gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &records, held),
        vec![(Kind::Index, 3)],
        "the released chunk embeds as index work, not query work"
    );
}

/// A chunk two searches hold stays held when one is dropped; the second
/// job's answer finishes it. Without the hold count, releasing the first
/// would requeue the chunk while the second still embeds it, and
/// `on_documents_done`'s `Held` assertion would panic the scheduler.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunk_two_searches_hold_stays_held_when_one_is_dropped() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let index_gate = fake.gate_index();
    let interactive_gate = fake.gate_interactive();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..40), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));

    let held = hash(8, 100);
    // Two searches' windows.
    rig.sched.barrier(REPO, true).await;
    rig.sched.barrier(REPO, true).await;
    let first = send_spot(&rig, vec![pending(8, 100).chunk]).await;
    let second = send_spot(&rig, vec![pending(8, 100).chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 33, false));
    wait_started(&mut log, &records, 34);
    drop(first);
    counts_of(&rig, WORKTREE).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 33, false));
    wait_cancelled(&mut log, &records, held, 0);

    interactive_gate.open();
    answered(second).await.expect("embeds fine");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 32, false));
    assert_cached(&rig, &[held]).await;
    rig.sched.barrier(REPO, false).await;
    rig.sched.barrier(REPO, false).await;

    index_gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &records, held),
        vec![(Kind::QueryDocument, 0)]
    );
}

/// `Stop` fails an unanswered `Documents` job and hands its held chunk
/// back to its queue; a query already out still finishes, the in-flight
/// requests finish, nothing is sent anew, and the scheduler exits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_fails_a_queued_search_part_and_requeues_its_held_chunk() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let index_gate = fake.gate_index();
    let interactive_gate = fake.gate_interactive();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..40), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));

    let query = send_query(&rig, "v1").await;
    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![pending(8, 100).chunk]).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(7, 0, 33, false));
    wait_started(&mut log, &records, 34);

    rig.sched.stop().await;
    let answer = answered(docs).await;
    assert!(
        matches!(answer, Err(EmbedError::SchedulerGone)),
        "the unanswered job fails: {answer:?}"
    );
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));
    wait_cancelled(&mut log, &records, hash(8, 100), 0);

    interactive_gate.open();
    tokio::time::timeout(TIMEOUT, query)
        .await
        .expect("the query is answered")
        .expect("query answered")
        .expect("embeds fine");

    index_gate.open();
    await_idle(&rig).await;
    assert!(
        rig.sched.counts_for(WORKTREE).await.is_err(),
        "the scheduler exited"
    );
    assert!(
        tokio::time::timeout(TIMEOUT, done.recv())
            .await
            .expect("the stage-1 channel closes")
            .is_none(),
        "stage 1 never completes"
    );
    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(
        embedded.iter().copied().collect::<HashSet<_>>(),
        hash_set((0..8).chain(16..40)),
        "only the 32 flights out at the stop embed"
    );
    assert!(
        !log.iter()
            .any(|r| r.kind == Kind::QueryDocument && r.hash == Some(hash(8, 100))),
        "the failed job's request was aborted"
    );
}

/// A search skips a chunk that got a vector inside its window: the job
/// is answered at once, with no request of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_skips_a_chunk_finished_inside_its_window() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(0, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;

    // The search's window; its vector sync, inside it, saw the chunk
    // uncached.
    rig.sched.barrier(REPO, true).await;
    gate.open();
    // The chunk was embedded and `item_done` ran inside the window.
    stage1(&mut done).await;

    // Answered at once, since the chunk is in the committed set.
    answered(send_spot(&rig, vec![pending(0, 100).chunk]).await)
        .await
        .expect("embeds fine");
    assert_cached(&rig, &[hash(0, 100)]).await;
    rig.sched.barrier(REPO, false).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_eq!(
        embeds_of(&mut log, &records, hash(0, 100)),
        vec![(Kind::Index, 3)]
    );
    assert!(
        log.iter().all(|r| r.urgency.get() != 0),
        "the job sent no request: {log:?}"
    );
}

// ---------------------------------------------------------------------
// Cancellation: the rewritten borrow/GPU-twin cases.
// ---------------------------------------------------------------------

/// A `Documents` job aborts the indexing flight of a chunk it takes — the
/// daemon cannot tell whether the server is running it or it is still
/// queued — and embeds it at `u=0` in its own request. The aborted
/// flight's late completion is stale: the item is held. (Rewrites
/// `a_search_borrows_*` and `a_search_embeds_its_gpu_chunk_again_*`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_cancels_an_in_flight_item_and_embeds_it_at_u0() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(0, 100));
    let interactive_gate = fake.gate_interactive();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let chunk = hash(0, 100);
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    wait_started(&mut log, &records, 1);

    rig.sched.barrier(REPO, true).await;
    let mut docs = send_spot(&rig, vec![pending(0, 100).chunk]).await;
    // Held now, and still counted in flight.
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));

    // The first item's first flight is item 0, generation 0: its late
    // completion changes nothing for the held item.
    rig.sched
        .tx()
        .send(SchedulerMsg::IndexDone {
            item: 0,
            generation: 0,
        })
        .await
        .expect("scheduler is up");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    assert!(matches!(
        docs.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    interactive_gate.open();
    answered(docs).await.expect("embeds fine");
    rig.sched.barrier(REPO, false).await;
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_cached(&rig, &[chunk]).await;

    // The aborted flight is gone: opening its gate embeds nothing.
    wait_cancelled(&mut log, &records, chunk, 3);
    gate.open();
    await_idle(&rig).await;
    assert_eq!(
        embeds_of(&mut log, &records, chunk),
        vec![(Kind::QueryDocument, 0)],
        "the search's u=0 request embeds the chunk its hold cancelled"
    );
}

/// A cancelled and requeued flight's late completion is ignored: the
/// item flies again on a new generation and finishes once, from it.
/// (Rewrites `a_gpu_twin_finished_*`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_flights_late_completion_is_ignored() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate(hash(0, 100));
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let chunk = hash(0, 100);
    let mut done = enqueue(&rig, small([0]), IndexClass::Background).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    wait_started(&mut log, &records, 1);

    // A stage-1 pass flips the item: its `u=7` flight (generation 0) is
    // aborted and the item flies again at `u=3` (generation 1).
    change_for(&rig, WORKTREE, vec![], vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    wait_started(&mut log, &records, 2);
    assert_eq!(starts_of(&mut log, &records, chunk), vec![7, 3]);

    // Generation 0's completion is stale: applying it would finish the
    // item and close stage 1 here.
    rig.sched
        .tx()
        .send(SchedulerMsg::IndexDone {
            item: 0,
            generation: 0,
        })
        .await
        .expect("scheduler is up");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 1, false));
    assert!(matches!(
        done.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));

    wait_cancelled(&mut log, &records, chunk, 7);
    gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    assert_cached(&rig, &[chunk]).await;
    assert_eq!(
        embeds_of(&mut log, &records, chunk),
        vec![(Kind::Index, 3)],
        "only the current flight embeds"
    );
}

// ---------------------------------------------------------------------
// The window and the wire: the new cases.
// ---------------------------------------------------------------------

/// The window holds the 8 shortest and the 24 longest remaining items;
/// the middle stays queued.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_window_holds_8_short_and_24_long() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    // Strictly increasing token counts: the short end is seq 0..=7, the
    // long end 16..=39, and 8..=15 stays queued.
    let tokens = |seq: u64| 100 + 10 * seq as u32;
    let mut done = enqueue(
        &rig,
        (0..40).map(|seq| pending(seq, tokens(seq))).collect(),
        IndexClass::Stage1,
    )
    .await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));
    wait_started(&mut log, &records, 32);
    let dispatched: HashSet<ContentHash> = log
        .iter()
        .filter(|r| r.kind == Kind::Started)
        .filter_map(|r| r.hash)
        .collect();
    let want: HashSet<ContentHash> = (0..8)
        .chain(16..40)
        .map(|seq| hash(seq, tokens(seq)))
        .collect();
    assert_eq!(
        dispatched, want,
        "the 8 shortest and 24 longest items are in flight"
    );
    gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
}

/// With the window full of background work, every slot a completion
/// frees goes to a queued stage-1 item before any queued background one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stage1_fills_the_window_before_background() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gates = gates(&mut fake, 0..50);
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    // 40 background items of another worktree fill the window — 0..=7
    // and 16..=39 — and leave 8..=15 queued.
    enqueue_for(&rig, OTHER, small(0..40), IndexClass::Background).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 8, 32, false));

    // A stage-1 enqueue lands: no slot is free, so it waits queued.
    let mut done = enqueue_for(&rig, WORKTREE, small(40..50), IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(10, 0, 0, false));

    // Ten background flights land: their slots refill from stage 1 only.
    for gate in gates[..8].iter().chain(&gates[16..18]) {
        gate.open();
    }
    await_counts(&rig, WORKTREE, counts(0, 0, 10, false)).await;
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 8, 22, false));
    wait_started(&mut log, &records, 42);
    for seq in 40..50 {
        assert_eq!(
            starts_of(&mut log, &records, hash(seq, 100)),
            vec![3],
            "every stage-1 refill flies at stage-1 urgency"
        );
    }

    for gate in &gates {
        gate.open();
    }
    stage1(&mut done).await;
    await_counts(&rig, OTHER, counts(0, 0, 0, false)).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    let embedded = index_hashes(&mut log, &records);
    assert_each_once(&embedded);
    assert_eq!(embedded.len(), 50);
}

/// A worktree's stage-1 activation aborts the background flights of its
/// items and requeues them to fly at `u=3`; the aborted `u=7` requests
/// embed nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_to_stage1_flip_cancels_and_requeues() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small(0..3), IndexClass::Background).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));
    wait_started(&mut log, &records, 3);

    // A stage-1 pass lands while the items fly at u=7.
    change_for(&rig, WORKTREE, vec![], vec![], IndexClass::Stage1).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 3, false));
    wait_started(&mut log, &records, 6);
    for seq in 0..3 {
        assert_eq!(
            starts_of(&mut log, &records, hash(seq, 100)),
            vec![7, 3],
            "the requeued flight goes at u=3"
        );
        wait_cancelled(&mut log, &records, hash(seq, 100), 7);
    }

    gate.open();
    stage1(&mut done).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 0, 0, true));
    for seq in 0..3 {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![(Kind::Index, 3)],
            "only the u=3 flight embeds"
        );
    }
}

/// An attached `index --wait` lifts its worktree's background work to
/// stage-1 urgency: the queued items move to the stage-1 queue and the
/// in-flight `u=7` requests are aborted and requeued at `u=3`. The last
/// waiter's detach drops the remaining queued items back to background;
/// another worktree's background items never move.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_waiter_lifts_background_work_until_it_detaches() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let index_gate = fake.gate_index();
    let rig = fake_rig(fake).await;
    let mut log = Vec::new();
    // WORKTREE fills the window — 0..=7 and 16..=39 in flight at `u=7`,
    // 8..=15 queued — and OTHER's ten stay queued behind it.
    let _done = enqueue(&rig, small(0..40), IndexClass::Background).await;
    let _other_done = enqueue_for(&rig, OTHER, small(40..50), IndexClass::Background).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 8, 32, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 10, 0, false));
    wait_started(&mut log, &records, 32);

    // The wait attaches: the queued items flip to stage 1, and the
    // in-flight `u=7` requests are aborted and requeued at `u=3`.
    let watch = rig
        .sched
        .watch_counts(WORKTREE)
        .await
        .expect("watch answered");
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(8, 0, 32, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 10, 0, false));
    wait_started(&mut log, &records, 64);
    for seq in (0..8).chain(16..40) {
        let hash = hash(seq, 100);
        assert_eq!(
            starts_of(&mut log, &records, hash),
            vec![7, 3],
            "the attached wait requeued the flight at stage-1 urgency"
        );
        wait_cancelled(&mut log, &records, hash, 7);
    }

    // Detaching drops the queued items back to background; the `u=3`
    // flights out already fly on. The detach lands on the next handled
    // message: the second `CountsFor` sees it.
    drop(watch);
    counts_of(&rig, WORKTREE).await;
    assert_eq!(counts_of(&rig, WORKTREE).await, counts(0, 8, 32, false));
    assert_eq!(counts_of(&rig, OTHER).await, counts(0, 10, 0, false));

    index_gate.open();
    await_idle(&rig).await;
    for seq in (0..8).chain(16..40) {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![(Kind::Index, 3)],
            "the lifted item embedded once, at stage-1 urgency"
        );
    }
    for seq in (8..16).chain(40..50) {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![(Kind::Index, 7)],
            "a queued item went back to background urgency"
        );
    }
}

/// Stage 1 flies at `u=3`, background at `u=7`, queries and `Documents`
/// jobs at `u=0`; a `Documents` job sends all its chunks in one request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn urgency_is_3_for_stage1_7_for_background_0_for_queries() {
    let (tx, records) = mpsc::channel();
    let rig = fake_rig(FakeEmbeddings::new(tx)).await;
    let mut log = Vec::new();
    let mut done = enqueue(&rig, small([0]), IndexClass::Stage1).await;
    stage1(&mut done).await;
    enqueue_for(&rig, OTHER, small([1]), IndexClass::Background).await;
    wait_indexed(&mut log, &records, hash(1, 100));
    query(&rig, "v1").await;
    rig.sched.barrier(REPO, true).await;
    answered(send_spot(&rig, vec![pending(2, 100).chunk, pending(3, 100).chunk]).await)
        .await
        .expect("embeds fine");
    rig.sched.barrier(REPO, false).await;

    assert_eq!(
        embeds_of(&mut log, &records, hash(0, 100)),
        vec![(Kind::Index, 3)]
    );
    assert_eq!(
        embeds_of(&mut log, &records, hash(1, 100)),
        vec![(Kind::Index, 7)]
    );
    for seq in [2, 3] {
        assert_eq!(
            embeds_of(&mut log, &records, hash(seq, 100)),
            vec![(Kind::QueryDocument, 0)],
            "documents embed at u=0"
        );
    }
    let queries: Vec<u8> = log
        .iter()
        .filter(|r| r.kind == Kind::Query)
        .map(|r| r.urgency.get())
        .collect();
    assert_eq!(queries, vec![0], "queries embed at u=0");
    assert!(
        log.iter()
            .filter(|r| r.kind == Kind::QueryDocument)
            .all(|r| r.batch_len == 2),
        "one request carries the job"
    );
}

/// A failed request is fatal: `Error::Server` goes to `run`, an
/// unanswered `Documents` job fails with it, the indexing window is
/// aborted, and the scheduler exits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_request_is_fatal_and_the_scheduler_exits() {
    let (tx, records) = mpsc::channel();
    let mut fake = FakeEmbeddings::new(tx);
    let failing = fake.gate(hash(0, 100));
    let _held = fake.gate(hash(1, 100));
    let _interactive = fake.gate_interactive();
    fake.fail(hash(0, 100), "the model died");
    let mut rig = fake_rig(fake).await;
    let mut log = Vec::new();
    let _done = enqueue(&rig, small(0..2), IndexClass::Stage1).await;
    rig.sched.barrier(REPO, true).await;
    let docs = send_spot(&rig, vec![pending(5, 100).chunk]).await;
    wait_started(&mut log, &records, 3);

    failing.open();
    let error = tokio::time::timeout(TIMEOUT, rig.fatal.recv())
        .await
        .expect("a fatal error arrives")
        .expect("the channel reports");
    let Error::Server(message) = error else {
        panic!("a server error, got {error:?}");
    };
    assert!(message.contains("the model died"), "{message}");
    let answer = answered(docs).await;
    assert!(
        matches!(&answer, Err(EmbedError::Model(m)) if m.contains("the model died")),
        "the unanswered job fails with the error: {answer:?}"
    );
    // Item 1's flight was aborted, so nothing is left out: the scheduler
    // exits while its gate stays shut.
    await_idle(&rig).await;
    assert!(
        rig.sched.counts_for(WORKTREE).await.is_err(),
        "the scheduler exited"
    );
    assert!(index_hashes(&mut log, &records).is_empty());
}

// ---------------------------------------------------------------------
// Fatal handling, through a real `ServerClient` on a hyper test server.
// ---------------------------------------------------------------------

/// What the test server recorded of a request.
struct Recorded {
    /// Which accepted connection the request came on, from 1.
    conn: usize,
    /// The request path.
    path: String,
    /// The `Priority` header, as sent.
    priority: Option<String>,
    /// The JSON body.
    body: serde_json::Value,
}

/// A hyper test server: every accepted connection serves requests that
/// `handler` answers — `None` kills the connection without a response.
struct TestServer {
    addr: SocketAddr,
    requests: mpsc::Receiver<Recorded>,
}

impl TestServer {
    /// The next request the server recorded.
    fn next(&self) -> Recorded {
        tokio::task::block_in_place(|| {
            self.requests
                .recv_timeout(TIMEOUT)
                .expect("the request arrives")
        })
    }

    /// The connections of every request recorded so far, in order.
    fn conns(&self) -> Vec<usize> {
        self.requests
            .try_iter()
            .map(|request| request.conn)
            .collect()
    }
}

fn serve<F>(handler: F) -> TestServer
where
    F: Fn(usize) -> Option<hyper::Response<http_body_util::Full<bytes::Bytes>>>
        + Send
        + Sync
        + 'static,
{
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.set_nonblocking(true).expect("nonblocking");
    let addr = listener.local_addr().expect("local addr");
    let listener = tokio::net::TcpListener::from_std(listener).expect("tokio listener");
    let (tx, requests) = mpsc::channel();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        let mut conn = 0usize;
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            conn += 1;
            let n = conn;
            let io = hyper_util::rt::TokioIo::new(stream);
            let handler = handler.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                let service = hyper::service::service_fn(
                    move |req: hyper::Request<hyper::body::Incoming>| {
                        let handler = handler.clone();
                        let tx = tx.clone();
                        async move {
                            let (parts, body) = req.into_parts();
                            let body = http_body_util::BodyExt::collect(body)
                                .await
                                .map(|c| c.to_bytes())
                                .unwrap_or_default();
                            let _ = tx.send(Recorded {
                                conn: n,
                                path: parts.uri.path().to_owned(),
                                priority: parts
                                    .headers
                                    .get("priority")
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_owned),
                                body: serde_json::from_slice(&body).unwrap_or_default(),
                            });
                            match handler(n) {
                                Some(response) => Ok::<_, std::io::Error>(response),
                                None => Err(std::io::Error::new(
                                    std::io::ErrorKind::ConnectionAborted,
                                    "killed",
                                )),
                            }
                        }
                    },
                );
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });
    TestServer { addr, requests }
}

/// A JSON response, with an optional `Server-Timing` header.
fn json_response(
    status: u16,
    body: serde_json::Value,
    timing: Option<&str>,
) -> hyper::Response<http_body_util::Full<bytes::Bytes>> {
    let mut builder = hyper::Response::builder()
        .status(status)
        .header("content-type", "application/json");
    if let Some(timing) = timing {
        builder = builder.header("server-timing", timing);
    }
    builder
        .body(http_body_util::Full::new(bytes::Bytes::from(
            body.to_string(),
        )))
        .expect("a response builds")
}

/// A standard `EmbeddingResponse` body carrying `vectors` as base64.
fn embeddings_body(vectors: &[&[f32]]) -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| serde_json::json!({
                "object": "embedding",
                "index": index,
                "embedding": encode_vector(vector),
            }))
            .collect::<Vec<_>>(),
        "model": repo_search_core::model::MODEL_ID,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    })
}

/// A standard error envelope body.
fn error_body(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "param": null,
            "code": null,
        }
    })
}

/// The client a test endpoint is built from.
fn client(addr: SocketAddr) -> ServerClient {
    ServerClient::new(&Endpoint::Tcp { addr, token: None }).expect("the client builds")
}

/// A transport error is retried once — the killed connection is followed
/// by a fresh one that fails with a status — then the failure is fatal:
/// `Error::Server` goes to `run`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transport_error_is_retried_once_then_fatal() {
    let server = serve(|conn| {
        if conn == 1 {
            None
        } else {
            Some(json_response(500, error_body("the model died"), None))
        }
    });
    let mut rig = rig(Arc::new(client(server.addr))).await;
    enqueue(&rig, small([0]), IndexClass::Stage1).await;
    let error = tokio::time::timeout(TIMEOUT, rig.fatal.recv())
        .await
        .expect("a fatal error arrives")
        .expect("the channel reports");
    let Error::Server(message) = error else {
        panic!("a server error, got {error:?}");
    };
    assert!(
        message.contains("the model died"),
        "the status detail surfaces: {message}"
    );
    // Both attempts were answered before the client gave up.
    assert_eq!(server.conns(), vec![1, 2], "one retry, on a new connection");
}

/// A non-2xx answer is fatal without a retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_5xx_is_fatal() {
    let server = serve(|_| {
        Some(json_response(
            503,
            error_body("the model is downloading"),
            None,
        ))
    });
    let mut rig = rig(Arc::new(client(server.addr))).await;
    enqueue(&rig, small([0]), IndexClass::Stage1).await;
    let error = tokio::time::timeout(TIMEOUT, rig.fatal.recv())
        .await
        .expect("a fatal error arrives")
        .expect("the channel reports");
    let Error::Server(message) = error else {
        panic!("a server error, got {error:?}");
    };
    assert!(
        message.contains("the model is downloading") && message.contains("503"),
        "the status detail surfaces: {message}"
    );
    assert_eq!(server.conns(), vec![1], "a non-2xx answer is never retried");
}

// ---------------------------------------------------------------------
// `ServerClient` unit tests against the hyper test server.
// ---------------------------------------------------------------------

/// The client decodes base64 vectors and sends the standard request at
/// the urgency given.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_client_decodes_base64() {
    let server = serve(|_| {
        Some(json_response(
            200,
            embeddings_body(&[&[1.0, 0.0, 0.0]]),
            None,
        ))
    });
    let answered = client(server.addr)
        .embed(vec!["fn main() {}".to_owned()], Urgency::new(3))
        .await
        .expect("embeds");
    assert_eq!(answered.vectors.len(), 1);
    assert!(
        (answered.vectors[0].as_slice()[0] - 1.0).abs() < 1e-6,
        "the base64 vector decodes"
    );
    let request = server.next();
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(request.priority.as_deref(), Some("u=3"));
    assert_eq!(request.body["model"], repo_search_core::model::MODEL_ID);
    assert_eq!(request.body["input"], serde_json::json!(["fn main() {}"]));
    assert_eq!(request.body["encoding_format"], serde_json::json!("base64"));
}

/// The `Server-Timing` header parses into `Answered::timing`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_client_parses_server_timing() {
    let server = serve(|_| {
        Some(json_response(
            200,
            embeddings_body(&[&[0.5, 0.5]]),
            Some("wait;dur=1.25, overtaken;desc=\"2\""),
        ))
    });
    let answered = client(server.addr)
        .embed(vec!["x".to_owned()], Urgency::new(0))
        .await
        .expect("embeds");
    assert_eq!(answered.timing.wait, Duration::from_micros(1250));
    assert_eq!(answered.timing.overtaken, 2);
}

/// A 503 answer surfaces the envelope's message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_client_surfaces_503_detail() {
    let server = serve(|_| {
        Some(json_response(
            503,
            error_body("the model is downloading"),
            None,
        ))
    });
    let error = client(server.addr)
        .embed(vec!["x".to_owned()], Urgency::new(0))
        .await
        .expect_err("a 503 fails");
    let repo_search::daemon::server::Error::Status { status, message } = error else {
        panic!("a status error, got {error:?}");
    };
    assert_eq!(status, 503);
    assert_eq!(message, "the model is downloading");
}

/// A killed connection — a transport error — is retried once on a fresh
/// connection, and the second attempt answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_client_retries_once_on_transport_error() {
    let server = serve(|conn| {
        if conn == 1 {
            None
        } else {
            Some(json_response(200, embeddings_body(&[&[1.0]]), None))
        }
    });
    let answered = client(server.addr)
        .embed(vec!["x".to_owned()], Urgency::new(0))
        .await
        .expect("the retry answers");
    assert_eq!(answered.vectors.len(), 1);
    assert_eq!(server.conns(), vec![1, 2]);
}

/// A 5xx answer is fatal without a retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_client_treats_5xx_as_fatal_without_retry() {
    let server = serve(|_| Some(json_response(500, error_body("boom"), None)));
    let error = client(server.addr)
        .embed(vec!["x".to_owned()], Urgency::new(0))
        .await
        .expect_err("a 500 fails");
    assert!(
        matches!(
            error,
            repo_search::daemon::server::Error::Status { status: 500, .. }
        ),
        "a status error: {error:?}"
    );
    assert_eq!(server.conns(), vec![1]);
}
