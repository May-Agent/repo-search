//! The ONNX embedding server: the same OpenAI-compatible HTTP API the
//! macOS Core ML server speaks — `POST /v1/embeddings`, `GET /v1/models`,
//! `GET /status`, the `Priority` header and the `Server-Timing` response —
//! over the [`OrtEmbedder`]s [`OrtEmbedder::open_devices`] opens, one per
//! device, each owned by its own thread.
//!
//! Where the Core ML server splits queries onto the ANE and bulk indexing
//! onto a power-gated GPU, this server has interchangeable device
//! sessions. A dispatcher thread owns the pending requests and hands each
//! idle device at most [`RUN_INPUTS`] same-bucket inputs, taken across the
//! pending requests of the most urgent class (the daemon sends one input
//! per request), re-picking between runs, so a bulk index batch cannot
//! park a query behind its full length.
//!
//! The unix socket carries no bearer check — its file permissions are the
//! check; TCP listeners authenticate when the server was given a token.

use std::convert::Infallible;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Body, Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use repo_search_core::UnitVector;
use repo_search_core::embed_api::{
    EmbeddingData, EmbeddingRequest, EncodingFormat, ErrorBody, ErrorDetail, GpuStatus, Input,
    LoadStage, ModelEntry, ModelStatus, ModelsResponse, ServerTiming, StatusBody, Urgency, Usage,
    encode_vector,
};
use tokio::sync::{mpsc, oneshot, watch};

use crate::{Encoder, Error as EmbedError, OrtEmbedder};

/// The request body cap: 32 MiB, the daemon-server contract's bound.
const BODY_LIMIT: usize = 32 << 20;

/// The dimensions Matryoshka truncation accepts.
const DIMENSIONS: [u32; 4] = [128, 256, 512, 768];

/// The most inputs a request may carry.
const MAX_INPUTS: usize = 2048;

/// The most inputs of one inference run: a request's batch is split into
/// runs of at most this, so an urgent request behind a bulk batch waits
/// one run, not the whole batch.
const RUN_INPUTS: usize = 32;

/// What the server is built from.
pub struct Config {
    /// The unix socket the daemon talks over, when the platform has one.
    #[cfg(unix)]
    pub unix: Option<tokio::net::UnixListener>,
    /// TCP listeners; empty when the server is unix-only.
    pub tcp: Vec<tokio::net::TcpListener>,
    /// The bearer token TCP listeners require; `None` means no auth.
    /// The unix socket needs none — its file permissions are the check.
    pub token: Option<String>,
    /// Exit after this long idle.
    pub idle_exit: Duration,
    /// The model directory the server loads (`manifest.json`,
    /// `tokenizer.json`, the ONNX file and its sidecar data).
    pub model_dir: PathBuf,
}

/// A failure that shuts the server down.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An embed call failed inside the session. There is no retry and no
    /// fallback.
    #[error("embedder failed: {0}")]
    Embedder(String),
    /// The listener or the signal handler failed.
    #[error("server I/O: {0}")]
    Io(#[from] io::Error),
}

/// The model's load state, as `/v1/` handlers see it.
enum Model {
    /// `OrtEmbedder::open` is running; the stage watch holds its progress.
    Loading,
    /// Ready to embed.
    Ready(Arc<Ready>),
    /// The load failed; a request retries it.
    Failed(String),
}

/// The loaded model's request-facing half.
struct Ready {
    /// The manifest's `model_id`.
    id: String,
    /// The tokenizer request inputs encode with.
    encoder: Encoder,
}

/// One embedding request's work for the dispatcher. Runs take same-bucket
/// inputs from anywhere in `remaining` — and from other pending jobs of the
/// same urgency — each carrying its original index, and finished vectors
/// return to their `slots`, so the reply is in input order however the
/// runs were spread over devices.
struct Job {
    /// Assigned by the dispatcher when the job arrives.
    id: u64,
    /// The request's RFC 9218 urgency; lower runs first.
    urgency: Urgency,
    /// When the request was submitted, the `wait` metric's origin.
    queued: Instant,
    /// Already-tokenized inputs not yet handed to a device, with their
    /// input indices, in input order.
    remaining: Vec<(usize, Vec<u32>)>,
    /// One slot per input, filled as its run finishes.
    slots: Vec<Option<UnitVector>>,
    /// Inputs handed to a device and not yet placed.
    outstanding: usize,
    /// The longest wait observed across the job's runs.
    wait: Duration,
    /// The lower-urgency jobs pending when this job first ran, counted
    /// once.
    overtaken: u64,
    /// Whether `overtaken` was already counted.
    counted: bool,
    /// The request's reply.
    reply: oneshot::Sender<Result<(Vec<UnitVector>, ServerTiming), String>>,
}

impl Job {
    fn new(
        urgency: Urgency,
        ids: Vec<Vec<u32>>,
        reply: oneshot::Sender<Result<(Vec<UnitVector>, ServerTiming), String>>,
    ) -> Self {
        Self {
            id: 0,
            urgency,
            queued: Instant::now(),
            slots: vec![None; ids.len()],
            remaining: ids.into_iter().enumerate().collect(),
            outstanding: 0,
            wait: Duration::ZERO,
            overtaken: 0,
            counted: false,
            reply,
        }
    }

    /// The bucket of the next input to hand out.
    fn lead_bucket(&self, buckets: &[usize]) -> usize {
        bucket_of(buckets, self.remaining[0].1.len())
    }

    /// Takes up to `room` remaining inputs of `bucket`, from anywhere in
    /// `remaining`, in input order.
    fn take_matching(
        &mut self,
        buckets: &[usize],
        bucket: usize,
        room: usize,
    ) -> Vec<(usize, Vec<u32>)> {
        let mut taken = Vec::with_capacity(room.min(self.remaining.len()));
        let mut kept = Vec::with_capacity(self.remaining.len());
        for input in std::mem::take(&mut self.remaining) {
            if taken.len() < room && bucket_of(buckets, input.1.len()) == bucket {
                taken.push(input);
            } else {
                kept.push(input);
            }
        }
        self.remaining = kept;
        self.outstanding += taken.len();
        taken
    }

    /// Writes one finished vector back to its input position.
    fn place(&mut self, index: usize, vector: UnitVector) {
        self.slots[index] = Some(vector);
        self.outstanding -= 1;
    }

    /// Whether every input is embedded and placed.
    fn is_done(&self) -> bool {
        self.remaining.is_empty() && self.outstanding == 0
    }

    /// Replies with the vectors in input order.
    fn finish(self) {
        let timing = ServerTiming {
            wait: self.wait,
            overtaken: self.overtaken,
        };
        let vectors = self
            .slots
            .into_iter()
            .map(|slot| slot.expect("a finished job filled every slot"))
            .collect();
        let _ = self.reply.send(Ok((vectors, timing)));
    }
}

/// The smallest of `buckets` (ascending) that holds `len` tokens.
fn bucket_of(buckets: &[usize], len: usize) -> usize {
    buckets
        .iter()
        .copied()
        .find(|&bucket| len <= bucket)
        .expect("the encoder capped every input")
}

/// The most urgent, oldest pending job that still has inputs to hand out.
fn pick(pending: &[Job]) -> Option<usize> {
    pending
        .iter()
        .enumerate()
        .filter(|(_, job)| !job.remaining.is_empty())
        .min_by_key(|(_, job)| (job.urgency, job.queued))
        .map(|(index, _)| index)
}

/// What the dispatcher receives: new jobs from handlers and finished runs
/// from device workers.
enum Msg {
    /// A request's job.
    Job(Job),
    /// A device worker finished a run.
    Done {
        /// The worker's index, idle again.
        worker: usize,
        /// The `(job, input index)` of each input of the run, matching
        /// the vectors.
        parts: Vec<(u64, usize)>,
        /// The run's vectors, or its fatal failure.
        result: Result<Vec<UnitVector>, String>,
    },
}

/// One run for a device worker: same-bucket inputs of one urgency, which
/// may come from several jobs.
struct Work {
    /// The `(job, input index)` of each batch input.
    parts: Vec<(u64, usize)>,
    batch: Vec<Vec<u32>>,
}

/// Assembles the next run: the most urgent, oldest job's next bucket,
/// filled up to [`RUN_INPUTS`] with that bucket's inputs from the jobs of
/// the same urgency, oldest first. A device is never left on one input
/// while same-bucket inputs wait in other requests — the daemon sends one
/// input per request — and a more urgent request still waits at most one
/// run, since lower urgencies never join its run.
fn assemble_run(pending: &mut [Job], buckets: &[usize]) -> Option<Work> {
    let lead = pick(pending)?;
    let urgency = pending[lead].urgency;
    let bucket = pending[lead].lead_bucket(buckets);
    let mut order: Vec<usize> = (0..pending.len())
        .filter(|&at| pending[at].urgency == urgency && !pending[at].remaining.is_empty())
        .collect();
    order.sort_by_key(|&at| pending[at].queued);
    let overtaken = pending
        .iter()
        .filter(|other| other.urgency > urgency)
        .count() as u64;
    let mut parts = Vec::with_capacity(RUN_INPUTS);
    let mut batch = Vec::with_capacity(RUN_INPUTS);
    for at in order {
        if batch.len() == RUN_INPUTS {
            break;
        }
        let job = &mut pending[at];
        let taken = job.take_matching(buckets, bucket, RUN_INPUTS - batch.len());
        if taken.is_empty() {
            continue;
        }
        if !job.counted {
            job.counted = true;
            job.wait = job.queued.elapsed();
            job.overtaken = overtaken;
        }
        for (index, ids) in taken {
            parts.push((job.id, index));
            batch.push(ids);
        }
    }
    Some(Work { parts, batch })
}

/// Everything a connection's handlers share.
struct Shared {
    /// The model's load state.
    state: watch::Receiver<Model>,
    /// The `open` pipeline's stage while the model loads.
    stage: watch::Receiver<LoadStage>,
    /// The loader's retry inbox: a `Failed` load retries on the next
    /// `/v1/` request.
    retry: std_mpsc::SyncSender<()>,
    /// The dispatcher's inbox.
    worker: std_mpsc::Sender<Msg>,
    /// The bearer token TCP listeners require.
    token: Option<String>,
    /// Handlers in flight; the idle exit waits for it to reach zero.
    in_flight: AtomicU64,
    /// The last request-handling moment; the idle exit counts from it and
    /// re-arms when it changes.
    activity: watch::Sender<Instant>,
}

/// Runs the server until it idle-exits, receives a shutdown signal, or
/// the embedder fails fatally.
///
/// # Errors
///
/// [`Error::Embedder`] when an embed call fails and [`Error::Io`] when a
/// listener or the signal handling fails.
pub async fn run(config: Config) -> Result<(), Error> {
    let (state_tx, state) = watch::channel(Model::Loading);
    let (stage_tx, stage) = watch::channel(LoadStage::Encoder);
    let (job_tx, job_rx) = std_mpsc::channel::<Msg>();
    let (retry_tx, retry_rx) = std_mpsc::sync_channel::<()>(1);
    let (fatal_tx, mut fatal_rx) = mpsc::channel::<String>(1);
    let (activity_tx, mut activity) = watch::channel(Instant::now());

    let dir = config.model_dir;
    let events = job_tx.clone();
    std::thread::spawn(move || {
        dispatcher(dir, state_tx, stage_tx, job_rx, events, retry_rx, fatal_tx);
    });

    let shared = Arc::new(Shared {
        state,
        stage,
        retry: retry_tx,
        worker: job_tx,
        token: config.token,
        in_flight: AtomicU64::new(0),
        activity: activity_tx,
    });

    #[cfg(unix)]
    if let Some(unix) = config.unix {
        let shared = shared.clone();
        tokio::spawn(async move {
            loop {
                match unix.accept().await {
                    Ok((stream, _)) => connect(TokioIo::new(stream), false, shared.clone()),
                    Err(error) => {
                        tracing::error!(%error, "unix accept failed");
                        return;
                    }
                }
            }
        });
    }
    for tcp in config.tcp {
        let shared = shared.clone();
        tokio::spawn(async move {
            loop {
                match tcp.accept().await {
                    Ok((stream, _)) => connect(TokioIo::new(stream), true, shared.clone()),
                    Err(error) => {
                        tracing::error!(%error, "tcp accept failed");
                        return;
                    }
                }
            }
        });
    }

    #[cfg(unix)]
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    loop {
        // The idle deadline: pushed by every request, suspended while a
        // handler is in flight.
        let idle = shared.in_flight.load(Ordering::Relaxed) == 0;
        let deadline = if idle {
            *activity.borrow() + config.idle_exit
        } else {
            Instant::now() + Duration::from_secs(86400)
        };
        tokio::select! {
            _ = tokio::time::sleep_until(deadline.into()) => {
                if shared.in_flight.load(Ordering::Relaxed) == 0 {
                    return Ok(());
                }
            }
            _ = activity.changed() => {}
            _ = tokio::signal::ctrl_c() => return Ok(()),
            _ = term.recv() => return Ok(()),
            message = fatal_rx.recv() => {
                return Err(Error::Embedder(
                    message.unwrap_or_else(|| "the embedding worker is gone".to_owned()),
                ));
            }
        }
    }
}

/// Serves one connection on the `h1` service.
fn connect<IO>(io: IO, auth: bool, shared: Arc<Shared>)
where
    IO: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    let service = service_fn(move |request| {
        let shared = shared.clone();
        async move { Ok::<_, Infallible>(handle(shared, auth, request).await) }
    });
    tokio::spawn(async move {
        let _ = http1::Builder::new().serve_connection(io, service).await;
    });
}

/// The dispatcher thread: loads the model on every device (retrying on
/// request), then hands the most urgent pending job's same-bucket runs of
/// at most [`RUN_INPUTS`] to idle device workers and collects the finished
/// runs. A failed embed is fatal: the job gets the error and the server
/// exits.
fn dispatcher(
    dir: PathBuf,
    state: watch::Sender<Model>,
    stage: watch::Sender<LoadStage>,
    inbox: std_mpsc::Receiver<Msg>,
    events: std_mpsc::Sender<Msg>,
    retry: std_mpsc::Receiver<()>,
    fatal: mpsc::Sender<String>,
) {
    let embedders = loop {
        match OrtEmbedder::open_devices(&dir, &stage) {
            Ok(embedders) => {
                // The model id and encoder are identical across devices.
                let ready = Arc::new(Ready {
                    id: embedders[0].model_id().to_owned(),
                    encoder: embedders[0].encoder(),
                });
                if state.send(Model::Ready(ready)).is_err() {
                    return;
                }
                break embedders;
            }
            Err(error) => {
                tracing::error!(%error, "model load failed");
                if state.send(Model::Failed(error.to_string())).is_err() {
                    return;
                }
                // A `/v1/` request on the failed state sends one retry.
                if retry.recv().is_err() {
                    return;
                }
            }
        }
    };

    tracing::info!(devices = embedders.len(), "embedding devices serving");
    let buckets: Vec<usize> = embedders[0].buckets().collect();
    let workers: Vec<std_mpsc::Sender<Work>> = embedders
        .into_iter()
        .enumerate()
        .map(|(worker, embedder)| {
            let (work_tx, work_rx) = std_mpsc::channel::<Work>();
            let events = events.clone();
            std::thread::spawn(move || device_worker(worker, embedder, work_rx, events));
            work_tx
        })
        .collect();
    drop(events);

    let mut idle: Vec<usize> = (0..workers.len()).collect();
    let mut pending: Vec<Job> = Vec::new();
    let mut next_id = 0u64;
    loop {
        // Hand runs to idle devices, most urgent job first.
        while !idle.is_empty() {
            let Some(work) = assemble_run(&mut pending, &buckets) else {
                break;
            };
            let worker = idle.pop().expect("idle is nonempty");
            if workers[worker].send(work).is_err() {
                // The worker thread is gone: its embedder panicked.
                let _ = fatal.try_send("a device worker is gone".to_owned());
                return;
            }
        }
        match inbox.recv() {
            Ok(message) => {
                if !absorb(message, &mut pending, &mut next_id, &mut idle, &fatal) {
                    return;
                }
            }
            Err(_) => return,
        }
        while let Ok(message) = inbox.try_recv() {
            if !absorb(message, &mut pending, &mut next_id, &mut idle, &fatal) {
                return;
            }
        }
    }
}

/// Applies one message to the dispatcher's state. `false` when a run
/// failed: the job got the error and `fatal` was sent.
fn absorb(
    message: Msg,
    pending: &mut Vec<Job>,
    next_id: &mut u64,
    idle: &mut Vec<usize>,
    fatal: &mpsc::Sender<String>,
) -> bool {
    match message {
        Msg::Job(mut job) => {
            job.id = *next_id;
            *next_id += 1;
            pending.push(job);
        }
        Msg::Done {
            worker,
            parts,
            result,
        } => {
            idle.push(worker);
            match result {
                Ok(vectors) => {
                    assert_eq!(
                        parts.len(),
                        vectors.len(),
                        "a run returns one vector per input"
                    );
                    for ((id, index), vector) in parts.iter().zip(vectors) {
                        let position = pending
                            .iter()
                            .position(|candidate| candidate.id == *id)
                            .expect("a job stays pending until its inputs are placed");
                        pending[position].place(*index, vector);
                    }
                    for (id, _) in &parts {
                        if let Some(position) = pending.iter().position(|job| job.id == *id)
                            && pending[position].is_done()
                        {
                            pending.swap_remove(position).finish();
                        }
                    }
                }
                Err(message) => {
                    for (id, _) in &parts {
                        if let Some(position) = pending.iter().position(|job| job.id == *id) {
                            let _ = pending
                                .swap_remove(position)
                                .reply
                                .send(Err(message.clone()));
                        }
                    }
                    let _ = fatal.try_send(message);
                    return false;
                }
            }
        }
    }
    true
}

/// A device worker: embeds each run on its own session and reports it.
fn device_worker(
    worker: usize,
    mut embedder: OrtEmbedder,
    work: std_mpsc::Receiver<Work>,
    events: std_mpsc::Sender<Msg>,
) {
    while let Ok(Work { parts, batch }) = work.recv() {
        let result = embedder
            .embed_tokens_batch(&batch)
            .map_err(|error| error.to_string());
        let failed = result.is_err();
        let done = Msg::Done {
            worker,
            parts,
            result,
        };
        if events.send(done).is_err() || failed {
            return;
        }
    }
}

/// Routes a request; auth and the error envelope apply to every branch.
async fn handle(
    shared: Arc<Shared>,
    auth: bool,
    request: Request<Incoming>,
) -> Response<Full<Bytes>> {
    if auth && let Some(token) = &shared.token {
        let presented = request
            .headers()
            .get(hyper::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| {
                v.get(..7)
                    .filter(|scheme| scheme.eq_ignore_ascii_case("bearer "))
                    .map(|_| &v[7..])
            });
        if presented
            .is_none_or(|p| !constant_time_eq::constant_time_eq(p.as_bytes(), token.as_bytes()))
        {
            return reply(
                StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                "missing or invalid bearer token",
                Some("invalid_api_key"),
                &[],
            );
        }
    }
    match (request.method(), request.uri().path()) {
        (&Method::POST, "/v1/embeddings") => embeddings(shared, request).await,
        (&Method::GET, "/v1/models") => models(&shared),
        (&Method::GET, "/status") => status(&shared),
        (_, "/v1/embeddings") | (_, "/v1/models") | (_, "/status") => reply(
            StatusCode::METHOD_NOT_ALLOWED,
            "invalid_request_error",
            "the endpoint does not support this method",
            None,
            &[],
        ),
        _ => reply(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "unknown path",
            None,
            &[],
        ),
    }
}

/// `POST /v1/embeddings`: readiness, body cap, parse, tokenize, submit,
/// encode.
async fn embeddings(shared: Arc<Shared>, request: Request<Incoming>) -> Response<Full<Bytes>> {
    let _busy = Busy::new(&shared);
    let urgency = Urgency::from_header(
        request
            .headers()
            .get("priority")
            .and_then(|v| v.to_str().ok()),
    );
    let body = request.into_body();
    // A declared length over the cap is refused before a byte is read.
    if body.size_hint().lower() > BODY_LIMIT as u64 {
        return too_large();
    }
    let body = Limited::new(body, BODY_LIMIT);
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return too_large(),
    };
    let parsed: EmbeddingRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(error) => {
            return reply(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid request body: {error}"),
                None,
                &[],
            );
        }
    };
    let ready = match ready(&shared) {
        Found::Ready(ready) => ready,
        Found::Reply(response) => return response,
    };
    if parsed.model != ready.id {
        return reply(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            &format!("the model {} does not exist", parsed.model),
            Some("model_not_found"),
            &[],
        );
    }
    let inputs = match parsed.input {
        Input::One(text) => vec![text],
        Input::Many(texts) => texts,
    };
    if inputs.is_empty() || inputs.iter().any(String::is_empty) {
        return reply(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "every input must be a non-empty string",
            None,
            &[],
        );
    }
    if inputs.len() > MAX_INPUTS {
        return reply(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("a request carries at most {MAX_INPUTS} inputs"),
            None,
            &[],
        );
    }
    let dimensions = match parsed.dimensions {
        None => None,
        Some(d) if DIMENSIONS.contains(&d) => Some(d),
        Some(d) => {
            return reply(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("dimensions must be one of {DIMENSIONS:?}, got {d}"),
                None,
                &[],
            );
        }
    };
    let encoder = ready.encoder.clone();
    let ids = match tokio::task::spawn_blocking(move || {
        inputs
            .iter()
            .map(|input| encoder.encode(input))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    {
        Ok(Ok(ids)) => ids,
        Ok(Err(EmbedError::TooLong { tokens, max })) => {
            return reply(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("input encodes to {tokens} tokens, over the {max}-token maximum"),
                Some("context_length_exceeded"),
                &[],
            );
        }
        Ok(Err(error)) => {
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &error.to_string(),
                None,
                &[],
            );
        }
        Err(error) => {
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &format!("the request could not be tokenized: {error}"),
                None,
                &[],
            );
        }
    };
    let prompt_tokens = ids.iter().map(Vec::len).sum::<usize>() as u32;
    let (reply_tx, reply_rx) = oneshot::channel();
    if shared
        .worker
        .send(Msg::Job(Job::new(urgency, ids, reply_tx)))
        .is_err()
    {
        return reply(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "the embedding worker is gone",
            None,
            &[],
        );
    }
    let (vectors, timing) = match reply_rx.await {
        Ok(Ok(answered)) => answered,
        Ok(Err(message)) => {
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &message,
                None,
                &[],
            );
        }
        Err(_) => {
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "the embedding worker is gone",
                None,
                &[],
            );
        }
    };
    let format = parsed.encoding_format.unwrap_or(EncodingFormat::Float);
    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            let vector = match dimensions {
                Some(d) => UnitVector::normalize(vector.as_slice()[..d as usize].to_vec()),
                None => vector,
            };
            let embedding = match format {
                EncodingFormat::Float => EmbeddingData::Float(vector.as_slice().to_vec()),
                EncodingFormat::Base64 => EmbeddingData::Base64(encode_vector(vector.as_slice())),
            };
            repo_search_core::embed_api::Embedding {
                object: repo_search_core::embed_api::EmbeddingTag::Embedding,
                index: index as u32,
                embedding,
            }
        })
        .collect();
    let body = serde_json::to_string(&repo_search_core::embed_api::EmbeddingResponse {
        object: repo_search_core::embed_api::ListTag::List,
        data,
        model: ready.id.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
    .expect("the response serializes");
    json(
        StatusCode::OK,
        body,
        &[("server-timing", &timing.to_header())],
    )
}

/// The model is ready, or the request already has its reply.
enum Found {
    /// `Model::Ready`.
    Ready(Arc<Ready>),
    /// A non-ready state's wire answer.
    Reply(Response<Full<Bytes>>),
}

/// Resolves the model's readiness for `/v1/` handlers. A `Failed` state
/// fires one retry — the loader's sync inbox, so a full channel means a
/// retry is already queued.
fn ready(shared: &Shared) -> Found {
    match &*shared.state.borrow() {
        Model::Ready(ready) => Found::Ready(ready.clone()),
        Model::Loading => Found::Reply(not_ready("the model is loading")),
        Model::Failed(message) => {
            let _ = shared.retry.try_send(());
            Found::Reply(reply(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                message,
                Some("model_load_failed"),
                &[("retry-after", "1")],
            ))
        }
    }
}

/// The 503 `model_not_ready` envelope with `Retry-After: 1`.
fn not_ready(message: &str) -> Response<Full<Bytes>> {
    reply(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        message,
        Some("model_not_ready"),
        &[("retry-after", "1")],
    )
}

/// `GET /v1/models`: the one model this server embeds.
fn models(shared: &Shared) -> Response<Full<Bytes>> {
    match ready(shared) {
        Found::Ready(ready) => json(
            StatusCode::OK,
            serde_json::to_string(&ModelsResponse {
                object: repo_search_core::embed_api::ListTag::List,
                data: vec![ModelEntry {
                    id: ready.id.clone(),
                    object: repo_search_core::embed_api::ModelTag::Model,
                    created: 0,
                    owned_by: "repo-search".to_owned(),
                }],
            })
            .expect("the response serializes"),
            &[],
        ),
        Found::Reply(response) => response,
    }
}

/// `GET /status`: the server's own report — the model's state and load
/// stage, and the GPU instance this server never has.
fn status(shared: &Shared) -> Response<Full<Bytes>> {
    let model = match &*shared.state.borrow() {
        Model::Ready(ready) => ModelStatus::Ready {
            id: ready.id.clone(),
        },
        Model::Loading => ModelStatus::Loading {
            stage: *shared.stage.borrow(),
        },
        Model::Failed(message) => ModelStatus::Failed {
            message: message.clone(),
        },
    };
    json(
        StatusCode::OK,
        serde_json::to_string(&StatusBody {
            model,
            gpu: GpuStatus::Absent { blocked: None },
        })
        .expect("the response serializes"),
        &[],
    )
}

/// A JSON reply with the error envelope when `r#type` is set.
fn reply(
    status: StatusCode,
    r#type: &'static str,
    message: &str,
    code: Option<&'static str>,
    headers: &[(&'static str, &str)],
) -> Response<Full<Bytes>> {
    let body = serde_json::to_string(&ErrorBody {
        error: ErrorDetail {
            message: message.to_owned(),
            r#type: r#type.to_owned(),
            param: None,
            code: code.map(str::to_owned),
        },
    })
    .expect("the error envelope serializes");
    json(status, body, headers)
}

/// A JSON reply with extra headers.
fn json(
    status: StatusCode,
    body: String,
    headers: &[(&'static str, &str)],
) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status;
    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );
    for (name, value) in headers {
        response.headers_mut().insert(
            *name,
            hyper::header::HeaderValue::from_str(value)
                .expect("header values are structured fields"),
        );
    }
    response
}

/// The 413 answer for a body over the cap.
fn too_large() -> Response<Full<Bytes>> {
    reply(
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
        "the request body is too large",
        None,
        &[],
    )
}

/// The in-flight tracker and activity stamp a handler holds.
struct Busy {
    /// The shared state it updates on drop.
    shared: Arc<Shared>,
}

impl Busy {
    /// Counts the handler in and stamps activity.
    fn new(shared: &Arc<Shared>) -> Self {
        shared.in_flight.fetch_add(1, Ordering::Relaxed);
        shared.activity.send_replace(Instant::now());
        Self {
            shared: shared.clone(),
        }
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        self.shared.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.shared.activity.send_replace(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUCKETS: [usize; 3] = [4, 8, 16];

    type Reply = oneshot::Receiver<Result<(Vec<UnitVector>, ServerTiming), String>>;

    fn job(urgency: u8, lens: &[usize]) -> (Job, Reply) {
        let (tx, rx) = oneshot::channel();
        let ids = lens.iter().map(|&len| vec![1; len]).collect();
        (Job::new(Urgency::new(urgency), ids, tx), rx)
    }

    fn vector(value: f32) -> UnitVector {
        UnitVector::normalize(vec![value, 1.0])
    }

    fn indices(run: &[(usize, Vec<u32>)]) -> Vec<usize> {
        run.iter().map(|(index, _)| *index).collect()
    }

    /// A job with a fixed id, as the dispatcher numbers them.
    fn numbered(id: u64, urgency: u8, lens: &[usize]) -> (Job, Reply) {
        let (mut job, rx) = job(urgency, lens);
        job.id = id;
        (job, rx)
    }

    #[test]
    fn take_matching_takes_one_bucket_from_anywhere_in_input_order() {
        let (mut job, _rx) = job(3, &[2, 10, 3, 12, 4, 1]);
        let bucket = job.lead_bucket(&BUCKETS);
        let taken = job.take_matching(&BUCKETS, bucket, RUN_INPUTS);
        assert_eq!(indices(&taken), [0, 2, 4, 5]);
        assert_eq!(job.outstanding, 4);
        assert_eq!(indices(&job.remaining), [1, 3]);
    }

    #[test]
    fn take_matching_is_capped_at_room() {
        let (mut job, _rx) = job(3, &[2; RUN_INPUTS + 5]);
        assert_eq!(job.take_matching(&BUCKETS, 4, RUN_INPUTS).len(), RUN_INPUTS);
        assert_eq!(job.remaining.len(), 5);
        assert_eq!(job.remaining[0].0, RUN_INPUTS);
    }

    #[test]
    fn results_return_to_original_positions() {
        let (mut job, mut rx) = job(3, &[2, 10, 3]);
        let first = job.take_matching(&BUCKETS, 4, RUN_INPUTS);
        let second = job.take_matching(&BUCKETS, 16, RUN_INPUTS);
        assert_eq!(indices(&first), [0, 2]);
        assert_eq!(indices(&second), [1]);
        assert!(job.remaining.is_empty() && !job.is_done());
        // The runs finish out of order.
        job.place(1, vector(2.0));
        assert!(!job.is_done());
        job.place(0, vector(0.0));
        job.place(2, vector(3.0));
        assert!(job.is_done());
        job.finish();
        let (vectors, _) = rx.try_recv().expect("replied").expect("ok");
        assert_eq!(vectors, [vector(0.0), vector(2.0), vector(3.0)]);
    }

    #[test]
    fn a_run_gathers_single_input_requests() {
        let mut pending = Vec::new();
        let mut replies = Vec::new();
        for id in 0..(RUN_INPUTS + 8) as u64 {
            let (job, rx) = numbered(id, 3, &[2]);
            pending.push(job);
            replies.push(rx);
        }
        let first = assemble_run(&mut pending, &BUCKETS).expect("work");
        assert_eq!(first.batch.len(), RUN_INPUTS);
        assert_eq!(first.parts.len(), RUN_INPUTS);
        let second = assemble_run(&mut pending, &BUCKETS).expect("work");
        assert_eq!(second.batch.len(), 8);
        assert!(assemble_run(&mut pending, &BUCKETS).is_none());
    }

    #[test]
    fn a_run_keeps_to_the_lead_bucket_and_urgency() {
        let (query, _a) = numbered(0, 0, &[10]);
        let (mut same_urgency, _b) = numbered(1, 0, &[2, 12]);
        same_urgency.queued = query.queued + Duration::from_millis(1);
        let (bulk, _c) = numbered(2, 3, &[12]);
        let mut pending = vec![bulk, same_urgency, query];
        // The query leads (bucket 16); the bucket-16 input of the other
        // urgency-0 request joins, the bucket-4 input and the bulk job
        // do not.
        let run = assemble_run(&mut pending, &BUCKETS).expect("work");
        let mut parts = run.parts.clone();
        parts.sort_unstable();
        assert_eq!(parts, [(0, 0), (1, 1)]);
        assert_eq!(run.batch.len(), 2);
    }

    #[test]
    fn a_failed_run_fails_every_job_it_carried() {
        let (a, mut rx_a) = numbered(0, 3, &[2]);
        let (b, mut rx_b) = numbered(1, 3, &[2]);
        let (c, mut rx_c) = numbered(2, 3, &[2, 12]);
        let mut pending = vec![a, b, c];
        let run = assemble_run(&mut pending, &BUCKETS).expect("work");
        let (fatal, mut fatal_rx) = mpsc::channel(1);
        let keep_going = absorb(
            Msg::Done {
                worker: 0,
                parts: run.parts,
                result: Err("boom".to_owned()),
            },
            &mut pending,
            &mut 3,
            &mut Vec::new(),
            &fatal,
        );
        assert!(!keep_going);
        for rx in [&mut rx_a, &mut rx_b, &mut rx_c] {
            assert_eq!(
                rx.try_recv().expect("replied").err(),
                Some("boom".to_owned())
            );
        }
        assert_eq!(fatal_rx.try_recv().expect("fatal"), "boom");
    }

    #[test]
    fn a_finished_run_completes_the_jobs_it_carried() {
        let (a, mut rx_a) = numbered(0, 3, &[2]);
        let (b, mut rx_b) = numbered(1, 3, &[2, 12]);
        let mut pending = vec![a, b];
        let run = assemble_run(&mut pending, &BUCKETS).expect("work");
        assert_eq!(run.parts, [(0, 0), (1, 0)]);
        let (fatal, _fatal_rx) = mpsc::channel(1);
        let mut idle = Vec::new();
        assert!(absorb(
            Msg::Done {
                worker: 1,
                parts: run.parts,
                result: Ok(vec![vector(1.0), vector(2.0)]),
            },
            &mut pending,
            &mut 2,
            &mut idle,
            &fatal,
        ));
        assert_eq!(idle, [1]);
        // `a` is whole; `b` still has its bucket-16 input.
        let (vectors, _) = rx_a.try_recv().expect("replied").expect("ok");
        assert_eq!(vectors, [vector(1.0)]);
        assert!(rx_b.try_recv().is_err());
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn pick_prefers_urgency_then_age_and_skips_dispatched_jobs() {
        let (mut spent, _a) = job(0, &[2]);
        spent.take_matching(&BUCKETS, 4, RUN_INPUTS);
        let (older, _b) = job(2, &[2]);
        let (mut newer, _c) = job(2, &[2]);
        newer.queued = older.queued + Duration::from_millis(1);
        let (urgent, _d) = job(1, &[2]);
        let mut pending = vec![spent, newer, older, urgent];
        // Index 0 has nothing left to hand out; index 3 is the most urgent.
        assert_eq!(pick(&pending), Some(3));
        pending.pop();
        // Equal urgency: the earlier-queued job (index 2) wins.
        assert_eq!(pick(&pending), Some(2));
        assert_eq!(pick(&[]), None);
    }
}
