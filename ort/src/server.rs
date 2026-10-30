//! The ONNX embedding server: the same OpenAI-compatible HTTP API the
//! macOS Core ML server speaks — `POST /v1/embeddings`, `GET /v1/models`,
//! `GET /status`, the `Priority` header and the `Server-Timing` response —
//! over one [`OrtEmbedder`] owned by its own thread.
//!
//! Where the Core ML server splits queries onto the ANE and bulk indexing
//! onto a power-gated GPU, this server has the one session. Its scheduler
//! runs at most [`RUN_INPUTS`] of the most urgent pending request per run
//! and re-picks between runs, so a bulk index batch cannot park a query
//! behind its full length.
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

/// One embedding request's work for the worker thread. Runs are consumed
/// off `inputs` in order, so `vectors` stays in input order.
struct Job {
    /// The request's RFC 9218 urgency; lower runs first.
    urgency: Urgency,
    /// When the request was submitted, the `wait` metric's origin.
    queued: Instant,
    /// Already-tokenized inputs not yet embedded, in input order.
    inputs: std::collections::VecDeque<Vec<u32>>,
    /// The finished vectors, in input order.
    vectors: Vec<UnitVector>,
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

/// Everything a connection's handlers share.
struct Shared {
    /// The model's load state.
    state: watch::Receiver<Model>,
    /// The `open` pipeline's stage while the model loads.
    stage: watch::Receiver<LoadStage>,
    /// The loader's retry inbox: a `Failed` load retries on the next
    /// `/v1/` request.
    retry: std_mpsc::SyncSender<()>,
    /// The worker thread's job inbox.
    worker: std_mpsc::Sender<Job>,
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
    let (job_tx, job_rx) = std_mpsc::channel::<Job>();
    let (retry_tx, retry_rx) = std_mpsc::sync_channel::<()>(1);
    let (fatal_tx, mut fatal_rx) = mpsc::channel::<String>(1);
    let (activity_tx, mut activity) = watch::channel(Instant::now());

    let dir = config.model_dir;
    std::thread::spawn(move || {
        worker(dir, state_tx, stage_tx, job_rx, retry_rx, fatal_tx);
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

/// The worker thread: loads the model (retrying on request), then runs
/// jobs in urgency order, splitting each job into same-bucket runs of at
/// most [`RUN_INPUTS`]. A failed embed is fatal: the job gets the error
/// and the server exits.
fn worker(
    dir: PathBuf,
    state: watch::Sender<Model>,
    stage: watch::Sender<LoadStage>,
    jobs: std_mpsc::Receiver<Job>,
    retry: std_mpsc::Receiver<()>,
    fatal: mpsc::Sender<String>,
) {
    let mut embedder = loop {
        match OrtEmbedder::open(&dir, &stage) {
            Ok(embedder) => {
                let ready = Arc::new(Ready {
                    id: embedder.model_id().to_owned(),
                    encoder: embedder.encoder(),
                });
                if state.send(Model::Ready(ready)).is_err() {
                    return;
                }
                break embedder;
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

    let mut pending: Vec<Job> = Vec::new();
    loop {
        if pending.is_empty() {
            match jobs.recv() {
                Ok(job) => pending.push(job),
                Err(_) => return,
            }
        }
        while let Ok(job) = jobs.try_recv() {
            pending.push(job);
        }
        // The most urgent, oldest pending job.
        let pick = pending
            .iter()
            .enumerate()
            .min_by_key(|(_, job)| (job.urgency, job.queued))
            .map(|(index, _)| index)
            .expect("pending is nonempty");
        let mut job = pending.swap_remove(pick);
        if !job.counted {
            job.counted = true;
            job.wait = job.queued.elapsed();
            job.overtaken = pending
                .iter()
                .filter(|other| other.urgency > job.urgency)
                .count() as u64;
        }
        // One run: the longest same-bucket prefix of the job's remaining
        // inputs, capped at RUN_INPUTS.
        let first = job.inputs.front().expect("jobs hold inputs");
        let bucket = embedder
            .bucket_of(first.len())
            .expect("the encoder capped every input");
        let mut take = 0usize;
        for ids in &job.inputs {
            if take == RUN_INPUTS || embedder.bucket_of(ids.len()).ok() != Some(bucket) {
                break;
            }
            take += 1;
        }
        let batch: Vec<Vec<u32>> = job.inputs.drain(..take).collect();
        match embedder.embed_tokens_batch(&batch) {
            Ok(vectors) => job.vectors.extend(vectors),
            Err(error) => {
                let message = error.to_string();
                let _ = job.reply.send(Err(message.clone()));
                let _ = fatal.try_send(message);
                return;
            }
        }
        if job.inputs.is_empty() {
            let timing = ServerTiming {
                wait: job.wait,
                overtaken: job.overtaken,
            };
            let _ = job.reply.send(Ok((job.vectors, timing)));
        } else {
            pending.push(job);
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
    let n = ids.len();
    if shared
        .worker
        .send(Job {
            urgency,
            queued: Instant::now(),
            inputs: ids.into_iter().collect(),
            vectors: Vec::with_capacity(n),
            wait: Duration::ZERO,
            overtaken: 0,
            counted: false,
            reply: reply_tx,
        })
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
