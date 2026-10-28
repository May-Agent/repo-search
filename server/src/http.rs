//! The HTTP boundary: the OpenAI-compatible `POST /v1/embeddings` and
//! `GET /v1/models`, `GET /status` (the server's own report: model state
//! with download progress and load stage, plus the GPU state watch), the
//! bearer-token check, the body cap and the `Priority`/`Server-Timing`
//! headers.
//!
//! Every handler resolves to a wire response; no error escapes as a
//! panic. The service is `h1` only: the daemon sends `u=` urgencies over
//! `Priority`, and keep-alive connections stay open between them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Body, Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::sync::{mpsc, oneshot, watch};

use repo_search_core::embed_api::{
    Embedding, EmbeddingData, EmbeddingResponse, EncodingFormat, Input, ListTag, ModelEntry,
    ModelsResponse, Urgency, encode_vector,
};
use repo_search_core::model::MODEL_ID;

use crate::backend::{Backend, LoadStage, TextEncoder};
use crate::loader::{LoaderMsg, ModelState};
use crate::scheduler::{GpuState, RequestId, SchedulerMsg, Submit};
use crate::{BODY_LIMIT, BusySource};

/// The dimensions Matryoshka truncation accepts.
const DIMENSIONS: [u32; 4] = [128, 256, 512, 768];

/// The most inputs a request may carry.
const MAX_INPUTS: usize = 2048;

/// Everything a connection's handlers share: the readiness watch, the
/// loader's inbox for `Retry`, the status watches, the bearer token and
/// the request counter.
pub struct Shared<B: Backend> {
    /// The model's current state.
    pub state: watch::Receiver<ModelState<B>>,
    /// The loader's inbox: a `Failed` state triggers one `Retry`.
    pub loader: mpsc::Sender<LoaderMsg>,
    /// The GPU instance's state for `GET /status`.
    pub gpu: watch::Receiver<GpuState>,
    /// The `open_ane` stage while the model loads, for `GET /status`.
    pub stage: watch::Receiver<LoadStage>,
    /// The bearer token TCP listeners require; the unix socket needs
    /// none.
    pub token: Option<String>,
    /// The next request's id.
    pub next_request: AtomicU64,
    /// Request tasks in flight: a busy source for the idle-exit tracker.
    pub in_flight: Arc<InFlight>,
}

/// Counts request handlers running; `busy` is `true` while one or more
/// are — the idle deadline counts requests, not keep-alive connections.
/// The count lives inside the watch, so `send_modify` makes the counter
/// and its busy derivation race-free. `Shared` holds it in an `Arc` for
/// the drop guards.
pub struct InFlight {
    count: watch::Sender<u64>,
}

impl InFlight {
    /// A fresh counter with its busy receiver, registered as a
    /// [`BusySource`] with the run loop.
    pub fn new() -> (Self, BusySource, watch::Receiver<u64>) {
        let (count, mut watch_rx) = watch::channel(0u64);
        let (busy_tx, busy_rx) = watch::channel(false);
        let count_rx = watch_rx.clone();
        tokio::spawn(async move {
            while watch_rx.changed().await.is_ok() {
                busy_tx.send_replace(*watch_rx.borrow() != 0);
            }
        });
        (Self { count }, BusySource { state: busy_rx }, count_rx)
    }

    /// Marks a request in flight until the guard drops.
    fn track(self: &Arc<Self>) -> InFlightGuard {
        self.count.send_modify(|c| *c += 1);
        InFlightGuard(self.clone())
    }
}

/// Drops the request's busy mark when the handler ends.
struct InFlightGuard(Arc<InFlight>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.count.send_modify(|c| *c -= 1);
    }
}

/// The accept loops: one per listener, feeding each connection into a
/// `GracefulShutdown`-watched `h1` service.
pub struct Conns {
    unix: tokio::net::UnixListener,
    tcp: Vec<tokio::net::TcpListener>,
    graceful: Arc<GracefulShutdown>,
}

/// Builds the connection registry; [`Conns::run`] then starts accepting.
pub fn serve(unix: tokio::net::UnixListener, tcp: Vec<tokio::net::TcpListener>) -> Conns {
    Conns {
        unix,
        tcp,
        graceful: Arc::new(GracefulShutdown::new()),
    }
}

/// The accept tasks [`Conns::run`] spawned.
pub struct Accepts {
    graceful: Arc<GracefulShutdown>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Conns {
    /// Starts the accept loops against `shared`; returns the shutdown
    /// handle and the task handles `run` aborts on exit.
    pub fn run<B: Backend>(self, shared: Arc<Shared<B>>) -> Accepts {
        let mut tasks = Vec::with_capacity(1 + self.tcp.len());
        tasks.push(tokio::spawn(accept_unix(
            self.unix,
            shared.clone(),
            self.graceful.clone(),
        )));
        for listener in self.tcp {
            tasks.push(tokio::spawn(accept_tcp(
                listener,
                shared.clone(),
                self.graceful.clone(),
            )));
        }
        Accepts {
            graceful: self.graceful,
            tasks,
        }
    }
}

impl Accepts {
    /// Stops accepting and closes every connection after its in-flight
    /// request ends.
    pub async fn shutdown(self) {
        for task in &self.tasks {
            task.abort();
        }
        for task in self.tasks {
            let _ = task.await;
        }
        let graceful = Arc::try_unwrap(self.graceful).expect("accept loops are aborted");
        graceful.shutdown().await;
    }
}

/// One accept loop for the unix socket: connections are trusted — the
/// socket's permissions are the check.
async fn accept_unix<B: Backend>(
    listener: tokio::net::UnixListener,
    shared: Arc<Shared<B>>,
    graceful: Arc<GracefulShutdown>,
) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => connect(
                TokioIo::new(stream),
                false,
                shared.clone(),
                graceful.clone(),
            ),
            Err(error) => {
                tracing::error!(%error, "unix accept failed");
                return;
            }
        }
    }
}

/// One accept loop for a TCP listener: connections authenticate when the
/// server has a token.
async fn accept_tcp<B: Backend>(
    listener: tokio::net::TcpListener,
    shared: Arc<Shared<B>>,
    graceful: Arc<GracefulShutdown>,
) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                connect(TokioIo::new(stream), true, shared.clone(), graceful.clone())
            }
            Err(error) => {
                tracing::error!(%error, "tcp accept failed");
                return;
            }
        }
    }
}

/// Serves one connection on the `h1` service under the graceful registry.
fn connect<B, IO>(io: IO, auth: bool, shared: Arc<Shared<B>>, graceful: Arc<GracefulShutdown>)
where
    B: Backend,
    IO: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    let service = service_fn(move |request| {
        let shared = shared.clone();
        async move { Ok::<_, std::convert::Infallible>(handle(shared, auth, request).await) }
    });
    let conn = hyper::server::conn::http1::Builder::new().serve_connection(io, service);
    tokio::spawn(graceful.watcher().watch(conn));
}

/// Routes a request; auth and the error envelope apply to every branch.
async fn handle<B: Backend>(
    shared: Arc<Shared<B>>,
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
        (&Method::GET, "/v1/models") => models(shared),
        (&Method::GET, "/status") => status(shared),
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
async fn embeddings<B: Backend>(
    shared: Arc<Shared<B>>,
    request: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let _in_flight = shared.in_flight.track();
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
    let parsed: repo_search_core::embed_api::EmbeddingRequest = match serde_json::from_slice(&bytes)
    {
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
    if parsed.model != MODEL_ID {
        return reply(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            &format!(
                "the model {parsed_model} does not exist",
                parsed_model = parsed.model
            ),
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
    let model = match ready(&shared) {
        Ready::Ready(model) => model,
        Ready::Reply(response) => return response,
    };
    let encoder = model.encoder.clone();
    let ids = match tokio::task::spawn_blocking(move || {
        inputs
            .iter()
            .map(|input| encoder.encode(input))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    {
        Ok(Ok(ids)) => ids,
        Ok(Err(crate::backend::EncodeError::TooLong(message))) => {
            return reply(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &message,
                Some("context_length_exceeded"),
                &[],
            );
        }
        Ok(Err(crate::backend::EncodeError::Failed(message))) => {
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &message,
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
    let request_id = RequestId(shared.next_request.fetch_add(1, Ordering::Relaxed));
    let (reply_tx, reply_rx) = oneshot::channel();
    let _cancel = CancelGuard {
        request: request_id,
        tx: model.scheduler.tx(),
    };
    if model
        .scheduler
        .submit(Submit {
            request: request_id,
            urgency,
            inputs: ids,
            reply: reply_tx,
        })
        .await
        .is_err()
    {
        return reply(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "the embedding scheduler is gone",
            None,
            &[],
        );
    }
    let answered = match reply_rx.await {
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
                "the embedding scheduler is gone",
                None,
                &[],
            );
        }
    };
    let format = parsed.encoding_format.unwrap_or(EncodingFormat::Float);
    let data = answered
        .vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            let vector = match dimensions {
                Some(d) => repo_search_core::UnitVector::normalize(
                    vector.as_slice()[..d as usize].to_vec(),
                ),
                None => vector,
            };
            let embedding = match format {
                EncodingFormat::Float => EmbeddingData::Float(vector.as_slice().to_vec()),
                EncodingFormat::Base64 => EmbeddingData::Base64(encode_vector(vector.as_slice())),
            };
            Embedding {
                object: repo_search_core::embed_api::EmbeddingTag::Embedding,
                index: index as u32,
                embedding,
            }
        })
        .collect();
    let body = serde_json::to_string(&EmbeddingResponse {
        object: ListTag::List,
        data,
        model: model.id.clone(),
        usage: repo_search_core::embed_api::Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
    .expect("the response serializes");
    json(
        StatusCode::OK,
        body,
        &[("server-timing", &answered.timing.to_header())],
    )
}

/// The model is ready, or the request already has its reply.
enum Ready<B: Backend> {
    /// `ModelState::Ready`.
    Ready(Arc<crate::loader::Model<B>>),
    /// A non-ready state's wire answer.
    Reply(Response<Full<Bytes>>),
}

/// Resolves the model's readiness for `/v1/` handlers. A `Failed` state
/// fires one `Retry` — a full inbox only delays it to the next request.
fn ready<B: Backend>(shared: &Shared<B>) -> Ready<B> {
    match &*shared.state.borrow() {
        ModelState::Ready(model) => Ready::Ready(model.clone()),
        ModelState::Absent => Ready::Reply(not_ready("the model has not started loading")),
        ModelState::Downloading { received, total } => {
            let message = match total {
                Some(total) => format!("downloading {received}/{total} bytes"),
                None => format!("downloading {received} bytes"),
            };
            Ready::Reply(not_ready(&message))
        }
        ModelState::Loading => Ready::Reply(not_ready("the model is loading")),
        ModelState::Failed(message) => {
            let _ = shared.loader.try_send(LoaderMsg::Retry);
            Ready::Reply(reply(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                message,
                Some("model_load_failed"),
                &[("retry-after", "1")],
            ))
        }
        ModelState::Fatal(message) => Ready::Reply(reply(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            message,
            Some("model_load_failed"),
            &[],
        )),
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
fn models<B: Backend>(shared: Arc<Shared<B>>) -> Response<Full<Bytes>> {
    match ready(&shared) {
        Ready::Ready(_) => json(
            StatusCode::OK,
            serde_json::to_string(&ModelsResponse {
                object: ListTag::List,
                data: vec![ModelEntry {
                    id: MODEL_ID.to_owned(),
                    object: repo_search_core::embed_api::ModelTag::Model,
                    created: 0,
                    owned_by: "repo-search".to_owned(),
                }],
            })
            .expect("the response serializes"),
            &[],
        ),
        Ready::Reply(response) => response,
    }
}

/// `GET /status`: the server's own report — the model's state (with the
/// archive download's `received`/`total` and the `open_ane` stage) and the
/// GPU instance's state.
fn status<B: Backend>(shared: Arc<Shared<B>>) -> Response<Full<Bytes>> {
    use repo_search_core::embed_api::{GpuStatus, ModelStatus, StatusBody};
    let model = match &*shared.state.borrow() {
        ModelState::Absent => ModelStatus::Absent,
        ModelState::Downloading { received, total } => ModelStatus::Downloading {
            received: *received,
            total: *total,
        },
        ModelState::Loading => ModelStatus::Loading {
            stage: load_stage(*shared.stage.borrow()),
        },
        ModelState::Ready(model) => ModelStatus::Ready {
            id: model.id.clone(),
        },
        ModelState::Failed(message) => ModelStatus::Failed {
            message: message.clone(),
        },
        ModelState::Fatal(message) => ModelStatus::Fatal {
            message: message.clone(),
        },
    };
    let gpu = match &*shared.gpu.borrow() {
        GpuState::Absent { blocked } => GpuStatus::Absent {
            blocked: blocked.clone(),
        },
        GpuState::Loading => GpuStatus::Loading,
        GpuState::Running => GpuStatus::Running,
    };
    json(
        StatusCode::OK,
        serde_json::to_string(&StatusBody { model, gpu }).expect("the status serializes"),
        &[],
    )
}

/// The wire `LoadStage` for the backend's stage enum.
fn load_stage(stage: LoadStage) -> repo_search_core::embed_api::LoadStage {
    use repo_search_core::embed_api::LoadStage as Wire;
    match stage {
        LoadStage::Encoder => Wire::Encoder,
        LoadStage::Open => Wire::Open,
        LoadStage::WarmUp => Wire::WarmUp,
    }
}

/// Cancels the scheduler request when the handler's future drops — a
/// closed connection while the request is queued.
struct CancelGuard {
    request: RequestId,
    tx: mpsc::Sender<SchedulerMsg>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        // A full inbox only delays the cancel to the next `is_closed`
        // purge, never loses it.
        let _ = self.tx.try_send(SchedulerMsg::Cancel(self.request));
    }
}

/// The error envelope as JSON text.
pub fn error_json(kind: &str, message: &str, code: Option<&str>) -> String {
    serde_json::to_string(&repo_search_core::embed_api::ErrorBody {
        error: repo_search_core::embed_api::ErrorDetail {
            message: message.to_owned(),
            r#type: kind.to_owned(),
            param: None,
            code: code.map(str::to_owned),
        },
    })
    .expect("the envelope serializes")
}

/// An error-envelope reply.
fn reply(
    status: StatusCode,
    kind: &str,
    message: &str,
    code: Option<&str>,
    headers: &[(&str, &str)],
) -> Response<Full<Bytes>> {
    json(status, error_json(kind, message, code), headers)
}

/// The 413 envelope both body-size checks — the declared length and
/// `Limited`'s read — send.
fn too_large() -> Response<Full<Bytes>> {
    reply(
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
        "the request body exceeds the 32 MiB cap",
        None,
        &[],
    )
}

/// The 503 `connection: close` response `refuse_http` serves.
pub fn refuse_503(body: String) -> Response<Full<Bytes>> {
    let mut response = json(StatusCode::SERVICE_UNAVAILABLE, body, &[]);
    response.headers_mut().insert(
        hyper::header::CONNECTION,
        hyper::header::HeaderValue::from_static("close"),
    );
    response
}

/// A JSON reply with `content-type` and extra headers.
fn json(status: StatusCode, body: String, headers: &[(&str, &str)]) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status;
    let map = response.headers_mut();
    map.insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );
    for (name, value) in headers {
        map.insert(
            name.parse::<hyper::header::HeaderName>()
                .expect("a server-built header name"),
            value
                .parse::<hyper::header::HeaderValue>()
                .expect("a server-built header value"),
        );
    }
    response
}
