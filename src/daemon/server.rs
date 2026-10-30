//! The daemon's HTTP client of the embedding server.
//!
//! [`ServerClient`] speaks the standard `POST /v1/embeddings` — the same
//! request a stock OpenAI client sends, plus the RFC 9218 `Priority`
//! header that carries the scheduler's urgency across the boundary.
//! Requests go over the server's unix socket in production; a TCP
//! endpoint serves the forwarded-socket case. A transport failure is
//! retried once — embedding is idempotent and the retry covers the
//! server's idle-exit race; a second failure or any non-2xx status is
//! fatal, with no fallback.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use repo_search_core::UnitVector;
use repo_search_core::embed_api::{
    EmbeddingData, EmbeddingRequest, EncodingFormat, ErrorBody, Input, ModelsResponse,
    ServerTiming, StatusBody, Urgency, decode_vector,
};
use repo_search_core::model::MODEL_ID;

/// The request's connect deadline; there is no total timeout, so a slow
/// answer under load is never cut off.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Idle pooled connections are dropped after this long.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Idle pooled connections kept for the request bursts of an index pass.
const POOL_MAX_IDLE: usize = 40;

/// Where the embedding server listens.
#[derive(Debug, Clone)]
pub enum Endpoint {
    /// A unix socket: launchd's `server.sock` in production.
    Unix(PathBuf),
    /// Loopback or forwarded TCP, with the optional bearer token a
    /// non-loopback server requires.
    Tcp {
        /// The server's `host:port`.
        addr: SocketAddr,
        /// The bearer token of `server.toml`, when the bind needs one.
        token: Option<String>,
    },
}

/// One embeddings answer: a vector per input, in input order, and the
/// request's `Server-Timing` metrics.
#[derive(Debug)]
pub struct Answered {
    /// The vectors, in input order.
    pub vectors: Vec<UnitVector>,
    /// How the server's scheduler ran the request.
    pub timing: ServerTiming,
}

/// The daemon's embeddings service — [`ServerClient`] in production; the
/// scheduler-level tests drive a recording fake.
pub trait Embeddings: Send + Sync + 'static {
    /// The request's failure.
    type Error: RequestError;

    /// Embeds `inputs` in one request at `urgency`.
    fn embed(
        &self,
        inputs: Vec<String>,
        urgency: Urgency,
    ) -> impl Future<Output = Result<Answered, Self::Error>> + Send;
}

/// An embed request's failure. Every variant is fatal to the daemon: the
/// one retry lives inside [`ServerClient`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request never got an answer — connect, write or read failed.
    /// Retried once, since the server may idle-exit mid-pool.
    #[error("embedding server: {0}")]
    Transport(#[source] reqwest::Error),
    /// A non-2xx status; the server's error message when it sent one.
    #[error("embedding server answered {status}: {message}")]
    Status {
        /// The HTTP status code.
        status: u16,
        /// `error.message` of the standard envelope, else the raw body.
        message: String,
        /// `error.code` of the standard envelope, when it sent one.
        code: Option<String>,
    },
    /// A 2xx answer that did not parse or carry base64 vectors.
    #[error("embedding server sent a malformed response: {0}")]
    Response(String),
}

impl Error {
    /// Whether the failure is a transport error: the one case the client
    /// retries.
    fn is_transport(&self) -> bool {
        matches!(self, Self::Transport(_))
    }
}

/// What an [`Embeddings`] failure means for its caller: only a client
/// error may fail one job; every other failure means the server is
/// unhealthy and is fatal to the daemon.
pub trait RequestError: std::error::Error + Send + Sync + 'static {
    /// Whether the request's own inputs were refused (a 4xx); the server
    /// itself is fine.
    fn is_client_error(&self) -> bool;
}

impl RequestError for Error {
    fn is_client_error(&self) -> bool {
        matches!(self, Self::Status { status, .. } if (400..500).contains(status))
    }
}

/// A reqwest client of the embedding server, built for the daemon's
/// request bursts: HTTP/1.1 only, no proxy, a pool sized to the
/// scheduler's in-flight window, a connect deadline and no total timeout.
pub struct ServerClient {
    client: reqwest::Client,
    url: String,
    token: Option<String>,
}

impl ServerClient {
    /// The client for `endpoint`.
    ///
    /// # Errors
    ///
    /// When the client cannot be built (a TLS or resolver failure).
    pub fn new(endpoint: &Endpoint) -> Result<Self, reqwest::Error> {
        let builder = reqwest::Client::builder()
            .http1_only()
            .no_proxy()
            .pool_max_idle_per_host(POOL_MAX_IDLE)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT);
        let (builder, url, token) = match endpoint {
            Endpoint::Unix(path) => (
                builder.unix_socket(path.clone()),
                "http://localhost".to_owned(),
                None,
            ),
            Endpoint::Tcp { addr, token } => (builder, format!("http://{addr}"), token.clone()),
        };
        Ok(Self {
            client: builder.build()?,
            url,
            token,
        })
    }

    /// One `POST /v1/embeddings` attempt: inputs at `urgency`, vectors as
    /// base64. Non-2xx answers come back with the envelope's message.
    async fn post(&self, inputs: &[String], urgency: Urgency) -> Result<Answered, Error> {
        let mut request = self
            .client
            .post(format!("{}/v1/embeddings", self.url))
            .header("priority", urgency.to_header())
            .json(&EmbeddingRequest {
                model: MODEL_ID.to_owned(),
                input: Input::Many(inputs.to_vec()),
                dimensions: None,
                encoding_format: Some(EncodingFormat::Base64),
                user: None,
            });
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.map_err(Error::Transport)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            let envelope = serde_json::from_slice::<ErrorBody>(&body).ok();
            let message = envelope
                .as_ref()
                .map(|body| body.error.message.clone())
                .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
            return Err(Error::Status {
                status: status.as_u16(),
                message,
                code: envelope.and_then(|body| body.error.code),
            });
        }
        let timing = response
            .headers()
            .get("server-timing")
            .and_then(|value| value.to_str().ok())
            .map(ServerTiming::from_header)
            .unwrap_or_default();
        let body = response.bytes().await.map_err(Error::Transport)?;
        let body: repo_search_core::embed_api::EmbeddingResponse =
            serde_json::from_slice(&body).map_err(|error| Error::Response(error.to_string()))?;
        let mut vectors = Vec::with_capacity(body.data.len());
        for entry in body.data {
            let EmbeddingData::Base64(text) = entry.embedding else {
                return Err(Error::Response(
                    "the server answered a non-base64 vector".to_owned(),
                ));
            };
            let floats =
                decode_vector(&text).map_err(|error| Error::Response(error.to_string()))?;
            vectors.push(UnitVector::normalize(floats));
        }
        Ok(Answered { vectors, timing })
    }
}

/// A `GET /v1/models` readiness probe's answer.
#[derive(Debug)]
pub enum Probe {
    /// The server is ready and lists these model ids.
    Listed(Vec<String>),
    /// `model_not_ready`; probe again after `retry_after`.
    NotReady {
        /// The server's `Retry-After`, 1 s when absent or unparsable.
        retry_after: Duration,
    },
    /// `model_load_failed`; the probe itself already fired the server
    /// loader's `Retry`, as any `/v1/` request on a `Failed` model does.
    Failed(String),
}

impl ServerClient {
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let request = self.client.get(format!("{}{path}", self.url));
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// `GET /v1/models` as a readiness probe: the listed ids, the
    /// not-ready delay or the load failure's message. Other statuses and
    /// transport failures come back as [`Error`].
    pub async fn probe(&self) -> Result<Probe, Error> {
        let response = self
            .get("/v1/models")
            .send()
            .await
            .map_err(Error::Transport)?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .map(Duration::from_secs_f64)
            .unwrap_or(Duration::from_secs(1));
        let body = response.bytes().await.map_err(Error::Transport)?;
        if status.is_success() {
            let body: ModelsResponse =
                serde_json::from_slice(&body).map_err(|e| Error::Response(e.to_string()))?;
            return Ok(Probe::Listed(
                body.data.into_iter().map(|entry| entry.id).collect(),
            ));
        }
        let envelope = serde_json::from_slice::<ErrorBody>(&body).ok();
        let message = envelope
            .as_ref()
            .map(|body| body.error.message.clone())
            .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
        match envelope.and_then(|body| body.error.code).as_deref() {
            Some("model_not_ready") => Ok(Probe::NotReady { retry_after }),
            Some("model_load_failed") => Ok(Probe::Failed(message)),
            _ => Err(Error::Status {
                status: status.as_u16(),
                message,
                code: None,
            }),
        }
    }

    /// `GET /status`: the server's own model and GPU report, for
    /// `search status`.
    pub async fn status(&self) -> Result<StatusBody, Error> {
        let response = self.get("/status").send().await.map_err(Error::Transport)?;
        let status = response.status();
        let body = response.bytes().await.map_err(Error::Transport)?;
        if !status.is_success() {
            let envelope = serde_json::from_slice::<ErrorBody>(&body).ok();
            let message = envelope
                .as_ref()
                .map(|body| body.error.message.clone())
                .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
            return Err(Error::Status {
                status: status.as_u16(),
                message,
                code: envelope.and_then(|body| body.error.code),
            });
        }
        serde_json::from_slice(&body).map_err(|e| Error::Response(e.to_string()))
    }
}

impl Embeddings for ServerClient {
    type Error = Error;

    /// Embeds `inputs`, retrying once on a transport error — a pooled
    /// connection to a server that idle-exited looks exactly like one.
    async fn embed(&self, inputs: Vec<String>, urgency: Urgency) -> Result<Answered, Error> {
        match self.post(&inputs, urgency).await {
            Err(error) if error.is_transport() => self.post(&inputs, urgency).await,
            result => result,
        }
    }
}
