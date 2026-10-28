//! A recording fake of the standard embeddings API, plus the
//! connection-level `serve` the `ServerClient` tests feed killed
//! connections and arbitrary answers.
//!
//! Every request is recorded — which connection it came on, its path,
//! `Priority`, `Authorization` and JSON body — on the channel the test
//! holds. [`FakeServer`] answers the three endpoints the daemon calls,
//! deriving each input's vector deterministically; [`serve`] and
//! [`serve_on`] hand each accepted connection's requests to the test's
//! own handler. Wire types come from `repo_search_core::embed_api`, so
//! nothing about the protocol is duplicated.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use repo_search_core::embed_api::{
    Embedding, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingTag, ErrorBody,
    ErrorDetail, GpuStatus, Input, ListTag, ModelEntry, ModelStatus, ModelTag, ModelsResponse,
    StatusBody, Usage, encode_vector,
};

const TIMEOUT: Duration = Duration::from_secs(30);

/// The width of the fake's deterministic vectors.
const DIM: usize = 64;

/// What the server recorded of a request.
pub struct Recorded {
    /// Which accepted connection the request came on, from 1.
    pub conn: usize,
    /// The request path.
    pub path: String,
    /// The `Priority` header, as sent.
    pub priority: Option<String>,
    /// The `Authorization` header, as sent.
    pub authorization: Option<String>,
    /// The JSON body.
    pub body: serde_json::Value,
}

/// One request, handed to a `serve` handler.
pub struct Call {
    /// The request method.
    pub method: String,
    /// The request path.
    pub path: String,
    /// The `Authorization` header, as sent.
    pub authorization: Option<String>,
    /// The request body.
    pub body: bytes::Bytes,
}

/// Where a test server listens.
pub enum Listen {
    /// A unix socket at the path.
    Unix(PathBuf),
    /// A TCP socket at the address.
    Tcp(SocketAddr),
}

impl Listen {
    /// An ephemeral loopback TCP port.
    pub fn loopback() -> Self {
        Self::Tcp(SocketAddr::from(([127, 0, 0, 1], 0)))
    }
}

/// Where a running test server ended up listening.
pub enum Address {
    /// A unix socket at the path.
    Unix(PathBuf),
    /// A TCP socket at the address.
    Tcp(SocketAddr),
}

impl Address {
    /// The TCP address; panics for a unix server.
    pub fn addr(&self) -> SocketAddr {
        match self {
            Self::Tcp(addr) => *addr,
            Self::Unix(path) => panic!("a unix server has no TCP address: {}", path.display()),
        }
    }

    /// The socket path; panics for a TCP server.
    pub fn path(&self) -> &Path {
        match self {
            Self::Unix(path) => path,
            Self::Tcp(addr) => panic!("a TCP server has no socket path: {addr}"),
        }
    }
}

/// A running server and its request record.
pub struct TestServer {
    /// Where it listens.
    pub address: Address,
    requests: mpsc::Receiver<Recorded>,
}

impl TestServer {
    /// The TCP address, for `Endpoint::Tcp`.
    pub fn addr(&self) -> SocketAddr {
        self.address.addr()
    }

    /// The socket path, for `Endpoint::Unix`.
    pub fn path(&self) -> &Path {
        self.address.path()
    }

    /// The next request the server recorded.
    pub fn next(&self) -> Recorded {
        tokio::task::block_in_place(|| {
            self.requests
                .recv_timeout(TIMEOUT)
                .expect("the request arrives")
        })
    }

    /// The connections of every request recorded so far, in order.
    pub fn conns(&self) -> Vec<usize> {
        self.requests
            .try_iter()
            .map(|request| request.conn)
            .collect()
    }

    /// Every request recorded so far, in order.
    pub fn recorded(&self) -> Vec<Recorded> {
        self.requests.try_iter().collect()
    }
}

/// A `TokioIo`-acceptable stream of either socket kind.
trait Io: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send {}
impl<T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send> Io for T {}

enum Incoming {
    Unix(tokio::net::UnixListener),
    Tcp(tokio::net::TcpListener),
}

impl Incoming {
    async fn accept(&self) -> std::io::Result<Box<dyn Io>> {
        match self {
            Self::Unix(listener) => listener
                .accept()
                .await
                .map(|(s, _)| Box::new(s) as Box<dyn Io>),
            Self::Tcp(listener) => listener
                .accept()
                .await
                .map(|(s, _)| Box::new(s) as Box<dyn Io>),
        }
    }
}

/// The response type the handlers build.
pub type Response = hyper::Response<http_body_util::Full<bytes::Bytes>>;

/// A TCP server on an ephemeral loopback port — `serve_on` at
/// [`Listen::loopback`].
pub fn serve<F>(handler: F) -> TestServer
where
    F: Fn(usize, &Call) -> Option<Response> + Send + Sync + 'static,
{
    serve_on(Listen::loopback(), handler)
}

/// A recording hyper test server on `listen`: every accepted connection
/// serves requests that `handler` answers — `None` kills the connection
/// without a response. The handler sees the connection number and the
/// request; every request is recorded first.
pub fn serve_on<F>(listen: Listen, handler: F) -> TestServer
where
    F: Fn(usize, &Call) -> Option<Response> + Send + Sync + 'static,
{
    let (incoming, address) = match listen {
        Listen::Unix(path) => {
            let listener = std::os::unix::net::UnixListener::bind(&path).expect("bind");
            listener.set_nonblocking(true).expect("nonblocking");
            (
                Incoming::Unix(tokio::net::UnixListener::from_std(listener).expect("adopt")),
                Address::Unix(path),
            )
        }
        Listen::Tcp(addr) => {
            let listener = std::net::TcpListener::bind(addr).expect("bind");
            listener.set_nonblocking(true).expect("nonblocking");
            let addr = listener.local_addr().expect("local addr");
            (
                Incoming::Tcp(tokio::net::TcpListener::from_std(listener).expect("adopt")),
                Address::Tcp(addr),
            )
        }
    };
    let (tx, requests) = mpsc::channel();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        let mut conn = 0usize;
        loop {
            let Ok(stream) = incoming.accept().await else {
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
                            let header = |name: &str| {
                                parts
                                    .headers
                                    .get(name)
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_owned)
                            };
                            let _ = tx.send(Recorded {
                                conn: n,
                                path: parts.uri.path().to_owned(),
                                priority: header("priority"),
                                authorization: header("authorization"),
                                body: serde_json::from_slice(&body).unwrap_or_default(),
                            });
                            let call = Call {
                                method: parts.method.as_str().to_owned(),
                                path: parts.uri.path().to_owned(),
                                authorization: header("authorization"),
                                body,
                            };
                            match handler(n, &call) {
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
    TestServer { address, requests }
}

/// A JSON response, with an optional `Server-Timing` header.
pub fn json_response(status: u16, body: serde_json::Value, timing: Option<&str>) -> Response {
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
pub fn embeddings_body(vectors: &[&[f32]]) -> serde_json::Value {
    serde_json::to_value(EmbeddingResponse {
        object: ListTag::List,
        data: vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| Embedding {
                object: EmbeddingTag::Embedding,
                index: index as u32,
                embedding: EmbeddingData::Base64(encode_vector(vector)),
            })
            .collect(),
        model: repo_search_core::model::MODEL_ID.to_owned(),
        usage: Usage {
            prompt_tokens: 1,
            total_tokens: 1,
        },
    })
    .expect("the response serializes")
}

/// A standard error envelope body.
pub fn error_body(message: &str) -> serde_json::Value {
    serde_json::to_value(ErrorBody {
        error: ErrorDetail {
            message: message.to_owned(),
            r#type: "server_error".to_owned(),
            param: None,
            code: None,
        },
    })
    .expect("the envelope serializes")
}

/// What the fake serves: a fixed model id and an optional required
/// bearer token.
pub struct FakeConfig {
    /// The model id `GET /v1/models` lists and `/status` reports ready.
    pub model_id: String,
    /// The token every request must carry as `Bearer`, when set; a
    /// missing or wrong one is a 401.
    pub token: Option<String>,
}

/// The standard-API fake on a unix or TCP socket.
pub struct FakeServer {
    server: TestServer,
}

impl FakeServer {
    /// Binds `path` and serves the fake on it.
    pub fn unix(path: &Path, config: FakeConfig) -> Self {
        Self {
            server: serve_on(Listen::Unix(path.to_path_buf()), standard(config)),
        }
    }

    /// Binds an ephemeral loopback port and serves the fake on it.
    pub fn tcp(config: FakeConfig) -> Self {
        Self {
            server: serve_on(Listen::loopback(), standard(config)),
        }
    }

    /// The socket path, for `Endpoint::Unix`.
    pub fn path(&self) -> &Path {
        self.server.path()
    }

    /// The TCP address, for `Endpoint::Tcp`.
    pub fn addr(&self) -> SocketAddr {
        self.server.addr()
    }

    /// The next request the fake recorded.
    pub fn next(&self) -> Recorded {
        self.server.next()
    }

    /// Every request recorded so far, in order.
    pub fn recorded(&self) -> Vec<Recorded> {
        self.server.recorded()
    }
}

/// The fake's handler: the three endpoints, behind the bearer check.
fn standard(config: FakeConfig) -> impl Fn(usize, &Call) -> Option<Response> {
    let config = Arc::new(config);
    move |_, call| {
        if let Some(token) = &config.token {
            let want = format!("Bearer {token}");
            if call.authorization.as_deref() != Some(want.as_str()) {
                return Some(error_response(401, "a bearer token is required"));
            }
        }
        match (call.method.as_str(), call.path.as_str()) {
            ("POST", "/v1/embeddings") => Some(embeddings(&call.body)),
            ("GET", "/v1/models") => Some(json_response(
                200,
                serde_json::to_value(ModelsResponse {
                    object: ListTag::List,
                    data: vec![ModelEntry {
                        id: config.model_id.clone(),
                        object: ModelTag::Model,
                        created: 0,
                        owned_by: "repo-search".to_owned(),
                    }],
                })
                .expect("the models list serializes"),
                None,
            )),
            ("GET", "/status") => Some(json_response(
                200,
                serde_json::to_value(StatusBody {
                    model: ModelStatus::Ready {
                        id: config.model_id.clone(),
                    },
                    gpu: GpuStatus::Running,
                })
                .expect("the status serializes"),
                None,
            )),
            _ => Some(error_response(404, "no such route")),
        }
    }
}

/// One embeddings answer: a deterministic vector per input, in input
/// order, base64 in the standard response shape.
fn embeddings(body: &bytes::Bytes) -> Response {
    let request: EmbeddingRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => {
            return error_response(400, &format!("could not parse the request: {error}"));
        }
    };
    let inputs = match request.input {
        Input::One(input) => vec![input],
        Input::Many(inputs) => inputs,
    };
    json_response(
        200,
        serde_json::to_value(EmbeddingResponse {
            object: ListTag::List,
            data: inputs
                .iter()
                .enumerate()
                .map(|(index, input)| Embedding {
                    object: EmbeddingTag::Embedding,
                    index: index as u32,
                    embedding: EmbeddingData::Base64(encode_vector(&vector(input))),
                })
                .collect(),
            model: request.model,
            usage: Usage {
                prompt_tokens: inputs.len() as u32,
                total_tokens: inputs.len() as u32,
            },
        })
        .expect("the response serializes"),
        None,
    )
}

/// A `status` error carrying the standard envelope.
fn error_response(status: u16, message: &str) -> Response {
    json_response(status, error_body(message), None)
}

/// A deterministic vector derived from the input, as
/// `support/fake_embeddings.rs` derives the scheduler fake's.
fn vector(input: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for word in input.split_whitespace() {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in word.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        v[hash as usize % DIM] += 1.0;
    }
    v
}
