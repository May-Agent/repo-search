//! The embedding server's HTTP API, driven end to end against a stub
//! backend over a temp unix socket and an ephemeral loopback listener.

use std::io::{Read as _, Write as _};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::time::Duration;

use repo_search_core::embed_api::{self, decode_vector};
use repo_search_core::model::{Fetch, MODEL_ID};
use repo_search_core::qos::QosClass;
use repo_search_coreml::GpuBlocked;
use repo_search_server::backend::LoadStage;
use repo_search_server::chip::ChipClass;
use repo_search_server::stub::{HoldHandle, Instance, Kind, Record, StubBackend};
use repo_search_server::{
    BusySource, Config, GpuState, LoaderConfig, LoaderMsg, ModelSource, ModelState, http, loader,
    run, spawn,
};
use tokio::sync::watch;

const TIMEOUT: Duration = Duration::from_secs(20);

/// A model source that answers `ensure` with a temp directory, optionally
/// after reporting `progress` and waiting on `gate`.
struct TempModel {
    dir: PathBuf,
    progress: Option<Fetch>,
    gate: Option<Arc<tokio::sync::Notify>>,
}

impl ModelSource for TempModel {
    fn model_id(&self) -> String {
        MODEL_ID.to_owned()
    }

    async fn ensure(&self, progress: &watch::Sender<Fetch>) -> Result<PathBuf, String> {
        if let Some(fetch) = self.progress {
            progress.send_replace(fetch);
        }
        if let Some(gate) = &self.gate {
            gate.notified().await;
        }
        Ok(self.dir.clone())
    }
}

/// A running server and its test handles.
struct Server {
    task: tokio::task::JoinHandle<Result<(), repo_search_server::Error>>,
    unix: PathBuf,
    tcp: SocketAddr,
    backend: Arc<StubBackend>,
    open_hold: HoldHandle,
    records: mpsc::Receiver<Record>,
    state: watch::Receiver<ModelState<StubBackend>>,
    stage: watch::Receiver<LoadStage>,
    gpu: watch::Receiver<GpuState>,
    _dir: tempfile::TempDir,
}

/// Spawns `run` with a stub backend: a temp unix socket, one loopback
/// listener, `token` when given, a 60s idle exit, and `ensure` answered
/// immediately with a temp dir unless `model` says otherwise.
async fn spawn_server(
    model: TempModel,
    token: Option<String>,
    idle_exit: Duration,
    configure: impl FnOnce(&mut StubBackend),
) -> Server {
    let dir = tempfile::tempdir().expect("temp dir");
    let unix = dir.path().join("server.sock");
    let unix = tokio::net::UnixListener::bind(&unix).expect("bind unix");
    let tcp = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind tcp");
    let tcp_addr = tcp.local_addr().expect("tcp addr");
    let (records_tx, records) = mpsc::channel();
    let mut backend = StubBackend::new(records_tx);
    let open_hold = backend.hold_open_ane();
    configure(&mut backend);
    let backend = Arc::new(backend);
    let running = spawn(Config {
        unix,
        tcp: vec![tcp],
        token,
        idle_exit,
        load_bound: Duration::from_secs(60),
        model,
        chip: ChipClass::Wide,
        backend: backend.clone(),
    })
    .await
    .expect("the server starts");
    Server {
        task: running.task,
        unix: dir.path().join("server.sock"),
        tcp: tcp_addr,
        backend,
        open_hold,
        records,
        state: running.state,
        stage: running.stage,
        gpu: running.gpu,
        _dir: dir,
    }
}

/// A reqwest client over the unix socket, as the daemon's is built.
fn unix_client(server: &Server) -> reqwest::Client {
    reqwest::Client::builder()
        .unix_socket(server.unix.clone())
        .no_proxy()
        .build()
        .expect("client")
}

/// A plain loopback client.
fn tcp_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client")
}

/// Waits until the gated `open_ane` call, releases it, then polls
/// `/v1/models` until the model is ready.
async fn until_ready(server: &Server) {
    server
        .open_hold
        .entered
        .recv_timeout(TIMEOUT)
        .expect("the load reached open_ane");
    server.open_hold.open();
    await_ready(server).await;
}

/// Waits until `watch` holds a value `matches` accepts, bounded by the
/// test's timeout.
async fn first<T>(watch: &watch::Receiver<T>, matches: impl Fn(&T) -> bool) {
    let mut rx = watch.clone();
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if matches(&rx.borrow_and_update()) {
                return;
            }
            rx.changed().await.expect("the watch lives");
        }
    })
    .await
    .expect("the state arrived");
}

/// Waits until the model's state — the watch `/status` serves — is
/// `Ready`.
async fn await_ready(server: &Server) {
    first(&server.state, |s| matches!(s, ModelState::Ready(_))).await;
}

/// A `POST /v1/embeddings` request body for `input` strings.
fn body(inputs: &serde_json::Value) -> String {
    serde_json::json!({ "model": MODEL_ID, "input": inputs }).to_string()
}

/// Drains `records` until `matches` fires; the log is returned.
fn wait_record(server: &Server, matches: impl Fn(&Record) -> bool) -> Vec<Record> {
    wait_until(&server.records, matches)
}

/// `wait_record` over a bare receiver.
fn wait_until(records: &mpsc::Receiver<Record>, matches: impl Fn(&Record) -> bool) -> Vec<Record> {
    let mut log = Vec::new();
    loop {
        let record = records.recv_timeout(TIMEOUT).expect("a record");
        let done = matches(&record);
        log.push(record);
        if done {
            return log;
        }
        assert!(log.len() < 500, "the record never came");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embeddings_float_base64_and_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let server = spawn_server(model, None, Duration::from_secs(600), |b| {
        b.set_dim(768);
    })
    .await;
    until_ready(&server).await;
    let client = unix_client(&server);

    // float
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(body(&serde_json::json!("hello world")))
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    assert!(response.headers().contains_key("server-timing"));
    let parsed: embed_api::EmbeddingResponse = response.json().await.expect("json");
    assert_eq!(parsed.model, MODEL_ID);
    assert_eq!(parsed.data.len(), 1);
    let embed_api::EmbeddingData::Float(vector) = &parsed.data[0].embedding else {
        panic!("expected a float vector");
    };
    assert_eq!(vector.len(), 768);
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "a unit vector");

    // base64
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(
            serde_json::json!({
                "model": MODEL_ID,
                "input": "hello world",
                "encoding_format": "base64",
            })
            .to_string(),
        )
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    let parsed: embed_api::EmbeddingResponse = response.json().await.expect("json");
    let embed_api::EmbeddingData::Base64(text) = &parsed.data[0].embedding else {
        panic!("expected a base64 vector");
    };
    assert_eq!(decode_vector(text).expect("decodes").len(), 768);

    // dimensions = 256 truncates and renormalizes; the input's stub
    // vector must have its nonzero element inside the first 256 dims, so
    // pick a token count whose `bag_of_ids` lands there.
    let tokens = (1..16)
        .find(|n| {
            repo_search_server::stub::bag_of_ids(&vec![0; *n], 768).as_slice()[..256]
                .iter()
                .any(|x| *x != 0.0)
        })
        .expect("a short input lands inside 256 dims");
    let input = "x".repeat(4 * (tokens - 1) + 1);
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(
            serde_json::json!({
                "model": MODEL_ID,
                "input": input,
                "dimensions": 256,
            })
            .to_string(),
        )
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    let parsed: embed_api::EmbeddingResponse = response.json().await.expect("json");
    let embed_api::EmbeddingData::Float(vector) = &parsed.data[0].embedding else {
        panic!("expected a float vector");
    };
    assert_eq!(vector.len(), 256);
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "renormalized to unit length");

    // many inputs answer in input order
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(body(&serde_json::json!(["first", "second", "third"])))
        .send()
        .await
        .expect("response");
    let parsed: embed_api::EmbeddingResponse = response.json().await.expect("json");
    assert_eq!(
        parsed.data.iter().map(|d| d.index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_errors_carry_the_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let server = spawn_server(model, None, Duration::from_secs(600), |_| {}).await;
    until_ready(&server).await;
    let client = unix_client(&server);

    async fn check(client: &reqwest::Client, raw: String, want: u16, code: Option<&str>) {
        let response = client
            .post("http://localhost/v1/embeddings")
            .body(raw)
            .send()
            .await
            .expect("response");
        assert_eq!(response.status().as_u16(), want);
        let error: embed_api::ErrorBody = response.json().await.expect("envelope");
        assert_eq!(error.error.code.as_deref(), code);
        assert_eq!(error.error.r#type, "invalid_request_error");
    }
    check(
        &client,
        r#"{"model":"m","input":"x","nope":1}"#.to_owned(),
        400,
        None,
    )
    .await;
    check(
        &client,
        serde_json::json!({"model": "not-the-model", "input": "x"}).to_string(),
        404,
        Some("model_not_found"),
    )
    .await;
    // over 512 tokens: the stub's len/4+1 ids pass 512 at 2049 chars
    check(
        &client,
        serde_json::json!({"model": MODEL_ID, "input": "x".repeat(3000)}).to_string(),
        400,
        Some("context_length_exceeded"),
    )
    .await;
    check(
        &client,
        serde_json::json!({"model": MODEL_ID, "input": []}).to_string(),
        400,
        None,
    )
    .await;
    check(&client, "not json".to_owned(), 400, None).await;
    check(
        &client,
        serde_json::json!({"model": MODEL_ID, "input": "x", "dimensions": 100}).to_string(),
        400,
        None,
    )
    .await;
    // an empty string input, and more than 2048 inputs, are 400s; `user`
    // is the standard optional field and is accepted.
    check(
        &client,
        serde_json::json!({"model": MODEL_ID, "input": ""}).to_string(),
        400,
        None,
    )
    .await;
    check(
        &client,
        serde_json::json!({"model": MODEL_ID, "input": vec!["x"; 2049]}).to_string(),
        400,
        None,
    )
    .await;
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(
            serde_json::json!({
                "model": MODEL_ID, "input": "x", "user": "someone",
            })
            .to_string(),
        )
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    // A body whose declared length is over the 32 MiB limit is refused
    // before a byte of it is read.
    let mut stream = std::os::unix::net::UnixStream::connect(&server.unix).expect("connect");
    stream
        .write_all(
            format!(
                "POST /v1/embeddings HTTP/1.1\r\nhost: x\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                repo_search_server::BODY_LIMIT + 1
            )
            .as_bytes(),
        )
        .expect("write");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let mut response = vec![0; 128];
    let n = stream
        .read(&mut response)
        .expect("the refused request answers");
    let head = String::from_utf8_lossy(&response[..n]);
    assert!(head.starts_with("HTTP/1.1 413"), "{head}");
    // GET on a POST endpoint is a 405; an unknown path a 404.
    let response = client
        .get("http://localhost/v1/embeddings")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 405);
    let response = client
        .post("http://localhost/nope")
        .body("{}")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 404);
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn priority_maps_to_the_requests_qos() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let server = spawn_server(model, None, Duration::from_secs(600), |_| {}).await;
    until_ready(&server).await;
    let client = unix_client(&server);
    for (header, want) in [
        (None, QosClass::UserInitiated),
        (Some("u=0"), QosClass::UserInteractive),
        (Some("u=7"), QosClass::Background),
    ] {
        let mut request = client
            .post("http://localhost/v1/embeddings")
            .body(body(&serde_json::json!("x")));
        if let Some(header) = header {
            request = request.header("priority", header);
        }
        let response = request.send().await.expect("response");
        assert_eq!(response.status(), 200);
        let log = wait_record(&server, |r| {
            matches!(r.kind, Kind::Index | Kind::Interactive)
        });
        assert_eq!(log.last().unwrap().qos, want, "priority {header:?}");
    }
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_requires_the_bearer_token_and_unix_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let server = spawn_server(
        model,
        Some("secret-token".to_owned()),
        Duration::from_secs(600),
        |_| {},
    )
    .await;
    until_ready(&server).await;
    let tcp = tcp_client();
    let url = format!("http://{}/v1/models", server.tcp);

    let response = tcp.get(&url).send().await.expect("response");
    assert_eq!(response.status(), 401);
    let error: embed_api::ErrorBody = response.json().await.expect("envelope");
    assert_eq!(error.error.code.as_deref(), Some("invalid_api_key"));
    let response = tcp
        .get(&url)
        .bearer_auth("wrong")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 401);
    let response = tcp
        .get(&url)
        .bearer_auth("secret-token")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    // The unix socket needs no token.
    let response = unix_client(&server)
        .get("http://localhost/v1/models")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn not_ready_then_models_and_status() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let server = spawn_server(model, None, Duration::from_secs(600), |_| {}).await;
    let client = unix_client(&server);
    // The load is gated inside `open_ane`: requests get 503 +
    // Retry-After, and /status reports the stage.
    server
        .open_hold
        .entered
        .recv_timeout(TIMEOUT)
        .expect("the load reached open_ane");
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(body(&serde_json::json!("x")))
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 503);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .unwrap()
            .to_str()
            .unwrap(),
        "1"
    );
    let error: embed_api::ErrorBody = response.json().await.expect("envelope");
    assert_eq!(error.error.code.as_deref(), Some("model_not_ready"));
    let response = client
        .get("http://localhost/v1/models")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 503);

    // `entered` already proved the state is `loading`; the stage the
    // /status handler reads is forwarded one hop behind the thread's.
    first(&server.stage, |stage| *stage == LoadStage::Open).await;
    let status = client
        .get("http://localhost/status")
        .send()
        .await
        .expect("response")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(status["model"]["type"], "loading");
    assert_eq!(status["model"]["stage"], "open");

    server.open_hold.open();
    await_ready(&server).await;
    let response = client
        .get("http://localhost/v1/models")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    let models: embed_api::ModelsResponse = response.json().await.expect("json");
    assert_eq!(models.data[0].id, MODEL_ID);
    assert_eq!(models.data[0].owned_by, "repo-search");

    let status = client
        .get("http://localhost/status")
        .send()
        .await
        .expect("response")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(status["model"]["type"], "ready");
    assert_eq!(status["model"]["id"], MODEL_ID);
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn status_reports_download_progress_and_gpu_state() {
    let dir = tempfile::tempdir().unwrap();
    let download_gate = Arc::new(tokio::sync::Notify::new());
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: Some(Fetch {
            received: 100,
            total: Some(400),
        }),
        gate: Some(download_gate.clone()),
    };
    let ane_gate = std::sync::Arc::new(std::sync::Mutex::new(None));
    let gpu_gate = std::sync::Arc::new(std::sync::Mutex::new(None));
    let (out_a, out_g) = (ane_gate.clone(), gpu_gate.clone());
    let server = spawn_server(model, None, Duration::from_secs(600), move |b| {
        *out_a.lock().unwrap() = Some(b.gate_first(Instance::Ane));
        *out_g.lock().unwrap() = Some(b.gate_first(Instance::Gpu));
    })
    .await;
    let client = unix_client(&server);
    // The download reports mid-flight: /status carries received/total.
    first(&server.state, |s| {
        matches!(
            s,
            ModelState::Downloading {
                received: 100,
                total: Some(400)
            }
        )
    })
    .await;
    let status = client
        .get("http://localhost/status")
        .send()
        .await
        .expect("response")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(status["model"]["type"], "downloading");
    assert_eq!(status["model"]["received"], 100);
    assert_eq!(status["model"]["total"], 400);

    // The server is up: headroom changes reach it. Block the GPU, finish
    // the download and the load, then read gpu.state.
    server
        .backend
        .push_headroom(Err(GpuBlocked::NotMains("Battery Power".to_owned())));
    download_gate.notify_one();
    until_ready(&server).await;
    let status = client
        .get("http://localhost/status")
        .send()
        .await
        .expect("response")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(status["gpu"]["type"], "absent");
    assert!(status["gpu"]["blocked"].is_string());

    // With the ANE in-flight on a gated filler call, the next Normal item
    // queues; restored headroom sends the GPU after it.
    server.backend.push_headroom(Ok(()));
    let filler = {
        let client = unix_client(&server);
        tokio::spawn(async move {
            client
                .post("http://localhost/v1/embeddings")
                .body(body(&serde_json::json!("ffffffff")))
                .send()
                .await
                .expect("response")
        })
    };
    wait_record(&server, |r| r.kind == Kind::Started);
    // The GPU's first call is gated too: while it holds, the request's
    // response is pending and /status reads `running`.
    let gpu_req = {
        let client = unix_client(&server);
        tokio::spawn(async move {
            client
                .post("http://localhost/v1/embeddings")
                .body(body(&serde_json::json!("x")))
                .header("priority", "u=3")
                .send()
                .await
                .expect("response")
        })
    };
    wait_record(&server, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Started
    });
    first(&server.gpu, |s| *s == GpuState::Running).await;
    let status = client
        .get("http://localhost/status")
        .send()
        .await
        .expect("response")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(status["gpu"]["type"], "running");
    let ane_gate = ane_gate.lock().unwrap().clone().unwrap();
    let gpu_gate = gpu_gate.lock().unwrap().clone().unwrap();
    std::thread::spawn(move || {
        ane_gate.wait();
        gpu_gate.wait();
    });
    assert_eq!(filler.await.unwrap().status(), 200);
    assert_eq!(gpu_req.await.unwrap().status(), 200);
    wait_record(&server, |r| {
        r.instance == Instance::Gpu && r.kind == Kind::Index
    });
    server.task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_closed_connection_embeds_nothing() {
    // Wire the pipeline by hand so the test owns the in-flight request
    // count — the signal that proves the raw request's handler ran,
    // submitted, and dropped.
    let dir = tempfile::tempdir().unwrap();
    let (records_tx, records) = mpsc::channel();
    let mut backend = StubBackend::new(records_tx);
    let gate = backend.gate_first(Instance::Ane);
    let backend = Arc::new(backend);
    let (_headroom_tx, headroom_rx) = watch::channel(Err(GpuBlocked::PowerSourceUnknown));
    let (loader_busy, _busy_rx) = watch::channel(false);
    let (busy_reg, _busy_reg_rx) = tokio::sync::mpsc::channel::<BusySource>(16);
    let (fatal_tx, _fatal_rx) = tokio::sync::mpsc::channel(64);
    let (gpu_tx, gpu_rx) = watch::channel(GpuState::Absent { blocked: None });
    let loader = loader::spawn(LoaderConfig {
        backend,
        model: Arc::new(TempModel {
            dir: dir.path().to_path_buf(),
            progress: None,
            gate: None,
        }),
        bound: Duration::from_secs(60),
        chip: ChipClass::Wide,
        headroom: headroom_rx,
        busy: loader_busy,
        busy_reg,
        fatal: fatal_tx,
        gpu_state: gpu_tx,
    });
    let (in_flight, _busy, mut count) = http::InFlight::new();
    let shared = Arc::new(http::Shared {
        state: loader.state.clone(),
        loader: loader.tx.clone(),
        gpu: gpu_rx,
        stage: loader.stage.clone(),
        token: None,
        next_request: AtomicU64::new(0),
        in_flight: Arc::new(in_flight),
    });
    let unix = tokio::net::UnixListener::bind(dir.path().join("s.sock")).expect("bind");
    let accepts = http::serve(unix, Vec::new()).run(shared);
    // Wait until the model answers ready, and take the scheduler's
    // queued-item watch: the signal that proves the raw request's item
    // was submitted and later left the queue unembedded.
    let model = tokio::time::timeout(TIMEOUT, async {
        let mut state = loader.state.clone();
        loop {
            if let ModelState::Ready(model) = &*state.borrow_and_update() {
                return model.clone();
            }
            state.changed().await.expect("the loader lives");
        }
    })
    .await
    .expect("the model became ready");
    let mut queued = model.scheduler.queued();
    let path = dir.path().join("s.sock");
    // A filler request takes the ANE's first (gated) call.
    let client = reqwest::Client::builder()
        .unix_socket(path.clone())
        .no_proxy()
        .build()
        .expect("client");
    let filler = tokio::spawn(async move {
        client
            .post("http://localhost/v1/embeddings")
            .body(body(&serde_json::json!("bbbbbbbb")))
            .send()
            .await
            .expect("response")
    });
    let log = wait_until(&records, |r| r.kind == Kind::Started);
    assert_eq!(log.len(), 1);
    // Its handler runs: count 1.
    while *count.borrow_and_update() < 1 {
        tokio::time::timeout(TIMEOUT, count.changed())
            .await
            .expect("the filler is tracked")
            .expect("the counter lives");
    }
    // The raw request writes fully, then the client closes: the handler
    // runs (count 2), submits, and its future drops (count 1) when hyper
    // sees the socket close.
    let request_body = body(&serde_json::json!("aaaa"));
    let mut stream = std::os::unix::net::UnixStream::connect(&path).expect("connect");
    let request = format!(
        "POST /v1/embeddings HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        request_body.len(),
        request_body
    );
    stream.write_all(request.as_bytes()).expect("write");
    while *count.borrow_and_update() < 2 {
        tokio::time::timeout(TIMEOUT, count.changed())
            .await
            .expect("the raw request is tracked")
            .expect("the counter lives");
    }
    // The handler submitted: its item sits in the scheduler's queue
    // behind the gated filler.
    while *queued.borrow_and_update() != 1 {
        tokio::time::timeout(TIMEOUT, queued.changed())
            .await
            .expect("the request was submitted")
            .expect("the scheduler lives");
    }
    drop(stream);
    while *count.borrow_and_update() != 1 {
        tokio::time::timeout(TIMEOUT, count.changed())
            .await
            .expect("the closed request's handler drops")
            .expect("the counter lives");
    }
    // The drop cancelled it: the queue emptied with the ANE still held,
    // so the item left without reaching the embedder.
    while *queued.borrow_and_update() != 0 {
        tokio::time::timeout(TIMEOUT, queued.changed())
            .await
            .expect("the queue emptied")
            .expect("the scheduler lives");
    }
    // The handler dropped: the queued item's reply is closed and the
    // cancel has landed. Open the gate; only the filler embeds.
    std::thread::spawn(move || gate.wait());
    assert_eq!(filler.await.unwrap().status(), 200);
    let client = reqwest::Client::builder()
        .unix_socket(path)
        .no_proxy()
        .build()
        .expect("client");
    let response = client
        .post("http://localhost/v1/embeddings")
        .body(body(&serde_json::json!("cccccccccccc")))
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    let log = wait_until(&records, |r| {
        matches!(r.kind, Kind::Index | Kind::Interactive) && r.ids == [0u32; 4]
    });
    assert!(
        log.iter()
            .all(|r| !(matches!(r.kind, Kind::Index | Kind::Interactive) && r.ids == [0u32; 2])),
        "the closed request's item never embedded: {log:?}"
    );
    accepts.shutdown().await;
    let (stopped, stopping) = tokio::sync::oneshot::channel();
    let _ = loader.tx.send(LoaderMsg::Shutdown { reply: stopped }).await;
    let _ = stopping.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_without_a_token_must_bind_loopback() {
    let dir = tempfile::tempdir().unwrap();
    let unix = tokio::net::UnixListener::bind(dir.path().join("s.sock")).expect("bind");
    let tcp = tokio::net::TcpListener::bind("0.0.0.0:0")
        .await
        .expect("bind");
    // A client already queued when the server refuses is answered, not
    // reset.
    let mut queued =
        std::net::TcpStream::connect(tcp.local_addr().expect("addr")).expect("connect");
    queued
        .write_all(b"GET /v1/models HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
        .expect("write");
    let (records_tx, _records) = mpsc::channel();
    let outcome = run(Config {
        unix,
        tcp: vec![tcp],
        token: None,
        idle_exit: Duration::from_secs(600),
        load_bound: Duration::from_secs(60),
        model: TempModel {
            dir: dir.path().to_path_buf(),
            progress: None,
            gate: None,
        },
        chip: ChipClass::Wide,
        backend: Arc::new(StubBackend::new(records_tx)),
    })
    .await;
    let Err(repo_search_server::Error::Listener(message)) = outcome else {
        panic!("a non-loopback unauthenticated listener is refused");
    };
    assert!(message.contains("not loopback"));
    queued
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let mut response = Vec::new();
    queued
        .read_to_end(&mut response)
        .expect("the queued connection is answered");
    let text = String::from_utf8(response).expect("utf8");
    assert!(text.starts_with("HTTP/1.1 503"), "{text}");
    assert!(text.contains("server_start_failed"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_flight_work_blocks_the_idle_exit() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let gate = std::sync::Arc::new(std::sync::Mutex::new(None));
    let out = gate.clone();
    let mut server = spawn_server(model, None, Duration::from_millis(300), move |b| {
        *out.lock().unwrap() = Some(b.gate_first(Instance::Ane));
    })
    .await;
    until_ready(&server).await;
    let gate = gate.lock().unwrap().clone().unwrap();
    // An embed held in flight keeps the server busy past the deadline.
    let client = unix_client(&server);
    let request = tokio::spawn(async move {
        client
            .post("http://localhost/v1/embeddings")
            .body(body(&serde_json::json!("x")))
            .send()
            .await
            .expect("response")
    });
    wait_record(&server, |r| r.kind == Kind::Started);
    assert!(
        tokio::time::timeout(Duration::from_secs(1), &mut server.task)
            .await
            .is_err(),
        "in-flight work holds the server open"
    );
    std::thread::spawn(move || gate.wait());
    assert_eq!(request.await.unwrap().status(), 200);
    // Once idle it exits inside the bound.
    tokio::time::timeout(TIMEOUT, &mut server.task)
        .await
        .expect("the server exits")
        .expect("not aborted")
        .expect("a clean exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn idle_exit_closes_even_keep_alive_connections() {
    let dir = tempfile::tempdir().unwrap();
    let model = TempModel {
        dir: dir.path().to_path_buf(),
        progress: None,
        gate: None,
    };
    let mut server = spawn_server(model, None, Duration::from_millis(300), |_| {}).await;
    until_ready(&server).await;
    // A keep-alive client pools the connection; the idle deadline still
    // fires and GracefulShutdown closes it.
    let client = unix_client(&server);
    let response = client
        .get("http://localhost/v1/models")
        .send()
        .await
        .expect("response");
    assert_eq!(response.status(), 200);
    tokio::time::timeout(TIMEOUT, &mut server.task)
        .await
        .expect("the server exits")
        .expect("not aborted")
        .expect("a clean exit");
    drop(client);
}
