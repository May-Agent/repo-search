//! `daemon::run` on a `Bound` listener against the fake standard-API
//! server — the portable daemon pieces end to end: a unix endpoint and a
//! TCP endpoint with a bearer token, the wrong-token failure, and the
//! `Bound` close path. On Linux these are the production pieces; here
//! they are exercised on macOS.

#[allow(dead_code)] // each test file uses its own slice of the fixtures
mod support;

use std::fs::TryLockError;
use std::path::Path;
use std::process::Stdio;
use std::sync::mpsc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use repo_search::daemon::listen::{Bound, Listener as _};
use repo_search::daemon::server::Endpoint;
use repo_search::protocol::{ErrorKind, Event, Final, GpuStatus, ModelStatus, Request, framed};
use repo_search_core::StateDir;
use repo_search_core::model::MODEL_ID;
use support::bound::spawn;
use support::client::{final_of, hello, request, search};
use support::embed_server::{FakeConfig, FakeServer};
use support::repo::{self, TestRepo};
use support::tokens::StubTokens;

const TIMEOUT: Duration = Duration::from_secs(60);

/// The marker an edit lands in the index, then the answer.
const MARKER: &str = "xylophonemarker";

/// `index --wait` answers `Indexed` with every chunk vectored.
async fn index_wait(socket: &Path, repo: &TestRepo) {
    let events = request(
        socket,
        &hello(),
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    match final_of(&events) {
        Final::Indexed { with_vector, total } => {
            assert_eq!(with_vector, total, "every chunk has a vector")
        }
        other => panic!("expected Indexed, got {other:?}"),
    }
}

/// An edit is embedded in the background; the marker search answers it.
async fn marker_search(socket: &Path, repo: &TestRepo) {
    repo::append(
        &repo.root,
        "src/edit.rs",
        &repo::rust_fn(&format!("{MARKER}_fn"), 500),
    );
    let events = request(socket, &hello(), search(&repo.root, MARKER)).await;
    let answer = match final_of(&events) {
        Final::Answer(answer) => answer,
        other => panic!("expected an Answer, got {other:?}"),
    };
    assert!(answer.has_results, "the marker is found: {events:?}");
    assert!(
        answer.output.contains(MARKER),
        "the answer carries the marker: {}",
        answer.output
    );
}

/// `status` reports what the fake serves: the model ready, the GPU
/// running — proof `GET /status` reached it.
async fn status_reports_server(socket: &Path, repo: &TestRepo) {
    let events = request(
        socket,
        &hello(),
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
        status.model,
        ModelStatus::Ready {
            model_id: MODEL_ID.to_owned()
        },
        "the model the fake lists"
    );
    assert_eq!(status.daemon.gpu, GpuStatus::Running, "the fake's /status");
}

/// (a): the daemon against the fake on a unix socket — `index --wait`,
/// a search answering with the edited marker, `status` reporting the
/// server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unix_socket_index_search_and_status() {
    let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
    let tmp = tempfile::tempdir().expect("tmp");
    let fake = FakeServer::unix(
        &tmp.path().join("server.sock"),
        FakeConfig {
            model_id: MODEL_ID.to_owned(),
            token: None,
        },
    );
    let socket = tmp.path().join("daemon.sock");
    let _daemon = spawn(
        &socket,
        StateDir::new(tmp.path().join("state")),
        Endpoint::Unix(fake.path().to_path_buf()),
        StubTokens::default(),
        Duration::from_secs(600),
    )
    .await;

    index_wait(&socket, &repo).await;
    marker_search(&socket, &repo).await;
    status_reports_server(&socket, &repo).await;
}

/// (b): the same over TCP with a token — and the fake saw `Bearer
/// <token>` on every request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_with_token_index_search_and_status() {
    let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
    let tmp = tempfile::tempdir().expect("tmp");
    let token = "t0ken".to_owned();
    let fake = FakeServer::tcp(FakeConfig {
        model_id: MODEL_ID.to_owned(),
        token: Some(token.clone()),
    });
    let socket = tmp.path().join("daemon.sock");
    let _daemon = spawn(
        &socket,
        StateDir::new(tmp.path().join("state")),
        Endpoint::Tcp {
            addr: fake.addr(),
            token: Some(token.clone()),
        },
        StubTokens::default(),
        Duration::from_secs(600),
    )
    .await;

    index_wait(&socket, &repo).await;
    marker_search(&socket, &repo).await;
    status_reports_server(&socket, &repo).await;

    let want = format!("Bearer {token}");
    let recorded = fake.recorded();
    assert!(!recorded.is_empty(), "the fake saw requests");
    for request in &recorded {
        assert_eq!(
            request.authorization.as_deref(),
            Some(want.as_str()),
            "every request carries the bearer token: {}",
            request.path
        );
    }
    assert!(
        recorded.iter().any(|r| r.path == "/v1/embeddings"),
        "the daemon embedded through the fake"
    );
}

/// (c): a wrong token — the fake 401s and the request fails `Model`
/// with the server's answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_wrong_token_fails_model() {
    let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
    let tmp = tempfile::tempdir().expect("tmp");
    let fake = FakeServer::tcp(FakeConfig {
        model_id: MODEL_ID.to_owned(),
        token: Some("right".to_owned()),
    });
    let socket = tmp.path().join("daemon.sock");
    let _daemon = spawn(
        &socket,
        StateDir::new(tmp.path().join("state")),
        Endpoint::Tcp {
            addr: fake.addr(),
            token: Some("wrong".to_owned()),
        },
        StubTokens::default(),
        Duration::from_secs(600),
    )
    .await;

    let events = request(&socket, &hello(), search(&repo.root, MARKER)).await;
    let report = match final_of(&events) {
        Final::Error(report) => report,
        other => panic!("expected an Error, got {other:?}"),
    };
    assert_eq!(report.kind, ErrorKind::Model);
    assert!(
        report.message.contains("401") && report.message.contains("bearer token"),
        "the 401 detail surfaces: {}",
        report.message
    );
}

/// A spawned `bound_holder`: reaped on drop, so a panicking test leaves
/// nothing running. `_stdin` is the lifeline its stdin watch reads.
struct Holder {
    child: std::process::Child,
    _stdin: std::process::ChildStdin,
}

impl Holder {
    /// Re-executes the test binary on the ignored `bound_holder` with
    /// the socket path in `REPO_SEARCH_TEST_HOST`; returns once it
    /// printed `BOUND` — its `Bound::acquire` has taken the lock.
    fn spawn(socket: &Path) -> Self {
        let exe = std::env::current_exe().expect("the test binary");
        let mut child = std::process::Command::new(exe)
            .args([
                "--exact",
                "bound_holder",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("REPO_SEARCH_TEST_HOST", socket)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("the holder spawns");
        let stdin = child.stdin.take().expect("piped");
        let (tx, rx) = mpsc::channel();
        let mut stdout = std::io::BufReader::new(child.stdout.take().expect("piped"));
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut line = String::new();
            while stdout.read_line(&mut line).unwrap_or(0) > 0 {
                if line.trim() == "BOUND" {
                    let _ = tx.send(());
                }
                line.clear();
            }
        });
        tokio::task::block_in_place(|| {
            rx.recv_timeout(TIMEOUT).expect("the holder prints BOUND");
        });
        Self {
            child,
            _stdin: stdin,
        }
    }

    fn kill(&mut self) {
        self.child.kill().expect("SIGKILL");
        self.child.wait().expect("wait");
    }
}

impl Drop for Holder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The ignored test re-executed as the lock's holder: `Bound::acquire`
/// on the socket `REPO_SEARCH_TEST_HOST` names, then waits to be
/// killed. The lock is the process's, so killing it is what releases it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "re-executed by bound_close_unlinks_refuses_and_holds_the_lock"]
async fn bound_holder() {
    let socket = std::env::var_os("REPO_SEARCH_TEST_HOST").expect("REPO_SEARCH_TEST_HOST");
    support::host::watch_stdin();
    let _bound = Bound::acquire(Path::new(&socket))
        .await
        .expect("the holder binds");
    // The harness left its `test ... ` prefix unterminated on this line.
    println!("\nBOUND");
    std::future::pending::<()>().await;
}

/// (d): the `Bound` close path — idle exit unlinks the socket, a queued
/// connection is answered `ShuttingDown`, and a second acquire waits out
/// the first daemon's process. Each part gets its own socket: the lock
/// rides on the `LifetimeLock` `close` returns, which `run` — and the
/// holder child — keeps until the daemon is done.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_close_unlinks_refuses_and_holds_the_lock() {
    let tmp = tempfile::tempdir().expect("tmp");
    let fake = FakeServer::unix(
        &tmp.path().join("server.sock"),
        FakeConfig {
            model_id: MODEL_ID.to_owned(),
            token: None,
        },
    );

    // Idle exit: `run` closes the `Bound`, which unlinks the socket.
    let idle = tmp.path().join("idle.sock");
    let daemon = spawn(
        &idle,
        StateDir::new(tmp.path().join("state")),
        Endpoint::Unix(fake.path().to_path_buf()),
        StubTokens::default(),
        Duration::from_millis(200),
    )
    .await;
    tokio::time::timeout(TIMEOUT, daemon)
        .await
        .expect("the daemon idle-exits")
        .expect("the daemon task joins")
        .expect("the daemon runs");
    assert!(!idle.exists(), "the close unlinked the socket");
    match std::os::unix::net::UnixStream::connect(&idle) {
        Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::NotFound),
        Ok(_) => panic!("the unlinked socket answers ENOENT"),
    }

    // The drain: a connection queued on the bound socket is answered
    // `ShuttingDown` by `close`.
    let drain = tmp.path().join("drain.sock");
    let bound = Bound::acquire(&drain).await.expect("acquire");
    let stream = tokio::net::UnixStream::connect(&drain)
        .await
        .expect("the client queues on the bound socket");
    let mut framed = framed(stream);
    framed
        .send(serde_json::to_string(&hello()).expect("hello serializes"))
        .await
        .expect("the hello writes");
    bound.close().await;
    let line = framed
        .next()
        .await
        .expect("the queued connection is answered")
        .expect("the answer decodes");
    match serde_json::from_str::<Event>(&line).expect("the frame parses") {
        Event::Final(Final::Error(report)) => {
            assert_eq!(report.kind, ErrorKind::ShuttingDown)
        }
        other => panic!("expected a ShuttingDown final, got {other:?}"),
    }
    assert!(!drain.exists(), "the close unlinked the socket");

    // The lifetime lock: a second `Bound::acquire` waits out the first
    // daemon's process — here a spawned holder child.
    let socket = tmp.path().join("lock.sock");
    let lock = socket.with_extension("lock");
    let mut holder = Holder::spawn(&socket);
    let (started_tx, started) = tokio::sync::oneshot::channel();
    let path = socket.clone();
    let acquiring = tokio::spawn(async move {
        let _ = started_tx.send(());
        Bound::acquire(&path).await
    });
    started.await.expect("the acquire started");
    assert!(!acquiring.is_finished(), "acquire waits on the held lock");
    assert!(
        holder.child.try_wait().expect("try_wait").is_none(),
        "the holder is alive"
    );
    // The lock itself reports the holder.
    let probe = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock)
        .expect("the lock file opens");
    assert!(
        matches!(probe.try_lock(), Err(TryLockError::WouldBlock)),
        "the holder owns the lock"
    );
    drop(probe);
    holder.kill();
    let bound = acquiring
        .await
        .expect("the acquire task joins")
        .expect("acquires once the holder exits");
    assert_eq!(
        std::fs::read_to_string(&lock)
            .expect("the lock file reads")
            .trim(),
        std::process::id().to_string(),
        "the lock names the new owner"
    );
    bound.close().await;
}
