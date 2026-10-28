//! Daemon tests on the real Core ML model: stage-1 query behaviour and the
//! kill-and-resume sequence against `CoreMlBackend`. They need
//! `REPO_SEARCH_MODEL_DIR` (the extracted archive) and `REPO_SEARCH_CORPUS`
//! (a checkout cloned into a temp dir), so they are `#[ignore]`d and run by
//! the release gate.

#![cfg(target_os = "macos")]

// `daemon.rs` uses the rest of the fixture.
#[path = "support/client.rs"]
mod client;
#[allow(dead_code)]
#[path = "support/host.rs"]
mod host;
#[path = "support/model.rs"]
mod model;

use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use repo_search::daemon::chip::ChipClass;
use repo_search::daemon::coreml::CoreMlBackend;
use repo_search::daemon::{self, Config};
use repo_search::model_store::{ModelSpec, ModelStore};
use repo_search::protocol::{Final, Request};
use repo_search_core::StateDir;
use tokio::net::UnixListener;

/// The corpus queries: behaviour descriptions, in the spirit of what a
/// WaterUI user cannot name.
const QUERIES: &[&str] = &[
    "react to a button press by navigating to a new screen",
    "animate a view's opacity over time",
    "observe a text field as the user types",
    "lay out child views in a vertical stack",
    "load a remote image into a view",
];

/// Queries 5 behaviour descriptions during stage 1: each answer is
/// non-empty, embeds chunks on the spot, and is never overtaken by indexing
/// work. `indexing_wait_us` is logged, never thresholded: the assertions
/// are structural, not latency.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real model; the release gate runs it"]
async fn stage_1_query_on_the_ane() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let tmp = tempfile::tempdir().expect("tmp");
    let worktree = model::clone_corpus(&tmp);
    let socket = tmp.path().join("daemon.sock");
    let listener = UnixListener::bind(&socket).expect("socket binds");
    let store = model::model_store(tmp.path().join("models"));
    let daemon = tokio::spawn(daemon::run(Config {
        listener,
        idle_exit: Duration::from_secs(600),
        load_bound: daemon::LOAD_BOUND,
        model_store: store,
        state: StateDir::new(tmp.path().join("state")),
        chip: ChipClass::Wide,
        backend: Arc::new(CoreMlBackend),
        build: client::build(),
    }));

    for query in QUERIES {
        let events =
            client::request(&socket, &client::hello(), client::search(&worktree, query)).await;
        let answer = match client::final_of(&events) {
            Final::Answer(answer) => answer,
            Final::Error(error) => panic!("{query:?} failed: {error:?}"),
            other => panic!("{query:?} answered with {other:?}"),
        };
        tracing::info!(
            query,
            indexing_wait_us = answer.stats.indexing_wait_us,
            "stage-1 query answered"
        );
        assert!(answer.has_results, "{query:?} answered empty");
        assert_eq!(
            answer.stats.indexing_overtaken, 0,
            "{query:?} was overtaken by indexing work"
        );
        assert!(
            answer.stats.embedded_on_the_spot > 0,
            "{query:?} embedded nothing on the spot: stage 1 must still run"
        );
    }

    daemon.abort();
}

/// The real-kill sequence of `kill_mid_index_resumes`, with the host
/// running `CoreMlBackend`: index to 500 committed vectors, SIGKILL the
/// host, and verify the replacement resumes from the committed count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real model and a real kill; the release gate runs it"]
async fn kill_and_resume_on_the_model() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let tmp = tempfile::tempdir().expect("tmp");
    let worktree = model::clone_corpus(&tmp);
    let model_cache = tmp.path().join("models");
    let state = StateDir::new(tmp.path().join("state"));
    let store = model::model_store(model_cache.clone());
    let model_id = store.model_id();
    let host_config = |sock: &str| HostConfig {
        socket: tmp.path().join(sock),
        model_cache: model_cache.clone(),
        state_dir: tmp.path().join("state"),
    };

    let mut host = Host::spawn(&host_config("daemon1.sock"));
    let socket = tmp.path().join("daemon1.sock");
    let mut stream = model::index_wait(&socket, &client::hello(), &worktree).await;
    model::await_coverage(&mut stream, 500).await;
    drop(stream);

    // A non-waiting index request runs the barrier, whose first step
    // flushes the vector writer; its `Final` landing means every embedded
    // chunk is committed, not still buffered.
    let events = client::request(
        &socket,
        &client::hello(),
        Request::Index {
            worktree: worktree.clone(),
            wait: false,
        },
    )
    .await;
    match client::final_of(&events) {
        Final::Indexed { .. } => {}
        other => panic!("the flush request failed: {other:?}"),
    }
    host.kill();

    let committed = model::committed(&state, &worktree, &worktree.join(".git"), &model_id).await;
    tracing::info!(committed, "committed vectors after the flush");
    // The barrier flushed the writer, so the 500 chunks coverage reported
    // are committed on disk.
    assert!(committed >= 500, "{committed} committed vectors");

    let _host = Host::spawn(&host_config("daemon2.sock"));
    model::assert_resume(
        &tmp.path().join("daemon2.sock"),
        &client::hello(),
        &worktree,
        committed,
    )
    .await;
}

/// Configuration handed to a re-executed `daemon_host_coreml` through
/// `REPO_SEARCH_TEST_HOST`.
#[derive(serde::Serialize, serde::Deserialize)]
struct HostConfig {
    socket: PathBuf,
    model_cache: PathBuf,
    state_dir: PathBuf,
}

/// The ignored test re-executed as the kill target; prints `READY` once its
/// socket is bound. Run under a bare `--ignored` without
/// `REPO_SEARCH_TEST_HOST` it panics, which `#[should_panic]` asserts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "re-executed by kill_and_resume_on_the_model"]
#[should_panic(expected = "REPO_SEARCH_TEST_HOST is set by kill_and_resume_on_the_model")]
async fn daemon_host_coreml() {
    let raw = std::env::var("REPO_SEARCH_TEST_HOST")
        .expect("REPO_SEARCH_TEST_HOST is set by kill_and_resume_on_the_model");
    let config: HostConfig = serde_json::from_str(&raw).expect("HostConfig json");
    host::watch_stdin();
    let store =
        ModelStore::new(config.model_cache, ModelSpec::embedding_gemma_2()).expect("model store");
    let listener = UnixListener::bind(&config.socket).expect("socket binds");
    // The harness left its `test ... ` prefix unterminated on this line.
    println!("\nREADY");
    daemon::run(Config {
        listener,
        idle_exit: Duration::from_secs(3600),
        load_bound: daemon::LOAD_BOUND,
        model_store: store,
        state: StateDir::new(config.state_dir),
        chip: ChipClass::Wide,
        backend: Arc::new(CoreMlBackend),
        build: client::build(),
    })
    .await
    .expect("the host runs");
}

/// A spawned `daemon_host_coreml`: the child, reaped on drop, and the
/// write end of the lifeline its stdin watch reads.
struct Host {
    child: std::process::Child,
    /// Held open and never written: the host exits when this end closes,
    /// however the holding process dies.
    _stdin: std::process::ChildStdin,
}

impl Host {
    /// Spawns a host and returns once it printed `READY`: its socket is
    /// bound, so a connect cannot race the bind.
    fn spawn(config: &HostConfig) -> Self {
        let exe = std::env::current_exe().expect("the test binary");
        let mut child = std::process::Command::new(exe)
            .args([
                "--exact",
                "daemon_host_coreml",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                "REPO_SEARCH_TEST_HOST",
                serde_json::to_string(config).expect("HostConfig json"),
            )
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("the daemon host spawns");
        let stdin = child.stdin.take().expect("piped");
        let (ready_tx, ready) = std::sync::mpsc::channel();
        let stdout = child.stdout.take().expect("piped");
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut lines = BufReader::new(stdout);
            let mut line = String::new();
            while lines.read_line(&mut line).unwrap_or(0) > 0 {
                if line.trim() == "READY" {
                    let _ = ready_tx.send(());
                }
                line.clear();
            }
        });
        tokio::task::block_in_place(|| {
            ready
                .recv_timeout(client::TIMEOUT)
                .expect("the host prints READY after binding");
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

impl Drop for Host {
    /// A panicking test must not leave the host running.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
