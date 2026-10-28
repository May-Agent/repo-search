//! End-to-end client tests: `cli::run` against the in-process stub daemon
//! through `FakeLaunchd`, with `Vec<u8>` terminals standing in for stdio.

#![cfg(target_os = "macos")]

#[allow(dead_code)]
mod support;

use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::mpsc;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::AsyncWrite;
use tokio::sync::mpsc::UnboundedSender;

use clap::Parser;
use repo_search::cli::{Cli, Env, Exit, Invocation, Terminal};
use repo_search::install::{AgentConfig, Plist, Timeouts};
use repo_search::launchd::{LaunchAgent, ServerAgent};
use repo_search::protocol::BuildId;
use std::net::{IpAddr, Ipv4Addr};
use support::client;
use support::daemon::Model;
use support::launchd::{FakeLaunchctl, FakeLaunchd};
use support::repo;
use support::stub::{Instance, StubBackend};

/// The test binary's `BuildId`, given to the fake's daemons and to the
/// `Hello` the test sends — they must match for the handshake to pass.
fn launchd(config: &AgentConfig) -> FakeLaunchd {
    FakeLaunchd::new(
        config.agent.socket().to_path_buf(),
        tokio::runtime::Handle::current(),
        client::build(),
    )
}

/// A test `AgentConfig`: `plist`, `lock`, `socket` and `stderr` inside the
/// temp dir, the production `user/<uid>` domain, the spec's test timeouts.
/// The server agent `cli::run` keeps current beside the daemon's. Its
/// plist and socket file are written at construction so every ensure
/// reads it as `Current` without touching the fake's single-job state.
fn server_config(tmp: &Path) -> AgentConfig {
    let config = AgentConfig {
        domain: repo_search::launchd::domain(),
        plist: tmp.join("LaunchAgents/com.github.may-agent.repo-search.server.cli-test.plist"),
        lock: tmp.join("support/install.lock"),
        agent: Plist::Server(ServerAgent {
            label: "com.github.may-agent.repo-search.server.cli-test".to_owned(),
            program: tmp.join("search"),
            socket: tmp.join("support/server.sock"),
            idle_exit: Duration::from_secs(600),
            build: "cli-test".to_owned(),
            model_cache: None,
            token_file: None,
            bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 41827,
            stderr: tmp.join("logs/server.log"),
        }),
        timeouts: Timeouts {
            teardown: Duration::from_secs(10),
            socket: Duration::from_secs(2),
        },
    };
    std::fs::create_dir_all(config.plist.parent().unwrap()).unwrap();
    std::fs::write(&config.plist, config.agent.render().unwrap()).unwrap();
    std::fs::create_dir_all(config.agent.socket().parent().unwrap()).unwrap();
    std::fs::write(config.agent.socket(), b"").unwrap();
    config
}

fn config(tmp: &tempfile::TempDir) -> AgentConfig {
    let root = tmp.path();
    AgentConfig {
        domain: repo_search::launchd::domain(),
        plist: root.join("LaunchAgents/com.github.may-agent.repo-search.test.plist"),
        lock: root.join("support/install.lock"),
        agent: Plist::Daemon(LaunchAgent {
            label: "com.github.may-agent.repo-search.cli-test".to_owned(),
            program: root.join("search"),
            socket: root.join("support/daemon.sock"),
            idle_exit: Duration::from_secs(600),
            model_cache: None,
            state_dir: None,
            stderr: root.join("logs/daemon.log"),
        }),
        timeouts: Timeouts {
            teardown: Duration::from_secs(10),
            socket: Duration::from_secs(2),
        },
    }
}

/// The `Env` `cli::run` gets in a test: the test cwd and config, the
/// fake's `Launchctl`, and the handshake matching the fake's daemons.
fn cli_env(config: AgentConfig, launchd: &FakeLaunchd, cwd: &Path) -> Env<FakeLaunchctl> {
    Env {
        cwd: cwd.to_path_buf(),
        server_agent: server_config(config.plist.parent().unwrap().parent().unwrap()),
        agent: config,
        launchctl: launchd.launchctl(),
        hello: client::hello(),
    }
}

fn vec_term() -> Terminal<Vec<u8>, Vec<u8>> {
    Terminal {
        stdout: Vec::new(),
        stderr: Vec::new(),
        stderr_tty: false,
    }
}

fn tty_term() -> Terminal<Vec<u8>, Vec<u8>> {
    Terminal {
        stderr_tty: true,
        ..vec_term()
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("program output is utf-8")
}

fn invocation(args: &[&str]) -> Invocation {
    Cli::try_parse_from(args)
        .expect("the arguments parse")
        .invocation()
}

/// A repo whose one function carries `MARKER` in its name and body.
fn marked_repo() -> support::repo::TestRepo {
    repo::repo(&[("src/lib.rs".into(), repo::rust_fn("xylophonemarker", 500))])
}

/// A first run installs the agent and the query answers: exit 0, stdout
/// is the rendered `Code:` section naming the marker, and the install is
/// announced on stderr.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_search_installs_and_answers() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let repo = marked_repo();
    let env = cli_env(config(&tmp), &launchd, &repo.root.join("src"));
    let mut term = vec_term();

    let exit =
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term).await;

    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr));
    let stdout = text(&term.stdout);
    assert!(
        stdout.starts_with("Code:"),
        "stdout starts with the Code section: {stdout}"
    );
    assert!(
        stdout.contains("xylophonemarker"),
        "the marker function is named: {stdout}"
    );
    assert!(
        text(&term.stderr).contains("installed launch agent"),
        "the install is announced: {}",
        text(&term.stderr)
    );
}

/// `--json` stdout is core's `{ code, docs, freshness }` plus the one
/// newline the client adds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn json_answer_is_core_json() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let repo = marked_repo();
    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();

    let exit = repo_search::cli::run(
        invocation(&["search", "xylophonemarker", "--json"]),
        env,
        &mut term,
    )
    .await;

    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr));
    let stdout = text(&term.stdout);
    assert!(stdout.ends_with('\n'), "json output ends with a newline");
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("stdout parses as JSON");
    assert!(
        json.get("code").is_some() && json.get("freshness").is_some(),
        "core's answer shape: {json}"
    );
}

/// A query matching nothing in lexical mode writes only the freshness
/// line and exits 1, grep's no-match code.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_results_exit_1() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let repo = marked_repo();
    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();

    let exit = repo_search::cli::run(
        invocation(&["search", "zzzqqqunmatched", "--mode", "lexical"]),
        env,
        &mut term,
    )
    .await;

    assert_eq!(exit, Exit::NoMatches, "stderr: {}", text(&term.stderr));
    assert_eq!(
        text(&term.stdout).matches('\n').count(),
        1,
        "stdout is the freshness line alone: {}",
        text(&term.stdout)
    );
}

/// A `cwd` that is not in a git worktree fails before any connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn outside_a_worktree_exits_2() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let outside = tempfile::tempdir().expect("non-git dir");
    let env = cli_env(config(&tmp), &launchd, outside.path());
    let mut term = vec_term();

    let exit =
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term).await;

    assert_eq!(exit, Exit::Error);
    assert!(
        text(&term.stderr).contains("is not inside a git worktree"),
        "stderr: {}",
        text(&term.stderr)
    );
}

/// `search status` after a search: the six text lines of section 6.3 and
/// the JSON shape, with the in-process daemon's pid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_text_and_json() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let repo = marked_repo();
    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();
    let exit =
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term).await;
    assert_eq!(exit, Exit::Ok, "the search indexes the worktree");

    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();
    let exit = repo_search::cli::run(invocation(&["search", "status"]), env, &mut term).await;
    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr));
    let stdout = text(&term.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        6,
        "worktree/index/pending/model/gpu/daemon: {lines:?}"
    );
    assert!(
        lines[5].contains(&format!("pid {}", std::process::id())),
        "the daemon is in process: {lines:?}"
    );

    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();
    let exit =
        repo_search::cli::run(invocation(&["search", "status", "--json"]), env, &mut term).await;
    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr));
    let json: serde_json::Value =
        serde_json::from_str(&text(&term.stdout)).expect("stdout parses as JSON");
    assert!(
        matches!(
            json["index"]["phase"].as_str(),
            Some("first_index") | Some("maintained")
        ),
        "index.phase: {json}"
    );
    assert_eq!(json["model"]["state"], "ready", "the search loaded it");
}

/// `search status` on a never-indexed worktree reports `index: null`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_before_any_search() {
    let tmp = tempfile::tempdir().expect("tmp");
    let launchd = launchd(&config(&tmp));
    let repo = marked_repo();
    let env = cli_env(config(&tmp), &launchd, &repo.root);
    let mut term = vec_term();

    let exit = repo_search::cli::run(invocation(&["search", "status"]), env, &mut term).await;

    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr));
    let stdout = text(&term.stdout);
    assert!(
        stdout.contains("index     none; the first search indexes this worktree"),
        "stdout: {stdout}"
    );
    assert!(
        !stdout.contains("pending"),
        "no pending line without an index: {stdout}"
    );
}

/// `index --wait` on a TTY: `embedding` progress redraws while coverage
/// climbs, the line is cleared before the answer, and the final `Indexed`
/// reads `(100%)`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn index_wait_reaches_full_coverage() {
    let tmp = tempfile::tempdir().expect("tmp");
    let mut backend = StubBackend::new(mpsc::channel().0);
    // Slow every embed call so the wait's coverage stream has events to
    // draw; the assertion is about the line, not the count.
    backend.set_instance_delay(Instance::Ane, Duration::from_millis(50));
    let config = config(&tmp);
    let launchd = FakeLaunchd::with_daemon(
        config.agent.socket().to_path_buf(),
        tokio::runtime::Handle::current(),
        client::build(),
        Model::Cached,
        backend,
    );
    let files: Vec<(String, String)> = (0..8)
        .map(|i| {
            (
                format!("src/f{i}.rs"),
                repo::rust_fn(&format!("xylophonemarker_{i}"), 500),
            )
        })
        .collect();
    let repo = repo::repo(&files);
    let env = cli_env(config, &launchd, &repo.root);
    let mut term = tty_term();

    let exit =
        repo_search::cli::run(invocation(&["search", "index", "--wait"]), env, &mut term).await;

    assert_eq!(exit, Exit::Ok, "stderr: {:?}", text(&term.stderr));
    assert!(
        text(&term.stdout).ends_with("(100%)\n"),
        "the wait answered at full coverage: {}",
        text(&term.stdout)
    );
    let stderr = text(&term.stderr);
    assert!(
        stderr.contains("embedding"),
        "coverage progress drew on the TTY: {stderr:?}"
    );
    assert!(
        term.stderr.ends_with(b"\r\x1b[2K"),
        "the line was cleared before the answer: {stderr:?}"
    );
}

/// The daemon runs with build A and the client sends build B: the
/// mismatch shuts the daemon down (`run` returns `Ok`) and the client
/// exits 2 — nothing relaunches an in-process daemon, so the retry ends
/// in a connect error. Launchd's relaunch is covered by the launchd
/// tests.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_build_shuts_the_daemon_down() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = config(&tmp);
    // No real file can carry this identity, so it differs from the
    // test binary's `BuildId` the client sends by construction.
    let stale = BuildId {
        dev: 0,
        ino: 0,
        size: 0,
        mtime_ns: 0,
    };
    let launchd = FakeLaunchd::new(
        config.agent.socket().to_path_buf(),
        tokio::runtime::Handle::current(),
        stale,
    );
    let repo = marked_repo();
    let env = cli_env(config, &launchd, &repo.root);
    let mut term = vec_term();

    let exit =
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term).await;

    assert_eq!(exit, Exit::Error, "stderr: {}", text(&term.stderr));
    let join = launchd
        .take_daemon_join()
        .expect("the bootstrap started a daemon");
    tokio::time::timeout(client::TIMEOUT, join)
        .await
        .expect("run returns after the mismatch")
        .expect("run task does not panic")
        .expect("a build mismatch is a clean shutdown");
}

/// A `bootstrap` failure is reported verbatim and no connection is
/// attempted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn install_failure_exits_2() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = config(&tmp);
    let launchd = FakeLaunchd::new(
        config.agent.socket().to_path_buf(),
        tokio::runtime::Handle::current(),
        client::build(),
    );
    launchd.fail_next_bootstrap("Bootstrap failed: 5: Input/output error");
    let repo = marked_repo();
    let env = cli_env(config, &launchd, &repo.root);
    let mut term = vec_term();

    let exit =
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term).await;

    assert_eq!(exit, Exit::Error);
    let stderr = text(&term.stderr);
    assert!(
        stderr.contains("bootstrap") && stderr.contains("Bootstrap failed: 5: Input/output error"),
        "launchctl's stderr is verbatim: {stderr}"
    );
    assert_eq!(
        launchd.calls().len(),
        2,
        "one service query and one bootstrap — no connection attempted"
    );
}

/// A daemon that fails after socket activation answers the waiting client
/// with its startup error; the client exits 2 naming the cause and the log.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_startup_failure_exits_2() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = config(&tmp);
    let log = config.agent.stderr().to_path_buf();
    let launchd = launchd(&config);
    launchd.bootstrap_failing_daemon("Apple Z1");
    let repo = marked_repo();
    let env = cli_env(config, &launchd, &repo.root);
    let mut term = vec_term();

    let exit = tokio::time::timeout(
        Duration::from_secs(30),
        repo_search::cli::run(invocation(&["search", "xylophonemarker"]), env, &mut term),
    )
    .await
    .expect("the client is answered instead of hanging");

    assert_eq!(exit, Exit::Error);
    let stderr = text(&term.stderr);
    assert!(
        stderr.contains(
            "the daemon failed to start: could not determine the host's chip class: \
             unrecognised chip brand \"Apple Z1\""
        ),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains(&log.display().to_string()),
        "stderr: {stderr}"
    );
}

/// A stderr that also forwards every write to a channel, so a test can
/// act once a line lands — here, release the download gate on the
/// announcement.
struct Spy {
    buf: Vec<u8>,
    tx: UnboundedSender<Vec<u8>>,
}

impl AsyncWrite for Spy {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.buf.extend_from_slice(buf);
        let _ = self.tx.send(buf.to_vec());
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Off a TTY, a model download is announced once as a plain line — never
/// redrawn — and `Reconciling`/`Coverage` print nothing. `serve_gated`
/// holds the download mid-body so the daemon's `Downloading` state
/// persists until sampled — without it a fast download could jump to
/// `Ready` before the client's watch sees it — and the gate releases when
/// the announcement lands on stderr.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_tty_shows_only_announcements() {
    let body = support::http::archive(&[("manifest.json", b"{}")]);
    let (release_tx, release) = mpsc::channel();
    let (url, _server) = support::http::serve_gated(body.len() as u64, body.clone(), release);
    let tmp = tempfile::tempdir().expect("tmp");
    let config = config(&tmp);
    let launchd = FakeLaunchd::with_daemon(
        config.agent.socket().to_path_buf(),
        tokio::runtime::Handle::current(),
        client::build(),
        Model::Served {
            url,
            sha: support::daemon::digest_of(&body),
        },
        StubBackend::new(mpsc::channel().0),
    );
    let repo = marked_repo();
    let env = cli_env(config, &launchd, &repo.root);

    let (tx, mut seen) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
    let mut term = Terminal {
        stdout: Vec::new(),
        stderr: Spy {
            buf: Vec::new(),
            tx,
        },
        stderr_tty: false,
    };
    let exit = {
        let run = std::pin::pin!(repo_search::cli::run(
            invocation(&["search", "xylophonemarker"]),
            env,
            &mut term,
        ));
        let release_on_announcement = async {
            let mut stderr = Vec::new();
            while let Some(chunk) = seen.recv().await {
                stderr.extend_from_slice(&chunk);
                if String::from_utf8_lossy(&stderr).contains("downloading the model") {
                    release_tx.send(()).expect("the gated server lives");
                    return;
                }
            }
        };
        let (exit, announced) = tokio::join!(
            run,
            tokio::time::timeout(Duration::from_secs(60), release_on_announcement),
        );
        announced.expect("the download announcement is written");
        exit
    };
    assert_eq!(exit, Exit::Ok, "stderr: {}", text(&term.stderr.buf));
    let stderr = text(&term.stderr.buf);
    assert_eq!(
        stderr.matches("search: downloading the model").count(),
        1,
        "the download is announced exactly once: {stderr}"
    );
    assert!(
        !stderr.contains('\r'),
        "off a TTY nothing is redrawn: {stderr:?}"
    );
}
