//! Real launchd tests: a per-pid `LaunchAgent` is installed into
//! `user/<uid>` through `ensure_agent`, and the daemon lifecycle — idle
//! exit, kill, relaunch on the next connection, reinstall — runs through
//! launchd itself. They need `REPO_SEARCH_MODEL_DIR` and
//! `REPO_SEARCH_CORPUS`, so they are `#[ignore]`d and run by the release
//! gate.

#![cfg(target_os = "macos")]

#[path = "support/client.rs"]
mod client;
#[path = "support/model.rs"]
mod model;
// `daemon.rs` uses the rest of the fixture.
#[allow(dead_code)]
#[path = "support/repo.rs"]
mod repo;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use repo_search::install::{self, AgentConfig, Check, SystemLaunchctl, ensure_agent};
use repo_search::kqueue::ExitWatch;
use repo_search::launchd::{LABEL, LaunchAgent};
use repo_search::protocol::{Final, Hello, Request};

/// The `search` binary the test agents run.
fn program() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_search"))
}

/// The handshake matching a daemon running `program` — the `search`
/// binary, not this test binary.
fn hello(program: &Path) -> Hello {
    client::hello_for(program)
}

/// Runs `launchctl` and returns its output.
fn launchctl(args: &[&str]) -> std::process::Output {
    std::process::Command::new("launchctl")
        .args(args)
        .output()
        .expect("launchctl runs")
}

/// The test config: label `<LABEL>.test-<pid>-<name>` — per test and per
/// process, because a stray agent from another run cannot collide with it —
/// and the plist, lock, socket and log all inside `dir`.
fn config(
    name: &str,
    dir: &Path,
    program: PathBuf,
    model_cache: PathBuf,
    idle_exit: Duration,
) -> AgentConfig {
    AgentConfig {
        domain: repo_search::launchd::domain(),
        plist: dir.join("agent.plist"),
        lock: dir.join("install.lock"),
        agent: LaunchAgent {
            label: format!("{LABEL}.test-{}-{name}", std::process::id()),
            program,
            socket: dir.join("daemon.sock"),
            idle_exit,
            model_cache: Some(model_cache),
            stderr: dir.join("daemon.log"),
        },
        timeouts: install::Timeouts {
            teardown: install::TEARDOWN_TIMEOUT,
            socket: install::SOCKET_TIMEOUT,
        },
    }
}

/// A bootstrapped launchd agent; booted out on drop.
struct Agent {
    label: String,
    socket: PathBuf,
    _tmp: tempfile::TempDir,
}

impl Agent {
    /// The `<domain>/<label>` bootout target.
    fn target(&self) -> String {
        format!("{}/{}", repo_search::launchd::domain(), self.label)
    }
}

/// Installs the test agent through `ensure_agent` — the production install
/// path — and returns once its socket is bound.
fn agent(name: &str, tmp: tempfile::TempDir, model_cache: PathBuf, idle_exit: Duration) -> Agent {
    let config = config(name, tmp.path(), program(), model_cache, idle_exit);
    let socket = config.agent.socket.clone();
    let label = config.agent.label.clone();
    tokio::task::block_in_place(|| ensure_agent(&config, &mut SystemLaunchctl, Check::Fast))
        .expect("the agent installs");
    Agent {
        label,
        socket,
        _tmp: tmp,
    }
}

impl Drop for Agent {
    /// Boot the test agent out of `user/<uid>` whether the test passed or
    /// panicked; a daemon it relaunched dies with it.
    fn drop(&mut self) {
        let _ = launchctl(&["bootout", &self.target()]);
    }
}

/// The daemon pid a `Status` request reports.
async fn daemon_pid(socket: &Path, hello: &Hello, worktree: &Path) -> u32 {
    let events = client::request(
        socket,
        hello,
        Request::Status {
            worktree: worktree.to_path_buf(),
        },
    )
    .await;
    match client::final_of(&events) {
        Final::Status(status) => status.daemon.pid,
        other => panic!("expected a Status, got {other:?}"),
    }
}

/// A query answers, `Status` names the daemon's pid, the process exits on
/// its 3 s idle deadline — waited for on kqueue — and the next connection
/// starts a replacement with a different pid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn relaunch_after_idle_exit() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());
    let agent = agent("idle", tmp, model_cache, Duration::from_secs(3));
    let socket = agent.socket.clone();
    let hello = hello(&program());
    let repo = repo::repo(&[("src/lib.rs".into(), repo::rust_fn("idle_probe", 500))]);

    let events = client::request(&socket, &hello, client::search(&repo.root, "a function")).await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the first query did not answer: {:?}",
        client::final_of(&events)
    );
    let first = daemon_pid(&socket, &hello, &repo.root).await;

    let exited = tokio::task::block_in_place(|| {
        ExitWatch::register(first)
            .expect("the daemon's exit registers")
            .wait(Duration::from_secs(30))
    });
    assert!(exited, "the daemon did not exit within 30 s of going idle");

    let events = client::request(&socket, &hello, client::search(&repo.root, "a function")).await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the second query did not answer: {:?}",
        client::final_of(&events)
    );
    let second = daemon_pid(&socket, &hello, &repo.root).await;
    assert_ne!(first, second, "launchd did not start a replacement daemon");
}

/// The `Index { wait: true }` sequence of `kill_and_resume_on_the_model`,
/// killed by `launchctl kill SIGKILL`: the relaunch comes from launchd on
/// the next connection to the same socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn kill_resumes_through_launchd() {
    let tmp = tempfile::tempdir().expect("tmp");
    let worktree = model::clone_corpus(&tmp);
    let model_cache = tmp.path().join("models");
    let store = model::model_store(model_cache.clone());
    let model_id = store.model_id();
    let agent = agent("kill", tmp, model_cache, Duration::from_secs(600));
    let socket = agent.socket.clone();
    let hello = hello(&program());

    let mut stream = model::index_wait(&socket, &hello, &worktree).await;
    model::await_coverage(&mut stream, 500).await;
    drop(stream);

    let out = launchctl(&["kill", "SIGKILL", &agent.target()]);
    assert!(
        out.status.success(),
        "launchctl kill {} failed: {}",
        agent.target(),
        String::from_utf8_lossy(&out.stderr)
    );

    let committed = model::committed(&worktree, &worktree.join(".git"), &model_id).await;
    assert!(committed >= 500, "{committed} committed vectors");

    model::assert_resume(&socket, &hello, &worktree, committed).await;
}

/// Install, search (so the daemon is running), then reinstall with the
/// `program` changed to a copy of the binary — as an installer replacing
/// it in place would: the old daemon's exit is awaited, so the
/// `bootstrap` after `bootout` does not hit `Bootstrap failed: 5`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn reinstall_waits_for_a_running_daemon() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());
    let repo = repo::repo(&[("src/lib.rs".into(), repo::rust_fn("reinstall_probe", 500))]);

    let mut config = config(
        "reinstall",
        tmp.path(),
        program(),
        model_cache,
        Duration::from_secs(600),
    );
    let socket = config.agent.socket.clone();
    let agent = Agent {
        label: config.agent.label.clone(),
        socket: socket.clone(),
        _tmp: tmp,
    };
    let mut launchctl = SystemLaunchctl;
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the first install succeeds");
    assert_eq!(outcome, install::Agent::Installed);

    let events = client::request(
        &socket,
        &hello(&program()),
        client::search(&repo.root, "a function"),
    )
    .await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the first query did not answer: {:?}",
        client::final_of(&events)
    );
    let first = daemon_pid(&socket, &hello(&program()), &repo.root).await;

    // A binary replaced at a new path: the plist's program changes, so the
    // install replaces the agent.
    let copy = agent._tmp.path().join("search-copy");
    std::fs::copy(program(), &copy).expect("the binary copies");
    config.agent.program = copy.clone();
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the reinstall succeeds without `Bootstrap failed: 5`");
    assert_eq!(
        outcome,
        install::Agent::Replaced(install::Reason::PlistChanged)
    );

    let hello = client::hello_for(&copy);
    let events = client::request(&socket, &hello, client::search(&repo.root, "a function")).await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the query after reinstall did not answer: {:?}",
        client::final_of(&events)
    );
    let second = daemon_pid(&socket, &hello, &repo.root).await;
    assert_ne!(first, second, "the reinstall did not replace the daemon");
}

/// An agent installed but never connected to has no running process; its
/// replacement runs `bootstrap` straight after `bootout`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn reinstall_of_an_idle_agent() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());

    let mut config = config(
        "idle-reinstall",
        tmp.path(),
        program(),
        model_cache,
        Duration::from_secs(600),
    );
    let agent = Agent {
        label: config.agent.label.clone(),
        socket: config.agent.socket.clone(),
        _tmp: tmp,
    };
    let mut launchctl = SystemLaunchctl;
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the first install succeeds");
    assert_eq!(outcome, install::Agent::Installed);

    let copy = agent._tmp.path().join("search-copy");
    std::fs::copy(program(), &copy).expect("the binary copies");
    config.agent.program = copy;
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the reinstall of an idle agent succeeds without `Bootstrap failed: 5`");
    assert_eq!(
        outcome,
        install::Agent::Replaced(install::Reason::PlistChanged)
    );
}

/// Two `search` processes racing a first install serialize on the lock:
/// both succeed and exactly one bootstraps.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn concurrent_installs_bootstrap_once() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());
    let config = config(
        "concurrent",
        tmp.path(),
        program(),
        model_cache,
        Duration::from_secs(600),
    );
    let agent = Agent {
        label: config.agent.label.clone(),
        socket: config.agent.socket.clone(),
        _tmp: tmp,
    };
    let barrier = Arc::new(Barrier::new(2));
    let config = Arc::new(config);
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let config = config.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                ensure_agent(&config, &mut SystemLaunchctl, Check::Fast)
            })
        })
        .collect();
    let outcomes: Vec<_> = threads
        .into_iter()
        .map(|t| t.join().expect("the installer joins"))
        .collect();
    for outcome in &outcomes {
        assert!(outcome.is_ok(), "a concurrent install failed: {outcome:?}");
    }
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, Ok(install::Agent::Installed)))
            .count(),
        1,
        "exactly one of the racers installs: {outcomes:?}"
    );
    assert!(agent.socket.exists(), "the install left the socket bound");
}
