//! Real launchd tests: a per-pid `LaunchAgent` is bootstrapped into
//! `gui/<uid>`, and the daemon lifecycle — idle exit, kill, relaunch on the
//! next connection — runs through launchd itself. They need a GUI launchd
//! domain, `REPO_SEARCH_MODEL_DIR` and `REPO_SEARCH_CORPUS`, so they are
//! `#[ignore]`d and run by the release gate.

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
use std::time::Duration;

use askama::Template;
use repo_search::kqueue::{DirWatch, ExitWatch};
use repo_search::launchd::{LABEL, LaunchAgent};
use repo_search::protocol::{Final, Request};

/// The calling process's uid, for the `gui/<uid>` domain.
fn uid() -> u32 {
    // SAFETY: getuid cannot fail.
    unsafe { libc::getuid() }
}

/// Runs `launchctl` and returns its output.
fn launchctl(args: &[&str]) -> std::process::Output {
    std::process::Command::new("launchctl")
        .args(args)
        .output()
        .expect("launchctl runs")
}

/// A bootstrapped launchd agent; booted out on drop.
struct Agent {
    label: String,
    socket: PathBuf,
    _tmp: tempfile::TempDir,
}

/// Renders the test agent's plist, writes it into `tmp` and bootstraps it
/// in `gui/<uid>`. The label is `<LABEL>.test-<pid>-<name>`: per test and
/// per process, because sequential bootstraps of one label race launchd's
/// asynchronous `bootout` teardown, and a stray agent from another run
/// cannot collide with it.
fn agent(name: &str, tmp: tempfile::TempDir, model_cache: PathBuf, idle_exit: Duration) -> Agent {
    let label = format!("{LABEL}.test-{}-{name}", std::process::id());
    let socket = tmp.path().join("daemon.sock");
    let document = LaunchAgent {
        label: label.clone(),
        program: PathBuf::from(env!("CARGO_BIN_EXE_search")),
        socket: socket.clone(),
        idle_exit,
        model_cache: Some(model_cache),
        stderr: tmp.path().join("daemon.log"),
    };
    let plist = tmp.path().join("agent.plist");
    std::fs::write(&plist, document.render().expect("the template renders")).expect("plist write");
    // Registered before `bootstrap` so the socket's creation — which
    // launchd performs asynchronously after the job loads — cannot be
    // missed.
    let watch = DirWatch::register(tmp.path()).expect("the socket dir watches");
    let domain = format!("gui/{}", uid());
    let out = launchctl(&[
        "bootstrap",
        &domain,
        plist.to_str().expect("utf-8 plist path"),
    ]);
    assert!(
        out.status.success(),
        "launchctl bootstrap {domain} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // `bootstrap` returns once the job is loaded; launchd materializes the
    // socket a moment later, and connecting before then reads ENOENT.
    let exists = tokio::task::block_in_place(|| watch.wait_for(&socket, Duration::from_secs(30)));
    assert!(exists, "launchd did not materialize {}", socket.display());
    Agent {
        label,
        socket,
        _tmp: tmp,
    }
}

impl Drop for Agent {
    /// Boot the test agent out of `gui/<uid>` whether the test passed or
    /// panicked; a daemon it relaunched dies with it.
    fn drop(&mut self) {
        let target = format!("gui/{}/{}", uid(), self.label);
        let _ = launchctl(&["bootout", &target]);
    }
}

/// The daemon pid a `Status` request reports.
async fn daemon_pid(socket: &Path, worktree: &Path) -> u32 {
    let events = client::request(
        socket,
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
    let repo = repo::repo(&[("src/lib.rs".into(), repo::rust_fn("idle_probe", 500))]);

    let events = client::request(&socket, client::search(&repo.root, "a function")).await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the first query did not answer: {:?}",
        client::final_of(&events)
    );
    let first = daemon_pid(&socket, &repo.root).await;

    let exited = tokio::task::block_in_place(|| {
        ExitWatch::register(first)
            .expect("the daemon's exit registers")
            .wait(Duration::from_secs(30))
    });
    assert!(exited, "the daemon did not exit within 30 s of going idle");

    let events = client::request(&socket, client::search(&repo.root, "a function")).await;
    assert!(
        matches!(client::final_of(&events), Final::Answer(_)),
        "the second query did not answer: {:?}",
        client::final_of(&events)
    );
    let second = daemon_pid(&socket, &repo.root).await;
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

    let mut stream = model::index_wait(&socket, &worktree).await;
    model::await_coverage(&mut stream, 500).await;
    drop(stream);

    let target = format!("gui/{}/{}", uid(), agent.label);
    let out = launchctl(&["kill", "SIGKILL", &target]);
    assert!(
        out.status.success(),
        "launchctl kill {target} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let committed = model::committed(&worktree, &worktree.join(".git"), &model_id).await;
    assert!(committed >= 500, "{committed} committed vectors");

    model::assert_resume(&socket, &worktree, committed).await;
}
