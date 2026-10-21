//! Install-flow tests over `FakeLaunchd`: `ensure_agent`'s decisions, its
//! launchctl sequence and its waits, without a real launchd.

#![cfg(target_os = "macos")]

#[allow(dead_code)]
mod support;

use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::sync::Barrier;
use std::time::Duration;

use askama::Template;
use repo_search::install::{
    Agent, AgentConfig, Check, InstallError, Reason, Timeouts, ensure_agent,
};
use repo_search::launchd::LaunchAgent;
use support::client;
use support::launchd::{Call, FakeLaunchd};

/// A test config: `plist`, `lock`, `socket` and `stderr` all inside the
/// temp dir, the production `user/<uid>` domain, the spec's test timeouts.
fn test_config(tmp: &tempfile::TempDir, program: &Path) -> AgentConfig {
    let root = tmp.path();
    AgentConfig {
        domain: repo_search::launchd::domain(),
        plist: root.join("LaunchAgents/com.github.may-agent.repo-search.test.plist"),
        lock: root.join("support/install.lock"),
        agent: LaunchAgent {
            label: "com.github.may-agent.repo-search.test".to_owned(),
            program: program.to_path_buf(),
            socket: root.join("support/daemon.sock"),
            idle_exit: Duration::from_secs(600),
            model_cache: None,
            stderr: root.join("logs/daemon.log"),
        },
        timeouts: Timeouts {
            teardown: Duration::from_secs(10),
            socket: Duration::from_secs(2),
        },
    }
}

fn program() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_search"))
}

/// The fake for `config`, with the daemon identity the test binary gives.
fn launchd(config: &AgentConfig) -> FakeLaunchd {
    FakeLaunchd::new(
        config.agent.socket.clone(),
        tokio::runtime::Handle::current(),
        client::build(),
    )
}

/// The plist file written for `config`: an agent previously installed for
/// `program`.
fn write_plist(config: &AgentConfig) {
    std::fs::create_dir_all(config.plist.parent().unwrap()).expect("the plist dir");
    std::fs::write(&config.plist, config.agent.render().unwrap()).expect("the plist writes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_run_installs() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    let mut lctl = launchd.launchctl();

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect("the first install succeeds");
    assert_eq!(outcome, Agent::Installed);

    assert_eq!(
        launchd.calls(),
        vec![
            Call::Service,
            Call::Bootstrap {
                plist_contents: config.agent.render().unwrap()
            },
        ],
        "a first run asks launchd once, then bootstraps"
    );
    assert_eq!(
        std::fs::read_to_string(&config.plist).unwrap(),
        config.agent.render().unwrap(),
        "the installed plist is the rendered agent"
    );
    assert!(
        config.agent.stderr.parent().unwrap().is_dir(),
        "the log directory is created for the daemon's StandardErrorPath"
    );
    assert!(
        std::fs::symlink_metadata(&config.agent.socket)
            .expect("the socket exists")
            .file_type()
            .is_socket(),
        "launchd bound the socket"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn current_agent_is_left_alone() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    let mut lctl = launchd.launchctl();

    tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect("the first install succeeds");
    assert_eq!(launchd.calls().len(), 2, "the install asked launchd twice");

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect("the fast check succeeds");
    assert_eq!(outcome, Agent::Current);
    assert_eq!(
        launchd.calls(),
        Vec::<Call>::new(),
        "Check::Fast makes no launchctl call"
    );

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Verify))
        .expect("the verify check succeeds");
    assert_eq!(outcome, Agent::Current);
    assert_eq!(
        launchd.calls(),
        vec![Call::Service],
        "Check::Verify asks launchd exactly once"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn moved_binary_replaces_after_teardown() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    // The installed plist names the binary's previous path.
    write_plist(&test_config(&tmp, Path::new("/defunct/search")));
    launchd.load_stand_in();
    let mut lctl = launchd.launchctl();

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect("the replacement succeeds");
    assert_eq!(outcome, Agent::Replaced(Reason::PlistChanged));
    let calls = launchd.calls();
    assert_eq!(
        calls.len(),
        3,
        "service check, bootout, bootstrap: {calls:?}"
    );
    assert!(matches!(calls[0], Call::Service));
    assert!(matches!(calls[1], Call::Bootout));
    assert!(
        matches!(calls[2], Call::Bootstrap { .. }),
        "the bootstrap ran, so the stand-in's exit was awaited first"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_socket_is_reinstalled() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    write_plist(&config);
    launchd.load_without_process();
    let mut lctl = launchd.launchctl();

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect("the reinstall succeeds");
    assert_eq!(outcome, Agent::Replaced(Reason::NotRunning));
    assert!(
        std::fs::symlink_metadata(&config.agent.socket)
            .expect("the socket exists")
            .file_type()
            .is_socket(),
        "launchd bound the socket"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_socket_file_is_removed() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    write_plist(&config);
    std::fs::create_dir_all(config.agent.socket.parent().unwrap()).expect("the socket dir");
    std::fs::write(&config.agent.socket, b"stale").expect("the stale socket file");
    let mut lctl = launchd.launchctl();

    let outcome = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Verify))
        .expect("the reinstall succeeds");
    assert_eq!(outcome, Agent::Replaced(Reason::NotRunning));
    let meta = std::fs::symlink_metadata(&config.agent.socket).expect("the socket exists");
    assert!(
        meta.file_type().is_socket(),
        "the stale file was replaced by launchd's real socket"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_first_runs_bootstrap_once() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    let barrier = Barrier::new(2);

    let outcomes = std::thread::scope(|scope| {
        // Two handles of one launchd, one moved into each thread — as two
        // `search` processes would race the first install.
        let [a, b] = [launchd.launchctl(), launchd.launchctl()];
        let config = &config;
        let barrier = &barrier;
        let threads: Vec<_> = [a, b]
            .into_iter()
            .map(|mut lctl| {
                scope.spawn(move || {
                    barrier.wait();
                    ensure_agent(config, &mut lctl, Check::Fast)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("the installer joins"))
            .collect::<Vec<_>>()
    });

    for outcome in &outcomes {
        assert!(outcome.is_ok(), "a concurrent install failed: {outcome:?}");
    }
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, Ok(Agent::Installed)))
            .count(),
        1,
        "exactly one of the racers installs: {outcomes:?}"
    );
    assert_eq!(
        launchd
            .calls()
            .iter()
            .filter(|c| matches!(c, Call::Bootstrap { .. }))
            .count(),
        1,
        "exactly one bootstrap happened"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bootstrap_failure_is_reported_verbatim() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    launchd.fail_next_bootstrap("Bootstrap failed: 5: Input/output error");
    let mut lctl = launchd.launchctl();

    let error = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect_err("the forced bootstrap failure propagates");
    let InstallError::Launchctl(inner) = error else {
        panic!("expected a Launchctl error, got {error:?}");
    };
    let text = inner.to_string();
    assert!(
        text.contains("bootstrap user/"),
        "the error names the launchctl call: {text}"
    );
    assert!(
        text.contains("Bootstrap failed: 5: Input/output error"),
        "the error carries launchd's stderr verbatim: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn socket_wait_times_out() {
    let tmp = tempfile::tempdir().expect("tmp");
    let config = test_config(&tmp, &program());
    let launchd = launchd(&config);
    launchd.bootstrap_without_socket();
    let mut lctl = launchd.launchctl();

    let error = tokio::task::block_in_place(|| ensure_agent(&config, &mut lctl, Check::Fast))
        .expect_err("a bootstrap that never binds the socket fails");
    match error {
        InstallError::Socket { path, after } => {
            assert_eq!(path, config.agent.socket);
            assert_eq!(after, config.timeouts.socket);
        }
        other => panic!("expected a Socket error, got {other:?}"),
    }
}
