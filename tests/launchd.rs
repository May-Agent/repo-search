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

use clap::Parser;
use repo_search::cli::{self, Cli, Exit, Terminal};
use repo_search::daemon::server::{Endpoint, ServerClient};
use repo_search_client::launcher::install::Plist;
use repo_search_client::launcher::install::{
    self, AgentConfig, Check, Launchctl, SystemLaunchctl, ensure_agent,
};
use repo_search_client::launcher::kqueue::ExitWatch;
use repo_search_client::launcher::launchd::{LABEL, LaunchAgent, SERVER_LABEL, ServerAgent};
use repo_search_client::launcher::{LaunchdAgents, SOCKET_TIMEOUT, TEARDOWN_TIMEOUT};
use repo_search_client::protocol::{Final, Hello, Request};
use repo_search_core::StateDir;

/// The `search` binary the test agents run.
fn program() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_search"))
}

/// A free TCP port for a test server agent, bound and released.
fn server_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("a probe bind")
        .local_addr()
        .unwrap()
        .port()
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

/// The embedding server's socket: the daemon's `server_socket` and the
/// server agent's `socket` are this one path.
fn server_socket(dir: &Path) -> PathBuf {
    dir.join("server.sock")
}

/// The test server agent's label, per test and per process like the
/// daemon's.
fn server_label(name: &str) -> String {
    format!("{SERVER_LABEL}.test-{}-{name}", std::process::id())
}

/// The test server config, the one place a server agent's config is built:
/// the plist, log and socket inside `dir`, `--build` naming `program` so a
/// started server accepts it, and `model_cache` the test's model store so
/// the model is present instead of downloaded on the runner.
fn server_config(
    name: &str,
    dir: &Path,
    program: &Path,
    model_cache: PathBuf,
    idle_exit: Duration,
) -> AgentConfig {
    AgentConfig {
        domain: repo_search_client::launcher::launchd::domain(),
        plist: dir.join("server-agent.plist"),
        lock: dir.join("install.lock"),
        agent: Plist::Server(ServerAgent {
            label: server_label(name),
            program: program.to_path_buf(),
            socket: server_socket(dir),
            idle_exit,
            build: client::hello_for(program).build.to_string(),
            model_cache: Some(model_cache),
            token_file: None,
            bind: std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            port: server_port(),
            stderr: dir.join("server.log"),
        }),
        timeouts: install::Timeouts {
            teardown: TEARDOWN_TIMEOUT,
            socket: SOCKET_TIMEOUT,
        },
    }
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
        domain: repo_search_client::launcher::launchd::domain(),
        plist: dir.join("agent.plist"),
        lock: dir.join("install.lock"),
        agent: Plist::Daemon(LaunchAgent {
            label: format!("{LABEL}.test-{}-{name}", std::process::id()),
            program,
            socket: dir.join("daemon.sock"),
            server_socket: server_socket(dir),
            idle_exit,
            model_cache: Some(model_cache),
            state_dir: Some(dir.join("state")),
            stderr: dir.join("daemon.log"),
        }),
        timeouts: install::Timeouts {
            teardown: TEARDOWN_TIMEOUT,
            socket: SOCKET_TIMEOUT,
        },
    }
}

/// A bootstrapped launchd agent; booted out on drop.
struct Agent {
    label: String,
    socket: PathBuf,
    /// The server agent the daemon queries, when the test answers queries.
    _server: Option<ServerGuard>,
    _tmp: tempfile::TempDir,
}

impl Agent {
    /// The `<domain>/<label>` bootout target.
    fn target(&self) -> String {
        target(&self.label)
    }
}

/// The `<domain>/<label>` bootout target of `label`.
fn target(label: &str) -> String {
    format!(
        "{}/{label}",
        repo_search_client::launcher::launchd::domain()
    )
}

/// A test server agent; booted out on drop, whether the test passed or
/// panicked. It is made before the install so a failed install is still
/// cleaned up.
struct ServerGuard {
    label: String,
}

impl ServerGuard {
    /// Installs the server agent of `config` through `ensure_agent`.
    fn install(config: &AgentConfig) -> Self {
        let guard = Self {
            label: config.agent.label().to_owned(),
        };
        tokio::task::block_in_place(|| ensure_agent(config, &mut SystemLaunchctl, Check::Fast))
            .expect("the server agent installs");
        guard
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        boot_out(&target(&self.label));
    }
}

/// Installs the test agent through `ensure_agent` — the production install
/// path — and returns once its socket is bound. The daemon embeds through
/// the embedding server, so the server agent is installed beside it as the
/// CLI's `LaunchdAgents` does; without it nothing listens on the daemon's
/// `server_socket` and every query fails.
fn agent(name: &str, tmp: tempfile::TempDir, model_cache: PathBuf, idle_exit: Duration) -> Agent {
    let server = ServerGuard::install(&server_config(
        name,
        tmp.path(),
        &program(),
        model_cache.clone(),
        Duration::from_secs(600),
    ));
    let config = config(name, tmp.path(), program(), model_cache, idle_exit);
    let socket = config.agent.socket().to_path_buf();
    let label = config.agent.label().to_owned();
    tokio::task::block_in_place(|| ensure_agent(&config, &mut SystemLaunchctl, Check::Fast))
        .expect("the agent installs");
    Agent {
        label,
        socket,
        _server: Some(server),
        _tmp: tmp,
    }
}

impl Drop for Agent {
    /// Boot the test agent out of `user/<uid>` whether the test passed or
    /// panicked; a daemon it relaunched dies with it. As in
    /// `ensure_agent`'s teardown, `bootout` returns before launchd has
    /// torn the job down, so a running process's exit is awaited — the
    /// label leaves `launchctl list` only once the teardown ends.
    /// Assertions run only when the test is not already unwinding, so a
    /// failure keeps its own panic.
    fn drop(&mut self) {
        boot_out(&self.target());
    }
}

/// Boots `target` out of `user/<uid>` and awaits its process's exit.
fn boot_out(target: &str) {
    let mut launchctl = SystemLaunchctl;
    let pid = launchctl
        .service(target)
        .ok()
        .flatten()
        .and_then(|service| service.pid);
    let exit = pid.and_then(|pid| ExitWatch::register(pid).ok());
    let booted = launchctl.bootout(target);
    if !std::thread::panicking() {
        booted.expect("the test agent boots out");
    }
    if let Some(exit) = exit {
        let exited = exit.wait(TEARDOWN_TIMEOUT);
        if !std::thread::panicking() {
            assert!(exited, "the test agent's process exits on bootout");
        }
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

    let state = StateDir::new(agent._tmp.path().join("state"));
    let committed = model::committed(&state, &worktree, &worktree.join(".git"), &model_id).await;
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
    let socket = config.agent.socket().to_path_buf();
    // The daemon answers the queries below through the server agent.
    let server = ServerGuard::install(&server_config(
        "reinstall",
        tmp.path(),
        &program(),
        tmp.path().join("models"),
        Duration::from_secs(600),
    ));
    let agent = Agent {
        label: config.agent.label().to_owned(),
        socket: socket.clone(),
        _server: Some(server),
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
    let Plist::Daemon(daemon_agent) = &mut config.agent else {
        unreachable!("the test config is a daemon agent")
    };
    daemon_agent.program = copy.clone();
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
        label: config.agent.label().to_owned(),
        socket: config.agent.socket().to_path_buf(),
        _server: None,
        _tmp: tmp,
    };
    let mut launchctl = SystemLaunchctl;
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the first install succeeds");
    assert_eq!(outcome, install::Agent::Installed);

    let copy = agent._tmp.path().join("search-copy");
    std::fs::copy(program(), &copy).expect("the binary copies");
    let Plist::Daemon(daemon_agent) = &mut config.agent else {
        unreachable!("the test config is a daemon agent")
    };
    daemon_agent.program = copy;
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
        label: config.agent.label().to_owned(),
        socket: config.agent.socket().to_path_buf(),
        _server: None,
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
/// A client invocation parsed the way `main` parses argv.
fn invocation(args: &[&str]) -> cli::Invocation {
    Cli::try_parse_from(args)
        .expect("the arguments parse")
        .invocation()
}

/// The `Env` `cli::run` gets against real launchd: the test `AgentConfig`,
/// the real `launchctl`, and the handshake matching the binary the plist
/// names — computed when the `Env` is built, so a replaced program yields
/// a new `BuildId`. The CLI installs the server agent `name` beside the
/// daemon's, built by `server_config` over the same model store.
fn cli_env(
    cwd: &Path,
    name: &str,
    config: AgentConfig,
    program: &Path,
    model_cache: PathBuf,
) -> cli::Env<LaunchdAgents<SystemLaunchctl>> {
    let dir = config.plist.parent().unwrap();
    let server = server_config(name, dir, program, model_cache, Duration::from_secs(600));
    cli::Env {
        cwd: cwd.to_path_buf(),
        launcher: LaunchdAgents::new(server, config, SystemLaunchctl),
        hello: hello(program),
    }
}

/// The `Vec<u8>` terminal `cli::run` writes into in tests; stderr is not
/// a TTY.
fn vec_term() -> Terminal<Vec<u8>, Vec<u8>> {
    Terminal {
        stdout: Vec::new(),
        stderr: Vec::new(),
        stderr_tty: false,
    }
}

/// `cli::run` end to end through real launchd: a run from a repo
/// subdirectory installs the agent, the daemon answers the query and the
/// run exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn install_answers_through_cli_run() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());
    let repo = repo::repo(&[("src/lib.rs".into(), repo::rust_fn("cli_probe", 500))]);
    let config = config(
        "cli",
        tmp.path(),
        program(),
        model_cache.clone(),
        Duration::from_secs(600),
    );
    // The CLI installs the server agent; the guard boots it out.
    let _agent = Agent {
        label: config.agent.label().to_owned(),
        socket: config.agent.socket().to_path_buf(),
        _server: Some(ServerGuard {
            label: server_label("cli"),
        }),
        _tmp: tmp,
    };

    let env = cli_env(
        &repo.root.join("src"),
        "cli",
        config,
        &program(),
        model_cache,
    );
    let mut term = vec_term();
    let exit = cli::run(invocation(&["search", "cli_probe"]), env, &mut term).await;
    assert_eq!(
        exit,
        Exit::Ok,
        "stderr: {}",
        String::from_utf8_lossy(&term.stderr)
    );
    let stdout = String::from_utf8(term.stdout).expect("stdout is utf-8");
    assert!(stdout.contains("cli_probe"), "stdout: {stdout}");
}

/// Install and search so the daemon runs, record its pid, then replace
/// the installed program file by renaming a fresh copy over it — a new
/// inode and mtime, as installers do (never write into the running
/// executable, which macOS may kill). The next `cli::run` carries the new
/// `BuildId`: the mismatched daemon shuts down, launchd restarts the
/// binary on disk, the query answers and `Status` reports a new pid. The
/// retry may wait out launchd's `ThrottleInterval`; that is expected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn replaced_binary_restarts_the_daemon() {
    let tmp = tempfile::tempdir().expect("tmp");
    let model_cache = tmp.path().join("models");
    let _store = model::model_store(model_cache.clone());
    let repo = repo::repo(&[("src/lib.rs".into(), repo::rust_fn("replace_probe", 500))]);

    // The plist's program is a copy in the temp dir, so the test can
    // replace the file launchd execs.
    let installed_program = tmp.path().join("search");
    std::fs::copy(program(), &installed_program).expect("the program copies");
    let dir = tmp.path().to_path_buf();
    let make_config = || {
        config(
            "replace",
            &dir,
            installed_program.clone(),
            model_cache.clone(),
            Duration::from_secs(600),
        )
    };
    let socket = make_config().agent.socket().to_path_buf();
    let agent = Agent {
        label: make_config().agent.label().to_owned(),
        socket: socket.clone(),
        // The CLI installs the server agent; the guard boots it out.
        _server: Some(ServerGuard {
            label: server_label("replace"),
        }),
        _tmp: tmp,
    };

    let env = cli_env(
        &repo.root,
        "replace",
        make_config(),
        &installed_program,
        model_cache.clone(),
    );
    let mut term = vec_term();
    let exit = cli::run(invocation(&["search", "replace_probe"]), env, &mut term).await;
    assert_eq!(
        exit,
        Exit::Ok,
        "stderr: {}",
        String::from_utf8_lossy(&term.stderr)
    );
    // This `Hello` was built for the pre-replacement file identity.
    let old_hello = hello(&installed_program);
    let first = daemon_pid(&socket, &old_hello, &repo.root).await;

    let replacement = agent._tmp.path().join("search-new");
    std::fs::copy(program(), &replacement).expect("the replacement copies");
    std::fs::rename(&replacement, &installed_program).expect("the replacement renames");

    let env = cli_env(
        &repo.root,
        "replace",
        make_config(),
        &installed_program,
        model_cache.clone(),
    );
    let mut term = vec_term();
    let exit = cli::run(invocation(&["search", "replace_probe"]), env, &mut term).await;
    assert_eq!(
        exit,
        Exit::Ok,
        "stderr: {}",
        String::from_utf8_lossy(&term.stderr)
    );
    let second = daemon_pid(&socket, &hello(&installed_program), &repo.root).await;
    assert_ne!(
        first, second,
        "the replaced binary restarted the daemon under launchd"
    );
}

/// The production agent with a `Listener` array of several sockets, so
/// launchd hands the daemon more than the one fd it accepts.
#[derive(askama::Template)]
#[template(path = "../tests/templates/two-listeners.plist", escape = "html")]
struct ListenerAgent {
    label: String,
    program: PathBuf,
    sockets: Vec<PathBuf>,
    stderr: PathBuf,
}

/// A daemon that fails after socket activation answers the client waiting
/// in the socket's queue with `Startup` instead of leaving it hanging. The
/// failure is the real `ListenerCount` check: the plist declares two
/// listeners. `cli::run` is not used because its install path would replace
/// the plist with the production one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn startup_failure_reaches_the_client() {
    use repo_search_client::client::{Client, ClientError};
    use repo_search_client::launcher::kqueue::DirWatch;
    use repo_search_client::protocol::{ErrorKind, ErrorReport};

    let tmp = tempfile::tempdir().expect("tmp");
    let config = config(
        "startup",
        tmp.path(),
        program(),
        tmp.path().join("models"),
        Duration::from_secs(600),
    );
    let socket = config.agent.socket().to_path_buf();
    let agent_plist = askama::Template::render(&ListenerAgent {
        label: config.agent.label().to_owned(),
        program: program(),
        sockets: vec![socket.clone(), tmp.path().join("second.sock")],
        stderr: config.agent.stderr().to_path_buf(),
    })
    .expect("the template renders");
    std::fs::write(&config.plist, agent_plist).expect("the plist writes");

    let watch = DirWatch::register(tmp.path()).expect("the dir registers");
    let agent = Agent {
        label: config.agent.label().to_owned(),
        socket: socket.clone(),
        _server: None,
        _tmp: tmp,
    };
    SystemLaunchctl
        .bootstrap(&config.domain, &config.plist)
        .expect("the agent bootstraps");
    assert!(
        watch.wait_for(&socket, SOCKET_TIMEOUT),
        "launchd binds the socket"
    );

    let result = tokio::time::timeout(
        Duration::from_secs(30),
        Client::connect(&agent.socket, &hello(&program())),
    )
    .await
    .expect("the client is answered instead of hanging");
    match result {
        Err(ClientError::Refused(ErrorReport {
            kind: ErrorKind::Startup,
            message,
        })) => assert!(message.contains('2'), "message: {message}"),
        Err(other) => panic!("expected a Startup refusal, got {other:?}"),
        Ok(_) => panic!("expected a Startup refusal, the handshake succeeded"),
    }
}

/// How long past an expected event the server test waits for it.
const MARGIN: Duration = Duration::from_secs(30);

/// The server agent's lifecycle: `ensure_agent` installs it, a unix
/// connect makes launchd run `search --server`, the 3 s idle exit takes
/// it down, the next connect relaunches a new pid, and a `--build`
/// change replaces the agent.
///
/// The server counts as busy while it downloads or loads the model, and
/// the load starts at once, so the idle exit is measured only after
/// `/status` reports the model ready — on a loaded machine the load
/// outlasts any idle deadline. The model store is populated so nothing
/// downloads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real launchd and model; the release gate runs it"]
async fn server_agent_lifecycle() {
    use repo_search_core::embed_api::{LOAD_BOUND, ModelStatus};

    let tmp = tempfile::tempdir().expect("tmp");
    let dir = tmp.path().to_path_buf();
    let _store = model::model_store(dir.join("models"));
    let idle_exit = Duration::from_secs(3);
    let make_config =
        || server_config("lifecycle", &dir, &program(), dir.join("models"), idle_exit);
    let config = make_config();
    let label = config.agent.label().to_owned();
    let socket = config.agent.socket().to_path_buf();
    let _agent = ServerGuard {
        label: label.clone(),
    };
    let _tmp = tmp;
    let mut launchctl = SystemLaunchctl;
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&config, &mut launchctl, Check::Fast))
            .expect("the server agent installs");
    assert_eq!(outcome, install::Agent::Installed);

    // A unix connect makes launchd start `search --server`; the status
    // poll is that connect, repeated until the model is ready: the load's
    // own bound plus a margin.
    let service = target(&label);
    let server_pid = |launchctl: &mut SystemLaunchctl| {
        launchctl
            .service(&service)
            .ok()
            .flatten()
            .and_then(|s| s.pid)
    };
    // Each poll is a request, so it resets the server's idle deadline.
    let status =
        ServerClient::new(&Endpoint::Unix(socket.clone())).expect("the status client builds");
    let deadline = std::time::Instant::now() + LOAD_BOUND + MARGIN;
    loop {
        match status
            .status()
            .await
            .expect("the server answers /status")
            .model
        {
            ModelStatus::Ready { .. } => break,
            ModelStatus::Failed { message } | ModelStatus::Fatal { message } => {
                panic!("the server's model failed to load: {message}")
            }
            loading => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the model was not ready within the load bound: {loading:?}"
                );
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    let first = server_pid(&mut launchctl).expect("the ready server runs");

    // The idle exit takes it down, measured from the last poll; the next
    // connect relaunches it.
    let exited = tokio::task::block_in_place(|| {
        ExitWatch::register(first)
            .expect("the server's exit registers")
            .wait(idle_exit + MARGIN)
    });
    assert!(
        exited,
        "the ready server did not exit within {MARGIN:?} past its idle exit"
    );
    std::os::unix::net::UnixStream::connect(&socket).expect("the reconnect lands");
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let second = loop {
        if let Some(pid) = server_pid(&mut launchctl)
            && pid != first
        {
            break pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "launchd did not relaunch the server"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_ne!(first, second, "launchd did not start a replacement server");

    // A `--build` change is a plist change: the agent is replaced.
    let mut rebuilt = make_config();
    let Plist::Server(server_agent) = &mut rebuilt.agent else {
        unreachable!()
    };
    server_agent.build = "other-build".to_owned();
    let outcome =
        tokio::task::block_in_place(|| ensure_agent(&rebuilt, &mut launchctl, Check::Fast))
            .expect("the rebuild replaces the agent");
    assert_eq!(
        outcome,
        install::Agent::Replaced(install::Reason::PlistChanged)
    );
}
