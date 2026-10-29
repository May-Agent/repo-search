//! Spawn-on-demand end to end: `cli::run` with an `OnDemand` launcher
//! whose command re-executes this test binary as `spawn_host`, which runs
//! the same `Bound::acquire`, ready-fd and `daemon::run` code as the
//! Linux `main` against the fake standard-API server. On Linux these are
//! the production pieces; here they are exercised on macOS too.
//!
//! The hosts are detached daemons, so a test learns about them only
//! through what the production protocol exposes: `daemon.lock` (alive,
//! and the pid it names), the socket, `daemon.cmd` and the log. Tests
//! kill only a pid read from a held `daemon.lock`.

#[allow(dead_code)] // each test file uses its own slice of the fixtures
mod support;

use std::fs::File;
use std::io::Write as _;
use std::os::fd::FromRawFd as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser as _;
use futures_util::future::join_all;
use repo_search::cli::{Cli, Env, Exit, Invocation, Launcher as _, Terminal};
use repo_search::client::ClientError;
use repo_search::daemon::listen::{Bound, LifetimeLock, Listener};
use repo_search::daemon::server::Endpoint;
use repo_search::daemon::{self, Config, StartupError};
use repo_search::protocol::{ErrorKind, ErrorReport};
use repo_search::spawn::{DaemonCommand, DaemonConfig, OnDemand, ServerTarget};
use repo_search_core::StateDir;
use repo_search_core::model::MODEL_ID;
use serde::{Deserialize, Serialize};
use support::client::hello;
use support::embed_server::{FakeConfig, FakeServer};
use support::lock::{self, signal};
use support::repo::{self, TestRepo};
use support::tokens::StubTokens;
use tokio::net::UnixListener;

/// A bound on every wait on a spawned process; the tests assert
/// structure, not latency.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The marker an edit lands in the index.
const MARKER: &str = "xylophonemarker";

/// What the re-executed `spawn_host` needs beyond the daemon flags the
/// launcher renders.
#[derive(Serialize, Deserialize)]
struct HostConfig {
    /// The daemon's state root.
    state: PathBuf,
    /// The idle deadline.
    idle_exit: Duration,
    /// When this file exists the host removes it and runs under a
    /// `BuildId` no client matches: the first host of an "upgrade".
    skew_marker: Option<PathBuf>,
    /// A socket the test listens on: once the host has closed its
    /// listener it connects here and lingers, holding the lifetime lock,
    /// until the test hangs up — a shutdown tail of the test's length.
    hold_socket: Option<PathBuf>,
}

/// The re-executed test acting as the daemon: it parses the flags the
/// launcher rendered after `--` with the real `Cli`, then does what
/// `bound_main` does with a `StubTokens`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "re-executed by the spawn tests through OnDemand"]
async fn spawn_host() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();
    let config: HostConfig = serde_json::from_str(
        &std::env::var("REPO_SEARCH_TEST_HOST").expect("REPO_SEARCH_TEST_HOST"),
    )
    .expect("the host config parses");
    let flags = std::env::args().skip_while(|arg| arg != "--").skip(1);
    let Invocation::Daemon {
        listen,
        ready_fd,
        server_socket,
        server_addr,
        server_token_file,
        ..
    } = Cli::try_parse_from(std::iter::once("search".to_owned()).chain(flags))
        .expect("the launcher's flags parse")
        .invocation()
    else {
        panic!("the launcher renders --daemon");
    };
    eprintln!("spawn_host started");

    let mut build = support::client::build();
    if let Some(marker) = &config.skew_marker
        && std::fs::remove_file(marker).is_ok()
    {
        build.mtime_ns += 1;
    }
    let bound = Bound::acquire(listen.expect("--listen"))
        .await
        .expect("the host binds");
    let ready = ready_fd.expect("--ready-fd");
    // The production `endpoint`: the token file is read and trimmed.
    let endpoint = match (server_socket, server_addr) {
        (Some(path), None) => Ok(Endpoint::Unix(path)),
        (None, Some(addr)) => match server_token_file {
            Some(path) => std::fs::read_to_string(&path)
                .map(|token| Endpoint::Tcp {
                    addr,
                    token: Some(token.trim_end_matches(['\n', '\r']).to_owned()),
                })
                .map_err(|source| StartupError::TokenFile { path, source }),
            None => Ok(Endpoint::Tcp { addr, token: None }),
        },
        _ => panic!("the launcher renders one server flag"),
    };
    let endpoint = match endpoint {
        Ok(endpoint) => endpoint,
        Err(error) => {
            let message = error.to_string();
            let _held = bound.refuse(&error).await;
            report(ready, &format!("failed: {message}\n"));
            std::process::exit(1);
        }
    };
    report(ready, "ready\n");
    let outcome = daemon::run(Config {
        listener: Lingering {
            bound,
            hold: config.hold_socket,
        },
        idle_exit: config.idle_exit,
        load_bound: daemon::LOAD_BOUND,
        tokens: StubTokens::default(),
        server: endpoint,
        model_id: MODEL_ID.to_owned(),
        state: StateDir::new(config.state),
        build,
    })
    .await;
    std::process::exit(i32::from(outcome.is_err()));
}

/// A [`Bound`] whose `close` lingers after the socket is gone: the
/// shutdown tail, stretched to however long the test holds `hold`.
struct Lingering {
    bound: Bound,
    hold: Option<PathBuf>,
}

impl Listener for Lingering {
    type Held = LifetimeLock;

    async fn accept(&self) -> std::io::Result<tokio::net::UnixStream> {
        self.bound.accept().await
    }

    async fn close(self) -> LifetimeLock {
        // The socket is unlinked and its queue drained; the lock stays.
        let held = self.bound.close().await;
        if let Some(path) = self.hold
            && let Ok(mut stream) = tokio::net::UnixStream::connect(path).await
        {
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut [0u8; 1]).await;
        }
        held
    }

    async fn refuse(self, error: &StartupError) -> std::io::Result<LifetimeLock> {
        self.bound.refuse(error).await
    }
}

/// One line on the ready fd, which the host then closes.
fn report(fd: i32, line: &str) {
    // SAFETY: the launcher hands the host ownership of this pipe end.
    let mut pipe = unsafe { File::from_raw_fd(fd) };
    pipe.write_all(line.as_bytes()).expect("the ready line");
}

/// One test's world: a repository, the fake server, the runtime dir the
/// launcher spawns into and the hosts' configuration.
struct World {
    repo: TestRepo,
    tmp: tempfile::TempDir,
    fake: FakeServer,
    host: HostConfig,
    /// Kills the daemon a failing test leaves running.
    _reaper: lock::Reaper,
}

impl World {
    fn new() -> Self {
        let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
        let tmp = tempfile::tempdir().expect("tmp");
        let fake = FakeServer::unix(
            &tmp.path().join("server.sock"),
            FakeConfig {
                model_id: MODEL_ID.to_owned(),
                token: None,
            },
        );
        std::fs::create_dir(tmp.path().join("run")).expect("run dir");
        let host = HostConfig {
            state: tmp.path().join("state"),
            idle_exit: Duration::from_secs(600),
            skew_marker: None,
            hold_socket: None,
        };
        Self {
            _reaper: lock::Reaper(tmp.path().join("run")),
            repo,
            tmp,
            fake,
            host,
        }
    }

    fn run_dir(&self) -> PathBuf {
        self.tmp.path().join("run")
    }

    fn socket(&self) -> PathBuf {
        self.run_dir().join("daemon.sock")
    }

    fn log(&self) -> PathBuf {
        self.tmp.path().join("daemon.log")
    }

    /// The default target: the fake's socket.
    fn target(&self) -> ServerTarget {
        ServerTarget::Socket(self.fake.path().to_path_buf())
    }

    /// An `OnDemand` whose command re-executes this binary as
    /// `spawn_host`, configured by `self.host`.
    fn launcher(&self, target: &ServerTarget) -> OnDemand {
        let exe = std::env::current_exe().expect("the test binary");
        let mut command = DaemonCommand::new(exe, &self.socket(), target);
        command.args.splice(
            0..0,
            [
                "--exact",
                "spawn_host",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                "--",
            ]
            .map(Into::into),
        );
        command.env.push((
            "REPO_SEARCH_TEST_HOST".into(),
            serde_json::to_string(&self.host)
                .expect("the host config serializes")
                .into(),
        ));
        OnDemand::new(command, self.socket(), self.log())
    }

    /// `cli::run` of `args` (after `search`) in the repository through a
    /// fresh launcher for `target`: the exit code, stdout and stderr.
    async fn client(&self, target: &ServerTarget, args: &[&str]) -> (Exit, String, String) {
        let invocation = Cli::try_parse_from(std::iter::once("search").chain(args.iter().copied()))
            .expect("the arguments parse")
            .invocation();
        let env = Env {
            cwd: self.repo.root.clone(),
            launcher: self.launcher(target),
            hello: hello(),
        };
        let mut term = Terminal {
            stdout: Vec::new(),
            stderr: Vec::new(),
            stderr_tty: false,
        };
        let exit = repo_search::cli::run(invocation, env, &mut term).await;
        (
            exit,
            String::from_utf8(term.stdout).expect("stdout is UTF-8"),
            String::from_utf8(term.stderr).expect("stderr is UTF-8"),
        )
    }

    /// The pid of the daemon answering `search status`.
    async fn status_pid(&self, target: &ServerTarget) -> u32 {
        let (exit, stdout, stderr) = self.client(target, &["status", "--json"]).await;
        assert_eq!(exit, Exit::Ok, "status answers: {stderr}");
        let status: serde_json::Value = serde_json::from_str(&stdout).expect("status is JSON");
        status["daemon"]["pid"]
            .as_u64()
            .expect("status names the daemon's pid") as u32
    }

    /// The pid `daemon.lock` names, when a daemon holds the lock.
    fn alive(&self) -> Option<u32> {
        lock::alive(&self.run_dir())
    }

    /// Waits for the daemon to exit by waiting for its lifetime lock:
    /// the kernel releases it with the process.
    async fn exited(&self) {
        let path = self.run_dir().join("daemon.lock");
        let wait = tokio::task::spawn_blocking(move || {
            File::open(path)
                .expect("the lock file exists")
                .lock_shared()
                .expect("the lock wait")
        });
        tokio::time::timeout(TIMEOUT, wait)
            .await
            .expect("the daemon exits")
            .expect("the wait joins");
    }

    /// How many hosts the log records starting.
    fn starts(&self) -> usize {
        std::fs::read_to_string(self.log())
            .unwrap_or_default()
            .matches("spawn_host started")
            .count()
    }
}

/// Polls `condition` until it holds. The conditions here are effects of
/// another process on the filesystem, which has no notification both
/// platforms share.
async fn eventually(what: &str, condition: impl Fn() -> bool) {
    tokio::time::timeout(TIMEOUT, async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

/// The first search spawns the daemon, which answers with the edited
/// marker; the daemon outlives the client and does not hold `spawn.lock`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_search_spawns_and_answers() {
    let world = World::new();
    repo::append(
        &world.repo.root,
        "src/edit.rs",
        &repo::rust_fn(&format!("{MARKER}_fn"), 500),
    );
    assert_eq!(world.alive(), None, "no daemon before the first search");

    let (exit, stdout, stderr) = world.client(&world.target(), &[MARKER]).await;
    assert_eq!(exit, Exit::Ok, "the search answers: {stderr}");
    assert!(stdout.contains(MARKER), "the marker is found: {stdout}");

    let pid = world.alive().expect("the daemon outlives the client");
    assert_eq!(world.starts(), 1);
    assert!(world.run_dir().join("daemon.cmd").exists());
    // The spawned host did not inherit `spawn.lock`: a client taking it
    // would deadlock otherwise.
    let spawn_lock = File::open(world.run_dir().join("spawn.lock")).expect("spawn.lock exists");
    spawn_lock
        .try_lock()
        .expect("the live daemon does not hold spawn.lock");
    drop(spawn_lock);
    assert_eq!(world.status_pid(&world.target()).await, pid);
}

/// N clients racing a missing daemon start exactly one host and are all
/// answered by it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_clients_spawn_one_host() {
    let world = World::new();
    let target = world.target();
    let pids = join_all((0..6).map(|_| world.status_pid(&target))).await;
    assert_eq!(world.starts(), 1, "one host started");
    let pid = world.alive().expect("the host is alive");
    assert!(pids.iter().all(|answered| *answered == pid), "{pids:?}");
}

/// A warm search spawns nothing and takes no lock: it answers while
/// another client holds `spawn.lock`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_warm_search_takes_no_lock() {
    let world = World::new();
    let target = world.target();
    let pid = world.status_pid(&target).await;
    let held = File::options()
        .write(true)
        .open(world.run_dir().join("spawn.lock"))
        .expect("spawn.lock exists");
    held.lock().expect("the test takes spawn.lock");
    let answered = tokio::time::timeout(TIMEOUT, world.status_pid(&target))
        .await
        .expect("a warm search does not wait for spawn.lock");
    assert_eq!(answered, pid);
    assert_eq!(world.starts(), 1);
}

/// The idle deadline fires: the daemon unlinks its socket and exits, and
/// the next search respawns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_exit_then_the_next_search_respawns() {
    let mut world = World::new();
    world.host.idle_exit = Duration::from_millis(300);
    let target = world.target();
    let first = world.status_pid(&target).await;
    world.exited().await;
    assert!(
        !world.socket().exists(),
        "the idle exit unlinked the socket"
    );

    let second = world.status_pid(&target).await;
    assert_ne!(first, second, "a new daemon answered");
    assert_eq!(world.starts(), 2);
}

/// `kill -9` leaves the socket file behind: the connect is refused, the
/// lock is free, and the next search respawns over the stale socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill_9_leaves_a_stale_socket_and_the_next_search_respawns() {
    let world = World::new();
    let target = world.target();
    let first = world.status_pid(&target).await;
    signal(first, libc::SIGKILL);
    world.exited().await;
    assert!(world.socket().exists(), "SIGKILL left the socket behind");

    let second = world.status_pid(&target).await;
    assert_ne!(first, second);
    assert_eq!(world.starts(), 2);
}

/// A client arriving while the old daemon is in its shutdown tail finds
/// no socket, spawns the new daemon, which waits on `daemon.lock`; the
/// client waits for it and is answered by the new daemon once the old
/// one is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_during_the_old_daemons_tail_waits_for_the_new_daemon() {
    let mut world = World::new();
    let hold = world.tmp.path().join("hold.sock");
    world.host.hold_socket = Some(hold.clone());
    // The first connection is the old daemon entering its tail; any later
    // one — the new daemon's — is dropped at once.
    let listener = UnixListener::bind(&hold).expect("the hold socket binds");
    let (entered_tx, entered) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let mut entered_tx = Some(entered_tx);
        while let Ok((stream, _)) = listener.accept().await {
            match entered_tx.take() {
                Some(tx) => {
                    let _ = tx.send(stream);
                }
                None => drop(stream),
            }
        }
    });
    let target = world.target();

    // Bring the old daemon up, then start its tail: SIGTERM, the socket
    // goes, and the daemon lingers on the hold.
    let report = world.launcher(&target).recover().await;
    assert_eq!(report.error, None);
    let old = world.alive().expect("the old daemon is up");
    signal(old, libc::SIGTERM);
    let held = tokio::time::timeout(TIMEOUT, entered)
        .await
        .expect("the old daemon reaches its tail")
        .expect("the hold socket accepts");
    assert!(!world.socket().exists(), "the tail has no socket");
    assert_eq!(world.alive(), Some(old), "the old daemon is in its tail");

    // The client finds no socket, and spawns a daemon that waits.
    let client = world.status_pid(&target);
    tokio::pin!(client);
    let waiting = eventually("the new daemon to wait on the lock", || {
        std::fs::read_to_string(world.log())
            .unwrap_or_default()
            .contains("waiting for the previous daemon to exit")
    });
    tokio::select! {
        answered = &mut client => panic!("answered {answered} while the old tail was held"),
        () = waiting => {}
    }
    assert_eq!(
        world.alive(),
        Some(old),
        "the old daemon still holds the lock"
    );

    // The old daemon finishes; the new one binds and answers.
    drop(held);
    let answered = client.await;
    assert_ne!(answered, old);
    assert_eq!(world.alive(), Some(answered));
    assert_eq!(world.starts(), 2);
}

/// A daemon built from another binary closes on the `Hello`: the client
/// recovers, spawning the build that is on disk now.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_build_mismatch_respawns_the_matching_build() {
    let mut world = World::new();
    let marker = world.tmp.path().join("skew");
    std::fs::write(&marker, "").expect("the skew marker");
    world.host.skew_marker = Some(marker);
    let target = world.target();

    let report = world.launcher(&target).recover().await;
    assert_eq!(report.error, None);
    let old = world.alive().expect("the skewed daemon is up");

    let answered = world.status_pid(&target).await;
    assert_ne!(answered, old, "the matching build answered");
    assert_eq!(world.starts(), 2);
}

/// A `daemon.toml` change renders another command: the next search
/// replaces the live daemon, which then talks to the new server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_daemon_toml_change_replaces_the_daemon() {
    let world = World::new();
    let other = FakeServer::unix(
        &world.tmp.path().join("other.sock"),
        FakeConfig {
            model_id: MODEL_ID.to_owned(),
            token: None,
        },
    );
    let toml = |path: &Path| {
        let text = format!("server_socket = {:?}\n", path.display().to_string());
        DaemonConfig::parse(Path::new("daemon.toml"), &text)
            .expect("daemon.toml parses")
            .server_or(PathBuf::new())
    };
    let first_target = toml(world.fake.path());
    let second_target = toml(other.path());

    let first = world.status_pid(&first_target).await;
    // The unchanged config leaves the daemon alone.
    assert_eq!(world.status_pid(&first_target).await, first);
    assert_eq!(world.starts(), 1);

    let second = world.status_pid(&second_target).await;
    assert_ne!(first, second, "the daemon was replaced");
    assert_eq!(world.starts(), 2);
    assert_eq!(world.alive(), Some(second));
    let recorded =
        std::fs::read_to_string(world.run_dir().join("daemon.cmd")).expect("daemon.cmd reads");
    assert!(
        recorded.contains(&other.path().display().to_string()),
        "daemon.cmd records the new server: {recorded}"
    );
}

/// A token file the daemon cannot read is the daemon's own startup error:
/// the client prints its `failed:` message and the log's path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreadable_token_file_shows_the_daemons_failure() {
    let world = World::new();
    let missing = world.tmp.path().join("missing.token");
    let target = ServerTarget::Tcp {
        addr: "127.0.0.1:1".parse().expect("an address"),
        token_file: Some(missing.clone()),
    };
    let (exit, _, stderr) = world.client(&target, &["status"]).await;
    assert_eq!(exit, Exit::Error);
    assert!(
        stderr.contains("the daemon failed to start")
            && stderr.contains(&missing.display().to_string())
            && stderr.contains(&world.log().display().to_string()),
        "{stderr}"
    );
    assert_eq!(world.alive(), None, "the failed daemon exited");
}

/// Which connect failures a fresh daemon can cure.
#[test]
fn recoverable_connect_failures() {
    let launcher = OnDemand::new(
        DaemonCommand::new(
            PathBuf::from("search"),
            Path::new("/run/daemon.sock"),
            &ServerTarget::Socket(PathBuf::from("/run/server.sock")),
        ),
        PathBuf::from("/run/daemon.sock"),
        PathBuf::from("/log"),
    );
    let refused = |kind| {
        ClientError::Refused(ErrorReport {
            kind,
            message: String::new(),
        })
    };
    let connect = |kind| ClientError::Connect(std::io::Error::from(kind));
    for cure in [
        connect(std::io::ErrorKind::NotFound),
        connect(std::io::ErrorKind::ConnectionRefused),
        ClientError::Io(std::io::Error::from(std::io::ErrorKind::ConnectionReset)),
        ClientError::Closed,
        refused(ErrorKind::ShuttingDown),
        refused(ErrorKind::ProtocolMismatch),
    ] {
        assert!(launcher.recoverable(&cure), "{cure:?} is recoverable");
    }
    for fatal in [
        ClientError::TooLarge,
        refused(ErrorKind::Startup),
        refused(ErrorKind::Model),
    ] {
        assert!(!launcher.recoverable(&fatal), "{fatal:?} is final");
    }
}
