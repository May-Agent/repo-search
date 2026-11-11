//! A host that depends only on `repo-search-client` with the `launcher`
//! feature starts a daemon from a program path it names, connects to it
//! and replaces a stale one. The daemon is a copy of this test binary
//! re-executed as `fake_daemon`, which binds the socket, holds the
//! lifetime lock and answers the `Hello` handshake the way the real
//! daemon does; the handshake's build id is the copy's own file.
//!
//! The spawn-on-demand launcher runs on macOS and Linux alike; the launchd
//! agents are tested in the `repo-search` crate against a fake `launchctl`.
#![cfg(feature = "launcher")]

use std::fs::File;
use std::io::Write as _;
use std::os::fd::FromRawFd as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::{SinkExt as _, StreamExt as _};
use repo_search_client::launcher::spawn::{OnDemand, ServerTarget, SpawnCommand};
use repo_search_client::launcher::{ConnectError, Launcher as _, connect};
use repo_search_client::protocol::{self, BuildId, ErrorKind, ErrorReport, Event, Final, Hello};
use tokio::net::UnixListener;

/// A bound on every wait on a spawned process; the tests assert
/// structure, not latency.
const TIMEOUT: Duration = Duration::from_secs(60);

/// The re-executed copy of this binary acting as the daemon. It parses the
/// flags the launcher rendered after `--`, takes `daemon.lock`, binds the
/// socket, reports `ready` on the ready fd and answers handshakes until it
/// is signalled or `TIMEOUT` passes.
#[tokio::test(flavor = "current_thread")]
#[ignore = "re-executed by the launcher tests through OnDemand"]
async fn fake_daemon() {
    let flags: Vec<String> = std::env::args()
        .skip_while(|arg| arg != "--")
        .skip(1)
        .collect();
    let value = |flag: &str| {
        let at = flags.iter().position(|arg| arg == flag).expect("the flag");
        flags[at + 1].clone()
    };
    let listen = PathBuf::from(value("--listen"));
    let ready: i32 = value("--ready-fd").parse().expect("a fd");
    // A launcher that starts the server has it ready before the daemon
    // starts: its socket is bound and accepts.
    if std::env::var_os(NEEDS_SERVER).is_some() {
        let server = PathBuf::from(value("--server-socket"));
        std::os::unix::net::UnixStream::connect(&server)
            .expect("the server is ready before the daemon starts");
    }

    let lock = File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(listen.with_extension("lock"))
        .expect("daemon.lock opens");
    // A replaced daemon's lock is released when its process exits.
    lock.lock().expect("the lifetime lock");
    lock.set_len(0).expect("the stale pid is cleared");
    (&lock)
        .write_all(std::process::id().to_string().as_bytes())
        .expect("the pid is written");
    let _ = std::fs::remove_file(&listen);
    let listener = UnixListener::bind(&listen).expect("the socket binds");
    let own = Hello::new(BuildId::of_current_exe().expect("the copy stats"));

    // SAFETY: the launcher hands the daemon ownership of this pipe end.
    let mut pipe = unsafe { File::from_raw_fd(ready) };
    pipe.write_all(b"ready\n").expect("the ready line");
    drop(pipe);

    let serve = async {
        loop {
            let (stream, _) = listener.accept().await.expect("accept");
            let mut framed = protocol::framed(stream);
            let Some(Ok(line)) = framed.next().await else {
                continue;
            };
            let theirs: Hello = serde_json::from_str(&line).expect("a Hello");
            let reply = if theirs == own {
                serde_json::to_string(&own)
            } else {
                serde_json::to_string(&Event::Final(Final::Error(ErrorReport {
                    kind: ErrorKind::ProtocolMismatch,
                    message: "another build".to_owned(),
                })))
            }
            .expect("the reply serializes");
            let _ = framed.send(reply).await;
        }
    };
    let _ = tokio::time::timeout(TIMEOUT, serve).await;
}

/// Set on a daemon that must find the server ready when it starts.
const NEEDS_SERVER: &str = "FAKE_DAEMON_NEEDS_SERVER";

/// The re-executed copy of this binary acting as the embedding server. It
/// parses the flags the launcher rendered after `--`, takes `server.lock`,
/// binds the socket, reports `ready` on the ready fd and accepts
/// connections until it is signalled or `TIMEOUT` passes. A `--build`
/// other than the copy's own is a failure on the ready fd, as in the
/// real server.
#[tokio::test(flavor = "current_thread")]
#[ignore = "re-executed by the launcher tests through OnDemand"]
async fn fake_server() {
    let flags: Vec<String> = std::env::args()
        .skip_while(|arg| arg != "--")
        .skip(1)
        .collect();
    let value = |flag: &str| {
        let at = flags.iter().position(|arg| arg == flag).expect("the flag");
        flags[at + 1].clone()
    };
    assert_eq!(flags[0], "--server");
    let socket = PathBuf::from(value("--socket"));
    let ready: i32 = value("--ready-fd").parse().expect("a fd");
    let build = BuildId::of_current_exe().expect("the copy stats");

    let lock = File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(socket.with_extension("lock"))
        .expect("server.lock opens");
    lock.lock().expect("the lifetime lock");
    lock.set_len(0).expect("the stale pid is cleared");
    (&lock)
        .write_all(std::process::id().to_string().as_bytes())
        .expect("the pid is written");
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).expect("the socket binds");

    // SAFETY: the launcher hands the server ownership of this pipe end.
    let mut pipe = unsafe { File::from_raw_fd(ready) };
    if value("--build") == build.to_string() {
        pipe.write_all(b"ready\n").expect("the ready line");
    } else {
        pipe.write_all(b"failed: another build\n")
            .expect("the failed line");
        return;
    }
    drop(pipe);

    let serve = async {
        loop {
            let _ = listener.accept().await.expect("accept");
        }
    };
    let _ = tokio::time::timeout(TIMEOUT, serve).await;
}

/// One test's world: a runtime dir and copies of this binary to launch.
struct World {
    tmp: tempfile::TempDir,
}

impl World {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("tmp");
        std::fs::create_dir(tmp.path().join("run")).expect("run dir");
        Self { tmp }
    }

    fn run_dir(&self) -> PathBuf {
        self.tmp.path().join("run")
    }

    fn socket(&self) -> PathBuf {
        self.run_dir().join("daemon.sock")
    }

    /// A daemon program: its own file, so its own build id.
    fn program(&self, name: &str) -> PathBuf {
        let program = self.tmp.path().join(name);
        std::fs::copy(std::env::current_exe().expect("the test binary"), &program)
            .expect("the program copies");
        program
    }

    /// The launcher for the daemon at `program`.
    fn launcher(&self, program: &Path) -> OnDemand {
        let target = ServerTarget::Socket(self.tmp.path().join("server.sock"));
        let command = SpawnCommand::daemon(program.to_path_buf(), &self.socket(), &target);
        OnDemand::new(
            reexec(command, "fake_daemon"),
            self.socket(),
            self.tmp.path().join("daemon.log"),
        )
    }

    /// The launcher for the daemon and the default server at `program`:
    /// the daemon targets the `server.sock` the launcher starts.
    fn launcher_with_server(&self, program: &Path) -> OnDemand {
        let socket = self.run_dir().join("server.sock");
        let target = ServerTarget::Socket(socket.clone());
        let mut daemon = reexec(
            SpawnCommand::daemon(program.to_path_buf(), &self.socket(), &target),
            "fake_daemon",
        );
        daemon.env.push((NEEDS_SERVER.into(), "1".into()));
        let build = BuildId::of_file(program).expect("the program stats");
        let server = reexec(
            SpawnCommand::server(program.to_path_buf(), &socket, &build),
            "fake_server",
        );
        OnDemand::new(daemon, self.socket(), self.tmp.path().join("daemon.log")).with_server(
            server,
            socket,
            self.tmp.path().join("server.log"),
        )
    }

    /// The pid `daemon.lock` names while a daemon holds the lock.
    fn alive(&self) -> Option<u32> {
        self.alive_unit("daemon")
    }

    /// The pid `server.lock` names while a server holds the lock.
    fn server_alive(&self) -> Option<u32> {
        self.alive_unit("server")
    }

    fn alive_unit(&self, name: &str) -> Option<u32> {
        let path = self.run_dir().join(format!("{name}.lock"));
        let lock = File::open(&path).ok()?;
        match lock.try_lock_shared() {
            Err(std::fs::TryLockError::WouldBlock) => {
                std::fs::read_to_string(&path).ok()?.trim().parse().ok()
            }
            _ => None,
        }
    }
}

impl Drop for World {
    /// Kills the daemon and server a test leaves running: each pid is
    /// read from a held lifetime lock, a process the test spawned.
    fn drop(&mut self) {
        for pid in [self.alive(), self.server_alive()].into_iter().flatten() {
            // SAFETY: `kill` takes plain integers.
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
    }
}

/// `command` run as the re-executed test `test` of this binary: the test
/// harness's arguments, then the launcher's flags after `--`.
fn reexec(mut command: SpawnCommand, test: &str) -> SpawnCommand {
    command.args.splice(
        0..0,
        [
            "--exact",
            test,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
            "--",
        ]
        .map(Into::into),
    );
    command
}

/// The handshake of the daemon program at `program`: its build id comes
/// from that file.
fn hello_of(program: &Path) -> Hello {
    Hello::new(BuildId::of_file(program).expect("the program stats"))
}

/// `prepare`, then `connect`, bounded; the notes told and the outcome.
async fn run(launcher: &mut OnDemand, hello: &Hello) -> (Vec<String>, Result<(), ConnectError>) {
    let prepared = launcher.prepare().await;
    let mut notes = Vec::new();
    let connected = tokio::time::timeout(
        TIMEOUT,
        connect(launcher, hello, prepared, async |note: &str| {
            notes.push(note.to_owned());
        }),
    )
    .await
    .expect("the connect finishes");
    (notes, connected.map(drop))
}

#[tokio::test(flavor = "current_thread")]
async fn starts_the_daemon_at_a_given_program_path_and_connects() {
    let world = World::new();
    let program = world.program("search-a");
    assert_eq!(world.alive(), None);

    let (notes, connected) = run(&mut world.launcher(&program), &hello_of(&program)).await;
    connected.expect("the first connect recovers by spawning the daemon");
    assert!(notes.is_empty(), "{notes:?}");
    let pid = world.alive().expect("the spawned daemon holds daemon.lock");

    // A second client finds the live daemon and spawns nothing.
    let (_, connected) = run(&mut world.launcher(&program), &hello_of(&program)).await;
    connected.expect("the warm connect");
    assert_eq!(world.alive(), Some(pid));
    let recorded =
        std::fs::read_to_string(world.run_dir().join("daemon.cmd")).expect("daemon.cmd reads");
    assert!(
        recorded.starts_with(&program.display().to_string()),
        "{recorded}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_daemon_started_from_another_program_is_replaced() {
    let world = World::new();
    let old = world.program("search-old");
    let new = world.program("search-new");

    let (_, connected) = run(&mut world.launcher(&old), &hello_of(&old)).await;
    connected.expect("the old program's daemon starts");
    let first = world.alive().expect("the old daemon is alive");

    // The live daemon was started from another program: `prepare`
    // replaces it, and the handshake is the new file's.
    let (_, connected) = run(&mut world.launcher(&new), &hello_of(&new)).await;
    connected.expect("the new program's daemon answers");
    let second = world.alive().expect("the new daemon is alive");
    assert_ne!(first, second, "the daemon was replaced");
}

#[tokio::test(flavor = "current_thread")]
async fn a_failed_spawn_is_the_launchers_error() {
    let world = World::new();
    let missing = world.tmp.path().join("no-such-program");
    let hello = hello_of(&world.program("search-a"));

    let (_, connected) = run(&mut world.launcher(&missing), &hello).await;
    assert!(
        matches!(&connected, Err(ConnectError::Launcher(message)) if message.contains("start the daemon")),
        "{connected:?}"
    );
}

/// Waits, bounded, until `alive` reports the process gone.
async fn gone(alive: impl Fn() -> Option<u32>) {
    tokio::time::timeout(TIMEOUT, async {
        while alive().is_some() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the process exits");
}

#[tokio::test(flavor = "current_thread")]
async fn starts_the_server_ready_before_the_daemon() {
    let world = World::new();
    let program = world.program("search-a");
    assert_eq!(world.server_alive(), None);

    // `fake_daemon` connects to the server's socket before it reports
    // ready, so the daemon starting at all shows the server was ready.
    let (notes, connected) = run(
        &mut world.launcher_with_server(&program),
        &hello_of(&program),
    )
    .await;
    connected.expect("the first connect starts the server, then the daemon");
    assert!(notes.is_empty(), "{notes:?}");
    let server = world.server_alive().expect("the server holds server.lock");
    let daemon = world.alive().expect("the daemon holds daemon.lock");
    let recorded =
        std::fs::read_to_string(world.run_dir().join("server.cmd")).expect("server.cmd reads");
    assert!(
        recorded.starts_with(&program.display().to_string()) && recorded.contains("\n--server\n"),
        "{recorded}"
    );

    // A second client finds both live and spawns nothing.
    let (_, connected) = run(
        &mut world.launcher_with_server(&program),
        &hello_of(&program),
    )
    .await;
    connected.expect("the warm connect");
    assert_eq!(world.server_alive(), Some(server));
    assert_eq!(world.alive(), Some(daemon));
}

#[tokio::test(flavor = "current_thread")]
async fn a_rebuild_replaces_the_daemon_and_the_server() {
    let world = World::new();
    let old = world.program("search-old");
    let new = world.program("search-new");

    let (_, connected) = run(&mut world.launcher_with_server(&old), &hello_of(&old)).await;
    connected.expect("the old build starts");
    let (server, daemon) = (
        world.server_alive().expect("the old server is alive"),
        world.alive().expect("the old daemon is alive"),
    );

    let (_, connected) = run(&mut world.launcher_with_server(&new), &hello_of(&new)).await;
    connected.expect("the new build answers");
    assert_ne!(
        world.server_alive(),
        Some(server),
        "the server was replaced"
    );
    assert!(world.server_alive().is_some());
    assert_ne!(world.alive(), Some(daemon), "the daemon was replaced");
}

#[tokio::test(flavor = "current_thread")]
async fn a_server_that_exited_is_started_again() {
    let world = World::new();
    let program = world.program("search-a");
    let (_, connected) = run(
        &mut world.launcher_with_server(&program),
        &hello_of(&program),
    )
    .await;
    connected.expect("both start");
    let (server, daemon) = (
        world.server_alive().expect("the server is alive"),
        world.alive().expect("the daemon is alive"),
    );

    // The server idles out while the daemon lives on.
    // SAFETY: `kill` takes plain integers; the pid came from a held
    // `server.lock`, a server this test spawned.
    unsafe { libc::kill(server as i32, libc::SIGKILL) };
    gone(|| world.server_alive()).await;

    let (_, connected) = run(
        &mut world.launcher_with_server(&program),
        &hello_of(&program),
    )
    .await;
    connected.expect("the warm connect");
    let restarted = world.server_alive().expect("the server is back");
    assert_ne!(restarted, server);
    assert_eq!(world.alive(), Some(daemon), "the daemon was left alone");
}

#[tokio::test(flavor = "current_thread")]
async fn a_launcher_without_a_server_starts_none() {
    let world = World::new();
    let program = world.program("search-a");
    let (_, connected) = run(&mut world.launcher(&program), &hello_of(&program)).await;
    connected.expect("the daemon starts");
    assert_eq!(world.server_alive(), None);
    assert!(!world.run_dir().join("server.lock").exists());
}
