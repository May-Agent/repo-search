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
use repo_search_client::launcher::spawn::{DaemonCommand, OnDemand, ServerTarget};
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
        let mut command = DaemonCommand::new(program.to_path_buf(), &self.socket(), &target);
        command.args.splice(
            0..0,
            [
                "--exact",
                "fake_daemon",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                "--",
            ]
            .map(Into::into),
        );
        OnDemand::new(command, self.socket(), self.tmp.path().join("daemon.log"))
    }

    /// The pid `daemon.lock` names while a daemon holds the lock.
    fn alive(&self) -> Option<u32> {
        let path = self.run_dir().join("daemon.lock");
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
    /// Kills the daemon a test leaves running: the pid is read from a
    /// held `daemon.lock`, a daemon the test spawned.
    fn drop(&mut self) {
        if let Some(pid) = self.alive() {
            // SAFETY: `kill` takes plain integers.
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
    }
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
