//! A launchd stand-in for the install and CLI tests. One thread owns the
//! fake service state; every `FakeLaunchctl` handle (`Launchctl + Clone +
//! Send`) talks to it over a request channel with a oneshot-style reply
//! channel, so two handles in two threads see one launchd — as two
//! `search` processes would.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use repo_search::install::{Launchctl, LaunchctlError, Service};
use repo_search::protocol::BuildId;
use tokio::task::JoinHandle;

use super::daemon::{self, Model};
use super::stub::StubBackend;

/// One `launchctl` call, reported to the test for sequence assertions.
#[derive(Debug, PartialEq, Eq)]
pub enum Call {
    /// `launchctl print <service>`.
    Service,
    /// `launchctl bootout <service>`.
    Bootout,
    /// `launchctl bootstrap <domain> <plist>`; carries the plist contents
    /// the bootstrap read off disk.
    Bootstrap { plist_contents: String },
}

/// What the fake's loaded job is backed by.
enum Process {
    /// Loaded, no process running.
    None,
    /// A `/bin/cat` with piped stdin standing in for a running daemon:
    /// `bootout` drops the stdin and the exit lands asynchronously, like a
    /// real daemon's SIGTERM shutdown.
    StandIn(std::process::Child),
    /// A stub daemon task started by `bootstrap`; the fake does not boot
    /// it out.
    Daemon(JoinHandle<Result<(), repo_search::daemon::Error>>),
}

struct Loaded {
    /// The plist contents the bootstrap read.
    #[allow(dead_code)]
    plist: String,
    process: Process,
}

enum Control {
    /// Load a `StandIn` job without a bootstrap.
    LoadStandIn,
    /// Load a job with no process without a bootstrap.
    LoadWithoutProcess,
    /// Make the next `bootstrap` fail with this stderr.
    FailNextBootstrap(String),
    /// Make `bootstrap` succeed without binding the socket.
    BootstrapWithoutSocket,
    /// Move the loaded daemon task's join handle to the caller.
    TakeDaemonJoin(mpsc::Sender<Option<JoinHandle<Result<(), repo_search::daemon::Error>>>>),
    /// End the owner thread.
    Shutdown,
}

enum Request {
    Service {
        service: String,
        reply: mpsc::Sender<Result<Option<Service>, LaunchctlError>>,
    },
    Bootout {
        service: String,
        reply: mpsc::Sender<Result<(), LaunchctlError>>,
    },
    Bootstrap {
        domain: String,
        plist: PathBuf,
        reply: mpsc::Sender<Result<(), LaunchctlError>>,
    },
    Control(Control),
}

/// An exit status as `wait(2)` encodes it, for fabricated failures.
fn exit(code: i32) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    ExitStatus::from_raw(code << 8)
}

fn failed(args: String, code: i32, stderr: &str) -> LaunchctlError {
    LaunchctlError::Failed {
        args,
        status: exit(code),
        stderr: stderr.to_owned(),
    }
}

/// The owner thread's state.
struct Owner {
    loaded: Option<Loaded>,
    /// A `StandIn` mid-teardown: `bootout` left it and its exit is still
    /// pending — launchd's asynchronous teardown.
    tearing_down: Option<std::process::Child>,
    fail_next_bootstrap: Option<String>,
    bootstrap_without_socket: bool,
    socket: PathBuf,
    runtime: tokio::runtime::Handle,
    models: tempfile::TempDir,
    /// The model layout the bootstrapped daemon's store gets.
    model: Model,
    /// The backend every bootstrapped daemon runs.
    backend: Arc<StubBackend>,
    build: BuildId,
}

impl Owner {
    fn service(&self) -> Option<Service> {
        self.loaded.as_ref().map(|loaded| Service {
            pid: match &loaded.process {
                Process::StandIn(child) => Some(child.id()),
                _ => None,
            },
        })
    }

    fn bootout(&mut self, args: String) -> Result<(), LaunchctlError> {
        let Some(loaded) = self.loaded.take() else {
            return Err(failed(
                args,
                36,
                "Boot-out failed: 36: Operation now in progress",
            ));
        };
        match loaded.process {
            Process::None => Ok(()),
            Process::StandIn(mut child) => {
                // Dropping the stdin ends `cat` asynchronously; the child
                // stays as `TearingDown` until it is reaped.
                drop(child.stdin.take());
                self.tearing_down = Some(child);
                Ok(())
            }
            Process::Daemon(_) => {
                panic!("the fake does not boot out an in-process daemon")
            }
        }
    }

    fn bootstrap(&mut self, args: String, plist_contents: String) -> Result<(), LaunchctlError> {
        if let Some(child) = &mut self.tearing_down {
            match child.try_wait() {
                Ok(Some(_)) => self.tearing_down = None,
                _ => {
                    return Err(failed(args, 5, "Bootstrap failed: 5: Input/output error"));
                }
            }
        }
        if self.loaded.is_some() {
            return Err(failed(args, 5, "Bootstrap failed: 5: Input/output error"));
        }
        if let Some(stderr) = self.fail_next_bootstrap.take() {
            return Err(failed(args, 5, &stderr));
        }
        let process = if self.bootstrap_without_socket {
            Process::None
        } else {
            // launchd binds the declared socket when the job loads; the
            // stub daemon then accepts on it.
            let listener = std::os::unix::net::UnixListener::bind(&self.socket)
                .expect("the fake binds the socket");
            let store = daemon::model_store(self.models.path().join("models"), &self.model);
            Process::Daemon(daemon::spawn_on(
                &self.runtime,
                listener,
                store,
                self.backend.clone(),
                Duration::from_secs(600),
                self.build,
            ))
        };
        self.loaded = Some(Loaded {
            plist: plist_contents,
            process,
        });
        Ok(())
    }

    /// Leaves nothing running: an unexited stand-in is killed and a loaded
    /// daemon task aborted.
    fn shutdown(&mut self) {
        if let Some(mut child) = self.tearing_down.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(loaded) = self.loaded.take() {
            match loaded.process {
                Process::StandIn(mut child) => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                Process::Daemon(task) => task.abort(),
                Process::None => {}
            }
        }
    }

    fn run(mut self, rx: mpsc::Receiver<Request>, calls: mpsc::Sender<Call>) {
        while let Ok(request) = rx.recv() {
            match request {
                Request::Service { service, reply } => {
                    let _ = calls.send(Call::Service);
                    let _ = reply.send(Ok(self.service()));
                    let _ = service;
                }
                Request::Bootout { service, reply } => {
                    let _ = calls.send(Call::Bootout);
                    let _ = reply.send(self.bootout(format!("bootout {service}")));
                }
                Request::Bootstrap {
                    domain,
                    plist,
                    reply,
                } => {
                    let plist_contents = std::fs::read_to_string(&plist)
                        .expect("the plist was written before bootstrap");
                    let _ = calls.send(Call::Bootstrap {
                        plist_contents: plist_contents.clone(),
                    });
                    let _ = reply.send(self.bootstrap(
                        format!("bootstrap {domain} {}", plist.display()),
                        plist_contents,
                    ));
                }
                Request::Control(control) => match control {
                    Control::TakeDaemonJoin(reply) => {
                        let handle =
                            self.loaded.as_mut().and_then(|loaded| {
                                match std::mem::replace(&mut loaded.process, Process::None) {
                                    Process::Daemon(join) => Some(join),
                                    other => {
                                        loaded.process = other;
                                        None
                                    }
                                }
                            });
                        let _ = reply.send(handle);
                    }
                    Control::LoadStandIn => {
                        let child = std::process::Command::new("/bin/cat")
                            .stdin(std::process::Stdio::piped())
                            .spawn()
                            .expect("the stand-in cat spawns");
                        self.loaded = Some(Loaded {
                            plist: String::new(),
                            process: Process::StandIn(child),
                        });
                    }
                    Control::LoadWithoutProcess => {
                        self.loaded = Some(Loaded {
                            plist: String::new(),
                            process: Process::None,
                        });
                    }
                    Control::FailNextBootstrap(stderr) => {
                        self.fail_next_bootstrap = Some(stderr);
                    }
                    Control::BootstrapWithoutSocket => {
                        self.bootstrap_without_socket = true;
                    }
                    Control::Shutdown => break,
                },
            }
        }
        self.shutdown();
    }
}

/// A launchd stand-in: owns its service state on one thread and records
/// every call for the test.
pub struct FakeLaunchd {
    tx: mpsc::Sender<Request>,
    calls: mpsc::Receiver<Call>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FakeLaunchd {
    /// A fake launchd for a service whose socket lives at `socket`; jobs
    /// it bootstraps run the stub daemon on `runtime` with `build`.
    pub fn new(socket: PathBuf, runtime: tokio::runtime::Handle, build: BuildId) -> Self {
        Self::with_daemon(
            socket,
            runtime,
            build,
            Model::Cached,
            StubBackend::new(mpsc::channel().0),
        )
    }

    /// A fake whose bootstrapped daemon runs `model` on `backend` — the
    /// tests that drive progress output need both knobs.
    pub fn with_daemon(
        socket: PathBuf,
        runtime: tokio::runtime::Handle,
        build: BuildId,
        model: Model,
        backend: StubBackend,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let (calls, calls_rx) = mpsc::channel();
        let owner = Owner {
            loaded: None,
            tearing_down: None,
            fail_next_bootstrap: None,
            bootstrap_without_socket: false,
            socket,
            runtime,
            models: tempfile::tempdir().expect("the fake's model cache"),
            model,
            backend: Arc::new(backend),
            build,
        };
        let thread = std::thread::spawn(move || owner.run(rx, calls));
        Self {
            tx,
            calls: calls_rx,
            thread: Some(thread),
        }
    }

    /// A `Launchctl` handle talking to this fake.
    pub fn launchctl(&self) -> FakeLaunchctl {
        FakeLaunchctl {
            tx: self.tx.clone(),
        }
    }

    /// The calls made so far, drained in order.
    pub fn calls(&self) -> Vec<Call> {
        self.calls.try_iter().collect()
    }

    /// Marks the service loaded, backed by a stand-in `/bin/cat` whose
    /// exit `bootout` leaves pending.
    pub fn load_stand_in(&self) {
        self.tx
            .send(Request::Control(Control::LoadStandIn))
            .expect("the launchd thread lives");
    }

    /// Marks the service loaded with no process running.
    pub fn load_without_process(&self) {
        self.tx
            .send(Request::Control(Control::LoadWithoutProcess))
            .expect("the launchd thread lives");
    }

    /// The next `bootstrap` fails with `stderr` verbatim.
    pub fn fail_next_bootstrap(&self, stderr: &str) {
        self.tx
            .send(Request::Control(Control::FailNextBootstrap(
                stderr.to_owned(),
            )))
            .expect("the launchd thread lives");
    }

    /// The next `bootstrap` succeeds without binding the socket.
    pub fn bootstrap_without_socket(&self) {
        self.tx
            .send(Request::Control(Control::BootstrapWithoutSocket))
            .expect("the launchd thread lives");
    }

    /// The loaded daemon task's join handle, when a bootstrap started one —
    /// the test awaits the daemon's `run` on it.
    pub fn take_daemon_join(&self) -> Option<JoinHandle<Result<(), repo_search::daemon::Error>>> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(Request::Control(Control::TakeDaemonJoin(reply)))
            .expect("the launchd thread lives");
        rx.recv().expect("the launchd thread answers")
    }
}

impl Drop for FakeLaunchd {
    fn drop(&mut self) {
        let _ = self.tx.send(Request::Control(Control::Shutdown));
        if let Some(thread) = self.thread.take() {
            // A panicked owner thread is already surfaced by the failed
            // reply `recv`; a second panic here would only abort the test
            // mid-unwind.
            if thread.join().is_err() && !std::thread::panicking() {
                panic!("the fake launchd thread panicked");
            }
        }
    }
}

/// The `Launchctl` handle to a [`FakeLaunchd`].
#[derive(Clone)]
pub struct FakeLaunchctl {
    tx: mpsc::Sender<Request>,
}

impl FakeLaunchctl {
    fn call<T>(
        &self,
        request: impl FnOnce(mpsc::Sender<Result<T, LaunchctlError>>) -> Request,
    ) -> Result<T, LaunchctlError> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(request(reply))
            .expect("the launchd thread lives");
        rx.recv().expect("the launchd thread answers")
    }
}

impl Launchctl for FakeLaunchctl {
    fn service(&mut self, service: &str) -> Result<Option<Service>, LaunchctlError> {
        let service = service.to_owned();
        self.call(|reply| Request::Service { service, reply })
    }

    fn bootout(&mut self, service: &str) -> Result<(), LaunchctlError> {
        let service = service.to_owned();
        self.call(|reply| Request::Bootout { service, reply })
    }

    fn bootstrap(&mut self, domain: &str, plist: &Path) -> Result<(), LaunchctlError> {
        let domain = domain.to_owned();
        let plist = plist.to_path_buf();
        self.call(|reply| Request::Bootstrap {
            domain,
            plist,
            reply,
        })
    }
}
