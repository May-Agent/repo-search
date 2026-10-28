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

use repo_search::daemon::server::Endpoint;
use repo_search::install::{Launchctl, LaunchctlError, Service};
use repo_search::protocol::BuildId;
use repo_search_core::StateDir;
use repo_search_server::stub::StubBackend;
use tokio::task::JoinHandle;

use super::daemon::{self, Model};
use super::tokens::StubTokens;

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
    /// A stub daemon-and-server task pair started by `bootstrap`; the
    /// fake does not boot it out.
    Daemon {
        daemon: JoinHandle<Result<(), repo_search::daemon::Error>>,
        server: JoinHandle<Result<(), repo_search_server::Error>>,
    },
    /// The server agent's job: a stub server task on the job's socket,
    /// started by `bootstrap`; the fake does not boot it out.
    Server(JoinHandle<Result<(), repo_search_server::Error>>),
    /// A task standing in for a daemon that fails after socket activation:
    /// on the first queued connection it answers every pending one with the
    /// startup error, as the demand-launched daemon would.
    Failing(JoinHandle<()>),
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
    /// Make the next `bootstrap` start a daemon that fails the chip check
    /// on this brand string.
    BootstrapFailingDaemon(String),
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
    failing_daemon: Option<String>,
    socket: PathBuf,
    /// `Some` for a daemon agent's job: the socket its own stub server
    /// binds and its daemon probes. `None` for the server agent's job,
    /// whose stub server binds `socket`.
    server_socket: Option<PathBuf>,
    runtime: tokio::runtime::Handle,
    models: tempfile::TempDir,
    /// The state root every bootstrapped daemon gets.
    state: tempfile::TempDir,
    /// The model layout the bootstrapped server's store gets.
    model: Model,
    /// The backend every bootstrapped server runs.
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
            Process::Daemon { .. } => {
                panic!("the fake does not boot out an in-process daemon")
            }
            Process::Server(_) => {
                panic!("the fake does not boot out an in-process server")
            }
            Process::Failing(task) => {
                task.abort();
                Ok(())
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
        } else if let Some(brand) = self.failing_daemon.take() {
            // launchd binds the socket; a connection in its queue makes it
            // launch the daemon, which fails and refuses the queue.
            let listener = std::os::unix::net::UnixListener::bind(&self.socket)
                .expect("the fake binds the socket");
            listener
                .set_nonblocking(true)
                .expect("the listener is nonblocking");
            Process::Failing(self.runtime.spawn(async move {
                // SAFETY: the std listener owns its fd, open until
                // `into_inner`.
                let listener = unsafe { tokio::io::unix::AsyncFd::register(listener) }
                    .expect("the listener registers");
                listener
                    .readable()
                    .await
                    .expect("the listener becomes readable")
                    .retain_ready();
                let error = repo_search::daemon::StartupError::StateDir(std::io::Error::other(
                    format!("simulated startup failure on {brand}"),
                ));
                repo_search::daemon::refuse(listener.into_inner(), &error).await;
            }))
        } else {
            // launchd binds the declared socket when the job loads; the
            // job's stub process then accepts on it.
            let listener = std::os::unix::net::UnixListener::bind(&self.socket)
                .expect("the fake binds the socket");
            match self.server_socket.clone() {
                // The server agent's job: the stub server alone.
                None => Process::Server(daemon::spawn_server_on(
                    &self.runtime,
                    listener,
                    self.server_setup(),
                )),
                // A daemon agent's job: the stub daemon, against a stub
                // server of its own. A path pre-created as the "socket
                // exists" marker is replaced.
                Some(server_socket) => {
                    let _ = std::fs::remove_file(&server_socket);
                    let server_listener = std::os::unix::net::UnixListener::bind(&server_socket)
                        .expect("the fake binds the server socket");
                    let server = daemon::spawn_server_on(
                        &self.runtime,
                        server_listener,
                        self.server_setup(),
                    );
                    let daemon = daemon::spawn_on(
                        &self.runtime,
                        listener,
                        daemon::Setup {
                            tokens: StubTokens::default(),
                            server: Endpoint::Unix(server_socket),
                            model_id: repo_search_core::model::MODEL_ID.to_owned(),
                            state: StateDir::new(self.state.path().to_path_buf()),
                            idle_exit: Duration::from_secs(600),
                            load_bound: repo_search::daemon::LOAD_BOUND,
                            build: self.build,
                        },
                    );
                    Process::Daemon { daemon, server }
                }
            }
        };
        self.loaded = Some(Loaded {
            plist: plist_contents,
            process,
        });
        Ok(())
    }

    /// The stub server every bootstrapped job runs: the fake's model
    /// layout and backend, on a wide chip.
    fn server_setup(&self) -> daemon::ServerSetup {
        daemon::ServerSetup {
            model_store: daemon::model_store(self.models.path().join("models"), &self.model),
            chip: repo_search_server::chip::ChipClass::Wide,
            backend: self.backend.clone(),
            idle_exit: Duration::from_secs(600),
            load_bound: repo_search_server::LOAD_BOUND,
        }
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
                Process::Daemon { daemon, server } => {
                    daemon.abort();
                    server.abort();
                }
                Process::Server(server) => server.abort(),
                Process::Failing(task) => task.abort(),
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
                                    Process::Daemon { daemon, .. } => Some(daemon),
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
                    Control::BootstrapFailingDaemon(brand) => {
                        self.failing_daemon = Some(brand);
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
    /// it bootstraps run the stub server on `server_socket` and the stub
    /// daemon against it, on `runtime` with `build`.
    pub fn new(
        socket: PathBuf,
        server_socket: PathBuf,
        runtime: tokio::runtime::Handle,
        build: BuildId,
    ) -> Self {
        Self::with_daemon(
            socket,
            server_socket,
            runtime,
            build,
            Model::Cached,
            StubBackend::new(mpsc::channel().0),
        )
    }

    /// A fake whose bootstrapped pair runs `model` on `backend` — the
    /// tests that drive progress output need both knobs.
    pub fn with_daemon(
        socket: PathBuf,
        server_socket: PathBuf,
        runtime: tokio::runtime::Handle,
        build: BuildId,
        model: Model,
        backend: StubBackend,
    ) -> Self {
        Self::spawn(Some(server_socket), socket, runtime, build, model, backend)
    }

    /// A fake launchd for the server agent, whose socket lives at
    /// `socket`: jobs it bootstraps run the stub server alone on it.
    pub fn server(socket: PathBuf, runtime: tokio::runtime::Handle, build: BuildId) -> Self {
        Self::spawn(
            None,
            socket,
            runtime,
            build,
            Model::Cached,
            StubBackend::new(mpsc::channel().0),
        )
    }

    /// Starts the owner thread; `server_socket` picks the job kind as
    /// [`Owner::server_socket`] describes.
    fn spawn(
        server_socket: Option<PathBuf>,
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
            failing_daemon: None,
            socket,
            server_socket,
            runtime,
            models: tempfile::tempdir().expect("the fake's model cache"),
            state: tempfile::tempdir().expect("the fake's state dir"),
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

    /// The next `bootstrap` loads a daemon that fails to start with an
    /// unrecognised chip `brand`: it answers the queued connections with
    /// the startup error and runs no further.
    pub fn bootstrap_failing_daemon(&self, brand: &str) {
        self.tx
            .send(Request::Control(Control::BootstrapFailingDaemon(
                brand.to_owned(),
            )))
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
