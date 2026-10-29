//! How the daemon's socket got here: launchd's activation or a path the
//! daemon bound itself under a lifetime lock.
//!
//! [`run`](crate::daemon::run) is generic over the [`Listener`] trait so
//! the two kinds share the accept loop and differ only in `close`: an
//! [`Activated`] socket belongs to launchd, so closing drops this
//! process's copy and leaves the backlog for the next activation; a
//! [`Bound`] socket is the daemon's own, so closing it makes the path
//! answer `ENOENT` and answers every queued connection `ShuttingDown`.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use tokio::net::{UnixListener, UnixStream};

use super::StartupError;
use crate::protocol::ErrorKind;

/// The socket `run` accepts on, however it got here.
pub trait Listener: Send + Sync + 'static {
    /// What `close` and `refuse` hand back for the caller to keep alive
    /// until the daemon's shutdown tail has finished: `()` when nothing
    /// must outlive the listener, a [`Bound`]'s [`LifetimeLock`] when
    /// the lifetime lock must stay held.
    type Held: Send;

    /// The next connection; every error is fatal to the accept loop.
    fn accept(&self) -> impl Future<Output = io::Result<UnixStream>> + Send;

    /// Stops accepting, in the way the listener's kind requires. `run`
    /// calls it right after the accept loop breaks, for every trigger,
    /// and holds what it returns until the tail has finished and `run`
    /// returns: a [`Bound`] unlinks its socket, answers the queued
    /// connections `ShuttingDown` and hands back its [`LifetimeLock`];
    /// an [`Activated`] only drops its copy — launchd's backlog stays
    /// for the next activation to drain.
    fn close(self) -> impl Future<Output = Self::Held> + Send;

    /// Stops before serving anything: the queued connections learn the
    /// startup error instead of `ShuttingDown`. Like [`close`](Listener::close),
    /// what it returns must be held until the caller exits.
    fn refuse(self, error: &StartupError) -> impl Future<Output = io::Result<Self::Held>> + Send;
}

/// launchd's activated socket.
pub struct Activated(pub UnixListener);

impl Listener for Activated {
    type Held = ();

    async fn accept(&self) -> io::Result<UnixStream> {
        self.0.accept().await.map(|(stream, _)| stream)
    }

    async fn close(self) {}

    async fn refuse(self, error: &StartupError) -> io::Result<()> {
        super::refuse(self.0.into_std()?, error).await;
        Ok(())
    }
}

/// A socket the daemon bound itself, under the lifetime lock `daemon.lock`
/// next to it. The lock proves the socket's owner — its contents are the
/// owner's pid — and `close`/`refuse` hand it back as the [`Held`](Listener::Held)
/// [`LifetimeLock`], held through the shutdown tail so a second
/// [`Bound::acquire`] waits out the first daemon instead of opening the
/// state files while the old one is still flushing. In the production
/// entry points the tail runs the process out, so the lock's release is
/// the process's exit.
pub struct Bound {
    listener: UnixListener,
    path: PathBuf,
    /// The open lock; `close`/`refuse` hand it to the caller.
    lock: File,
}

/// A [`Bound`]'s lifetime lock, held past `close`/`refuse`. [`run`]
/// keeps it until the shutdown tail has finished and it returns; the
/// production entry points drop it only by exiting, so the flock's
/// release — what a second [`Bound::acquire`] waits on — stays after
/// every flush.
///
/// [`run`]: crate::daemon::run
pub struct LifetimeLock {
    /// The open lock file, held for its `Drop`: closing it releases the
    /// flock.
    _file: File,
}

impl Bound {
    /// Takes the lifetime lock next to `path`, then unlinks any stale
    /// socket and binds. When another daemon holds the lock the acquire
    /// waits it out, so the new daemon serves only once the old one's
    /// tail has run and its process is gone.
    ///
    /// Every step but adopting the listener into the runtime is blocking
    /// filesystem work — the open, the lock (and its wait), the pid
    /// write, the stale-socket unlink, the bind — so the whole
    /// acquisition runs on the blocking pool in one task; only
    /// [`UnixListener::from_std`] runs on the executor.
    ///
    /// # Errors
    ///
    /// When the lock file cannot be opened, locked or written, the stale
    /// socket cannot be removed, or the socket cannot be bound.
    pub async fn acquire(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (listener, lock, path) = tokio::task::spawn_blocking(move || {
            // `daemon.sock`'s lock is `daemon.lock`, next to it.
            let lock_path = path.with_extension("lock");
            let mut lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)?;
            match lock.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => {
                    // The previous daemon is in its shutdown tail; the
                    // client's ready deadline is what the user sees, so
                    // the wait blocks.
                    tracing::info!("waiting for the previous daemon to exit");
                    lock.lock()?;
                }
                Err(TryLockError::Error(error)) => return Err(error),
            }
            lock.set_len(0)?;
            lock.write_all(format!("{}\n", std::process::id()).as_bytes())?;
            // Holding the lock proves nobody owns a stale socket file.
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            let listener = std::os::unix::net::UnixListener::bind(&path)?;
            listener.set_nonblocking(true)?;
            Ok::<_, io::Error>((listener, lock, path))
        })
        .await
        .expect("the acquire does not panic")?;
        Ok(Self {
            listener: UnixListener::from_std(listener)?,
            path,
            lock,
        })
    }

    /// The shared stop: unlink `path` so new connects get `ENOENT`, then
    /// drain the backlog nonblocking and answer each queued connection
    /// `kind` with `message`, `refuse`'s shape. The lock comes back as
    /// the [`Held`](Listener::Held) value — the caller, not `stop`,
    /// decides how long it lives: `run` holds it until the shutdown tail
    /// has finished, and production exits after it.
    async fn stop(self, kind: ErrorKind, message: String) -> (LifetimeLock, io::Result<()>) {
        let Self {
            listener,
            path,
            lock,
        } = self;
        let held = LifetimeLock { _file: lock };
        if let Err(error) = std::fs::remove_file(&path) {
            tracing::error!(%error, path = %path.display(), "could not unlink the daemon socket");
        }
        let drained = match listener.into_std() {
            Ok(listener) => {
                super::answer_queued(listener, kind, message).await;
                Ok(())
            }
            Err(error) => Err(error),
        };
        (held, drained)
    }
}

impl Listener for Bound {
    type Held = LifetimeLock;

    async fn accept(&self) -> io::Result<UnixStream> {
        self.listener.accept().await.map(|(stream, _)| stream)
    }

    async fn close(self) -> LifetimeLock {
        let (held, drained) = self
            .stop(
                ErrorKind::ShuttingDown,
                "the daemon is shutting down".to_owned(),
            )
            .await;
        if let Err(error) = drained {
            tracing::error!(%error, "could not answer the queued connections");
        }
        held
    }

    async fn refuse(self, error: &StartupError) -> io::Result<LifetimeLock> {
        let (held, drained) = self.stop(ErrorKind::Startup, error.to_string()).await;
        drained.map(|()| held)
    }
}
