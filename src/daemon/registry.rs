//! The worktree and repository registry.
//!
//! One task owns one [`RepoShared`] per git common dir — the shared vector
//! cache plus its writer, opened lazily on the first request that needs
//! them — and spawns one worktree owner per root. The connection task
//! resolves paths; `Get` compares the resolution with the stored
//! [`OwnerEntry`] so a removed-and-recreated worktree at the same path gets
//! a fresh owner. On shutdown it drops the owner inboxes and writer handles
//! and hands the writer tasks back so `run` can await the close-flush.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use repo_search_core::StateDir;
use repo_search_core::store::VectorCache;
use tokio::sync::{mpsc, oneshot, watch};

use super::model::ModelState;
use super::scheduler::{RepoId, WorktreeId};
use super::worktree::{self, OwnerConfig, OwnerMsg};
use super::writer::{self, WriterHandle};
use super::{Backend, BusySource, Error};

/// A repository's shared state: the vector cache and its writer task,
/// opened lazily on the first `Search` or `Index` — a `Status` never
/// creates the repository's cache directory.
struct RepoShared {
    /// The scheduler's repository id.
    id: RepoId,
    /// The opened half, once any request needed it.
    open: Option<RepoOpen>,
}

struct RepoOpen {
    /// The shared vector cache.
    cache: VectorCache,
    /// The vector writer's handle.
    writer: WriterHandle,
    /// The writer task, awaited by `run` at shutdown.
    task: tokio::task::JoinHandle<()>,
}

/// What a `Status` cannot open but a `Search` or `Index` needs: the
/// repository's cache and writer.
#[derive(Clone)]
pub struct RepoHandle {
    /// The scheduler's repository id.
    pub id: RepoId,
    /// The shared vector cache.
    pub cache: VectorCache,
    /// The vector writer's handle.
    pub writer: WriterHandle,
}

/// A live owner and the resolution it was spawned for; a root whose
/// `(dev, ino)`, `common_dir` or `git_dir` no longer matches is a
/// different tree and gets a fresh owner.
struct OwnerEntry {
    /// The scheduler's worktree id.
    id: WorktreeId,
    /// The owner inbox.
    tx: mpsc::Sender<OwnerMsg>,
    /// The canonical git common dir at spawn.
    common_dir: PathBuf,
    /// The canonical git dir at spawn.
    git_dir: PathBuf,
    /// `(dev, ino)` of the worktree root at spawn.
    root_id: (u64, u64),
}

/// A worktree's resolved paths, computed by the connection task —
/// `git rev-parse` plus canonicalization off the actor.
pub struct ResolvedWorktree {
    /// The canonicalized request path.
    pub requested: PathBuf,
    /// `--show-toplevel`, canonicalized: the worktree root.
    pub toplevel: PathBuf,
    /// `--git-common-dir`, canonicalized.
    pub common_dir: PathBuf,
    /// `--absolute-git-dir`, canonicalized.
    pub git_dir: PathBuf,
    /// `(dev, ino)` of `toplevel` right now.
    pub root_id: (u64, u64),
}

/// The registry's inbox.
pub enum RegistryMsg {
    /// Resolve `resolved` to its owner, creating it on first sight.
    Get {
        /// The request's resolved paths.
        resolved: Box<ResolvedWorktree>,
        /// The owner, or why there is none.
        reply: oneshot::Sender<Result<OwnerHandle, GetError>>,
    },
    /// The repository's cache and writer, opened lazily on first need.
    Shared {
        /// The canonical git common dir.
        common_dir: PathBuf,
        /// The repo's shared handles, or why they cannot open.
        reply: oneshot::Sender<Result<RepoHandle, GetError>>,
    },
    /// An owner exited because its root is gone; forget the entry if it is
    /// still this owner's.
    Forget {
        /// The canonical worktree root.
        worktree: PathBuf,
        /// The exiting owner's id.
        id: WorktreeId,
    },
    /// How many worktrees have owners, for `DaemonStatus`.
    Count {
        /// Where the count goes.
        reply: oneshot::Sender<u32>,
    },
    /// Drop every owner inbox and writer handle, then hand back the writer
    /// tasks for `run` to await.
    Shutdown {
        /// The writer task join handles.
        reply: oneshot::Sender<Vec<tokio::task::JoinHandle<()>>>,
    },
}

/// Why a `Get` or `Shared` failed.
#[derive(Debug, thiserror::Error)]
pub enum GetError {
    /// The path is not a worktree root.
    #[error("{0} is not a repository worktree root")]
    NotAWorktree(PathBuf),
    /// `git`, the vector cache or the busy tracker failed.
    #[error("{0}")]
    Internal(String),
}

/// A resolved owner's inbox; consumed by `dispatch` so the sender's close
/// is how the owner learns the registry is gone.
pub struct OwnerHandle {
    tx: mpsc::Sender<OwnerMsg>,
}

impl OwnerHandle {
    /// Hand `msg` to the owner; the handle is spent, so the registry's
    /// sender stays the only one.
    ///
    /// # Errors
    ///
    /// When the owner is gone.
    pub async fn dispatch(self, msg: OwnerMsg) -> Result<(), OwnerGone> {
        self.tx.send(msg).await.map_err(|_| OwnerGone)
    }
}

/// The owner is gone.
#[derive(Debug, thiserror::Error)]
#[error("the worktree owner is gone")]
pub struct OwnerGone;

/// The registry task's handle.
#[derive(Clone)]
pub struct RegistryHandle {
    tx: mpsc::Sender<RegistryMsg>,
}

impl RegistryHandle {
    /// The owner for `resolved`, creating it as needed.
    ///
    /// # Errors
    ///
    /// [`GetError::NotAWorktree`] when the path is not a repository root, or
    /// [`GetError::Internal`] when the cache fails. `None` when the registry
    /// is gone.
    pub async fn get(&self, resolved: ResolvedWorktree) -> Option<Result<OwnerHandle, GetError>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(RegistryMsg::Get {
                resolved: Box::new(resolved),
                reply,
            })
            .await
            .ok()?;
        rx.await.ok()
    }

    /// The repository's cache and writer, opened lazily.
    ///
    /// # Errors
    ///
    /// [`GetError::Internal`] when the cache cannot open. `None` when the
    /// registry is gone.
    pub async fn shared(&self, common_dir: PathBuf) -> Option<Result<RepoHandle, GetError>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(RegistryMsg::Shared { common_dir, reply })
            .await
            .ok()?;
        rx.await.ok()
    }

    /// Forget `worktree`'s owner when it is still `id`'s — the owner saw
    /// its root removed and exited.
    pub async fn forget(&self, worktree: PathBuf, id: WorktreeId) {
        let _ = self.tx.send(RegistryMsg::Forget { worktree, id }).await;
    }

    /// How many worktrees have owners.
    pub async fn count(&self) -> Option<u32> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(RegistryMsg::Count { reply }).await.ok()?;
        rx.await.ok()
    }

    /// Close every owner inbox and writer handle, returning the writer
    /// tasks to await.
    pub async fn shutdown(&self) -> Vec<tokio::task::JoinHandle<()>> {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(RegistryMsg::Shutdown { reply }).await.is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }
}

/// What the registry needs to build owners and writers.
pub struct RegistryConfig<B: Backend> {
    /// The daemon's state root, handed to every owner and shared cache.
    pub state: StateDir,
    /// The model-id vectors are stored under.
    pub model_id: String,
    /// The model readiness watch, handed to every owner.
    pub model: watch::Receiver<ModelState<B>>,
    /// Daemon start, handed to every owner.
    pub started: Instant,
    /// The shutdown watch, handed to every owner.
    pub stopping: watch::Receiver<bool>,
    /// The busy-source registrar, handed to every writer.
    pub busy: mpsc::Sender<BusySource>,
    /// The fatal channel, handed to every writer.
    pub fatal: mpsc::Sender<Error>,
    /// Marker tying the config to `B`.
    pub marker: std::marker::PhantomData<B>,
}

/// Spawns the registry task and returns its handle.
pub fn spawn<B: Backend>(config: RegistryConfig<B>) -> RegistryHandle {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(
        Registry {
            owners: HashMap::new(),
            repos: HashMap::new(),
            next_worktree: 0,
            next_repo: 0,
            handle: RegistryHandle { tx: tx.clone() },
            config,
            rx,
        }
        .run(),
    );
    RegistryHandle { tx }
}

struct Registry<B: Backend> {
    /// Owner entries by canonical worktree root.
    owners: HashMap<PathBuf, OwnerEntry>,
    /// Shared repository state by canonical git common dir.
    repos: HashMap<PathBuf, RepoShared>,
    next_worktree: u64,
    next_repo: u64,
    /// A self-handle, handed to owners for `DaemonStatus::worktrees` and
    /// `Forget`.
    handle: RegistryHandle,
    config: RegistryConfig<B>,
    rx: mpsc::Receiver<RegistryMsg>,
}

impl<B: Backend> Registry<B> {
    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                RegistryMsg::Get { resolved, reply } => {
                    let _ = reply.send(self.get(*resolved).await);
                }
                RegistryMsg::Shared { common_dir, reply } => {
                    let _ = reply.send(self.shared(&common_dir).await);
                }
                RegistryMsg::Forget { worktree, id } => {
                    if self.owners.get(&worktree).is_some_and(|e| e.id == id) {
                        self.owners.remove(&worktree);
                    }
                }
                RegistryMsg::Count { reply } => {
                    let _ = reply.send(self.owners.len() as u32);
                }
                RegistryMsg::Shutdown { reply } => {
                    self.owners.clear();
                    let tasks = std::mem::take(&mut self.repos)
                        .into_values()
                        .filter_map(|shared| shared.open.map(|open| open.task))
                        .collect();
                    let _ = reply.send(tasks);
                    return;
                }
            }
        }
    }

    async fn get(&mut self, w: ResolvedWorktree) -> Result<OwnerHandle, GetError> {
        if w.toplevel != w.requested {
            return Err(GetError::NotAWorktree(w.requested));
        }
        if !self.repos.contains_key(&w.common_dir) {
            let id = RepoId(self.next_repo);
            self.next_repo += 1;
            self.repos
                .insert(w.common_dir.clone(), RepoShared { id, open: None });
        }
        let same = self.owners.get(&w.toplevel).is_some_and(|entry| {
            entry.common_dir == w.common_dir
                && entry.git_dir == w.git_dir
                && entry.root_id == w.root_id
        });
        if !same {
            // A removed-and-recreated tree at the same path: drop the stale
            // entry; the old owner exits once its inbox closes.
            self.owners.remove(&w.toplevel);
            let repo = self.repos[&w.common_dir].id;
            let id = WorktreeId(self.next_worktree);
            self.next_worktree += 1;
            let tx = worktree::spawn(OwnerConfig {
                state: self.config.state.clone(),
                id,
                root: w.toplevel.clone(),
                common_dir: w.common_dir.clone(),
                git_dir: w.git_dir.clone(),
                repo,
                model_id: self.config.model_id.clone(),
                model: self.config.model.clone(),
                registry: self.handle.clone(),
                started: self.config.started,
                stopping: self.config.stopping.clone(),
            });
            self.owners.insert(
                w.toplevel.clone(),
                OwnerEntry {
                    id,
                    tx,
                    common_dir: w.common_dir,
                    git_dir: w.git_dir,
                    root_id: w.root_id,
                },
            );
        }
        let tx = self.owners[&w.toplevel].tx.clone();
        Ok(OwnerHandle { tx })
    }

    async fn shared(&mut self, common_dir: &Path) -> Result<RepoHandle, GetError> {
        let shared = self.repos.get_mut(common_dir).ok_or_else(|| {
            GetError::Internal(format!("unknown repository {}", common_dir.display()))
        })?;
        if shared.open.is_none() {
            let cache = VectorCache::open(&self.config.state, common_dir)
                .await
                .map_err(|e| GetError::Internal(e.to_string()))?;
            let (busy_tx, busy_rx) = watch::channel(false);
            self.config
                .busy
                .send(BusySource { state: busy_rx })
                .await
                .map_err(|_| GetError::Internal("the daemon is shutting down".to_owned()))?;
            let (writer, task) = writer::spawn(
                cache.clone(),
                self.config.model_id.clone(),
                self.config.fatal.clone(),
                busy_tx,
            );
            shared.open = Some(RepoOpen {
                cache,
                writer,
                task,
            });
        }
        let open = shared.open.as_ref().expect("opened above");
        Ok(RepoHandle {
            id: shared.id,
            cache: open.cache.clone(),
            writer: open.writer.clone(),
        })
    }
}
