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
use std::sync::Arc;
use std::time::Instant;

use repo_search_core::StateDir;
use repo_search_core::store::VectorCache;
use tokio::sync::{mpsc, oneshot, watch};

use super::collection::{self, CollectionConfig, CollectionMsg};
use super::counter::Tokens;
use super::link::EmbeddingState;
use super::scheduler::{RepoId, WorktreeId};
use super::server::ServerClient;
use super::worktree::{self, OwnerConfig, OwnerMsg};
use super::writer::{self, WriterHandle};
use super::{BusySource, Error};

/// A repository's shared state: the vector cache and its writer task,
/// opened lazily on the first `Search` or `Index` — a `Status` never
/// creates the repository's cache directory.
struct RepoShared {
    /// The scheduler's repository id.
    id: RepoId,
    /// Where the cache lives.
    home: CacheHome,
    /// The opened half, once any request needed it.
    open: Option<RepoOpen>,
}

/// Whose vector cache a [`RepoShared`] holds: a git repository's, shared
/// by its worktrees, or a document collection's own.
enum CacheHome {
    /// The cache of the repository whose common dir is the entry's key.
    Repository,
    /// The cache of the named collection; the entry's key is the
    /// collection's directory.
    Collection(String),
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
    /// Resolve the collection `name` to its owner; with `create`, a
    /// collection never written is created on first sight.
    Collection {
        /// The collection's name.
        name: String,
        /// Whether a missing collection is created; a read leaves it unset.
        create: bool,
        /// The owner's inbox, or why there is none.
        reply: oneshot::Sender<Result<CollectionHandle, GetError>>,
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
    /// A read named a collection nothing was ever written to.
    #[error("the collection {0:?} does not exist")]
    NoSuchCollection(String),
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

/// A resolved collection owner's inbox; consumed by `dispatch` like
/// [`OwnerHandle`].
pub struct CollectionHandle {
    tx: mpsc::Sender<CollectionMsg>,
}

impl CollectionHandle {
    /// Hand `msg` to the owner.
    ///
    /// # Errors
    ///
    /// When the owner is gone.
    pub async fn dispatch(self, msg: CollectionMsg) -> Result<(), OwnerGone> {
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

    /// The owner of the collection `name`; with `create`, a collection
    /// never written is created, otherwise it is an error and nothing is
    /// created on disk or in the registry.
    ///
    /// # Errors
    ///
    /// [`GetError::NoSuchCollection`] when `create` is unset and the
    /// collection does not exist, [`GetError::Internal`] when its directory
    /// cannot be inspected or created. `None` when the registry is gone.
    pub async fn collection(
        &self,
        name: String,
        create: bool,
    ) -> Option<Result<CollectionHandle, GetError>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(RegistryMsg::Collection {
                name,
                create,
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
pub struct RegistryConfig<T: Tokens> {
    /// The daemon's state root, handed to every owner and shared cache.
    pub state: StateDir,
    /// The model-id vectors are stored under.
    pub model_id: String,
    /// The model readiness watch, handed to every owner.
    pub model: watch::Receiver<EmbeddingState<T>>,
    /// Daemon start, handed to every owner.
    pub started: Instant,
    /// The shutdown watch, handed to every owner.
    pub stopping: watch::Receiver<bool>,
    /// The busy-source registrar, handed to every writer.
    pub busy: mpsc::Sender<BusySource>,
    /// The fatal channel, handed to every writer.
    pub fatal: mpsc::Sender<Error>,
    /// The embedding server's client; owners read `GET /status` through
    /// it for the GPU report.
    pub server: Arc<ServerClient>,
    /// Marker tying the config to `T`.
    pub marker: std::marker::PhantomData<T>,
}

/// Spawns the registry task and returns its handle.
pub fn spawn<T: Tokens>(config: RegistryConfig<T>) -> RegistryHandle {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(
        Registry {
            owners: HashMap::new(),
            collections: HashMap::new(),
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

struct Registry<T: Tokens> {
    /// Owner entries by canonical worktree root.
    owners: HashMap<PathBuf, OwnerEntry>,
    /// Collection owners' inboxes by name.
    collections: HashMap<String, mpsc::Sender<CollectionMsg>>,
    /// Shared repository state by canonical git common dir, and each
    /// collection's by its directory.
    repos: HashMap<PathBuf, RepoShared>,
    next_worktree: u64,
    next_repo: u64,
    /// A self-handle, handed to owners for `DaemonStatus::worktrees` and
    /// `Forget`.
    handle: RegistryHandle,
    config: RegistryConfig<T>,
    rx: mpsc::Receiver<RegistryMsg>,
}

impl<T: Tokens> Registry<T> {
    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                RegistryMsg::Get { resolved, reply } => {
                    let _ = reply.send(self.get(*resolved).await);
                }
                RegistryMsg::Collection {
                    name,
                    create,
                    reply,
                } => {
                    let _ = reply.send(self.collection(name, create).await);
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
                    self.collections.clear();
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
            self.repos.insert(
                w.common_dir.clone(),
                RepoShared {
                    id,
                    home: CacheHome::Repository,
                    open: None,
                },
            );
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
                server: self.config.server.clone(),
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

    async fn collection(
        &mut self,
        name: String,
        create: bool,
    ) -> Result<CollectionHandle, GetError> {
        if let Some(tx) = self.collections.get(&name) {
            return Ok(CollectionHandle { tx: tx.clone() });
        }
        let state = &self.config.state;
        let docs = state.collection_docs(&name);
        if create {
            tokio::fs::create_dir_all(&docs)
                .await
                .map_err(|e| GetError::Internal(format!("{}: {e}", docs.display())))?;
        } else {
            match tokio::fs::metadata(&docs).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(GetError::NoSuchCollection(name));
                }
                Err(e) => return Err(GetError::Internal(format!("{}: {e}", docs.display()))),
            }
        }
        let key = state.collection(&name);
        let repo = RepoId(self.next_repo);
        self.next_repo += 1;
        self.repos.insert(
            key.clone(),
            RepoShared {
                id: repo,
                home: CacheHome::Collection(name.clone()),
                open: None,
            },
        );
        let worktree = WorktreeId(self.next_worktree);
        self.next_worktree += 1;
        let tx = collection::spawn(CollectionConfig {
            state: state.clone(),
            name: name.clone(),
            key,
            id: worktree,
            repo,
            model_id: self.config.model_id.clone(),
            model: self.config.model.clone(),
            registry: self.handle.clone(),
            stopping: self.config.stopping.clone(),
        });
        self.collections.insert(name, tx.clone());
        Ok(CollectionHandle { tx })
    }

    async fn shared(&mut self, common_dir: &Path) -> Result<RepoHandle, GetError> {
        let shared = self.repos.get_mut(common_dir).ok_or_else(|| {
            GetError::Internal(format!("unknown repository {}", common_dir.display()))
        })?;
        if shared.open.is_none() {
            let state = &self.config.state;
            let cache = match &shared.home {
                CacheHome::Repository => VectorCache::open(state, common_dir).await,
                CacheHome::Collection(name) => {
                    VectorCache::open_at(&state.collection_docs(name), state.collection_cache(name))
                        .await
                }
            }
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
