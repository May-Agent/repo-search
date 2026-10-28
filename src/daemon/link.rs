//! The embedding link: the tokenizer load and the server-readiness probe.
//!
//! One task owns `watch::Sender<EmbeddingState>` and runs the link
//! sequence: `Tokens::load` — the tokenizer download and counter load,
//! publishing [`EmbeddingState::Downloading`] — then the `GET /v1/models`
//! probe loop, which keeps the state filled from the server's
//! `GET /status` until it lists the model id, spawning the index
//! scheduler and publishing [`EmbeddingState::Ready`]. A failure
//! publishes [`EmbeddingState::Failed`]; a request that finds it sends
//! [`LoaderMsg::Retry`], which starts the sequence over — each probe on a
//! `Failed` server fires that loader's own `Retry`, as requests do. A
//! model-id mismatch or a link outliving the bound publishes
//! [`EmbeddingState::Fatal`] — which no `Retry` restarts — and reports to
//! `run` over the fatal channel. On [`LoaderMsg::Shutdown`] the task
//! cancels a running attempt, stops the scheduler and waits for its
//! in-flight requests to report.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};

use repo_search_core::embed_api;
use repo_search_core::model::Fetch;

use super::counter::Tokens;
use super::scheduler::{self, SchedulerHandle};
use super::server::{Probe, ServerClient};
use super::{BusySource, Error};

/// Where the embedding service is on its way to readiness. The states
/// mirror the model loader's of the in-process pipeline: `Downloading`
/// covers the daemon's own tokenizer download, then the server archive's
/// as `GET /status` reports it.
pub enum EmbeddingState<T: Tokens> {
    /// `Tokens::load` has not produced a counter yet.
    Absent,
    /// The tokenizer or the server's archive is downloading.
    Downloading {
        /// Bytes received so far.
        received: u64,
        /// The declared length, when known.
        total: Option<u64>,
    },
    /// The tokenizer is loaded; the server is loading the model.
    Loading,
    /// Ready to embed.
    Ready(Arc<Embedding<T>>),
    /// Download, load or probe failed; a request finding it sends
    /// [`LoaderMsg::Retry`].
    Failed(String),
    /// The link failed fatally — a model-id mismatch or a link past the
    /// bound — and the daemon is shutting down.
    Fatal(String),
}

/// The counter, scheduler and server client the ready link hands out.
pub struct Embedding<T: Tokens> {
    /// The model id vectors are stored under.
    pub id: String,
    /// The token counter for chunking, output budgets and queue order.
    pub counter: Arc<T::Counter>,
    /// The index scheduler over the server's HTTP API.
    pub scheduler: SchedulerHandle,
    /// The client `search status` reads `GET /status` through.
    pub server: Arc<ServerClient>,
}

/// The link's inbox.
#[derive(Debug)]
pub enum LoaderMsg {
    /// Restart the link sequence after an [`EmbeddingState::Failed`].
    Retry,
    /// Stop the scheduler and wait for its in-flight requests, then reply.
    Shutdown {
        /// Signalled once the scheduler has exited.
        reply: oneshot::Sender<()>,
    },
}

/// The link's handle: the readiness watch and the inbox.
pub struct Link<T: Tokens> {
    /// The link's current state.
    pub state: watch::Receiver<EmbeddingState<T>>,
    /// The link's inbox sender.
    pub tx: mpsc::Sender<LoaderMsg>,
}

/// What the link is built from.
pub struct LinkConfig<T: Tokens> {
    /// The tokenizer source; `load` starts at once.
    pub tokens: T,
    /// The embedding server's client, shared by the scheduler and
    /// `search status`.
    pub server: Arc<ServerClient>,
    /// The model-id `GET /v1/models` must list; another list is fatal.
    pub model_id: String,
    /// The cap on one link attempt — download plus probe wait.
    pub bound: Duration,
    /// The link's busy state for the idle-exit tracker.
    pub busy: watch::Sender<bool>,
    /// Where busy sources register with the idle-exit tracker.
    pub busy_reg: mpsc::Sender<BusySource>,
    /// Fatal errors go to `run` here.
    pub fatal: mpsc::Sender<Error>,
}

/// Spawns the link task; it starts the link sequence at once.
pub fn spawn<T: Tokens>(config: LinkConfig<T>) -> Link<T> {
    let LinkConfig {
        tokens,
        server,
        model_id,
        bound,
        busy,
        busy_reg,
        fatal,
    } = config;
    let (state_tx, state) = watch::channel(EmbeddingState::Absent);
    let (tx, mut rx) = mpsc::channel::<LoaderMsg>(64);
    tokio::spawn(async move {
        let tokens = Arc::new(tokens);
        let mut join: Option<tokio::task::JoinHandle<Attempt<T>>>;
        // The running attempt's cancel sender; `Some` exactly while `join`
        // is. Cancelling reaches a download or a probe sleep; the
        // scheduler an attempt already spawned is stopped the usual way.
        let mut cancel: Option<oneshot::Sender<()>>;
        let mut scheduler: Option<SchedulerHandle> = None;
        let (fetch_tx, mut fetch_rx) = watch::channel(Fetch {
            received: 0,
            total: None,
        });
        macro_rules! attempt {
            () => {{
                let (cancel_tx, cancel_rx) = oneshot::channel();
                cancel = Some(cancel_tx);
                join = Some(tokio::spawn(link(Parts {
                    tokens: tokens.clone(),
                    server: server.clone(),
                    model_id: model_id.clone(),
                    bound,
                    fetch: fetch_tx.clone(),
                    busy_reg: busy_reg.clone(),
                    fatal: fatal.clone(),
                    state: state_tx.clone(),
                    cancel: cancel_rx,
                })));
                busy.send_replace(true);
            }};
        }
        attempt!();
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    None | Some(LoaderMsg::Shutdown { .. }) => {
                        let reply = match msg {
                            Some(LoaderMsg::Shutdown { reply }) => Some(reply),
                            _ => None,
                        };
                        if let Some(cancel) = cancel.take() {
                            let _ = cancel.send(());
                        }
                        if let Some(task) = join.take() {
                            match task.await {
                                Ok(Ok((_embedding, handle))) => stop_scheduler(handle).await,
                                Ok(Err(fail)) => {
                                    if let Some(handle) = fail.scheduler() {
                                        stop_scheduler(handle).await;
                                    }
                                }
                                Err(_) => {}
                            }
                        }
                        if let Some(handle) = scheduler.take() {
                            stop_scheduler(handle).await;
                        }
                        busy.send_replace(false);
                        if let Some(reply) = reply {
                            let _ = reply.send(());
                        }
                        return;
                    }
                    // A Retry is honored only while the state is `Failed`
                    // and no attempt runs.
                    Some(LoaderMsg::Retry) => {
                        if matches!(*state_tx.borrow(), EmbeddingState::Failed(_))
                            && join.is_none()
                        {
                            attempt!();
                        }
                    }
                },
                changed = fetch_rx.changed(), if join.is_some() => {
                    if changed.is_ok() {
                        let fetch = *fetch_rx.borrow();
                        let _ = state_tx.send(EmbeddingState::Downloading {
                            received: fetch.received,
                            total: fetch.total,
                        });
                    }
                }
                result = async { join.as_mut().expect("set").await }, if join.is_some() => {
                    join = None;
                    cancel = None;
                    let mut fatal_message = None;
                    let state = match result {
                        Ok(Ok((embedding, handle))) => {
                            scheduler = Some(handle);
                            EmbeddingState::Ready(embedding)
                        }
                        Ok(Err(LinkFail::Retryable { message, scheduler: handle })) => {
                            if let Some(handle) = handle {
                                stop_scheduler(handle).await;
                            }
                            EmbeddingState::Failed(message)
                        }
                        Ok(Err(LinkFail::Fatal { message, scheduler: handle })) => {
                            if let Some(handle) = handle {
                                stop_scheduler(handle).await;
                            }
                            fatal_message = Some(message.clone());
                            EmbeddingState::Fatal(message)
                        }
                        Ok(Err(LinkFail::Cancelled)) => unreachable!(
                            "the cancel signal fires only on shutdown, which awaits the attempt itself"
                        ),
                        Err(_) => EmbeddingState::Failed("the link task ended".to_owned()),
                    };
                    busy.send_replace(false);
                    let _ = state_tx.send(state);
                    if let Some(message) = fatal_message {
                        let _ = fatal.send(Error::Server(message)).await;
                    }
                }
            }
        }
    });
    Link { state, tx }
}

/// Stops an attempt's scheduler and waits for its exit — its `Stop` lets
/// in-flight requests finish and the task's end closes the channel.
async fn stop_scheduler(handle: SchedulerHandle) {
    let tx = handle.tx();
    handle.stop().await;
    tx.closed().await;
}

/// One link attempt's outcome.
type Attempt<T> = Result<(Arc<Embedding<T>>, SchedulerHandle), LinkFail>;

/// A link failure; `Retryable` publishes `Failed`, `Fatal` also reports
/// to `run`. `Cancelled` is the shutdown path.
enum LinkFail {
    /// Shutdown cancelled the attempt.
    Cancelled,
    /// `Failed`, and a request may retry.
    Retryable {
        /// The error text.
        message: String,
        /// The scheduler the attempt already spawned, to be stopped.
        scheduler: Option<SchedulerHandle>,
    },
    /// `Fatal` — a model-id mismatch or the bound passing — and the
    /// daemon exits.
    Fatal {
        /// The error text.
        message: String,
        /// The scheduler the attempt already spawned, to be stopped.
        scheduler: Option<SchedulerHandle>,
    },
}

impl LinkFail {
    fn scheduler(self) -> Option<SchedulerHandle> {
        match self {
            Self::Cancelled => None,
            Self::Retryable { scheduler, .. } | Self::Fatal { scheduler, .. } => scheduler,
        }
    }
}

fn retryable(message: String) -> LinkFail {
    LinkFail::Retryable {
        message,
        scheduler: None,
    }
}

/// What one link attempt is built from.
struct Parts<T: Tokens> {
    tokens: Arc<T>,
    server: Arc<ServerClient>,
    model_id: String,
    bound: Duration,
    fetch: watch::Sender<Fetch>,
    busy_reg: mpsc::Sender<BusySource>,
    fatal: mpsc::Sender<Error>,
    state: watch::Sender<EmbeddingState<T>>,
    cancel: oneshot::Receiver<()>,
}

/// Publishes the server-reported model state, when it has one, so a
/// client watching progress sees the archive download and the load.
fn publish_server_state<T: Tokens>(
    state: &watch::Sender<EmbeddingState<T>>,
    server: &embed_api::ModelStatus,
) {
    use embed_api::ModelStatus as Server;
    let next = match server {
        Server::Absent => EmbeddingState::Absent,
        Server::Downloading { received, total } => EmbeddingState::Downloading {
            received: *received,
            total: *total,
        },
        Server::Loading { .. } | Server::Ready { .. } => EmbeddingState::Loading,
        Server::Failed { message } | Server::Fatal { message } => {
            EmbeddingState::Failed(message.clone())
        }
    };
    let _ = state.send(next);
}

/// One link attempt: `Tokens::load`, then the probe loop — `GET
/// /v1/models` until it lists [`MODEL_ID`], keeping the state filled from
/// `GET /status` — then the index scheduler. Everything is a task here,
/// so `cancel` may cut any phase short on shutdown.
async fn link<T: Tokens>(parts: Parts<T>) -> Attempt<T> {
    let Parts {
        tokens,
        server,
        model_id,
        bound,
        fetch,
        busy_reg,
        fatal,
        state,
        mut cancel,
    } = parts;
    let deadline = tokio::time::Instant::now() + bound;
    let counter = tokio::select! {
        counter = tokens.load(&fetch) => counter.map_err(retryable)?,
        _ = &mut cancel => return Err(LinkFail::Cancelled),
        () = tokio::time::sleep_until(deadline) => return Err(timed_out(bound)),
    };
    let _ = state.send(EmbeddingState::Loading);
    let counter = Arc::new(counter);
    // The probe loop: `GET /v1/models` answers listed, not-ready or
    // failed; `GET /status` keeps the published state current between
    // probes. A `Failed` answer ends the attempt — the probe itself
    // retried the server's load — and a client's `Retry` re-probes.
    loop {
        let probe = tokio::select! {
            probe = server.probe() => probe,
            _ = &mut cancel => return Err(LinkFail::Cancelled),
            () = tokio::time::sleep_until(deadline) => return Err(timed_out(bound)),
        };
        let after = match probe {
            Ok(Probe::Listed(ids)) => {
                if ids.iter().any(|id| id == &model_id) {
                    break;
                }
                return Err(LinkFail::Fatal {
                    message: format!(
                        "the embedding server lists {ids:?}, this daemon expects {model_id}"
                    ),
                    scheduler: None,
                });
            }
            Ok(Probe::NotReady { retry_after }) => retry_after,
            Ok(Probe::Failed(message)) => return Err(retryable(message)),
            Err(error) => return Err(retryable(error.to_string())),
        };
        if let Ok(report) = server.status().await {
            publish_server_state(&state, &report.model);
        }
        tokio::select! {
            () = tokio::time::sleep(after) => {}
            _ = &mut cancel => return Err(LinkFail::Cancelled),
            () = tokio::time::sleep_until(deadline) => return Err(timed_out(bound)),
        }
    }
    let (sched_busy_tx, sched_busy_rx) = watch::channel(false);
    if busy_reg
        .send(BusySource {
            state: sched_busy_rx,
        })
        .await
        .is_err()
    {
        return Err(retryable("the busy aggregator is gone".to_owned()));
    }
    let scheduler = scheduler::spawn(scheduler::SchedulerConfig {
        embeddings: server.clone(),
        query_input: counter.clone(),
        fatal,
        busy: sched_busy_tx,
    });
    Ok((
        Arc::new(Embedding {
            id: model_id,
            counter,
            scheduler: scheduler.clone(),
            server,
        }),
        scheduler,
    ))
}

/// The link outlived `bound`: fatal — the client gets the error and no
/// `Retry` restarts the attempt.
fn timed_out(bound: Duration) -> LinkFail {
    tracing::error!(bound_s = bound.as_secs(), "the embedding link timed out");
    LinkFail::Fatal {
        message: format!(
            "the embedding link exceeded the {}s bound waiting for the server",
            bound.as_secs()
        ),
        scheduler: None,
    }
}
