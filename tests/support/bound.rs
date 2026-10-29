//! `daemon::run` with `StubTokens` on a `Bound` listener: the daemon of
//! the Linux entry point, in the test's process.

use std::path::Path;
use std::time::Duration;

use repo_search::daemon::listen::Bound;
use repo_search::daemon::server::Endpoint;
use repo_search::daemon::{self, Config, Error};
use repo_search_core::StateDir;
use repo_search_core::model::MODEL_ID;
use tokio::task::JoinHandle;

use super::client;
use super::tokens::StubTokens;

/// `daemon::run` on a `Bound` listener at `socket`. The acquire is
/// awaited before the handle is returned, so the socket exists when the
/// test goes on.
pub async fn spawn(
    socket: &Path,
    state: StateDir,
    endpoint: Endpoint,
    tokens: StubTokens,
    idle_exit: Duration,
) -> JoinHandle<Result<(), Error>> {
    let listener = Bound::acquire(socket).await.expect("the daemon binds");
    tokio::spawn(daemon::run(Config {
        listener,
        idle_exit,
        load_bound: daemon::LOAD_BOUND,
        tokens,
        server: endpoint,
        model_id: MODEL_ID.to_owned(),
        state,
        build: client::build(),
    }))
}
