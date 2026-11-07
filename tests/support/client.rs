//! The client side of a test daemon connection: the `Hello` the harness's
//! in-process daemons expect, the canned `search` request, and the
//! request/stream helpers every socket-level test shares.
#![allow(dead_code)]

use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use repo_search_client::client::{Client, ClientError, Responses};
use repo_search_client::protocol::{
    BuildId, Event, Final, Format, Hello, Request, SearchRequest, WireMode,
};

/// A single event's deadline: a cold model load and a reconcile can take a
/// while, and the tests assert structure, not latency.
pub const TIMEOUT: Duration = Duration::from_secs(180);

/// The identity of the running test binary: what the harness's in-process
/// daemons put in `Config::build` and what the matching client `Hello`
/// carries.
pub fn build() -> BuildId {
    BuildId::of_current_exe().expect("the test binary stats")
}

/// The handshake matching [`build`]: right for the in-process and
/// re-executed test daemons, which run this same binary.
pub fn hello() -> Hello {
    Hello::new(build())
}

/// The handshake matching a daemon running the `search` binary at
/// `program` — the real launchd tests, where the daemon is not this test
/// binary.
pub fn hello_for(program: &Path) -> Hello {
    Hello::new(BuildId::of_file(program).expect("the daemon binary stats"))
}

/// A `search` request for `query` on `worktree`.
pub fn search(worktree: &Path, query: &str) -> Request {
    Request::Search(SearchRequest {
        worktree: worktree.to_path_buf(),
        query: query.to_owned(),
        mode: WireMode::Hybrid,
        paths: vec![],
        langs: vec![],
        kinds: vec![],
        symbol: None,
        budget: 4000,
        files_only: false,
        format: Format::Json,
    })
}

/// Sends `request` and collects events to the `Final` frame.
pub async fn request(socket: &Path, hello: &Hello, request: Request) -> Vec<Event> {
    let client = Client::connect(socket, hello)
        .await
        .expect("connect to the daemon");
    let responses = client.request(request).await.expect("request sends");
    tokio::time::timeout(TIMEOUT, responses.collect::<Vec<_>>())
        .await
        .expect("the reply stream does not stall")
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("every frame decodes")
}

/// The events' `Final`.
pub fn final_of(events: &[Event]) -> &Final {
    match events.last() {
        Some(Event::Final(f)) => f,
        other => panic!("the last event is the Final frame, got {other:?}"),
    }
}

/// Sends `request` and returns the live event stream, uncollected.
pub async fn stream_request(socket: &Path, hello: &Hello, request: Request) -> Responses {
    let client = Client::connect(socket, hello).await.expect("connect");
    client.request(request).await.expect("request sends")
}

/// Connects to a socket launchd has just created and handshakes.
///
/// launchd creates a `Listener` socket's path with `bind` and calls
/// `listen` only after setting the path's owner and mode, so the path can
/// exist before the socket accepts, and no event marks the `listen`. A
/// connect in between is refused without queueing anything — launchd
/// launches nothing for it — so a refused connect is tried again every
/// [`LISTEN_RETRY`] until `within` has passed, and the last refusal is
/// returned then. Every other outcome returns at once.
pub async fn connect_once_listening(
    socket: &Path,
    hello: &Hello,
    within: Duration,
) -> Result<Client, ClientError> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match Client::connect(socket, hello).await {
            Err(ClientError::Connect(error))
                if error.kind() == std::io::ErrorKind::ConnectionRefused
                    && tokio::time::Instant::now() < deadline =>
            {
                tokio::time::sleep(LISTEN_RETRY).await;
            }
            result => return result,
        }
    }
}

/// The pause between [`connect_once_listening`]'s refused attempts: far
/// longer than launchd's few calls between `bind` and `listen`.
const LISTEN_RETRY: Duration = Duration::from_millis(1);

/// Reads `stream` to its `Final` frame, dropping any progress frames.
pub async fn final_event(stream: &mut Responses) -> Final {
    loop {
        if let Event::Final(f) = next_event(stream).await {
            break f;
        }
    }
}

/// The next frame on a live stream within `TIMEOUT`.
pub async fn next_event(stream: &mut Responses) -> Event {
    tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("a frame arrives")
        .expect("the stream stays open")
        .expect("the frame decodes")
}
