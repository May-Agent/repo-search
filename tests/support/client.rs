//! The client side of a test daemon connection: the `Hello` the harness's
//! in-process daemons expect, the canned `search` request, and the
//! request/stream helpers every socket-level test shares.
#![allow(dead_code)]

use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use repo_search::client::{Client, Responses};
use repo_search::protocol::{
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
        include_ignored: false,
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
