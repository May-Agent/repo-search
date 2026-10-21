//! The library client of the `search` daemon.
//!
//! [`Client::connect`] performs the `Hello` handshake; [`Client::request`]
//! sends the connection's one request and returns the streamed reply. The
//! caller ensures the agent is installed and current; the client knows
//! nothing about launchd.

use std::io;
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::sink::SinkExt;
use futures_util::stream::{Stream, StreamExt};
use serde::Serialize;
use thiserror::Error;
use tokio::net::UnixStream;
use tokio_util::codec::LinesCodecError;

use crate::protocol::{self, Connection, ErrorKind, ErrorReport, Event, Final, Hello, Request};

/// One wire message as a frame line: compact JSON, always serializable.
fn line(value: &impl Serialize) -> String {
    serde_json::to_string(value).expect("wire types serialize")
}

fn codec_error(err: LinesCodecError) -> ClientError {
    match err {
        LinesCodecError::MaxLineLengthExceeded => ClientError::TooLarge,
        LinesCodecError::Io(err) => ClientError::Io(err),
    }
}

/// A connection to the daemon after a successful handshake.
pub struct Client {
    framed: Connection,
}

impl Client {
    /// Connect to the socket at `path` and exchange `hello` for the
    /// daemon's own.
    ///
    /// A [`ErrorKind::ProtocolMismatch`] refusal is retried once on a fresh
    /// connection: the mismatched daemon shuts down, so the retry reaches the
    /// binary that is on disk now.
    pub async fn connect(socket: &Path, hello: &Hello) -> Result<Self, ClientError> {
        match Self::handshake(socket, hello).await {
            Err(ClientError::Refused(report)) if report.kind == ErrorKind::ProtocolMismatch => {
                Self::handshake(socket, hello).await
            }
            result => result,
        }
    }

    /// Send `Hello`, then accept only the peer's identical `Hello` back; a
    /// `Final` frame means the daemon refused the handshake.
    async fn handshake(socket: &Path, hello: &Hello) -> Result<Self, ClientError> {
        let stream = UnixStream::connect(socket)
            .await
            .map_err(ClientError::Connect)?;
        let mut framed = protocol::framed(stream);
        framed.send(line(hello)).await.map_err(codec_error)?;
        let line = framed
            .next()
            .await
            .ok_or(ClientError::Closed)?
            .map_err(codec_error)?;
        match serde_json::from_str::<Final>(&line) {
            Ok(Final::Error(report)) => Err(ClientError::Refused(report)),
            Ok(_) => Err(ClientError::Protocol(
                "the handshake was answered with a non-error final frame",
            )),
            Err(_) => match serde_json::from_str::<Hello>(&line) {
                Ok(peer) if peer == *hello => Ok(Self { framed }),
                Ok(_) => Err(ClientError::Protocol(
                    "the peer's Hello does not match the client's",
                )),
                Err(err) => Err(ClientError::Frame(err)),
            },
        }
    }

    /// Send the connection's one request; the returned stream yields
    /// `Progress` frames and then the closing `Final`.
    pub async fn request(self, request: Request) -> Result<Responses, ClientError> {
        let mut framed = self.framed;
        framed.send(line(&request)).await.map_err(codec_error)?;
        Ok(Responses {
            framed,
            done: false,
        })
    }
}

/// The daemon's streamed reply; ends after the `Final` frame.
pub struct Responses {
    framed: Connection,
    done: bool,
}

impl Stream for Responses {
    type Item = Result<Event, ClientError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        let item = match Pin::new(&mut self.framed).poll_next(cx) {
            Poll::Ready(Some(Ok(line))) => match serde_json::from_str::<Event>(&line) {
                Ok(event) => {
                    self.done = matches!(event, Event::Final(_));
                    Ok(event)
                }
                Err(err) => Err(ClientError::Frame(err)),
            },
            Poll::Ready(Some(Err(err))) => Err(codec_error(err)),
            Poll::Ready(None) => Err(ClientError::Closed),
            Poll::Pending => return Poll::Pending,
        };
        if item.is_err() {
            self.done = true;
        }
        Poll::Ready(Some(item))
    }
}

/// Failure modes of [`Client::connect`], [`Client::request`] and the
/// [`Responses`] stream.
#[derive(Debug, Error)]
pub enum ClientError {
    /// The `UnixStream::connect` itself failed.
    #[error("cannot connect to the daemon socket: {0}")]
    Connect(io::Error),
    /// Socket or stream I/O after the connect.
    #[error("daemon connection failed: {0}")]
    Io(#[from] io::Error),
    /// The daemon refused the handshake or a request.
    #[error("daemon reported an error: {0:?}")]
    Refused(ErrorReport),
    /// A received frame exceeded the protocol's maximum frame length.
    #[error("a frame from the daemon exceeds the maximum frame length")]
    TooLarge,
    /// The connection closed before the reply's final frame.
    #[error("the daemon closed the connection mid-reply")]
    Closed,
    /// The peer sent a frame out of order.
    #[error("protocol violation: {0}")]
    Protocol(&'static str),
    /// A received frame was not valid protocol JSON.
    #[error("invalid frame from the daemon: {0}")]
    Frame(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use crate::protocol::{BuildId, Format, PROTOCOL, Progress, StatusReport, WireMode};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    /// The test client's handshake; the test servers echo it back.
    fn hello() -> Hello {
        Hello::new(BuildId {
            dev: 1,
            ino: 2,
            size: 3,
            mtime_ns: 4,
        })
    }

    /// A raw test server: accepts one connection, checks the client's Hello,
    /// answers it, then replies to the request with `events` and closes.
    async fn serve(listener: UnixListener, events: Vec<Event>) -> (Hello, Request) {
        let (stream, _) = listener.accept().await.unwrap();
        let (read, mut write) = stream.into_split();
        let mut lines = BufReader::new(read);

        let mut buf = String::new();
        lines.read_line(&mut buf).await.unwrap();
        let hello: Hello = serde_json::from_str(&buf).unwrap();
        let reply = line(&hello);
        write.write_all(reply.as_bytes()).await.unwrap();
        write.write_all(b"\n").await.unwrap();

        buf.clear();
        lines.read_line(&mut buf).await.unwrap();
        let request: Request = serde_json::from_str(&buf).unwrap();
        for event in events {
            write.write_all(line(&event).as_bytes()).await.unwrap();
            write.write_all(b"\n").await.unwrap();
        }
        (hello, request)
    }

    fn status_report() -> StatusReport {
        StatusReport {
            worktree: PathBuf::from("/repo"),
            index: None,
            model: crate::protocol::ModelStatus::Ready {
                model_id: "ab12".into(),
            },
            daemon: crate::protocol::DaemonStatus {
                version: env!("CARGO_PKG_VERSION").into(),
                pid: 4242,
                uptime_s: 3,
                worktrees: 1,
                gpu: crate::protocol::GpuStatus::Absent { blocked: None },
            },
        }
    }

    #[tokio::test]
    async fn handshake_then_request_streams_events() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("daemon.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let reply = Event::Final(Final::Status(status_report()));
        let server = tokio::spawn(serve(
            listener,
            vec![Event::Progress(Progress::Reconciling), reply.clone()],
        ));

        let client = Client::connect(&socket, &hello()).await.unwrap();
        let request = Request::Search(crate::protocol::SearchRequest {
            worktree: PathBuf::from("/repo"),
            query: "where are errors handled".into(),
            mode: WireMode::Lexical,
            paths: vec![],
            langs: vec!["rust".into()],
            kinds: vec![],
            symbol: None,
            budget: 4000,
            files_only: false,
            include_ignored: false,
            format: Format::Text,
        });
        let responses = client.request(request.clone()).await.unwrap();
        let events: Vec<Event> = responses.map(|e| e.unwrap()).collect().await;

        assert_eq!(events, vec![Event::Progress(Progress::Reconciling), reply]);
        let (hello, got) = server.await.unwrap();
        assert_eq!(hello.protocol, PROTOCOL);
        assert_eq!(hello.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(got, request);
    }

    #[tokio::test]
    async fn protocol_mismatch_is_retried_once_then_returned() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("daemon.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let refusal = line(&Final::Error(ErrorReport {
            kind: ErrorKind::ProtocolMismatch,
            message: "daemon is protocol 2".into(),
        }));
        let (counted, connections) = std::sync::mpsc::channel();
        let server = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                counted.send(()).unwrap();
                let (read, mut write) = stream.into_split();
                let mut buf = String::new();
                BufReader::new(read).read_line(&mut buf).await.unwrap();
                write.write_all(refusal.as_bytes()).await.unwrap();
                write.write_all(b"\n").await.unwrap();
            }
        });

        let err = match Client::connect(&socket, &hello()).await {
            Ok(_) => panic!("the handshake should have been refused"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            ClientError::Refused(ErrorReport {
                kind: ErrorKind::ProtocolMismatch,
                ..
            })
        ));
        // The refusal was retried exactly once on a fresh connection.
        connections.recv().unwrap();
        connections.recv().unwrap();
        assert!(
            connections.try_recv().is_err(),
            "a third connection was made"
        );
        server.abort();
    }
}
