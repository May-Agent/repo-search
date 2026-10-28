//! The daemon's wire protocol.
//!
//! One request per connection over a Unix socket, framed as JSON lines: each
//! frame is one compact `serde_json` value, so a frame is exactly one line
//! and plain `nc -U` can talk to the daemon. The client opens with [`Hello`],
//! sends one [`Request`], and reads [`Event::Progress`] frames until the
//! closing [`Event::Final`], which always ends the stream.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use repo_search_core::search::Mode;
use repo_search_core::store::Phase;
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tokio_util::codec::{Framed, LinesCodec};

/// Wire format version, bumped whenever a wire type changes.
pub const PROTOCOL: u32 = 3;

/// The largest frame a peer may send, in bytes.
pub const MAX_FRAME: usize = 64 << 20;

/// A framed daemon connection: one compact JSON value per line.
pub type Connection = Framed<UnixStream, LinesCodec>;

/// Frame a socket stream with the line codec bounded by [`MAX_FRAME`].
pub fn framed(stream: UnixStream) -> Connection {
    Framed::new(stream, LinesCodec::new_with_max_length(MAX_FRAME))
}

/// The identity of the executable file a process runs. A binary rebuilt or
/// reinstalled at the same path has a different one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildId {
    /// The filesystem the file lives on (`st_dev`).
    pub dev: u64,
    /// The file's inode (`st_ino`).
    pub ino: u64,
    /// The file's size in bytes (`st_size`).
    pub size: u64,
    /// The file's modification time in nanoseconds (`st_mtime`,
    /// `st_mtime_nsec`).
    pub mtime_ns: i128,
}

impl BuildId {
    /// The identity of the file at `path`, from `stat`.
    pub fn of_file(path: &Path) -> io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path)?;
        Ok(Self {
            dev: meta.dev(),
            ino: meta.ino(),
            size: meta.size(),
            mtime_ns: meta.mtime() as i128 * 1_000_000_000 + i128::from(meta.mtime_nsec()),
        })
    }

    /// The identity of the running executable, at its canonical path.
    pub fn of_current_exe() -> io::Result<Self> {
        Self::of_file(&std::env::current_exe()?.canonicalize()?)
    }
}

impl fmt::Display for BuildId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.ino, self.mtime_ns)
    }
}

/// The client's first frame. The daemon answers with its own `Hello` when
/// all three fields are equal, and with `Final::Error` of kind
/// [`ErrorKind::ProtocolMismatch`] on any difference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    /// The sender's [`PROTOCOL`].
    pub protocol: u32,
    /// The sender's binary version (`CARGO_PKG_VERSION`).
    pub version: String,
    /// The sender's binary file's identity.
    pub build: BuildId,
}

impl Hello {
    /// This process's handshake: this [`PROTOCOL`], this binary's version
    /// and `build`.
    pub fn new(build: BuildId) -> Self {
        Self {
            protocol: PROTOCOL,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            build,
        }
    }
}

/// The connection's one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Answer a query.
    Search(SearchRequest),
    /// Report index, model and daemon state.
    Status {
        /// Absolute worktree root (`git rev-parse --show-toplevel`).
        worktree: PathBuf,
    },
    /// Reconcile now.
    Index {
        /// Absolute worktree root.
        worktree: PathBuf,
        /// Block until every chunk has a vector.
        wait: bool,
    },
}

/// A `search <query>` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// Absolute worktree root (`git rev-parse --show-toplevel`).
    pub worktree: PathBuf,
    /// Free-text query.
    pub query: String,
    /// Retrieval channels.
    pub mode: WireMode,
    /// Path globs; empty means unrestricted.
    pub paths: Vec<String>,
    /// Language names; empty means unrestricted.
    pub langs: Vec<String>,
    /// Syntactic kinds; empty means unrestricted.
    pub kinds: Vec<String>,
    /// Symbol path glob such as `Environment::*`.
    pub symbol: Option<String>,
    /// Output token budget.
    pub budget: u32,
    /// Ranked file list only.
    pub files_only: bool,
    /// Include files `.gitignore` excludes.
    pub include_ignored: bool,
    /// How `Answer::output` is rendered.
    pub format: Format,
}

/// `search::Mode` on the wire; core types carry no serde derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireMode {
    /// BM25 and vector KNN, fused.
    Hybrid,
    /// FTS only.
    Lexical,
    /// Vector KNN only.
    Semantic,
}

impl From<Mode> for WireMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Hybrid => Self::Hybrid,
            Mode::Lexical => Self::Lexical,
            Mode::Semantic => Self::Semantic,
        }
    }
}

impl From<WireMode> for Mode {
    fn from(mode: WireMode) -> Self {
        match mode {
            WireMode::Hybrid => Self::Hybrid,
            WireMode::Lexical => Self::Lexical,
            WireMode::Semantic => Self::Semantic,
        }
    }
}

/// `store::Phase` on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WirePhase {
    /// The first pass over the worktree has not finished.
    FirstIndex,
    /// The index has been complete once and is kept up to date.
    Maintained,
}

impl From<Phase> for WirePhase {
    fn from(phase: Phase) -> Self {
        match phase {
            Phase::FirstIndex => Self::FirstIndex,
            Phase::Maintained => Self::Maintained,
        }
    }
}

impl From<WirePhase> for Phase {
    fn from(phase: WirePhase) -> Self {
        match phase {
            WirePhase::FirstIndex => Self::FirstIndex,
            WirePhase::Maintained => Self::Maintained,
        }
    }
}

/// How the daemon renders `Answer::output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Format {
    /// `Results::to_text()`.
    Text,
    /// `Results::to_json()`.
    Json,
}

/// One server→client frame.
///
/// Adjacent tagging: `Progress` and `Final` are themselves internally tagged,
/// so an internal tag here would emit a second `type` key that serde refuses
/// to deserialize back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum Event {
    /// Work is proceeding; any number may precede the `Final`.
    Progress(Progress),
    /// The reply's last frame; the daemon closes the connection after it.
    Final(Final),
}

/// An in-progress report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Progress {
    /// The model archive is downloading.
    ModelDownload {
        /// Bytes received so far.
        received: u64,
        /// The server's declared length, when known.
        total: Option<u64>,
    },
    /// The model is being loaded.
    ModelLoading,
    /// The freshness barrier is running.
    Reconciling,
    /// Vector coverage; `index --wait` only, at most every 250 ms.
    Coverage {
        /// Chunks with a vector.
        with_vector: u64,
        /// Chunks in total.
        total: u64,
    },
}

/// The reply's last frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Final {
    /// A rendered query answer.
    Answer(Answer),
    /// A `status` reply.
    Status(StatusReport),
    /// An `index --wait` reply once every chunk has a vector.
    Indexed {
        /// Chunks with a vector.
        with_vector: u64,
        /// Chunks in total.
        total: u64,
    },
    /// The request failed.
    Error(ErrorReport),
}

/// A query's rendered answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    /// `Results::to_text()` or `Results::to_json()`, per `SearchRequest::format`.
    pub output: String,
    /// Whether the result list was non-empty (drives the grep exit code).
    pub has_results: bool,
    /// Where the query spent its time.
    pub stats: QueryStats,
}

/// Timing and preemption counters of one query, in microseconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryStats {
    /// The freshness barrier's duration.
    pub barrier_us: u64,
    /// Time the query waited on the indexing thread in flight.
    pub indexing_wait_us: u64,
    /// Index jobs that overtook the query; must be 0.
    pub indexing_overtaken: u32,
    /// Lexical candidates the query embedded itself.
    pub embedded_on_the_spot: u64,
    /// End-to-end duration.
    pub total_us: u64,
}

/// A failed request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorReport {
    /// Which failure; the client's handling keys off it.
    pub kind: ErrorKind,
    /// Human-readable cause.
    pub message: String,
}

/// The failure kind of an [`ErrorReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ErrorKind {
    /// `Hello` protocol or version differed; the daemon shuts down so the
    /// next connection launches the binary on disk.
    ProtocolMismatch,
    /// The request could not be understood.
    BadRequest,
    /// The worktree path is not a repository root.
    NotAWorktree,
    /// Model download or load failed.
    Model,
    /// The query failed.
    Search,
    /// The daemon is exiting.
    ShuttingDown,
    /// The serialized `Answer` exceeded [`MAX_FRAME`].
    TooLarge,
    /// Any other failure.
    Internal,
    /// The daemon could not start; it exits after answering.
    Startup,
}

/// A `status` reply for one worktree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusReport {
    /// The queried worktree.
    pub worktree: PathBuf,
    /// `None`: never indexed (no index file under the state dir).
    pub index: Option<IndexStatus>,
    /// Where the model is.
    pub model: ModelStatus,
    /// The daemon itself.
    pub daemon: DaemonStatus,
}

/// One worktree's index state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexStatus {
    /// First index or maintained.
    pub phase: WirePhase,
    /// Chunks in total.
    pub chunks_total: u64,
    /// Chunks with a vector.
    pub chunks_with_vector: u64,
    /// Stage-1 items waiting for an embedding thread.
    pub queued_stage1: u64,
    /// Stage-2 items waiting for an embedding thread.
    pub queued_background: u64,
    /// Items an embedding thread is working on.
    pub in_flight: u64,
}

/// Where the model is on its way to readiness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelStatus {
    /// The archive is downloading.
    Downloading {
        /// Bytes received so far.
        received: u64,
        /// The server's declared length, when known.
        total: Option<u64>,
    },
    /// The model is being loaded.
    Loading,
    /// Ready to embed.
    Ready {
        /// The model build's identity (the archive hash).
        model_id: String,
    },
    /// Download or load failed; the next request retries.
    Failed {
        /// Human-readable cause.
        message: String,
    },
}

/// The daemon process itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonStatus {
    /// The daemon binary's version.
    pub version: String,
    /// Its process id.
    pub pid: u32,
    /// Seconds since it started.
    pub uptime_s: u64,
    /// Worktrees it is serving.
    pub worktrees: u32,
    /// The GPU embedding instance.
    pub gpu: GpuStatus,
}

/// The GPU embedding instance's state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GpuStatus {
    /// No GPU instance exists.
    Absent {
        /// Why it cannot be created, when the check has run.
        blocked: Option<String>,
    },
    /// Being created (a multi-second cold start).
    Loading,
    /// Embedding stage-1 index work.
    Running,
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures_util::stream::StreamExt;
    use tokio::io::AsyncWriteExt;
    use tokio_util::codec::LinesCodecError;

    fn round_trip<T>(value: &T)
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let line = serde_json::to_string(value).unwrap();
        assert!(!line.contains('\n'), "frame is not one line: {line}");
        let back: T = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("{line} does not parse back: {e}"));
        assert_eq!(*value, back, "frame {line} did not round-trip");
    }

    fn answer() -> Answer {
        Answer {
            output: "src/lib.rs:1: fn main".into(),
            has_results: true,
            stats: QueryStats {
                barrier_us: 12,
                indexing_wait_us: 34,
                indexing_overtaken: 0,
                embedded_on_the_spot: 7,
                total_us: 900,
            },
        }
    }

    fn status_report() -> StatusReport {
        StatusReport {
            worktree: PathBuf::from("/repo"),
            index: Some(IndexStatus {
                phase: WirePhase::FirstIndex,
                chunks_total: 1000,
                chunks_with_vector: 250,
                queued_stage1: 600,
                queued_background: 0,
                in_flight: 8,
            }),
            model: ModelStatus::Downloading {
                received: 10,
                total: Some(20),
            },
            daemon: DaemonStatus {
                version: "0.0.0".into(),
                pid: 4242,
                uptime_s: 3,
                worktrees: 1,
                gpu: GpuStatus::Absent {
                    blocked: Some("not on mains".into()),
                },
            },
        }
    }

    #[test]
    fn every_message_round_trips() {
        round_trip(&Hello {
            protocol: PROTOCOL,
            version: "0.0.0".into(),
            build: BuildId {
                dev: 1,
                ino: 2,
                size: 3,
                mtime_ns: 4,
            },
        });
        round_trip(&Request::Search(SearchRequest {
            worktree: PathBuf::from("/repo"),
            query: "where are errors handled".into(),
            mode: WireMode::Hybrid,
            paths: vec!["src/**".into()],
            langs: vec!["rust".into()],
            kinds: vec!["fn".into(), "struct".into()],
            symbol: Some("Environment::*".into()),
            budget: 4000,
            files_only: false,
            include_ignored: true,
            format: Format::Json,
        }));
        round_trip(&Request::Status {
            worktree: PathBuf::from("/repo"),
        });
        round_trip(&Request::Index {
            worktree: PathBuf::from("/repo"),
            wait: true,
        });
        for mode in [WireMode::Hybrid, WireMode::Lexical, WireMode::Semantic] {
            round_trip(&mode);
        }
        for format in [Format::Text, Format::Json] {
            round_trip(&format);
        }
        for phase in [WirePhase::FirstIndex, WirePhase::Maintained] {
            round_trip(&phase);
        }
        for progress in [
            Progress::ModelDownload {
                received: 5,
                total: None,
            },
            Progress::ModelLoading,
            Progress::Reconciling,
            Progress::Coverage {
                with_vector: 3,
                total: 9,
            },
        ] {
            round_trip(&Event::Progress(progress));
        }
        for final_ in [
            Final::Answer(answer()),
            Final::Status(status_report()),
            Final::Indexed {
                with_vector: 9,
                total: 9,
            },
            Final::Error(ErrorReport {
                kind: ErrorKind::ProtocolMismatch,
                message: "daemon is 0.0.0 protocol 1".into(),
            }),
        ] {
            round_trip(&Event::Final(final_));
        }
        for kind in [
            ErrorKind::ProtocolMismatch,
            ErrorKind::BadRequest,
            ErrorKind::NotAWorktree,
            ErrorKind::Model,
            ErrorKind::Search,
            ErrorKind::ShuttingDown,
            ErrorKind::TooLarge,
            ErrorKind::Internal,
            ErrorKind::Startup,
        ] {
            round_trip(&ErrorReport {
                kind,
                message: "x".into(),
            });
        }
        for model in [
            ModelStatus::Downloading {
                received: 0,
                total: None,
            },
            ModelStatus::Loading,
            ModelStatus::Ready {
                model_id: "ab12".into(),
            },
            ModelStatus::Failed {
                message: "sha256 mismatch".into(),
            },
        ] {
            round_trip(&model);
        }
        for gpu in [
            GpuStatus::Absent { blocked: None },
            GpuStatus::Loading,
            GpuStatus::Running,
        ] {
            round_trip(&gpu);
        }
    }

    #[test]
    fn hello_carries_the_build() {
        let build = BuildId {
            dev: 1,
            ino: 2,
            size: 3,
            mtime_ns: 4,
        };
        let hello = Hello::new(build);
        assert_eq!(hello.protocol, PROTOCOL);
        assert_eq!(hello.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(hello.build, build);
        round_trip(&hello);
    }

    #[test]
    fn build_id_tracks_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bin");
        std::fs::write(&path, b"aaaa").expect("the file is written");
        let first = BuildId::of_file(&path).expect("the file stats");
        assert_eq!(
            first,
            BuildId::of_file(&path).expect("the file stats again"),
            "two stats of an unchanged file are one identity"
        );
        // A rename replacement, as installers do: same size, new inode and
        // mtime.
        let replacement = dir.path().join("bin.new");
        std::fs::write(&replacement, b"bbbb").expect("the replacement is written");
        std::fs::rename(&replacement, &path).expect("the replacement renames over");
        assert_ne!(
            first,
            BuildId::of_file(&path).expect("the replacement stats"),
            "a same-size rename replacement is a different build"
        );
    }

    #[test]
    fn wire_mirrors_convert_both_ways() {
        for mode in [Mode::Hybrid, Mode::Lexical, Mode::Semantic] {
            assert_eq!(Mode::from(WireMode::from(mode)), mode);
        }
        for phase in [Phase::FirstIndex, Phase::Maintained] {
            assert_eq!(Phase::from(WirePhase::from(phase)), phase);
        }
    }

    #[tokio::test]
    async fn a_frame_over_max_is_refused() {
        let (peer, mut writer) = UnixStream::pair().unwrap();
        let send = tokio::spawn(async move {
            let _ = writer.write_all(&vec![b'x'; MAX_FRAME + 1]).await;
        });
        let mut connection = framed(peer);
        assert!(matches!(
            connection.next().await,
            Some(Err(LinesCodecError::MaxLineLengthExceeded))
        ));
        // The peer may still be writing into the socket buffer.
        send.abort();
    }
}
