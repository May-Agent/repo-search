//! Rendering the daemon's `Final` to stdout and mapping failures to the
//! user's `search: <message>` line. The wire types are not the user
//! contract: [`StatusOutput`] and [`IndexedOutput`] define the `--json`
//! shapes, and `templates/status.txt` the status text — the same
//! separation daemon-11 drew for the search answer.

use std::io;
use std::path::{Path, PathBuf};

use askama::Template;
use repo_search_core::search::Coverage;
use serde::Serialize;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::{Exit, Terminal, mb, tell};
use crate::client::ClientError;
use crate::protocol::{
    Answer, ErrorKind, ErrorReport, GpuStatus, IndexStatus, ModelStatus, StatusReport, WirePhase,
};

/// Writes `bytes` to stdout and returns `exit`. `BrokenPipe` means the
/// reader stopped reading (`search q | head`), so the exit code is the
/// reply's and nothing more is written; any other stdout error is
/// reported on stderr and exits 2.
async fn out<O, E>(term: &mut Terminal<O, E>, bytes: &[u8], exit: Exit) -> Exit
where
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    let result = async {
        term.stdout.write_all(bytes).await?;
        term.stdout.flush().await
    };
    match result.await {
        Ok(()) => exit,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => exit,
        Err(error) => {
            tell(
                &mut term.stderr,
                &format!("cannot write the answer: {error}"),
            )
            .await;
            Exit::Error
        }
    }
}

/// A search answer to stdout: the daemon rendered `Answer::output` — the
/// client does not parse or reformat it — and the grep exit code is the
/// answer's. Core's JSON ends without a newline, so `--json` adds one.
pub async fn answer<O, E>(answer: &Answer, json: bool, term: &mut Terminal<O, E>) -> Exit
where
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    let exit = if answer.has_results {
        Exit::Ok
    } else {
        Exit::NoMatches
    };
    let mut bytes = answer.output.clone().into_bytes();
    if json {
        bytes.push(b'\n');
    }
    out(term, &bytes, exit).await
}

/// The `search status` JSON and text shape.
#[derive(Debug, Serialize, Template)]
#[template(path = "status.txt", escape = "none")]
pub struct StatusOutput {
    /// The queried worktree.
    pub worktree: PathBuf,
    /// `None`: never indexed; the text reads `none;` and JSON `null`.
    pub index: Option<IndexOutput>,
    /// Where the model is.
    pub model: ModelOutput,
    /// The GPU embedding instance.
    pub gpu: GpuOutput,
    /// The daemon process.
    pub daemon: DaemonOutput,
}

/// One worktree's index state.
#[derive(Debug, Serialize)]
pub struct IndexOutput {
    /// `first_index` or `maintained`.
    pub phase: IndexPhase,
    /// Chunks in total.
    pub chunks: u64,
    /// Chunks with a vector.
    pub with_vector: u64,
    /// `with_vector / chunks`, serialized as a fraction.
    pub coverage: Coverage,
    /// Embedding work outstanding.
    pub pending: Pending,
}

/// `phase` in the `status` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexPhase {
    /// The first pass over the worktree has not finished.
    FirstIndex,
    /// The index has been complete once and is kept up to date.
    Maintained,
}

/// Embedding work outstanding.
#[derive(Debug, Serialize)]
pub struct Pending {
    /// Stage-1 items waiting for an embedding thread.
    pub stage1: u64,
    /// Stage-2 items waiting for an embedding thread.
    pub background: u64,
    /// Items an embedding thread is working on.
    pub in_flight: u64,
}

/// `model`, tagged by `state`.
#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ModelOutput {
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
        id: String,
    },
    /// Download or load failed; the next request retries.
    Failed {
        /// Human-readable cause.
        message: String,
    },
}

/// `gpu`, tagged by `state`.
#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GpuOutput {
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

/// The daemon process itself.
#[derive(Debug, Serialize)]
pub struct DaemonOutput {
    /// Its process id.
    pub pid: u32,
    /// The daemon binary's version.
    pub version: String,
    /// Seconds since it started.
    pub uptime_s: u64,
    /// Worktrees it is serving.
    pub worktrees: u32,
}

impl IndexPhase {
    /// The status text's phase words.
    fn text(self) -> &'static str {
        match self {
            Self::FirstIndex => "first index",
            Self::Maintained => "maintained",
        }
    }
}

impl StatusOutput {
    /// The `index` line, or `pending`'s contents.
    pub fn index_line(&self) -> String {
        match &self.index {
            Some(index) => format!(
                "{}, {} of {} chunks embedded ({}%)",
                index.phase.text(),
                index.with_vector,
                index.chunks,
                index.coverage.percent()
            ),
            None => "none; the first search indexes this worktree".to_owned(),
        }
    }

    /// The `pending` line; `index` is `Some` wherever the template calls it.
    pub fn pending_line(&self) -> String {
        let pending = &self
            .index
            .as_ref()
            .expect("the template guards this call on `index`")
            .pending;
        format!(
            "{} stage 1, {} background, {} in flight",
            pending.stage1, pending.background, pending.in_flight
        )
    }

    /// The `model` line.
    pub fn model_line(&self) -> String {
        match &self.model {
            ModelOutput::Downloading { received, total } => match total {
                Some(total) => format!("downloading, {} of {} MB", mb(*received), mb(*total)),
                None => format!("downloading, {} MB", mb(*received)),
            },
            ModelOutput::Loading => "loading".to_owned(),
            ModelOutput::Ready { id } => format!("ready ({})", short_id(id)),
            ModelOutput::Failed { message } => format!("failed: {message}"),
        }
    }

    /// The `gpu` line.
    pub fn gpu_line(&self) -> String {
        match &self.gpu {
            GpuOutput::Absent { blocked } => match blocked {
                Some(blocked) => format!("absent: {blocked}"),
                None => "absent".to_owned(),
            },
            GpuOutput::Loading => "loading".to_owned(),
            GpuOutput::Running => "running".to_owned(),
        }
    }

    /// The `daemon` line.
    pub fn daemon_line(&self) -> String {
        format!(
            "pid {}, version {}, up {} s, {} worktrees",
            self.daemon.pid, self.daemon.version, self.daemon.uptime_s, self.daemon.worktrees
        )
    }
}

/// The first 12 characters of the model id.
fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}

impl From<WirePhase> for IndexPhase {
    fn from(phase: WirePhase) -> Self {
        match phase {
            WirePhase::FirstIndex => Self::FirstIndex,
            WirePhase::Maintained => Self::Maintained,
        }
    }
}

impl From<IndexStatus> for IndexOutput {
    fn from(status: IndexStatus) -> Self {
        Self {
            phase: status.phase.into(),
            chunks: status.chunks_total,
            with_vector: status.chunks_with_vector,
            coverage: Coverage::new(status.chunks_with_vector, status.chunks_total),
            pending: Pending {
                stage1: status.queued_stage1,
                background: status.queued_background,
                in_flight: status.in_flight,
            },
        }
    }
}

impl From<ModelStatus> for ModelOutput {
    fn from(status: ModelStatus) -> Self {
        match status {
            ModelStatus::Downloading { received, total } => Self::Downloading { received, total },
            ModelStatus::Loading => Self::Loading,
            ModelStatus::Ready { model_id } => Self::Ready { id: model_id },
            ModelStatus::Failed { message } => Self::Failed { message },
        }
    }
}

impl From<GpuStatus> for GpuOutput {
    fn from(status: GpuStatus) -> Self {
        match status {
            GpuStatus::Absent { blocked } => Self::Absent { blocked },
            GpuStatus::Loading => Self::Loading,
            GpuStatus::Running => Self::Running,
        }
    }
}

impl From<StatusReport> for StatusOutput {
    fn from(report: StatusReport) -> Self {
        Self {
            worktree: report.worktree,
            index: report.index.map(IndexOutput::from),
            model: report.model.into(),
            gpu: report.daemon.gpu.into(),
            daemon: DaemonOutput {
                pid: report.daemon.pid,
                version: report.daemon.version,
                uptime_s: report.daemon.uptime_s,
                worktrees: report.daemon.worktrees,
            },
        }
    }
}

/// The `search index` JSON shape.
#[derive(Debug, Serialize)]
pub struct IndexedOutput {
    /// The reconciled worktree.
    pub worktree: PathBuf,
    /// Chunks in total.
    pub chunks: u64,
    /// Chunks with a vector.
    pub with_vector: u64,
    /// `with_vector / chunks`, serialized as a fraction.
    pub coverage: Coverage,
}

/// The `status.txt` rendering of `output`. Askama drops the template
/// file's trailing newline; the reply's last line still ends with one.
fn status_text(output: &StatusOutput) -> String {
    let mut text = output.render().expect("the status template renders");
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// A `search status` reply to stdout: the JSON shape or the
/// `status.txt` rendering. Exits 0.
pub async fn status<O, E>(report: &StatusReport, json: bool, term: &mut Terminal<O, E>) -> Exit
where
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    let output = StatusOutput::from(report.clone());
    let text = if json {
        format!(
            "{}\n",
            serde_json::to_string(&output).expect("StatusOutput serializes")
        )
    } else {
        status_text(&output)
    };
    out(term, text.as_bytes(), Exit::Ok).await
}

/// An `index` reply to stdout: `indexed N chunks, M embedded (P%)`, or
/// the [`IndexedOutput`] JSON. Exits 0.
pub async fn indexed<O, E>(
    worktree: &Path,
    with_vector: u64,
    total: u64,
    json: bool,
    term: &mut Terminal<O, E>,
) -> Exit
where
    O: AsyncWrite + Unpin,
    E: AsyncWrite + Unpin,
{
    let coverage = Coverage::new(with_vector, total);
    let text = if json {
        let output = IndexedOutput {
            worktree: worktree.to_path_buf(),
            chunks: total,
            with_vector,
            coverage,
        };
        format!(
            "{}\n",
            serde_json::to_string(&output).expect("IndexedOutput serializes")
        )
    } else {
        format!(
            "indexed {total} chunks, {with_vector} embedded ({}%)\n",
            coverage.percent()
        )
    };
    out(term, text.as_bytes(), Exit::Ok).await
}

/// The `search: <message>` for a daemon-reported error — `Final::Error`,
/// or [`ClientError::Refused`] from the handshake. `log` is the daemon's
/// `StandardErrorPath`.
pub fn daemon_error(report: &ErrorReport, log: &Path) -> String {
    match report.kind {
        ErrorKind::ProtocolMismatch => format!(
            "the daemon did not restart into this build: {}; see {}",
            report.message,
            log.display()
        ),
        ErrorKind::BadRequest => format!("the daemon rejected the request: {}", report.message),
        ErrorKind::NotAWorktree => report.message.clone(),
        ErrorKind::Model => format!("the model is unavailable: {}", report.message),
        ErrorKind::Search => format!("the search failed: {}", report.message),
        ErrorKind::ShuttingDown => {
            "the daemon shut down before answering; run the command again".to_owned()
        }
        ErrorKind::TooLarge => {
            "the answer exceeds the 64 MiB frame limit; lower --budget".to_owned()
        }
        ErrorKind::Internal => {
            format!("daemon error: {}; see {}", report.message, log.display())
        }
        ErrorKind::Startup => format!(
            "the daemon failed to start: {}; see {}",
            report.message,
            log.display()
        ),
    }
}

/// The `search: <message>` for a client-side failure.
pub fn client_error(error: &ClientError, socket: &Path, log: &Path) -> String {
    match error {
        ClientError::Connect(error) => format!(
            "cannot connect to {}: {error}; see {}",
            socket.display(),
            log.display()
        ),
        ClientError::Io(error) => format!("the daemon connection failed: {error}"),
        ClientError::Refused(report) => daemon_error(report, log),
        ClientError::Closed => format!(
            "the daemon closed the connection before answering; see {}",
            log.display()
        ),
        ClientError::TooLarge => {
            "a reply from the daemon exceeds the 64 MiB frame limit".to_owned()
        }
        ClientError::Protocol(_) | ClientError::Frame(_) => format!("protocol error: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocol::{DaemonStatus, IndexStatus};

    fn report(model: ModelStatus, gpu: GpuStatus, index: Option<IndexStatus>) -> StatusReport {
        StatusReport {
            worktree: PathBuf::from("/Users/me/src/waterui"),
            index,
            model,
            daemon: DaemonStatus {
                version: "0.1.0".into(),
                pid: 4242,
                uptime_s: 192,
                worktrees: 2,
                gpu,
            },
        }
    }

    fn indexed_status() -> IndexStatus {
        IndexStatus {
            phase: WirePhase::FirstIndex,
            chunks_total: 34500,
            chunks_with_vector: 12345,
            queued_stage1: 600,
            queued_background: 0,
            in_flight: 8,
        }
    }

    fn text(output: &StatusOutput) -> String {
        status_text(output)
    }

    #[test]
    fn status_text_indexed() {
        let output = StatusOutput::from(report(
            ModelStatus::Ready {
                model_id: "ab12cd34ef56ff00".into(),
            },
            GpuStatus::Absent {
                blocked: Some("not on mains".into()),
            },
            Some(indexed_status()),
        ));
        assert_eq!(
            text(&output),
            "worktree  /Users/me/src/waterui\n\
             index     first index, 12345 of 34500 chunks embedded (35%)\n\
             pending   600 stage 1, 0 background, 8 in flight\n\
             model     ready (ab12cd34ef56)\n\
             gpu       absent: not on mains\n\
             daemon    pid 4242, version 0.1.0, up 192 s, 2 worktrees\n"
        );
    }

    #[test]
    fn status_text_never_indexed() {
        let output = StatusOutput::from(report(ModelStatus::Loading, GpuStatus::Loading, None));
        assert_eq!(
            text(&output),
            "worktree  /Users/me/src/waterui\n\
             index     none; the first search indexes this worktree\n\
             model     loading\n\
             gpu       loading\n\
             daemon    pid 4242, version 0.1.0, up 192 s, 2 worktrees\n"
        );
    }

    #[test]
    fn status_text_every_model_and_gpu_state() {
        let models = [
            (
                ModelStatus::Downloading {
                    received: 123_400_000,
                    total: Some(310_200_000),
                },
                "model     downloading, 123.4 of 310.2 MB\n",
            ),
            (
                ModelStatus::Downloading {
                    received: 123_400_000,
                    total: None,
                },
                "model     downloading, 123.4 MB\n",
            ),
            (ModelStatus::Loading, "model     loading\n"),
            (
                ModelStatus::Ready {
                    model_id: "ab12cd34ef56ff00".into(),
                },
                "model     ready (ab12cd34ef56)\n",
            ),
            (
                ModelStatus::Failed {
                    message: "checksum mismatch".into(),
                },
                "model     failed: checksum mismatch\n",
            ),
        ];
        for (model, expected) in models {
            let output =
                StatusOutput::from(report(model, GpuStatus::Running, Some(indexed_status())));
            assert!(
                text(&output).contains(expected),
                "missing {expected:?} in:\n{}",
                text(&output)
            );
        }
        let gpus = [
            (
                GpuStatus::Absent {
                    blocked: Some("not on mains".into()),
                },
                "gpu       absent: not on mains\n",
            ),
            (GpuStatus::Absent { blocked: None }, "gpu       absent\n"),
            (GpuStatus::Loading, "gpu       loading\n"),
            (GpuStatus::Running, "gpu       running\n"),
        ];
        for (gpu, expected) in gpus {
            let output =
                StatusOutput::from(report(ModelStatus::Loading, gpu, Some(indexed_status())));
            assert!(
                text(&output).contains(expected),
                "missing {expected:?} in:\n{}",
                text(&output)
            );
        }
        let output = StatusOutput::from(report(
            ModelStatus::Loading,
            GpuStatus::Running,
            Some(IndexStatus {
                phase: WirePhase::Maintained,
                ..indexed_status()
            }),
        ));
        assert!(
            text(&output).contains("index     maintained, 12345 of 34500 chunks embedded (35%)\n"),
            "{}",
            text(&output)
        );
    }

    #[test]
    fn status_json_keys_and_tags() {
        let output = StatusOutput::from(report(
            ModelStatus::Downloading {
                received: 5,
                total: Some(10),
            },
            GpuStatus::Absent {
                blocked: Some("not on mains".into()),
            },
            Some(indexed_status()),
        ));
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&output).unwrap()).unwrap();
        assert_eq!(json["worktree"], "/Users/me/src/waterui");
        assert_eq!(json["index"]["phase"], "first_index");
        assert_eq!(json["index"]["chunks"], 34500);
        assert_eq!(json["index"]["with_vector"], 12345);
        assert_eq!(
            json["index"]["coverage"].as_f64().unwrap(),
            12345.0 / 34500.0
        );
        assert_eq!(json["index"]["pending"]["stage1"], 600);
        assert_eq!(json["index"]["pending"]["in_flight"], 8);
        assert_eq!(json["model"]["state"], "downloading");
        assert_eq!(json["model"]["received"], 5);
        assert_eq!(json["model"]["total"], 10);
        assert_eq!(json["gpu"]["state"], "absent");
        assert_eq!(json["gpu"]["blocked"], "not on mains");
        assert_eq!(json["daemon"]["pid"], 4242);
        assert_eq!(json["daemon"]["uptime_s"], 192);
        assert_eq!(json["daemon"]["worktrees"], 2);

        let output = StatusOutput::from(report(
            ModelStatus::Ready {
                model_id: "ab12".into(),
            },
            GpuStatus::Running,
            None,
        ));
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&output).unwrap()).unwrap();
        assert!(json["index"].is_null(), "a never-indexed worktree is null");
        assert_eq!(json["model"]["state"], "ready");
        assert_eq!(json["model"]["id"], "ab12");
        assert_eq!(json["gpu"]["state"], "running");
    }

    #[test]
    fn every_error_kind_has_its_message() {
        let log = Path::new("/log/daemon.log");
        let cases: [(ErrorKind, &str); 8] = [
            (
                ErrorKind::ProtocolMismatch,
                "the daemon did not restart into this build: why; see /log/daemon.log",
            ),
            (
                ErrorKind::BadRequest,
                "the daemon rejected the request: why",
            ),
            (ErrorKind::NotAWorktree, "why"),
            (ErrorKind::Model, "the model is unavailable: why"),
            (ErrorKind::Search, "the search failed: why"),
            (
                ErrorKind::ShuttingDown,
                "the daemon shut down before answering; run the command again",
            ),
            (
                ErrorKind::TooLarge,
                "the answer exceeds the 64 MiB frame limit; lower --budget",
            ),
            (
                ErrorKind::Internal,
                "daemon error: why; see /log/daemon.log",
            ),
        ];
        for (kind, expected) in cases {
            let report = ErrorReport {
                kind,
                message: "why".into(),
            };
            assert_eq!(daemon_error(&report, log), expected, "{kind:?}");
        }
    }
}
