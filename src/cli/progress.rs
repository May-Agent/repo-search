//! The client's stderr progress: one `\r`-rewritten status line on a TTY,
//! two one-time announcement lines off it.

use repo_search_core::search::Coverage;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::mb;
use repo_search_client::protocol::Progress;

/// The one-line text a `Progress` draws on a TTY. The texts carry no
/// paths, so a line never wraps and `\r` always returns to its start.
fn line(progress: &Progress) -> String {
    match progress {
        Progress::ModelDownload { received, total } => match total {
            Some(total) => format!(
                "downloading model  {} / {} MB  {}%",
                mb(*received),
                mb(*total),
                u32::try_from(u128::from(*received) * 100 / u128::from(*total))
                    .expect("a share of at most 100")
            ),
            None => format!("downloading model  {} MB", mb(*received)),
        },
        Progress::ModelLoading => "loading model".to_owned(),
        Progress::Reconciling => "reconciling".to_owned(),
        Progress::Coverage { with_vector, total } => format!(
            "embedding  {} / {} chunks  {}%",
            with_vector,
            total,
            Coverage::new(*with_vector, *total).percent()
        ),
    }
}

/// Stderr progress for one request.
///
/// On a TTY every event redraws one line in place (`\r\x1b[2K<text>`, no
/// newline) — and only when the text changed — which [`clear`] erases
/// before any stdout write or error message lands. Off a TTY there is no
/// redrawing: only the two waits a user cannot otherwise explain are
/// announced, once each per run — the first model download and the first
/// model load. `Reconciling` and `Coverage` print nothing off a TTY: they
/// arrive on every query or many times a second, and a log reader gains
/// nothing from them. Write errors are ignored; there is nowhere left to
/// report them.
pub struct ProgressLine {
    tty: bool,
    shown: Option<String>,
    announced_download: bool,
    announced_loading: bool,
}

impl ProgressLine {
    /// A progress line for `stderr`, which is or is not a `tty`.
    pub fn new(tty: bool) -> Self {
        Self {
            tty,
            shown: None,
            announced_download: false,
            announced_loading: false,
        }
    }

    /// Shows `progress`.
    pub async fn show(&mut self, progress: &Progress, stderr: &mut (impl AsyncWrite + Unpin)) {
        if self.tty {
            let text = line(progress);
            if self.shown.as_deref() == Some(text.as_str()) {
                return;
            }
            let _ = stderr
                .write_all(format!("\r\x1b[2K{text}").as_bytes())
                .await;
            self.shown = Some(text);
            return;
        }
        let announcement = match progress {
            Progress::ModelDownload { total, .. } if !self.announced_download => {
                self.announced_download = true;
                Some(match total {
                    Some(total) => {
                        format!("search: downloading the model ({} MB)", mb(*total))
                    }
                    None => "search: downloading the model".to_owned(),
                })
            }
            Progress::ModelLoading if !self.announced_loading => {
                self.announced_loading = true;
                Some("search: loading the model".to_owned())
            }
            _ => None,
        };
        if let Some(announcement) = announcement {
            let _ = stderr
                .write_all(format!("{announcement}\n").as_bytes())
                .await;
        }
    }

    /// Erases the drawn line; a no-op when nothing is shown.
    pub async fn clear(&mut self, stderr: &mut (impl AsyncWrite + Unpin)) {
        if self.tty && self.shown.is_some() {
            let _ = stderr.write_all(b"\r\x1b[2K").await;
            self.shown = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn download(received: u64, total: u64) -> Progress {
        Progress::ModelDownload {
            received,
            total: Some(total),
        }
    }

    fn text(stderr: &[u8]) -> String {
        String::from_utf8(stderr.to_vec()).expect("stderr is utf-8")
    }

    #[tokio::test]
    async fn tty_redraws_only_when_the_text_changes() {
        let mut stderr = Vec::new();
        let mut line = ProgressLine::new(true);
        line.show(&download(100, 400), &mut stderr).await;
        line.show(&download(100, 400), &mut stderr).await;
        line.show(&download(200, 400), &mut stderr).await;
        assert_eq!(
            text(&stderr),
            "\r\x1b[2Kdownloading model  0.0 / 0.0 MB  25%\r\x1b[2Kdownloading model  0.0 / 0.0 MB  50%",
            "an identical event draws nothing"
        );
    }

    #[tokio::test]
    async fn tty_clear_erases_a_drawn_line() {
        let mut stderr = Vec::new();
        let mut line = ProgressLine::new(true);
        line.clear(&mut stderr).await;
        assert!(stderr.is_empty(), "clearing an empty line writes nothing");
        line.show(&Progress::Reconciling, &mut stderr).await;
        line.show(
            &Progress::Coverage {
                with_vector: 35,
                total: 100,
            },
            &mut stderr,
        )
        .await;
        line.clear(&mut stderr).await;
        assert_eq!(
            text(&stderr),
            "\r\x1b[2Kreconciling\r\x1b[2Kembedding  35 / 100 chunks  35%\r\x1b[2K",
            "the answer never lands on a half-drawn line"
        );
    }

    #[tokio::test]
    async fn off_tty_announces_download_and_load_once() {
        let mut stderr = Vec::new();
        let mut line = ProgressLine::new(false);
        for progress in [
            download(50, 400),
            download(100, 400),
            Progress::ModelDownload {
                received: 400,
                total: None,
            },
            Progress::ModelLoading,
            Progress::ModelLoading,
            Progress::Reconciling,
            Progress::Coverage {
                with_vector: 3,
                total: 12,
            },
        ] {
            line.show(&progress, &mut stderr).await;
        }
        assert_eq!(
            text(&stderr),
            "search: downloading the model (0.0 MB)\nsearch: loading the model\n",
            "only the two unexplained waits are announced, once each"
        );
    }

    #[tokio::test]
    async fn off_tty_an_unknown_total_announces_no_size() {
        let mut stderr = Vec::new();
        let mut line = ProgressLine::new(false);
        line.show(
            &Progress::ModelDownload {
                received: 10,
                total: None,
            },
            &mut stderr,
        )
        .await;
        assert_eq!(text(&stderr), "search: downloading the model\n");
    }
}
