//! First-use download of the published Core ML model archive.
//!
//! The archive is pinned to an immutable Hugging Face commit and a known
//! sha256. [`ModelStore::ensure`] downloads it once into the per-user cache,
//! verifies the hash before unpacking, and is a no-op when the model is
//! already present.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use repo_search_core::model::{ARCHIVE, Fetch, TOKENIZER};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::io::AsyncWriteExt;

/// Upper bound on the archive size, far above the published archive.
const MAX_ARCHIVE_BYTES: u64 = 4 << 30;

/// Deadline for establishing the archive connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Deadline between received bytes; a stalled stream fails while a merely
/// slow one keeps running.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// A sha256 digest; its hex form is used only for directory names and error
/// text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Parse `text` as 64 hex digits.
    pub fn from_hex(text: &str) -> Option<Self> {
        let mut digest = [0u8; 32];
        hex::decode_to_slice(text, &mut digest).ok()?;
        Some(Self(digest))
    }

    /// The sha256 of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Lowercase hex, e.g. for a directory name.
    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// The per-user cache root the model archive and the tokenizer share:
/// `models/` under the OS cache root ([`UserPaths`](repo_search_client::launcher::paths::UserPaths)).
fn user_cache_root() -> Result<PathBuf, Error> {
    Ok(repo_search_client::launcher::paths::UserPaths::user()?
        .cache
        .join("models"))
}

/// A model archive to fetch and the digest that verifies it.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Download URL, pinned to an immutable upstream revision.
    pub url: String,
    /// Expected sha256 of the archive bytes.
    pub sha256: Sha256Digest,
}

impl ModelSpec {
    /// The published EmbeddingGemma 2 Core ML archive, from
    /// [`repo_search_core::model`]'s model card.
    pub fn embedding_gemma_2() -> Self {
        Self {
            url: ARCHIVE.url.to_owned(),
            sha256: Sha256Digest::from_hex(ARCHIVE.sha256)
                .expect("pinned archive digest is valid hex"),
        }
    }
}

/// HTTP client configuration shared by production and tests: a connect
/// deadline and an idle-read deadline, never a total timeout that would cut
/// off a large archive mid-download.
fn client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
}

/// A cache of verified, unpacked model archives.
pub struct ModelStore {
    /// Cache root; each archive unpacks to a `<sha256>/` subdirectory.
    root: PathBuf,
    spec: ModelSpec,
    /// The store's own client, built through [`client_builder`].
    client: reqwest::Client,
}

impl ModelStore {
    /// A store under `root` serving `spec`.
    pub fn new(root: PathBuf, spec: ModelSpec) -> Result<Self, Error> {
        Ok(Self {
            root,
            spec,
            client: client_builder().build()?,
        })
    }

    /// The per-user store: `models/` under the OS cache root.
    pub fn user_cache() -> Result<Self, Error> {
        Self::new(user_cache_root()?, ModelSpec::embedding_gemma_2())
    }

    /// Directory the archive unpacks to.
    pub fn dir(&self) -> PathBuf {
        self.root.join(self.spec.sha256.hex())
    }

    /// The model build's identity: the archive's sha256, the `model_id` the
    /// embedder and the vector cache name vectors with.
    pub fn model_id(&self) -> String {
        self.spec.sha256.hex()
    }

    /// Path to the unpacked model, downloading it on first use.
    ///
    /// Idempotent: an existing directory is always a complete unpack, because
    /// the archive contents are staged in a sibling temp directory and moved
    /// into place with one rename. Concurrent calls serialize on a per-archive
    /// lock file: the caller holding the lock re-checks the final directory,
    /// sweeps leftovers of killed runs, downloads, and unpacks; callers that
    /// were waiting on the lock then find the finished directory.
    ///
    /// `progress` receives a [`Fetch`] after each downloaded chunk.
    pub async fn ensure(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<PathBuf, Error> {
        let dir = self.dir();
        if tokio::fs::try_exists(&dir).await? {
            return Ok(dir);
        }
        let lock_path = self.root.join(format!("{}.lock", self.spec.sha256));
        let root = self.root.clone();
        let dest = dir.clone();
        // The lock and its re-check run on the blocking pool; the guard file
        // comes back locked and stays that way for the whole install.
        let lock = tokio::task::spawn_blocking(move || -> io::Result<Option<std::fs::File>> {
            std::fs::create_dir_all(&root)?;
            let lock = std::fs::File::create(lock_path)?;
            lock.lock()?;
            if dest.exists() {
                return Ok(None);
            }
            sweep_stale(&root)?;
            Ok(Some(lock))
        })
        .await??;
        let Some(_lock) = lock else {
            return Ok(dir);
        };
        let archive = self.download(progress).await?;
        self.unpack(archive, &dir).await?;
        Ok(dir)
    }

    /// Stream the archive into a temp file inside the cache root, returning it
    /// only once its sha256 matches the pinned digest. The temp file deletes
    /// itself on drop; error paths run that drop on the blocking pool.
    async fn download(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<NamedTempFile, Error> {
        fetch_verified(
            &self.client,
            &self.spec.url,
            self.spec.sha256,
            &self.root,
            progress,
        )
        .await
    }

    /// Unpack a verified archive into `dir` via a sibling staging directory.
    /// The caller holds the archive lock, so a rename error is a real error.
    async fn unpack(&self, archive: NamedTempFile, dir: &Path) -> Result<(), Error> {
        let root = self.root.clone();
        let dest = dir.to_path_buf();
        tokio::task::spawn_blocking(move || -> io::Result<()> {
            let staging = tempfile::tempdir_in(&root)?;
            unpack_archive(archive.as_file(), staging.path())?;
            std::fs::rename(staging.path(), &dest)?;
            let _ = staging.keep();
            Ok(())
        })
        .await??;
        tracing::info!(dir = %dir.display(), "model archive unpacked");
        Ok(())
    }
}

/// A tokenizer file to fetch and the digest that verifies it.
#[derive(Debug, Clone)]
pub struct TokenizerSpec {
    /// Download URL, pinned to an immutable upstream revision.
    pub url: String,
    /// Expected sha256 of the file bytes.
    pub sha256: Sha256Digest,
}

impl TokenizerSpec {
    /// The tokenizer from [`repo_search_core::model`]'s model card: the
    /// upstream revision the archive ships the same bytes under.
    pub fn embedding_gemma_2() -> Self {
        Self {
            url: TOKENIZER.url.to_owned(),
            sha256: Sha256Digest::from_hex(TOKENIZER.sha256)
                .expect("pinned tokenizer digest is valid hex"),
        }
    }
}

/// A cache of the pinned `tokenizer.json` for the daemon's counter.
///
/// The file lands at `<root>/tokenizer/<sha256>.json`, written by an
/// atomic rename of the verified temp file — a present path is always a
/// complete, verified download, so an existing file is a cache hit.
pub struct TokenizerStore {
    /// Cache root, shared with the model archive cache.
    root: PathBuf,
    spec: TokenizerSpec,
    /// The store's own client, built through [`client_builder`].
    client: reqwest::Client,
}

impl TokenizerStore {
    /// A store under `root` serving `spec`.
    pub fn new(root: PathBuf, spec: TokenizerSpec) -> Result<Self, Error> {
        Ok(Self {
            root,
            spec,
            client: client_builder().build()?,
        })
    }

    /// The per-user store, in the same cache root as the model archive.
    pub fn user_cache() -> Result<Self, Error> {
        Self::new(user_cache_root()?, TokenizerSpec::embedding_gemma_2())
    }

    /// The cached tokenizer's path.
    pub fn path(&self) -> PathBuf {
        self.root
            .join("tokenizer")
            .join(format!("{}.json", self.spec.sha256.hex()))
    }

    /// Path to the cached tokenizer, downloading it on first use.
    ///
    /// `progress` receives a [`Fetch`] after each downloaded chunk.
    pub async fn ensure(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<PathBuf, Error> {
        let path = self.path();
        if tokio::fs::try_exists(&path).await? {
            return Ok(path);
        }
        // `fetch_verified` stages its temp file in `root`.
        let dir = path.parent().expect("path has a parent").to_path_buf();
        tokio::fs::create_dir_all(dir).await?;
        let temp = fetch_verified(
            &self.client,
            &self.spec.url,
            self.spec.sha256,
            &self.root,
            progress,
        )
        .await?;
        // The verified bytes are moved into place with one rename; a
        // racing install writes the identical bytes.
        tokio::task::spawn_blocking(move || temp.persist(path).map_err(|e| e.error)).await??;
        Ok(self.path())
    }
}

/// Stream `url` into a temp file inside `root`, returning it only once its
/// sha256 matches `sha256`. The temp file deletes itself on drop; error
/// paths run that drop on the blocking pool. Shared by [`ModelStore`],
/// [`TokenizerStore`] and the server's [`repo_search_server::ModelSource`].
pub async fn fetch_verified(
    client: &reqwest::Client,
    url: &str,
    sha256: Sha256Digest,
    root: &Path,
    progress: &tokio::sync::watch::Sender<Fetch>,
) -> Result<NamedTempFile, Error> {
    let mut response = client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(Error::Status {
            url: url.to_owned(),
            status: response.status(),
        });
    }
    let total = response.content_length();
    if let Some(size) = total
        && size > MAX_ARCHIVE_BYTES
    {
        return Err(Error::TooLarge {
            size,
            limit: MAX_ARCHIVE_BYTES,
        });
    }
    let root = root.to_path_buf();
    let (temp, writer) = tokio::task::spawn_blocking(move || {
        let temp = NamedTempFile::new_in(&root)?;
        let writer = temp.reopen()?;
        io::Result::Ok((temp, writer))
    })
    .await??;
    let mut writer = tokio::fs::File::from_std(writer);
    let mut hasher = Sha256::new();
    let mut received = 0u64;
    tracing::info!(url, "downloading model archive");
    let streamed: Result<(), Error> = async {
        while let Some(chunk) = response.chunk().await? {
            received += chunk.len() as u64;
            if received > MAX_ARCHIVE_BYTES {
                return Err(Error::TooLarge {
                    size: received,
                    limit: MAX_ARCHIVE_BYTES,
                });
            }
            hasher.update(&chunk);
            writer.write_all(&chunk).await?;
            progress.send_replace(Fetch { received, total });
        }
        writer.flush().await?;
        Ok(())
    }
    .await;
    let digest = match streamed {
        Ok(()) => Sha256Digest(hasher.finalize().into()),
        Err(e) => {
            discard(temp).await;
            return Err(e);
        }
    };
    if digest != sha256 {
        discard(temp).await;
        return Err(Error::HashMismatch {
            expected: sha256,
            actual: digest,
        });
    }
    Ok(temp)
}

#[cfg(target_os = "macos")]
impl repo_search_server::ModelSource for ModelStore {
    fn model_id(&self) -> String {
        self.model_id()
    }

    async fn ensure(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<PathBuf, String> {
        self.ensure(progress).await.map_err(|e| e.to_string())
    }
}

/// Run a temp archive's delete-on-drop off the async runtime.
async fn discard(temp: NamedTempFile) {
    let _ = tokio::task::spawn_blocking(move || drop(temp)).await;
}

/// Remove temp files and staging directories a killed run left behind. The
/// caller holds the archive lock, so every `.tmp*` entry present is stale.
fn sweep_stale(root: &Path) -> io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(".tmp"))
        {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Unpack `archive` into `staging`, syncing every extracted file and then the
/// staging directory itself so the rename that follows exposes durable
/// contents. Anything but regular files and directories is rejected.
fn unpack_archive(archive: &std::fs::File, staging: &Path) -> io::Result<()> {
    let decoder = zstd::stream::read::Decoder::new(archive)?;
    let mut tar = tar::Archive::new(decoder);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let kind = entry.header().entry_type();
        if !matches!(kind, tar::EntryType::Regular | tar::EntryType::Directory) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported tar entry type {kind:?}: {}", path.display()),
            ));
        }
        if !entry.unpack_in(staging)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tar entry escapes the staging directory: {}",
                    path.display()
                ),
            ));
        }
        if kind == tar::EntryType::Regular {
            std::fs::File::open(staging.join(&path))?.sync_all()?;
        }
    }
    std::fs::File::open(staging)?.sync_all()?;
    Ok(())
}

/// Failure modes of [`ModelStore::ensure`].
#[derive(Debug, Error)]
pub enum Error {
    /// The downloaded bytes did not match the pinned archive digest.
    #[error("model archive sha256 mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// The pinned digest.
        expected: Sha256Digest,
        /// The digest of the downloaded bytes.
        actual: Sha256Digest,
    },
    /// The server answered a non-success status.
    #[error("download of {url} failed: {status}")]
    Status {
        /// The request URL.
        url: String,
        /// The response status.
        status: reqwest::StatusCode,
    },
    /// The archive exceeds the size bound.
    #[error("model archive size {size} exceeds the {limit} byte bound")]
    TooLarge {
        /// The declared or received size.
        size: u64,
        /// The configured bound.
        limit: u64,
    },
    /// HTTP transport error.
    #[error("model archive download failed: {0}")]
    Download(#[from] reqwest::Error),
    /// Filesystem or archive error.
    #[error("model store error: {0}")]
    Io(#[from] io::Error),
    /// A blocking-pool task could not complete.
    #[error("model store task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
}

#[cfg(test)]
#[path = "../tests/support/http.rs"]
mod http;

#[cfg(test)]
mod tests {
    use super::http::{archive, serve_once, symlink_archive};
    use super::*;

    use tokio::sync::watch;

    /// A store whose client comes through the same builder as production,
    /// with system proxies disabled so the local test server is reached
    /// directly.
    fn store(root: PathBuf, url: String, sha256: Sha256Digest) -> ModelStore {
        ModelStore {
            root,
            spec: ModelSpec { url, sha256 },
            client: client_builder().no_proxy().build().unwrap(),
        }
    }

    fn digest_of(bytes: &[u8]) -> Sha256Digest {
        Sha256Digest(Sha256::digest(bytes).into())
    }

    /// The progress sink `ensure` reports to, seeded with zero progress.
    fn sink() -> (watch::Sender<Fetch>, watch::Receiver<Fetch>) {
        watch::channel(Fetch {
            received: 0,
            total: None,
        })
    }

    /// The lock file is the one entry a failed or finished run may leave in
    /// the cache root; everything else must be gone.
    fn assert_only_lock_remains(cache: &Path) {
        let names: Vec<String> = std::fs::read_dir(cache)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(
            names.iter().all(|name| name.ends_with(".lock")),
            "unexpected cache entries: {names:?}"
        );
    }

    #[tokio::test]
    async fn downloads_verifies_and_unpacks() {
        let body = archive(&[
            ("manifest.json", b"{\"embedding_dim\": 768}"),
            ("eg2-coreml.mlmodelc/model.mlmodel", b"weights"),
        ]);
        let (url, server) = serve_once("200 OK", body.len() as u64, body.clone());
        let root = tempfile::tempdir().unwrap();
        let store = store(root.path().join("models"), url, digest_of(&body));
        let (progress, mut received) = sink();

        let dir = store.ensure(&progress).await.unwrap();

        assert_eq!(dir, store.dir());
        received.borrow_and_update();
        assert_eq!(
            *received.borrow(),
            Fetch {
                received: body.len() as u64,
                total: Some(body.len() as u64)
            }
        );
        assert_eq!(
            std::fs::read(dir.join("manifest.json")).unwrap(),
            b"{\"embedding_dim\": 768}"
        );
        assert_eq!(
            std::fs::read(dir.join("eg2-coreml.mlmodelc/model.mlmodel")).unwrap(),
            b"weights"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn corrupt_download_is_rejected_and_leaves_nothing() {
        let good = archive(&[("manifest.json", b"{}")]);
        let corrupt = archive(&[("manifest.json", b"corrupted")]);
        let (url, server) = serve_once("200 OK", corrupt.len() as u64, corrupt.clone());
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("models");
        let store = store(cache.clone(), url, digest_of(&good));
        let (progress, _) = sink();

        let err = store.ensure(&progress).await.unwrap_err();

        match err {
            Error::HashMismatch { expected, actual } => {
                assert_eq!(expected, digest_of(&good));
                assert_eq!(actual, digest_of(&corrupt));
            }
            e => panic!("expected HashMismatch, got {e}"),
        }
        assert!(!store.dir().exists());
        assert_only_lock_remains(&cache);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn cached_model_skips_download() {
        let root = tempfile::tempdir().unwrap();
        let store = store(
            root.path().to_path_buf(),
            "http://127.0.0.1:1/unreachable".into(),
            Sha256Digest([0; 32]),
        );
        let dir = store.dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker"), b"sentinel").unwrap();
        let (progress, _) = sink();

        assert_eq!(store.ensure(&progress).await.unwrap(), dir);
        assert_eq!(std::fs::read(dir.join("marker")).unwrap(), b"sentinel");
    }

    #[tokio::test]
    async fn interrupted_download_leaves_no_partial_dir() {
        let body = archive(&[("big.bin", &[7u8; 8192])]);
        let sent = body.len() / 2;
        let (url, server) = serve_once("200 OK", body.len() as u64, body[..sent].to_vec());
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("models");
        let store = store(cache.clone(), url, digest_of(&body));
        let (progress, _) = sink();

        let err = store.ensure(&progress).await.unwrap_err();

        assert!(matches!(err, Error::Download(_)), "unexpected error: {err}");
        assert!(!store.dir().exists());
        assert_only_lock_remains(&cache);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oversized_archive_is_refused() {
        let body = archive(&[("manifest.json", b"{}")]);
        let (url, server) = serve_once("200 OK", MAX_ARCHIVE_BYTES + 1, body);
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("models");
        let store = store(cache.clone(), url, digest_of(b""));
        let (progress, _) = sink();

        let err = store.ensure(&progress).await.unwrap_err();

        match err {
            Error::TooLarge { size, limit } => {
                assert_eq!(size, MAX_ARCHIVE_BYTES + 1);
                assert_eq!(limit, MAX_ARCHIVE_BYTES);
            }
            e => panic!("expected TooLarge, got {e}"),
        }
        assert!(!store.dir().exists());
        server.join().unwrap();
    }

    #[tokio::test]
    async fn http_error_status_is_reported() {
        let (url, server) = serve_once("503 Service Unavailable", 0, Vec::new());
        let root = tempfile::tempdir().unwrap();
        let store = store(root.path().join("models"), url, digest_of(b""));
        let (progress, _) = sink();

        let err = store.ensure(&progress).await.unwrap_err();

        match err {
            Error::Status { status, .. } => {
                assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
            }
            e => panic!("expected Status, got {e}"),
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn symlink_entries_are_rejected() {
        let body = symlink_archive();
        let (url, server) = serve_once("200 OK", body.len() as u64, body.clone());
        let root = tempfile::tempdir().unwrap();
        let store = store(root.path().join("models"), url, digest_of(&body));
        let (progress, _) = sink();

        let err = store.ensure(&progress).await.unwrap_err();

        match err {
            Error::Io(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
            e => panic!("unexpected error: {e}"),
        }
        assert!(!store.dir().exists());
        server.join().unwrap();
    }

    #[tokio::test]
    async fn concurrent_ensure_downloads_once() {
        let body = archive(&[("manifest.json", b"{}")]);
        // The one-shot server answers exactly one request: a second download
        // would hang and fail the test.
        let (url, server) = serve_once("200 OK", body.len() as u64, body.clone());
        let root = tempfile::tempdir().unwrap();
        let store = store(root.path().join("models"), url, digest_of(&body));
        let (progress, _) = sink();

        let (first, second) = tokio::join!(store.ensure(&progress), store.ensure(&progress));

        assert_eq!(first.unwrap(), store.dir());
        assert_eq!(second.unwrap(), store.dir());
        assert_eq!(
            std::fs::read(store.dir().join("manifest.json")).unwrap(),
            b"{}"
        );
        server.join().unwrap();
    }
    fn tokenizer_store(root: PathBuf, url: String, sha256: Sha256Digest) -> TokenizerStore {
        TokenizerStore {
            root,
            spec: TokenizerSpec { url, sha256 },
            client: client_builder().no_proxy().build().unwrap(),
        }
    }

    #[tokio::test]
    async fn tokenizer_downloads_verifies_and_caches() {
        let body = b"{\"tokenizer\": true}".to_vec();
        let (url, server) = serve_once("200 OK", body.len() as u64, body.clone());
        let root = tempfile::tempdir().unwrap();
        let store = tokenizer_store(root.path().join("models"), url, digest_of(&body));
        let (progress, mut received) = sink();

        let path = store.ensure(&progress).await.unwrap();

        assert_eq!(path, store.path());
        assert_eq!(
            path,
            root.path()
                .join("models/tokenizer")
                .join(format!("{}.json", digest_of(&body).hex()))
        );
        assert_eq!(std::fs::read(&path).unwrap(), body);
        received.borrow_and_update();
        assert_eq!(
            *received.borrow(),
            Fetch {
                received: body.len() as u64,
                total: Some(body.len() as u64)
            }
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn tokenizer_cache_hit_never_touches_the_server() {
        let body = b"{}".to_vec();
        let (url, server) = serve_once("200 OK", body.len() as u64, body.clone());
        let root = tempfile::tempdir().unwrap();
        let store = tokenizer_store(root.path().join("models"), url, digest_of(&body));
        let (progress, _) = sink();
        let path = store.ensure(&progress).await.unwrap();
        server.join().unwrap();
        // The one-shot server is spent; a second ensure must be a hit.
        assert_eq!(store.ensure(&progress).await.unwrap(), path);
    }

    #[tokio::test]
    async fn a_tokenizer_mismatch_is_rejected_and_leaves_nothing() {
        let corrupt = b"corrupted".to_vec();
        let (url, server) = serve_once("200 OK", corrupt.len() as u64, corrupt.clone());
        let root = tempfile::tempdir().unwrap();
        let store = tokenizer_store(root.path().join("models"), url, digest_of(b"{}"));
        let (progress, _) = sink();

        let Err(Error::HashMismatch { .. }) = store.ensure(&progress).await else {
            panic!("a corrupt tokenizer is rejected");
        };
        assert!(!store.path().exists());
        server.join().unwrap();
    }
}
