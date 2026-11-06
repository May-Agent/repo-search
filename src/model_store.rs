//! First-use download of the model files [`repo_search_core::model`] pins.
//!
//! The model archive and the daemon's tokenizer install through one
//! [`ondevice::Store`] in the per-user cache, downloading through the
//! crate's own reqwest client; these wrappers fix which pin each serves and
//! report progress as [`Fetch`] on a `watch` channel.

use std::io;
use std::path::PathBuf;

use futures_util::StreamExt as _;
use futures_util::stream::BoxStream;
use ondevice::{Archive, File, Pin, Progress, Response, Store, Transport};
use repo_search_core::model::{ARCHIVE, Fetch, TOKENIZER};

/// An [`ensure`](ModelStore::ensure) failure.
pub type Error = ondevice::Error<reqwest::Error>;

/// Why a store could not be opened.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// The per-user cache root could not be resolved.
    #[error("could not resolve the model cache: {0}")]
    Root(#[from] io::Error),
    /// The HTTP client could not be built.
    #[error("could not build the download client: {0}")]
    Client(#[from] reqwest::Error),
}

/// The store's [`Transport`]: a reqwest client with system proxies, a
/// non-success status answered as an error.
pub struct Reqwest(reqwest::Client);

impl Reqwest {
    /// A transport over a fresh client.
    ///
    /// # Errors
    ///
    /// When the TLS backend or the resolver cannot be initialised.
    pub fn new() -> reqwest::Result<Self> {
        Ok(Self(reqwest::Client::builder().build()?))
    }
}

impl Transport for Reqwest {
    type Error = reqwest::Error;
    type Chunk = bytes::Bytes;
    type Body = BoxStream<'static, reqwest::Result<bytes::Bytes>>;

    async fn get(&self, url: &str) -> reqwest::Result<Response<Self::Body>> {
        let response = self.0.get(url).send().await?.error_for_status()?;
        Ok(Response {
            length: response.content_length(),
            body: futures_util::stream::unfold(response, |mut response| async move {
                let chunk = response.chunk().await.transpose()?;
                Some((chunk, response))
            })
            .boxed(),
        })
    }
}

/// The per-user cache root the model archive and the tokenizer share:
/// `models/` under the OS cache root ([`UserPaths`](repo_search_client::launcher::paths::UserPaths)).
fn user_cache_root() -> io::Result<PathBuf> {
    Ok(repo_search_client::launcher::paths::UserPaths::user()?
        .cache
        .join("models"))
}

/// Reports a store's [`Progress`] as the [`Fetch`] `progress` carries.
fn report(progress: &tokio::sync::watch::Sender<Fetch>) -> impl FnMut(Progress) + Send + '_ {
    |Progress { received, total }| {
        progress.send_replace(Fetch {
            received,
            total: Some(total),
        });
    }
}

/// The cache serving one model archive.
pub struct ModelStore {
    store: Store<Reqwest>,
    pin: Pin<'static, Archive>,
}

impl ModelStore {
    /// A store under `root` serving `pin`.
    ///
    /// # Errors
    ///
    /// When the download client cannot be built.
    pub fn new(root: PathBuf, pin: Pin<'static, Archive>) -> Result<Self, OpenError> {
        Ok(Self {
            store: Store::new(root, Reqwest::new()?),
            pin,
        })
    }

    /// The per-user store serving the Core ML archive.
    ///
    /// # Errors
    ///
    /// When the per-user cache root cannot be resolved or the download
    /// client cannot be built.
    pub fn user_cache() -> Result<Self, OpenError> {
        Self::new(user_cache_root()?, ARCHIVE)
    }

    /// The per-user store serving the ONNX archive the non-macOS server
    /// loads.
    ///
    /// # Errors
    ///
    /// As [`user_cache`](Self::user_cache).
    #[cfg(not(target_os = "macos"))]
    pub fn user_cache_onnx() -> Result<Self, OpenError> {
        Self::new(user_cache_root()?, repo_search_core::model::ONNX_ARCHIVE)
    }

    /// The per-user store serving the int8 ONNX archive.
    ///
    /// # Errors
    ///
    /// As [`user_cache`](Self::user_cache).
    #[cfg(not(target_os = "macos"))]
    pub fn user_cache_onnx_int8() -> Result<Self, OpenError> {
        Self::new(
            user_cache_root()?,
            repo_search_core::model::ONNX_INT8_ARCHIVE,
        )
    }

    /// The directory the archive unpacks to.
    pub fn dir(&self) -> PathBuf {
        self.store.path(&self.pin)
    }

    /// The model build's identity: the archive's sha256, the `model_id` the
    /// embedder and the vector cache name vectors with.
    pub fn model_id(&self) -> String {
        self.pin.sha256.to_string()
    }

    /// The unpacked model directory, downloading it on first use.
    /// `progress` receives a [`Fetch`] after each downloaded chunk.
    ///
    /// # Errors
    ///
    /// The [`ondevice::Store::ensure`] failures.
    pub async fn ensure(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<PathBuf, Error> {
        self.store.ensure(&self.pin, report(progress)).await
    }
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

/// The cache serving the pinned `tokenizer.json` for the daemon's
/// counter, in the same root as the model archive.
pub struct TokenizerStore {
    store: Store<Reqwest>,
    pin: Pin<'static, File>,
}

impl TokenizerStore {
    /// A store under `root` serving `pin`.
    ///
    /// # Errors
    ///
    /// When the download client cannot be built.
    pub fn new(root: PathBuf, pin: Pin<'static, File>) -> Result<Self, OpenError> {
        Ok(Self {
            store: Store::new(root, Reqwest::new()?),
            pin,
        })
    }

    /// The per-user store serving [`TOKENIZER`].
    ///
    /// # Errors
    ///
    /// When the per-user cache root cannot be resolved or the download
    /// client cannot be built.
    pub fn user_cache() -> Result<Self, OpenError> {
        Self::new(user_cache_root()?, TOKENIZER)
    }

    /// The cached tokenizer's path.
    pub fn path(&self) -> PathBuf {
        self.store.path(&self.pin)
    }

    /// The cached tokenizer, downloading it on first use. `progress`
    /// receives a [`Fetch`] after each downloaded chunk.
    ///
    /// # Errors
    ///
    /// The [`ondevice::Store::ensure`] failures.
    pub async fn ensure(
        &self,
        progress: &tokio::sync::watch::Sender<Fetch>,
    ) -> Result<PathBuf, Error> {
        self.store.ensure(&self.pin, report(progress)).await
    }
}
