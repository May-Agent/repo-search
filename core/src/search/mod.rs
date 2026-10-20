//! Retrieval and output: query analysis, BM25 and vector candidates, reciprocal
//! rank fusion into a code list and a docs list, grouping by file, filters, a
//! token budget, and the text and JSON renderings.
//!
//! A [`Retriever`] answers [`Request`]s against one worktree's [`Index`] and
//! the repository's vector cache. The caller runs the freshness barrier
//! (reconcile) first and passes its outcome in; the retriever embeds the
//! query, and on the spot the best lexical candidates that still lack a
//! vector, through the caller's [`EmbedService`].

mod embedder;
mod fuse;
mod output;
mod retriever;
mod scope;
mod terms;
mod vectors;

pub use embedder::EmbedService;
pub use fuse::{Channel, Fused, RRF_K, fuse};
pub use output::{
    CodeChunk, DocChunk, FileHit, Freshness, Reconciliation, Results, Unparsable, UnparsableReason,
};
pub use retriever::Retriever;
pub use scope::Scope;
pub use terms::{QueryTerms, SymbolName};

use crate::knn;
use crate::store;

/// Candidates each channel contributes to fusion: the KNN scan's result
/// count, [`knn::RESULTS`], so both channels fill the same list.
pub const CANDIDATES: u32 = knn::RESULTS as u32;

/// Lexical candidates lacking a vector that a query embeds on the spot, per
/// list.
pub const ON_THE_SPOT: usize = 50;

/// The default token budget.
pub const DEFAULT_BUDGET: usize = 4000;

/// Which channels answer a query (`--mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    /// BM25 and vector KNN, fused.
    #[default]
    Hybrid,
    /// BM25 only; nothing is embedded.
    Lexical,
    /// Vector KNN only.
    Semantic,
}

/// The restricting filters (`--path`, `--lang`, `--kind`, `--symbol`); empty
/// means unrestricted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filters {
    /// Path globs; a file matches when any glob matches it or a directory
    /// above it.
    pub paths: Vec<String>,
    /// Language names (`rust`, `markdown`, ...).
    pub langs: Vec<String>,
    /// Syntactic kinds (`fn`, `struct`, ...) or `doc`.
    pub kinds: Vec<String>,
    /// A symbol path glob such as `Environment::*`.
    pub symbol: Option<String>,
}

/// One query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The free-text query.
    pub query: String,
    /// The channels to use.
    pub mode: Mode,
    /// The filters.
    pub filters: Filters,
    /// Tokens of output to fill, counted by the caller's [`TokenCounter`].
    pub budget: usize,
    /// Emit ranked files only (`-l`).
    pub files_only: bool,
    /// Search chunks of ignored files too (`--include-ignored`); the caller's
    /// reconcile pass must have been run with the same flag.
    pub include_ignored: bool,
}

impl Request {
    /// A hybrid query with the default budget and no filters.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            mode: Mode::Hybrid,
            filters: Filters::default(),
            budget: DEFAULT_BUDGET,
            files_only: false,
            include_ignored: false,
        }
    }
}

/// Counts the tokens of the model the answer is read by; implemented by the
/// caller, since the library carries no tokenizer.
pub trait TokenCounter {
    /// The number of tokens in `text`.
    fn count(&self, text: &str) -> usize;
}

/// A filter that cannot be applied.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// `--lang` named a language this build does not know.
    #[error("unknown language {0:?}")]
    UnknownLanguage(String),

    /// `--kind` named a kind that is neither `doc`, a short name such as
    /// `fn`, nor a node kind the index holds.
    #[error("unknown kind {0:?}")]
    UnknownKind(String),

    /// A `--path` glob did not compile.
    #[error("bad path glob {pattern:?}: {source}")]
    BadGlob {
        /// The offending pattern.
        pattern: String,
        /// Why it failed.
        source: globset::Error,
    },
}

/// Failures of a query.
#[derive(Debug, thiserror::Error)]
pub enum Error<E: std::error::Error + 'static> {
    /// The query holds no text.
    #[error("the query is empty")]
    EmptyQuery,

    /// A filter is not valid.
    #[error(transparent)]
    Filter(#[from] FilterError),

    /// The index or the vector cache failed.
    #[error(transparent)]
    Store(#[from] store::Error),

    /// The in-memory vector index failed.
    #[error(transparent)]
    Knn(#[from] knn::Error),

    /// The caller's embedder failed.
    #[error("embedder: {0}")]
    Embed(#[source] E),

    /// Reading a file failed.
    #[error("reading {path}: {source}")]
    Io {
        /// Worktree-relative path.
        path: String,
        /// The underlying failure.
        source: std::io::Error,
    },

    /// A chunk no longer matches the working tree: the file changed after the
    /// freshness barrier, which must run before every query.
    #[error("{path}:{start}-{end} no longer matches the indexed chunk; reconcile before searching")]
    Stale {
        /// Worktree-relative path.
        path: String,
        /// First line of the chunk.
        start: u32,
        /// Last line of the chunk.
        end: u32,
    },

    /// The blocking worker thread could not be started.
    #[error("spawning a search worker: {0}")]
    Spawn(std::io::Error),

    /// The blocking worker exited without answering; a panic on it surfaces
    /// here, never in the caller.
    #[error("search worker exited without answering")]
    WorkerGone,
}
