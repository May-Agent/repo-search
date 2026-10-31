//! The per-collection owner task.
//!
//! A document collection is an index the caller fills. It is built from the
//! worktree pipeline's own pieces by giving the collection a directory of
//! files: each document is one Markdown file, so the reconciler's walk, its
//! content-hash diff, the documentation chunker, the keyword index with its
//! BM25 statistics, the vector cache and the retriever's fusion and ranking
//! all apply unchanged. An upsert writes the files whose bytes changed, a
//! delete removes them, and a reconcile pass then re-chunks exactly the
//! files that changed; chunks left without a vector are embedded through
//! the scheduler before the write is answered.
//!
//! A document's file is named by the hash of its id and starts with a front
//! matter block holding the id, which the chunker leaves out of every
//! chunk; its title is the file's first heading, so every chunk's heading
//! path starts with it. One task per collection handles its inbox one
//! message at a time, which the `Retriever`'s `&mut` search requires.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use askama::Template;
use futures_util::stream::{self, StreamExt, TryStreamExt};
use repo_search_core::reconcile::{self, Reconciler, Report};
use repo_search_core::search::{self, EmbedService, Filters, Reconciliation, Retriever};
use repo_search_core::store::{Index, VectorCache};
use repo_search_core::{CHUNKER_FINGERPRINT, StateDir};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, watch};

use crate::wire::ToCore;
use repo_search_client::protocol::{
    Answer, CollectionReport, CollectionSearch, Document, ErrorKind, Event, Final, Format,
    QueryStats,
};

use super::counter::Tokens;
use super::link::{Embedding, EmbeddingState};
use super::registry::RegistryHandle;
use super::scheduler::{RepoId, WorktreeId};
use super::worktree::{AneHandle, BudgetCounter, error, search_failure, send, shutting_down};
use super::writer::WriterHandle;

/// Files written, removed or read concurrently for one request.
const FILE_CONCURRENCY: usize = 16;

/// Chunks sent to the scheduler in one `Documents` job: a write that adds
/// a large corpus embeds in jobs of this size, one after the other.
const EMBED_BATCH: usize = 64;

/// Every file of the collection's directory is indexed: the walk's ignore
/// rules must not hide a document because a parent directory of the state
/// dir carries one.
const RECONCILE: reconcile::Options = reconcile::Options {
    include_ignored: true,
};

/// The collection owner's inbox: one message per connection's request.
#[derive(Debug)]
pub enum CollectionMsg {
    /// Put documents into the collection.
    Upsert {
        /// The documents.
        documents: Vec<Document>,
        /// Events sink.
        events: mpsc::Sender<Event>,
    },
    /// Remove documents from the collection.
    Delete {
        /// The ids to remove.
        ids: Vec<String>,
        /// Events sink.
        events: mpsc::Sender<Event>,
    },
    /// Query the collection.
    Search {
        /// The wire request.
        request: CollectionSearch,
        /// Events sink.
        events: mpsc::Sender<Event>,
    },
}

impl CollectionMsg {
    fn events(&self) -> mpsc::Sender<Event> {
        match self {
            Self::Upsert { events, .. }
            | Self::Delete { events, .. }
            | Self::Search { events, .. } => events.clone(),
        }
    }
}

/// Everything the registry hands a new collection owner.
pub struct CollectionConfig<T: Tokens> {
    /// The daemon's state root: the collection lives under it.
    pub state: StateDir,
    /// The collection's name.
    pub name: String,
    /// The collection's directory, the key its cache and writer are
    /// registered under.
    pub key: PathBuf,
    /// The scheduler's id for the collection, under which its query
    /// statistics accumulate.
    pub id: WorktreeId,
    /// The scheduler's repository id of the collection's own cache.
    pub repo: RepoId,
    /// The model-id vectors are stored under.
    pub model_id: String,
    /// The model readiness watch.
    pub model: watch::Receiver<EmbeddingState<T>>,
    /// The registry, for the collection's cache and writer.
    pub registry: RegistryHandle,
    /// Set when the daemon begins shutdown.
    pub stopping: watch::Receiver<bool>,
}

/// Spawns the owner task; the returned sender is its inbox.
pub fn spawn<T: Tokens>(config: CollectionConfig<T>) -> mpsc::Sender<CollectionMsg> {
    let (tx, rx) = mpsc::channel(64);
    let root = config.state.collection_docs(&config.name);
    tokio::spawn(
        Collection {
            state: config.state,
            name: config.name,
            key: config.key,
            id: config.id,
            repo: config.repo,
            model_id: config.model_id,
            model: config.model,
            registry: config.registry,
            stopping: config.stopping,
            rx,
            root,
            open: None,
        }
        .run(),
    );
    tx
}

struct Collection<T: Tokens> {
    state: StateDir,
    name: String,
    key: PathBuf,
    id: WorktreeId,
    repo: RepoId,
    model_id: String,
    model: watch::Receiver<EmbeddingState<T>>,
    registry: RegistryHandle,
    stopping: watch::Receiver<bool>,
    rx: mpsc::Receiver<CollectionMsg>,
    /// The directory the documents are files in.
    root: PathBuf,
    open: Option<Open>,
}

/// The collection's stores, opened by the first request and synced to the
/// directory.
struct Open {
    index: Index,
    writer: WriterHandle,
    cache: VectorCache,
    reconciler: Reconciler,
    /// Built by the first search from the reconciler's books; every later
    /// pass's presence follows.
    retriever: Option<Retriever<AneHandle>>,
    /// The last pass, with coverage counted after its embedding.
    report: Report,
}

impl<T: Tokens> Collection<T> {
    async fn run(mut self) {
        let mut stopping = self.stopping.clone();
        loop {
            tokio::select! {
                msg = self.rx.recv() => {
                    let Some(msg) = msg else { break };
                    let events = msg.events();
                    // The request runs to completion; a client hang-up
                    // cancels only the search's retrieval.
                    let quit = tokio::select! {
                        _ = self.handle(msg) => false,
                        _ = stopping.changed() => {
                            send(&events, shutting_down()).await;
                            true
                        }
                    };
                    if quit {
                        break;
                    }
                }
                _ = stopping.changed() => break,
            }
        }
    }

    /// The `Ready` model, or `None` while it is still on its way.
    fn ready(&self) -> Option<Arc<Embedding<T>>> {
        match &*self.model.borrow() {
            EmbeddingState::Ready(model) => Some(model.clone()),
            _ => None,
        }
    }

    async fn handle(&mut self, msg: CollectionMsg) {
        let events = msg.events();
        let Some(model) = self.ready() else {
            send(
                &events,
                error(ErrorKind::Model, "the model is not ready".into()),
            )
            .await;
            return;
        };
        if let Err(message) = self.ensure_open(&model).await {
            send(&events, error(ErrorKind::Internal, message)).await;
            return;
        }
        match msg {
            CollectionMsg::Upsert { documents, events } => {
                self.upsert(&model, documents, &events).await;
            }
            CollectionMsg::Delete { ids, events } => self.delete(&model, ids, &events).await,
            CollectionMsg::Search { request, events } => {
                self.search(&model, request, &events).await;
            }
        }
    }

    /// Opens the stores on first use and brings them to the directory's
    /// state, embedding any chunk a crash left without a vector.
    async fn ensure_open(&mut self, model: &Arc<Embedding<T>>) -> Result<(), String> {
        if self.open.is_some() {
            return Ok(());
        }
        let shared = self
            .registry
            .shared(self.key.clone())
            .await
            .ok_or_else(|| "the daemon is shutting down".to_owned())?
            .map_err(|e| e.to_string())?;
        let index = Index::open_at(&self.root, self.state.collection_index(&self.name))
            .await
            .map_err(|e| e.to_string())?;
        let reconciler = Reconciler::new(
            index.clone(),
            shared.cache.clone(),
            self.model_id.clone(),
            self.root.clone(),
        );
        self.open = Some(Open {
            index,
            writer: shared.writer,
            cache: shared.cache,
            reconciler,
            retriever: None,
            report: Report::default(),
        });
        if let Err(message) = self.sync(model).await {
            self.open = None;
            return Err(message);
        }
        Ok(())
    }

    /// Reconciles the index with the directory, then embeds every chunk
    /// without a vector. Returns how many chunks it embedded. The pass runs
    /// inside a scheduler window, which a `Documents` job lands in.
    async fn sync(&mut self, model: &Arc<Embedding<T>>) -> Result<u64, String> {
        let sched = model.scheduler.clone();
        sched.barrier(self.repo, true).await;
        let synced = self.sync_in_window(model).await;
        sched.barrier(self.repo, false).await;
        synced
    }

    async fn sync_in_window(&mut self, model: &Arc<Embedding<T>>) -> Result<u64, String> {
        let (worktree, repo) = (self.id, self.repo);
        let open = self.open.as_mut().expect("opened by ensure_open");
        open.writer
            .flush()
            .await
            .map_err(|e| format!("vector writer flush: {e}"))?;
        let reconciled = open
            .reconciler
            .reconcile(CHUNKER_FINGERPRINT, RECONCILE, model.counter.clone())
            .await
            .map_err(|e| e.to_string())?;
        if let Some(retriever) = &open.retriever {
            retriever
                .follow(reconciled.presence)
                .await
                .map_err(|e| e.to_string())?;
        }
        let known = reconciled
            .changes
            .map(|changes| changes.added)
            .unwrap_or_default();
        let pending = open
            .reconciler
            .missing(true)
            .await
            .map_err(|e| e.to_string())?
            .pending(known)
            .await
            .map_err(|e| e.to_string())?
            .chunks;
        let ane = AneHandle {
            worktree,
            repo,
            model_id: model.id.clone(),
            sched: model.scheduler.tx(),
            writer: open.writer.clone(),
        };
        let mut skipped = 0u64;
        for batch in pending.chunks(EMBED_BATCH) {
            skipped += ane
                .documents(batch.to_vec())
                .await
                .map_err(|e| e.to_string())?;
        }
        let counts = open
            .reconciler
            .counts(true)
            .await
            .map_err(|e| e.to_string())?;
        let mut report = reconciled.report;
        report.chunks_total = counts.chunks_total;
        report.chunks_with_vector = counts.chunks_with_vector;
        open.report = report;
        Ok(pending.len() as u64 - skipped)
    }

    async fn upsert(
        &mut self,
        model: &Arc<Embedding<T>>,
        documents: Vec<Document>,
        events: &mpsc::Sender<Event>,
    ) {
        if let Err(message) = distinct_ids(documents.iter().map(|d| d.id.as_str())) {
            send(events, error(ErrorKind::BadRequest, message)).await;
            return;
        }
        let root = self.root.clone();
        let written: Result<Vec<bool>, io::Error> = stream::iter(documents)
            .map(move |document| write_document(root.clone(), document))
            .buffer_unordered(FILE_CONCURRENCY)
            .try_collect()
            .await;
        match written {
            Ok(written) => {
                let changed = written.iter().filter(|&&written| written).count() as u64;
                let unchanged = written.len() as u64 - changed;
                self.report(model, changed, unchanged, events).await;
            }
            Err(e) => {
                send(events, error(ErrorKind::Internal, e.to_string())).await;
            }
        }
    }

    async fn delete(
        &mut self,
        model: &Arc<Embedding<T>>,
        ids: Vec<String>,
        events: &mpsc::Sender<Event>,
    ) {
        if let Err(message) = distinct_ids(ids.iter().map(String::as_str)) {
            send(events, error(ErrorKind::BadRequest, message)).await;
            return;
        }
        let root = self.root.clone();
        let removed: Result<Vec<bool>, io::Error> = stream::iter(ids)
            .map(move |id| remove_document(root.clone(), id))
            .buffer_unordered(FILE_CONCURRENCY)
            .try_collect()
            .await;
        match removed {
            Ok(removed) => {
                let changed = removed.iter().filter(|&&removed| removed).count() as u64;
                let unchanged = removed.len() as u64 - changed;
                self.report(model, changed, unchanged, events).await;
            }
            Err(e) => {
                send(events, error(ErrorKind::Internal, e.to_string())).await;
            }
        }
    }

    /// Syncs after a write and answers it.
    async fn report(
        &mut self,
        model: &Arc<Embedding<T>>,
        changed: u64,
        unchanged: u64,
        events: &mpsc::Sender<Event>,
    ) {
        let embedded = match self.sync(model).await {
            Ok(embedded) => embedded,
            Err(message) => {
                send(events, error(ErrorKind::Internal, message)).await;
                return;
            }
        };
        let report = &self.open.as_ref().expect("opened by ensure_open").report;
        let reply = CollectionReport {
            changed,
            unchanged,
            embedded,
            chunks_total: report.chunks_total,
            chunks_with_vector: report.chunks_with_vector,
        };
        tracing::info!(collection = %self.name, changed, unchanged, embedded, "collection written");
        let _ = send(events, Event::Final(Final::Collection(reply))).await;
    }

    async fn search(
        &mut self,
        model: &Arc<Embedding<T>>,
        request: CollectionSearch,
        events: &mpsc::Sender<Event>,
    ) {
        let total_start = Instant::now();
        let open = self.open.as_mut().expect("opened by ensure_open");
        if open.retriever.is_none() {
            let threads = match std::thread::available_parallelism() {
                Ok(threads) => threads,
                Err(e) => {
                    send(events, error(ErrorKind::Internal, e.to_string())).await;
                    return;
                }
            };
            let ane = AneHandle {
                worktree: self.id,
                repo: self.repo,
                model_id: model.id.clone(),
                sched: model.scheduler.tx(),
                writer: open.writer.clone(),
            };
            let pairs = open
                .reconciler
                .books()
                .expect("a sync that succeeded leaves books")
                .pairs();
            match Retriever::new(
                open.index.clone(),
                open.cache.clone(),
                ane,
                self.root.clone(),
                threads,
                pairs,
            ) {
                Ok(retriever) => open.retriever = Some(retriever),
                Err(e) => {
                    send(events, error(ErrorKind::Internal, e.to_string())).await;
                    return;
                }
            }
        }
        // Reset the accumulated query stats so the answer reports only its
        // own.
        if model.scheduler.take_stats(self.id).await.is_err() {
            send(events, shutting_down()).await;
            return;
        }
        // Only the documentation list: a collection holds no code, and the
        // code query is never embedded.
        let core_request = search::Request {
            query: request.query,
            mode: request.mode.to_core(),
            filters: Filters {
                kinds: vec!["doc".to_owned()],
                ..Filters::default()
            },
            budget: request.budget as usize,
            files_only: request.ids_only,
            include_ignored: true,
        };
        let reconciliation = Reconciliation::new(self.name.clone(), &open.report);
        let counter = BudgetCounter::<T>(model.counter.clone());
        // The retrieval's window, as a worktree search's: its on-the-spot
        // `Documents` jobs land inside it.
        model.scheduler.barrier(self.repo, true).await;
        let result = tokio::select! {
            result = open
                .retriever
                .as_mut()
                .expect("built above")
                .search(&core_request, reconciliation, &counter) => Some(result),
            _ = events.closed() => None,
        };
        model.scheduler.barrier(self.repo, false).await;
        let Some(result) = result else { return };
        let stats = match model.scheduler.take_stats(self.id).await {
            Ok(stats) => stats,
            Err(_) => {
                send(events, shutting_down()).await;
                return;
            }
        };
        let mut results = match result {
            Ok(results) => results,
            Err(e) => {
                send(events, search_failure(&e)).await;
                return;
            }
        };
        if let Err(message) = self.name_documents(&mut results).await {
            send(events, error(ErrorKind::Internal, message)).await;
            return;
        }
        let total_us = total_start.elapsed().as_micros() as u64;
        tracing::info!(collection = %self.name, total_us, "collection search answered");
        let answer = Answer {
            output: match request.format {
                Format::Text => results.to_text(),
                Format::Json => results.to_json(),
            },
            has_results: !results.is_empty(),
            stats: QueryStats {
                barrier_us: 0,
                indexing_wait_us: stats.wait.as_micros() as u64,
                indexing_overtaken: stats.overtaken,
                embedded_on_the_spot: results.embedded_on_the_spot,
                total_us,
            },
        };
        let _ = send(events, Event::Final(Final::Answer(answer))).await;
    }

    /// Replaces each hit's file name with the id of the document the file
    /// holds.
    async fn name_documents(&self, results: &mut search::Results) -> Result<(), String> {
        let Some(docs) = &mut results.docs else {
            return Ok(());
        };
        let root = self.root.clone();
        let files: Vec<String> = docs.iter().map(|hit| hit.path.clone()).collect();
        let ids: Vec<String> = stream::iter(files)
            .map(move |file| read_id(root.clone(), file))
            .buffered(FILE_CONCURRENCY)
            .try_collect()
            .await?;
        for (hit, id) in docs.iter_mut().zip(ids) {
            hit.path = id;
        }
        Ok(())
    }
}

/// Fails when `ids` holds an empty id or one twice: both name no single
/// document.
fn distinct_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        if id.is_empty() {
            return Err("a document id is empty".to_owned());
        }
        if !seen.insert(id) {
            return Err(format!("document id {id:?} appears twice in the request"));
        }
    }
    Ok(())
}

/// The document `id`'s file name: the hash of the id, which any string
/// has and no filesystem folds together.
fn file_name(id: &str) -> String {
    format!("{}.md", hex::encode(Sha256::digest(id.as_bytes())))
}

/// A document as the file that holds it.
#[derive(Template)]
#[template(path = "collection-document.md")]
struct DocumentFile<'a> {
    /// The id as a JSON string, one line of valid YAML.
    id: String,
    /// The title on one line; empty for no heading.
    title: String,
    text: &'a str,
}

/// The bytes of `document`'s file.
fn render(document: &Document) -> String {
    DocumentFile {
        id: serde_json::to_string(&document.id).expect("a string serializes"),
        title: document
            .title
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        text: &document.text,
    }
    .render()
    .expect("the template renders its own typed context")
}

/// The id the file `src` holds its document under.
fn document_id(src: &str) -> Option<String> {
    let mut lines = src.lines();
    if lines.next()? != "---" {
        return None;
    }
    serde_json::from_str(lines.next()?.strip_prefix("id: ")?).ok()
}

/// The id of the document the file `file` of the collection holds.
async fn read_id(root: PathBuf, file: String) -> Result<String, String> {
    let src = tokio::fs::read_to_string(root.join(&file))
        .await
        .map_err(|e| format!("{file}: {e}"))?;
    document_id(&src).ok_or_else(|| format!("{file} holds no document id"))
}

/// Writes `document`'s file unless it already holds exactly these bytes;
/// whether it wrote. The file appears whole under its name by a rename,
/// so a reconcile pass never reads half of one.
async fn write_document(root: PathBuf, document: Document) -> io::Result<bool> {
    let path = root.join(file_name(&document.id));
    let bytes = render(&document).into_bytes();
    match tokio::fs::read(&path).await {
        Ok(stored) if stored == bytes => return Ok(false),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut partial = path.clone().into_os_string();
    partial.push(".tmp");
    let partial = PathBuf::from(partial);
    tokio::fs::write(&partial, &bytes).await?;
    tokio::fs::rename(&partial, &path).await?;
    Ok(true)
}

/// Removes the file of the document `id`; whether the collection held it.
async fn remove_document(root: PathBuf, id: String) -> io::Result<bool> {
    match tokio::fs::remove_file(root.join(file_name(&id))).await {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(id: &str, title: &str, text: &str) -> Document {
        Document {
            id: id.to_owned(),
            title: title.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_file_names_its_document() {
        let id = "owner/repo#12 \"quoted\"\nline";
        let file = render(&document(id, "A title", "Body <b>text</b> & more"));
        assert_eq!(document_id(&file).as_deref(), Some(id));
        assert!(
            file.ends_with("# A title\n\nBody <b>text</b> & more"),
            "the text is written unescaped after the title heading: {file:?}"
        );
    }

    #[test]
    fn a_title_is_one_line_and_an_empty_one_has_no_heading() {
        let file = render(&document("a", "two\nlines  here", "text"));
        assert!(file.contains("\n# two lines here\n\ntext"), "{file:?}");
        let file = render(&document("a", "  ", "text"));
        assert!(file.ends_with("---\ntext"), "{file:?}");
    }

    #[test]
    fn file_names_differ_per_id_and_are_case_safe() {
        assert_ne!(file_name("A"), file_name("a"));
        let name = file_name("a");
        assert_eq!(name, name.to_lowercase());
        assert_eq!(name, file_name("a"));
    }

    #[test]
    fn ids_must_be_distinct_and_non_empty() {
        assert!(distinct_ids(["a", "b"].into_iter()).is_ok());
        assert!(distinct_ids(["a", "a"].into_iter()).is_err());
        assert!(distinct_ids([""].into_iter()).is_err());
    }
}
