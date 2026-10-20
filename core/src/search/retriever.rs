//! The query pipeline: candidates, on-the-spot embedding, fusion, ranking,
//! text and budget.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;

use tokio::sync::oneshot;

use crate::qos::{QosClass, set_current_thread};
use crate::reconcile::PendingChunk;
use crate::store::{ChunkFilter, ChunkId, ChunkRow, ContentHash, Index, Side, VectorCache};
use crate::{QueryTask, UnitVector};

use super::EmbedService;
use super::fuse::{Fused, fuse};
use super::output::{
    self, Freshness, RankedChunk, ReadError, Reconciliation, Results, Selector, Step,
};
use super::scope::Scope;
use super::terms::{QueryTerms, SymbolName};
use super::vectors::Vectors;
use super::{CANDIDATES, Error, Mode, ON_THE_SPOT, Request, TokenCounter};

/// Chunk texts read per request once the selection needs one: the next few
/// units of the list are read together so that a file shared by neighbours is
/// read once.
const TEXT_WINDOW: usize = 8;

/// Member kinds that never define their symbol: an `impl` block names a type
/// without defining it.
const NON_DEFINITION_KINDS: &[&str] = &["impl_item"];

/// Answers queries against one worktree.
///
/// Embeds through the caller's [`EmbedService`] and owns the in-memory vector
/// indexes of the worktree's chunks, one per list (on a thread of its own),
/// kept in step with the index and the cache by every search. The thread does
/// not block the executor, and a search dropped half way leaves it usable.
pub struct Retriever<S: EmbedService> {
    index: Index,
    cache: VectorCache,
    service: S,
    worktree: PathBuf,
    threads: NonZeroUsize,
    vectors: Vectors,
}

impl<S: EmbedService> Retriever<S> {
    /// A retriever over `worktree`, whose chunks `index` stores, extending
    /// `cache` through `service`. KNN scans run on `knn_threads` threads
    /// (the performance cores, see [`qos::performance_cores`](crate::qos)).
    ///
    /// # Errors
    ///
    /// When a worker thread cannot be spawned.
    pub fn new(
        index: Index,
        cache: VectorCache,
        service: S,
        worktree: PathBuf,
        knn_threads: NonZeroUsize,
    ) -> io::Result<Self> {
        Ok(Self {
            index,
            cache,
            service,
            worktree,
            threads: knn_threads,
            vectors: Vectors::spawn()?,
        })
    }

    /// Answers `request`. `reconciliation` is what the caller's freshness
    /// barrier just did; every chunk returned is read from the working tree
    /// and checked against its indexed hash. Searches are one at a time
    /// (`&mut self`): each brings the vector indexes up to date first.
    ///
    /// # Errors
    ///
    /// When the query is empty, a filter is invalid, the index, cache,
    /// embedder or filesystem fails, or a file changed after the barrier
    /// ([`Error::Stale`], reported only for a chunk the answer reaches).
    pub async fn search(
        &mut self,
        request: &Request,
        reconciliation: Reconciliation,
        counter: &impl TokenCounter,
    ) -> Result<Results, Error<S::Error>> {
        let query = request.query.trim();
        if query.is_empty() {
            return Err(Error::EmptyQuery);
        }
        let terms = QueryTerms::parse(query);
        let use_bm25 = request.mode != Mode::Semantic;
        let use_vector = request.mode != Mode::Lexical;

        let (files, kinds, has_ignored) = tokio::join!(
            async {
                if request.filters.paths.is_empty() {
                    Ok(Vec::new())
                } else {
                    self.index.file_states().await
                }
            },
            async {
                if request.filters.kinds.is_empty() {
                    Ok(Vec::new())
                } else {
                    self.index.kinds().await
                }
            },
            async {
                if request.include_ignored {
                    Ok(true)
                } else {
                    self.index.has_ignored().await
                }
            },
        );
        let files = files?;
        let kinds = kinds?;
        // With no ignored file indexed there is nothing to hide, so an
        // unfiltered search stays unrestricted and needs no mask.
        let include_ignored = request.include_ignored || !has_ignored?;
        let scope = Scope::resolve(&request.filters, include_ignored, &files, &kinds)?;
        let docs_present = match scope.docs {
            Some(_) => self.index.count(Side::Docs, include_ignored).await? > 0,
            None => false,
        };
        let filters = [scope.code, scope.docs.filter(|_| docs_present)];

        // Lexical candidates, query vectors and the in-memory index sync are
        // independent.
        let model = self.service.model_id().to_owned();
        let bm25_expr = terms.fts_match().filter(|_| use_bm25);
        let (bm25_code, bm25_docs, query_code, query_docs, synced) = tokio::join!(
            bm25::<S::Error>(&self.index, bm25_expr.as_deref(), filters[0].as_ref()),
            bm25::<S::Error>(&self.index, bm25_expr.as_deref(), filters[1].as_ref()),
            self.query_vector(
                QueryTask::CodeRetrieval,
                query,
                use_vector && filters[0].is_some()
            ),
            self.query_vector(
                QueryTask::QuestionAnswering,
                query,
                use_vector && filters[1].is_some()
            ),
            async {
                if use_vector {
                    self.vectors
                        .sync::<S::Error>(&self.index, &self.cache, &model)
                        .await
                } else {
                    Ok(())
                }
            },
        );
        synced?;
        let bm25_lists = [bm25_code?, bm25_docs?];
        let query_vectors = [query_code?, query_docs?];

        // Lexical candidates without a vector are embedded now, so the vector
        // channel does not miss the chunks BM25 found.
        let mut embedded = 0;
        if use_vector && use_bm25 && reconciliation.chunks_with_vector < reconciliation.chunks_total
        {
            embedded = self
                .embed_on_the_spot(&bm25_lists, &model, include_ignored)
                .await?;
        }

        let (vector_code, vector_docs) = tokio::join!(
            self.vector_candidates(Side::Code, query_vectors[0].as_ref(), filters[0].as_ref()),
            self.vector_candidates(Side::Docs, query_vectors[1].as_ref(), filters[1].as_ref()),
        );
        let fused = [
            fuse(&bm25_lists[0], &vector_code?),
            fuse(&bm25_lists[1], &vector_docs?),
        ];

        let mut ranked = self.rank(fused, terms.symbols()).await?;
        let freshness = Freshness::new(reconciliation, embedded);

        let mut selector = Selector::new(request.budget, request.files_only);
        loop {
            match selector.advance([&ranked[0], &ranked[1]], counter)? {
                Step::Done => break,
                Step::Needs { side, from } => {
                    self.read_window(&mut ranked[side as usize], from).await?;
                }
            }
        }
        let [code_groups, doc_groups] = selector.finish();
        Ok(Results {
            code: output::code_hits(&ranked[0], code_groups),
            docs: output::doc_hits(&ranked[1], doc_groups),
            freshness,
        })
    }

    async fn query_vector(
        &self,
        task: QueryTask,
        query: &str,
        wanted: bool,
    ) -> Result<Option<UnitVector>, Error<S::Error>> {
        if !wanted {
            return Ok(None);
        }
        self.service
            .query(task, query.to_owned())
            .await
            .map(Some)
            .map_err(Error::Embed)
    }

    /// Embeds the lexical candidates that have no vector among the first
    /// [`ON_THE_SPOT`] of each list, stores them and adds them to the
    /// in-memory indexes. Returns how many chunk rows gained a vector.
    ///
    /// The window is taken first and filtered second: it is the best
    /// candidates, however many of them already have a vector. A candidate
    /// whose file changed since the barrier is left out of the batch — it
    /// stays lexical-only and the output reports it only if it is reached.
    async fn embed_on_the_spot(
        &self,
        bm25_lists: &[Vec<ChunkId>; 2],
        model: &str,
        include_ignored: bool,
    ) -> Result<u64, Error<S::Error>> {
        let mut missing: Vec<ChunkRow> = Vec::new();
        let mut seen: HashSet<ContentHash> = HashSet::new();
        let mut sides: HashMap<ContentHash, Vec<Side>> = HashMap::new();
        // The two lists' reads are independent, on the index and the vectors
        // actors alike: the rows are fetched together, then what is missing.
        let (code_rows, doc_rows) = tokio::join!(
            self.index
                .chunks(bm25_lists[0].iter().copied().take(ON_THE_SPOT).collect()),
            self.index
                .chunks(bm25_lists[1].iter().copied().take(ON_THE_SPOT).collect()),
        );
        let rows = [code_rows?, doc_rows?];
        let [code_hashes, doc_hashes] = rows
            .each_ref()
            .map(|rows| rows.iter().map(|row| row.content_hash).collect::<Vec<_>>());
        let (code_lacking, doc_lacking) = tokio::join!(
            self.vectors.lacking::<S::Error>(code_hashes),
            self.vectors.lacking::<S::Error>(doc_hashes),
        );
        let lacking = [
            code_lacking?.into_iter().collect::<HashSet<_>>(),
            doc_lacking?.into_iter().collect::<HashSet<_>>(),
        ];
        for (rows, lacking) in rows.into_iter().zip(lacking) {
            for row in &rows {
                let side = Side::of_kind(&row.kind);
                let entry = sides.entry(row.content_hash).or_default();
                if !entry.contains(&side) {
                    entry.push(side);
                }
            }
            missing.extend(
                rows.into_iter()
                    .filter(|row| lacking.contains(&row.content_hash))
                    .filter(|row| seen.insert(row.content_hash)),
            );
        }
        if missing.is_empty() {
            return Ok(0);
        }
        let texts = read_texts::<S::Error>(&self.worktree, missing.clone()).await?;
        let pending: Vec<PendingChunk> = missing
            .iter()
            .zip(texts)
            .filter_map(|(row, text)| match text {
                Ok(text) => Some(PendingChunk {
                    path: row.path.clone(),
                    title: row.title.clone(),
                    text,
                    lines: row.lines.clone(),
                    content_hash: row.content_hash,
                }),
                Err(error) => {
                    tracing::debug!(
                        path = %row.path,
                        ?error,
                        "on-the-spot embedding skipped an unreadable chunk"
                    );
                    None
                }
            })
            .collect();
        if pending.is_empty() {
            return Ok(0);
        }
        let vectors = self
            .service
            .documents(pending.clone())
            .await
            .map_err(Error::Embed)?;
        assert_eq!(
            vectors.len(),
            pending.len(),
            "the embedder must return one vector per chunk"
        );
        self.cache
            .put_many(
                model,
                pending
                    .iter()
                    .map(|p| p.content_hash)
                    .zip(vectors)
                    .collect(),
            )
            .await?;
        let hashes: Vec<ContentHash> = pending.iter().map(|p| p.content_hash).collect();
        let stored = self.cache.get_many(model, &hashes).await?;
        self.vectors
            .adopt::<S::Error>(
                &self.cache,
                model,
                stored
                    .into_iter()
                    .flat_map(|v| {
                        sides[&v.content_hash]
                            .iter()
                            .map(move |&side| (v.content_hash, v.slot, side))
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
            .await?;
        // The coverage figure counts chunk rows: every row holding one of the
        // embedded hashes, on either side, gained a vector.
        let unrestricted = |side| ChunkFilter {
            side,
            include_ignored,
            paths: None,
            extensions: Vec::new(),
            kinds: Vec::new(),
            symbol: None,
        };
        let (code, docs) = tokio::join!(
            self.index
                .chunks_with_hashes(hashes.clone(), unrestricted(Side::Code)),
            self.index
                .chunks_with_hashes(hashes.clone(), unrestricted(Side::Docs)),
        );
        let covered = code?.iter().map(Vec::len).sum::<usize>() as u64
            + docs?.iter().map(Vec::len).sum::<usize>() as u64;
        Ok(covered)
    }

    /// The chunks of `side` nearest to `query` among those that pass
    /// `filter`, best first; empty when the side is out of scope, the mode
    /// has no vector channel, or no vector exists yet.
    ///
    /// The filter is applied inside the scan, so a restrictive filter still
    /// yields the best vectors within it, not the filter's share of the best
    /// vectors overall.
    async fn vector_candidates(
        &self,
        side: Side,
        query: Option<&UnitVector>,
        filter: Option<&ChunkFilter>,
    ) -> Result<Vec<ChunkId>, Error<S::Error>> {
        let (Some(query), Some(filter)) = (query, filter) else {
            return Ok(Vec::new());
        };
        let restrict = if filter.is_unrestricted() {
            None
        } else {
            Some(self.index.hashes_matching(filter.clone()).await?)
        };
        let hashes = self
            .vectors
            .search::<S::Error>(side, query.clone(), self.threads.get(), restrict)
            .await?;
        // Hits are mapped to this worktree's chunks of the side that pass the
        // filter (a content hash can be shared with a chunk outside it).
        let per_hash = self
            .index
            .chunks_with_hashes(hashes, filter.clone())
            .await?;
        let mut seen = HashSet::new();
        Ok(per_hash
            .into_iter()
            .flatten()
            .filter(|id| seen.insert(*id))
            .collect())
    }

    /// Loads the rows of the fused chunks and places definitions of the
    /// symbols the query names (see [`place_definitions`]). Texts stay
    /// unread until the selection reaches them.
    async fn rank(
        &self,
        fused: [Vec<Fused>; 2],
        symbols: &[SymbolName],
    ) -> Result<[Vec<RankedChunk>; 2], Error<S::Error>> {
        let ids: Vec<ChunkId> = fused.iter().flatten().map(|f| f.id).collect();
        let mut rows: HashMap<ChunkId, ChunkRow> = self
            .index
            .chunks(ids)
            .await?
            .into_iter()
            .map(|row| (row.id, row))
            .collect();

        let mut lists: [Vec<RankedChunk>; 2] = [Vec::new(), Vec::new()];
        for (list, fused) in lists.iter_mut().zip(fused) {
            // A row missing here was removed after the candidates were read.
            *list = fused
                .into_iter()
                .filter_map(|f| {
                    Some(RankedChunk {
                        row: rows.remove(&f.id)?,
                        score: f.score,
                        channels: f.channels,
                        text: None,
                    })
                })
                .collect();
            let ids: Vec<ChunkId> = list.iter().map(|c| c.row.id).collect();
            for symbol in symbols {
                let mentions: HashSet<ChunkId> = match symbol.mention_expr() {
                    Some(expr) => self
                        .index
                        .matching(expr, ids.clone())
                        .await?
                        .into_iter()
                        .collect(),
                    None => HashSet::new(),
                };
                place_definitions(list, symbol, &mentions);
            }
        }
        Ok(lists)
    }

    /// Reads the texts of the units from `from` on in the next window of
    /// `list`. A failure to read one unit is kept with it: it is an error
    /// only if the selection reaches that unit.
    async fn read_window(
        &self,
        list: &mut [RankedChunk],
        from: usize,
    ) -> Result<(), Error<S::Error>> {
        let window = from..list.len().min(from + TEXT_WINDOW);
        let rows = list[window.clone()].iter().map(|c| c.row.clone()).collect();
        let texts = read_texts::<S::Error>(&self.worktree, rows).await?;
        for (chunk, text) in list[window].iter_mut().zip(texts) {
            chunk.text = Some(text);
        }
        Ok(())
    }
}

/// Whether a chunk defines `symbol`: one of its members has a symbol path
/// ending in the name and is not an `impl` block.
fn is_definition(chunk: &RankedChunk, symbol: &SymbolName) -> bool {
    chunk.row.members.iter().any(|m| {
        !m.symbol.is_empty()
            && !NON_DEFINITION_KINDS.contains(&m.kind.as_str())
            && symbol.names(&m.symbol)
    })
}

/// Ranks the definitions of `symbol` above its uses, without reordering
/// anything else.
///
/// `list` is in fusion order and `mentions` holds the chunks whose text spells
/// the symbol. The best-ranked chunk that mentions the symbol without
/// defining it is the one a definition has to beat: each definition ranked
/// below it moves to just ahead of it, in its own relative order, and takes
/// that chunk's score so that scores keep descending down the list. A
/// definition already ranked above every such chunk stays, and no chunk that
/// does not mention the symbol is ever passed. A list in which nothing
/// mentions the symbol, or none of the mentions is outranked by a definition,
/// is unchanged.
fn place_definitions(
    list: &mut Vec<RankedChunk>,
    symbol: &SymbolName,
    mentions: &HashSet<ChunkId>,
) {
    let Some(first_use) = list
        .iter()
        .position(|c| mentions.contains(&c.row.id) && !is_definition(c, symbol))
    else {
        return;
    };
    let below: Vec<usize> = (first_use + 1..list.len())
        .filter(|&i| is_definition(&list[i], symbol))
        .collect();
    if below.is_empty() {
        return;
    }
    let score = list[first_use].score;
    // Taken from the back so earlier indices stay valid.
    let mut moved: Vec<RankedChunk> = below.iter().rev().map(|&i| list.remove(i)).collect();
    moved.reverse();
    for chunk in &mut moved {
        chunk.score = chunk.score.max(score);
    }
    list.splice(first_use..first_use, moved);
}

async fn bm25<E: std::error::Error + 'static>(
    index: &Index,
    expr: Option<&str>,
    filter: Option<&ChunkFilter>,
) -> Result<Vec<ChunkId>, Error<E>> {
    match (expr, filter) {
        (Some(expr), Some(filter)) => Ok(index
            .bm25(expr.to_owned(), filter.clone(), CANDIDATES)
            .await?),
        _ => Ok(Vec::new()),
    }
}

/// The text of each of `rows` in order, read from the working tree on a
/// worker thread and checked against the indexed hash. A row that cannot be
/// read has its own error.
async fn read_texts<E: std::error::Error + 'static>(
    root: &Path,
    rows: Vec<ChunkRow>,
) -> Result<Vec<Result<String, ReadError>>, Error<E>> {
    let root = root.to_path_buf();
    run_blocking(move || {
        let mut files: HashMap<&str, Vec<u8>> = HashMap::new();
        rows.iter()
            .map(|row| read_text(&root, row, &mut files))
            .collect()
    })
    .await
}

fn read_text<'a>(
    root: &Path,
    row: &'a ChunkRow,
    files: &mut HashMap<&'a str, Vec<u8>>,
) -> Result<String, ReadError> {
    let stale = || ReadError::Stale {
        path: row.path.clone(),
        start: *row.lines.start(),
        end: *row.lines.end(),
    };
    if !files.contains_key(row.path.as_str()) {
        let bytes = match fs::read(root.join(&row.path)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(stale()),
            Err(e) => {
                return Err(ReadError::Io {
                    path: row.path.clone(),
                    kind: e.kind(),
                    message: e.to_string(),
                });
            }
        };
        files.insert(&row.path, bytes);
    }
    let bytes = &files[row.path.as_str()];
    let mut text = Vec::new();
    for (i, member) in row.members.iter().enumerate() {
        if i > 0 {
            text.push(b'\n');
        }
        let range = usize::try_from(member.bytes.start)
            .ok()
            .zip(usize::try_from(member.bytes.end).ok());
        let slice = range.and_then(|(s, e)| bytes.get(s..e)).ok_or_else(stale)?;
        text.extend_from_slice(slice);
    }
    let text = String::from_utf8(text).map_err(|_| stale())?;
    if ContentHash::of_chunk(&row.title, &text) != row.content_hash {
        return Err(stale());
    }
    Ok(text)
}

/// Runs blocking `work` on its own thread and awaits the result.
async fn run_blocking<T: Send + 'static, E: std::error::Error + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Error<E>> {
    let (tx, rx) = oneshot::channel();
    thread::Builder::new()
        .name("repo-search-query".into())
        .spawn(move || {
            set_current_thread(QosClass::UserInteractive);
            let _ = tx.send(work());
        })
        .map_err(Error::Spawn)?;
    rx.await.map_err(|_| Error::WorkerGone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fuse::Channel;
    use crate::store::{ContentHash, MemberRow};

    fn chunk(id: i64, symbol: &str, kind: &str, score: f64) -> RankedChunk {
        RankedChunk {
            row: ChunkRow {
                id: ChunkId(id),
                path: format!("f{id}.rs"),
                title: format!("f{id}.rs › {symbol}"),
                kind: kind.to_owned(),
                lines: 1..=2,
                content_hash: ContentHash::of_chunk("t", &id.to_string()),
                members: vec![MemberRow {
                    bytes: 0..1,
                    kind: kind.to_owned(),
                    symbol: symbol.to_owned(),
                }],
            },
            score,
            channels: vec![Channel::Bm25],
            text: None,
        }
    }

    fn name(s: &str) -> SymbolName {
        QueryTerms::parse(s).symbols()[0].clone()
    }

    fn order(list: &[RankedChunk]) -> Vec<i64> {
        list.iter().map(|c| c.row.id.0).collect()
    }

    fn mentions(ids: &[i64]) -> HashSet<ChunkId> {
        ids.iter().copied().map(ChunkId).collect()
    }

    #[test]
    fn a_definition_moves_just_ahead_of_the_best_use() {
        // 1 is unrelated and better; 2 and 3 use the symbol; 4 defines it.
        let mut list = vec![
            chunk(1, "other", "function_item", 0.9),
            chunk(2, "caller", "function_item", 0.8),
            chunk(3, "caller2", "function_item", 0.7),
            chunk(4, "ComputedSignal", "struct_item", 0.1),
        ];
        place_definitions(&mut list, &name("ComputedSignal"), &mentions(&[2, 3, 4]));
        assert_eq!(order(&list), [1, 4, 2, 3]);
        // The definition took the score of the use it now precedes.
        assert_eq!(list[1].score, 0.8);
        assert!(list.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn a_definition_already_above_every_use_stays() {
        let mut list = vec![
            chunk(1, "ComputedSignal", "struct_item", 0.9),
            chunk(2, "caller", "function_item", 0.8),
        ];
        place_definitions(&mut list, &name("ComputedSignal"), &mentions(&[1, 2]));
        assert_eq!(order(&list), [1, 2]);
        assert_eq!(list[0].score, 0.9);
    }

    #[test]
    fn without_a_use_nothing_moves() {
        let mut list = vec![
            chunk(1, "other", "function_item", 0.9),
            chunk(2, "ComputedSignal", "struct_item", 0.5),
        ];
        place_definitions(&mut list, &name("ComputedSignal"), &mentions(&[2]));
        assert_eq!(order(&list), [1, 2]);
    }

    #[test]
    fn an_impl_block_is_a_use_not_a_definition() {
        let mut list = vec![
            chunk(1, "ComputedSignal", "impl_item", 0.9),
            chunk(2, "ComputedSignal", "struct_item", 0.5),
        ];
        place_definitions(&mut list, &name("ComputedSignal"), &mentions(&[1, 2]));
        assert_eq!(order(&list), [2, 1]);
    }

    #[test]
    fn several_definitions_keep_their_relative_order() {
        let mut list = vec![
            chunk(1, "use_site", "function_item", 0.9),
            chunk(2, "x", "function_item", 0.8),
            chunk(3, "env_insert", "function_item", 0.7),
            chunk(4, "y", "function_item", 0.6),
            chunk(5, "mod_a::env_insert", "function_item", 0.5),
        ];
        place_definitions(&mut list, &name("env_insert"), &mentions(&[1, 3, 5]));
        assert_eq!(order(&list), [3, 5, 1, 2, 4]);
        assert!(list.windows(2).all(|w| w[0].score >= w[1].score));
    }
}
