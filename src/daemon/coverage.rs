//! `index --wait`'s coverage, read from the worktree owner's books.
//!
//! The wait runs in a task the owner spawns, so it cannot borrow the
//! owner's [`Reconciler`]. It asks instead: the owner answers each [`Ask`]
//! between requests, from the books its barriers keep. An answer reads only
//! the vectors the cache stored since the books last looked — no chunk row
//! and no hash lookup — so a progress frame costs the same at any index
//! size. The books stay single-owned; the wait holds no index or cache
//! handle at all.

use repo_search_core::reconcile::{Counts, Reconciler};
use tokio::sync::{mpsc, oneshot};

/// Asks queued at once; each wait keeps at most one outstanding.
const ASKS: usize = 8;

/// A wait's ask for the coverage of its scope now.
#[derive(Debug)]
pub struct Ask {
    reply: oneshot::Sender<Result<Counts, String>>,
}

impl Ask {
    /// Answers from `reconciler`'s books in `index --wait`'s scope — never
    /// ignored files, as its barrier. `None` is an owner whose index is not
    /// open: a reopen after a recreated `.repo-search/` failed.
    pub async fn answer(self, reconciler: Option<&mut Reconciler>) {
        let counts = match reconciler {
            Some(reconciler) => reconciler.counts(false).await.map_err(|e| e.to_string()),
            None => Err("the worktree's index is not open".to_owned()),
        };
        if let Ok(counts) = &counts {
            tracing::debug!(
                chunk_rows = counts.reads.chunk_rows,
                cache_probes = counts.reads.cache_probes,
                cache_slots = counts.reads.cache_slots,
                "index wait counted"
            );
        }
        // A wait that hung up no longer needs the answer.
        let _ = self.reply.send(counts);
    }
}

/// A wait's handle on the owner's books; the owner keeps one so its inbox
/// of asks never closes while it runs.
#[derive(Debug, Clone)]
pub struct Asker {
    tx: mpsc::Sender<Ask>,
}

/// Why an ask has no counts.
#[derive(Debug, thiserror::Error)]
pub enum AskError {
    /// The owner is gone, so its books are.
    #[error("the worktree owner is gone")]
    OwnerGone,
    /// The owner's books could not be read.
    #[error("{0}")]
    Count(String),
}

impl Asker {
    /// The coverage now, as the owner's books count it.
    ///
    /// # Errors
    ///
    /// When the owner is gone or its books could not be read.
    pub async fn now(&self) -> Result<Counts, AskError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Ask { reply })
            .await
            .map_err(|_| AskError::OwnerGone)?;
        rx.await
            .map_err(|_| AskError::OwnerGone)?
            .map_err(AskError::Count)
    }
}

/// The owner's asker and the inbox its asks arrive on.
#[must_use]
pub fn channel() -> (Asker, mpsc::Receiver<Ask>) {
    let (tx, rx) = mpsc::channel(ASKS);
    (Asker { tx }, rx)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;
    use std::sync::Arc;

    use repo_search_core::reconcile::{Options, Reads};
    use repo_search_core::store::{ContentHash, Index, VectorCache};
    use repo_search_core::{CHUNKER_FINGERPRINT, TokenCounter, UnitVector};

    use super::*;

    const MODEL: &str = "test-model";

    /// Reports zero tokens, so the chunk cap never binds.
    struct CapOff;

    impl TokenCounter for CapOff {
        fn count(&self, _: &str) -> usize {
            0
        }

        fn fits(&self, _: &str, _: &[&str], _: usize) -> bool {
            true
        }

        fn fit_end(&self, _: &str, _: &[&str], _: usize) -> Option<usize> {
            None
        }

        fn document_tokens(&self, _: &str, _: &str) -> usize {
            0
        }
    }

    /// A worktree of `files` modules of three functions, each above the
    /// chunker's pack threshold so it is a chunk of its own.
    fn write_modules(root: &Path, files: usize) {
        for i in 0..files {
            let module: String = ["alpha", "beta", "gamma"]
                .iter()
                .map(|name| {
                    let body = format!("{name} {i} ").repeat(60);
                    format!("pub fn {name}_{i}() -> &'static str {{\n    \"{body}\"\n}}\n\n")
                })
                .collect();
            std::fs::write(root.join(format!("m{i}.rs")), module).expect("module written");
        }
    }

    /// Stores a vector for each of `hashes`, as the writer would.
    async fn store(cache: &VectorCache, hashes: &[ContentHash]) {
        let vectors = hashes
            .iter()
            .map(|hash| (*hash, UnitVector::normalize(vec![1.0, 2.0])))
            .collect();
        cache.put_many(MODEL, vectors).await.expect("stored");
    }

    /// Rows of `rows` whose hash is in `cached`: coverage recounted.
    fn recount(rows: &[ContentHash], cached: &HashSet<ContentHash>) -> u64 {
        rows.iter().filter(|hash| cached.contains(hash)).count() as u64
    }

    /// What every frame of one simulated `index --wait` read, with the
    /// owner answering from a reconciler that ran one barrier, and how many
    /// distinct hashes the index holds.
    async fn frame_reads(files: usize) -> (Vec<Reads>, u64) {
        let tree = tempfile::tempdir().expect("tempdir");
        let common = tempfile::tempdir().expect("tempdir");
        write_modules(tree.path(), files);
        let index = Index::open(tree.path()).await.expect("index");
        let cache = VectorCache::open(common.path()).await.expect("cache");
        let mut reconciler = Reconciler::new(index.clone(), cache.clone(), MODEL, tree.path());
        reconciler
            .reconcile(CHUNKER_FINGERPRINT, Options::default(), Arc::new(CapOff))
            .await
            .expect("the barrier's pass");

        // The owner: its books answer each ask, as `Owner::run` does.
        let (asker, mut asks) = channel();
        let owner = tokio::spawn(async move {
            while let Some(ask) = asks.recv().await {
                ask.answer(Some(&mut reconciler)).await;
            }
        });

        let rows = index.chunk_hashes(false).await.expect("rows");
        let mut distinct: Vec<ContentHash> = Vec::new();
        let mut seen = HashSet::new();
        for hash in &rows {
            if seen.insert(*hash) {
                distinct.push(*hash);
            }
        }
        assert!(distinct.len() >= 3 * files, "three chunks per module");

        let mut reads = Vec::new();
        let mut cached = HashSet::new();
        let mut frame = async |cached: &HashSet<ContentHash>| {
            let counts = asker.now().await.expect("the owner answers");
            assert_eq!(counts.chunks_total, rows.len() as u64);
            assert_eq!(counts.chunks_with_vector, recount(&rows, cached));
            reads.push(counts.reads);
        };
        // Nothing embedded yet, then three vectors at a time — the same
        // number at every size — then nothing more.
        frame(&cached).await;
        for batch in distinct.chunks(3).take(4) {
            store(&cache, batch).await;
            cached.extend(batch.iter().copied());
            frame(&cached).await;
            frame(&cached).await;
        }
        // The rest at once, as a drained queue leaves it for `Indexed`.
        let rest = &distinct[12..];
        store(&cache, rest).await;
        cached.extend(rest.iter().copied());
        frame(&cached).await;
        assert_eq!(cached.len(), distinct.len(), "every hash is stored");

        drop(asker);
        owner.await.expect("the owner ends with its last asker");
        (reads, distinct.len() as u64)
    }

    #[tokio::test]
    async fn a_frame_reads_only_the_vectors_stored_since_the_last() {
        let three = Reads {
            chunk_rows: 0,
            cache_probes: 0,
            cache_slots: 3,
        };
        for files in [5, 50] {
            let (reads, distinct) = frame_reads(files).await;
            let rest = distinct - 12;
            assert_eq!(
                reads,
                [
                    Reads::default(),
                    three,
                    Reads::default(),
                    three,
                    Reads::default(),
                    three,
                    Reads::default(),
                    three,
                    Reads::default(),
                    Reads {
                        cache_slots: rest,
                        ..Reads::default()
                    },
                ],
                "no frame reads a chunk row or looks up a hash, at {files} files"
            );
        }
    }

    #[tokio::test]
    async fn an_owner_without_an_index_answers_an_error() {
        let (asker, mut asks) = channel();
        let owner = tokio::spawn(async move {
            let ask = asks.recv().await.expect("one ask");
            ask.answer(None).await;
        });
        assert!(matches!(asker.now().await, Err(AskError::Count(_))));
        owner.await.expect("the owner answered");
        assert!(matches!(asker.now().await, Err(AskError::OwnerGone)));
    }
}
