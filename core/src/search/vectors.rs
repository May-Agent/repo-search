//! The worktree's in-memory vector indexes, owned by a thread of their own.
//!
//! Searches and updates are messages to that thread, so no future ever holds
//! an index: a query dropped half way leaves nothing outstanding, and the
//! blocking work (building, inserting, scanning) stays off the executor.
//!
//! The thread also keeps the books of what it loaded. A chunk of the
//! worktree is *loaded* (it has a vector in an index) or *absent* (the cache
//! had none when last asked). The cache is asked about a chunk once, when the
//! chunk is first seen; after that only vectors stored *since* (slots from
//! the high-water mark on) can turn an absent chunk into a loaded one.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::PathBuf;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use crate::UnitVector;
use crate::knn::{self, Calibration, KnnIndex, calibration_for, set_user_interactive};
use crate::store::{ContentHash, Index, Side, VectorCache};

use super::Error;

/// Bound on the request queue.
const QUEUE: usize = 16;

/// Vectors read from the cache per request while loading an index.
const LOAD_BATCH: usize = 2048;

/// What the indexes of one model are built from.
struct Build {
    calibration: Calibration,
    file: PathBuf,
    /// The cache's vector count when the calibration was taken.
    vectors: u64,
}

/// What a sync has to look up.
struct Plan {
    /// Chunks the cache was never asked about.
    fresh: Vec<(ContentHash, Side)>,
    /// Where to look for vectors stored since the last sync; `None` before
    /// the first one.
    since: Option<u64>,
}

enum Adopted {
    Done,
    /// There are no indexes yet; resend with a [`Build`].
    NeedsBuild(Vec<(ContentHash, u64, Side)>),
}

enum Request {
    Plan {
        current: Vec<(ContentHash, Side)>,
        cache_vectors: u64,
        reply: oneshot::Sender<Plan>,
    },
    Adopt {
        vectors: Vec<(ContentHash, u64, Side)>,
        build: Option<Build>,
        reply: oneshot::Sender<Result<Adopted, knn::Error>>,
    },
    Settle {
        missing: Vec<(ContentHash, Side)>,
        high_water: u64,
        reply: oneshot::Sender<()>,
    },
    Lacking {
        hashes: Vec<ContentHash>,
        reply: oneshot::Sender<Vec<ContentHash>>,
    },
    Search {
        side: Side,
        query: UnitVector,
        threads: usize,
        restrict: Option<Vec<ContentHash>>,
        reply: oneshot::Sender<Vec<ContentHash>>,
    },
}

/// Handle to the thread owning the indexes; it exits when the handle drops.
pub(super) struct Vectors {
    tx: mpsc::Sender<Request>,
}

impl Vectors {
    pub(super) fn spawn() -> io::Result<Self> {
        let (tx, mut rx) = mpsc::channel(QUEUE);
        thread::Builder::new()
            .name("repo-search-vectors-index".into())
            .spawn(move || {
                set_user_interactive();
                let mut state = State::default();
                while let Some(request) = rx.blocking_recv() {
                    state.handle(request);
                }
            })?;
        Ok(Self { tx })
    }

    async fn ask<T, E: std::error::Error + 'static>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> Request,
    ) -> Result<T, Error<E>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(make(reply))
            .await
            .map_err(|_| Error::WorkerGone)?;
        rx.await.map_err(|_| Error::WorkerGone)
    }

    /// Brings the indexes in step with the worktree's chunks and the cache:
    /// drops vectors of chunks that are gone, loads those of chunks seen for
    /// the first time and those stored since the last sync, and starts over
    /// once the cache has doubled since the calibration, so that a
    /// calibration from short chunks does not stick.
    ///
    /// A sync cut short by a dropped future leaves the books consistent; the
    /// next one finishes the work.
    pub(super) async fn sync<E: std::error::Error + 'static>(
        &self,
        index: &Index,
        cache: &VectorCache,
        model: &str,
    ) -> Result<(), Error<E>> {
        let Some(info) = cache.model(model).await? else {
            return Ok(());
        };
        let current = index.hash_sides().await?;
        let mut sides: HashMap<ContentHash, Vec<Side>> = HashMap::new();
        for (hash, side) in &current {
            sides.entry(*hash).or_default().push(*side);
        }
        let plan: Plan = self
            .ask(|reply| Request::Plan {
                current,
                cache_vectors: info.vectors,
                reply,
            })
            .await?;

        let mut missing = Vec::new();
        for batch in plan.fresh.chunks(LOAD_BATCH) {
            let hashes: Vec<ContentHash> = batch.iter().map(|(hash, _)| *hash).collect();
            let found = cache.get_many(model, &hashes).await?;
            let found_hashes: HashSet<ContentHash> = found.iter().map(|v| v.content_hash).collect();
            missing.extend(
                batch
                    .iter()
                    .filter(|(h, _)| !found_hashes.contains(h))
                    .copied(),
            );
            self.adopt(
                cache,
                model,
                found
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
        }
        if let Some(from) = plan.since
            && info.next_slot > from
        {
            let stored = cache.since(model, from).await?;
            let vectors = stored
                .into_iter()
                .flat_map(|v| {
                    sides
                        .get(&v.content_hash)
                        .into_iter()
                        .flatten()
                        .map(move |&side| (v.content_hash, v.slot, side))
                        .collect::<Vec<_>>()
                })
                .collect();
            self.adopt(cache, model, vectors).await?;
        }
        self.ask(|reply| Request::Settle {
            missing,
            high_water: info.next_slot,
            reply,
        })
        .await
    }

    /// Adds vectors, given as `(hash, slot, side)`, to the indexes; one that
    /// is already loaded is skipped. The first adoption builds the indexes
    /// from the cache's calibration.
    pub(super) async fn adopt<E: std::error::Error + 'static>(
        &self,
        cache: &VectorCache,
        model: &str,
        vectors: Vec<(ContentHash, u64, Side)>,
    ) -> Result<(), Error<E>> {
        if vectors.is_empty() {
            return Ok(());
        }
        let adopted: Result<Adopted, knn::Error> = self
            .ask(|reply| Request::Adopt {
                vectors,
                build: None,
                reply,
            })
            .await?;
        let Adopted::NeedsBuild(vectors) = adopted? else {
            return Ok(());
        };
        let calibration = calibration_for(cache, model).await?;
        let info = cache
            .model(model)
            .await?
            .ok_or_else(|| knn::Error::NoVectors(model.to_owned()))?;
        let build = Build {
            calibration,
            file: info.file,
            vectors: info.vectors,
        };
        let adopted: Result<Adopted, knn::Error> = self
            .ask(|reply| Request::Adopt {
                vectors,
                build: Some(build),
                reply,
            })
            .await?;
        match adopted? {
            Adopted::Done => Ok(()),
            Adopted::NeedsBuild(_) => unreachable!("a build was supplied"),
        }
    }

    /// The chunks among `hashes` that have no vector loaded.
    pub(super) async fn lacking<E: std::error::Error + 'static>(
        &self,
        hashes: Vec<ContentHash>,
    ) -> Result<Vec<ContentHash>, Error<E>> {
        self.ask(|reply| Request::Lacking { hashes, reply }).await
    }

    /// The hashes of the vectors of `side` nearest to `query`, best first;
    /// with `restrict`, only among those hashes. Empty before any vector is
    /// loaded.
    pub(super) async fn search<E: std::error::Error + 'static>(
        &self,
        side: Side,
        query: UnitVector,
        threads: usize,
        restrict: Option<Vec<ContentHash>>,
    ) -> Result<Vec<ContentHash>, Error<E>> {
        self.ask(|reply| Request::Search {
            side,
            query,
            threads,
            restrict,
            reply,
        })
        .await
    }
}

/// The books and indexes, touched only by the thread.
#[derive(Default)]
struct State {
    /// Code, then docs; built by the first adoption.
    indexes: Option<[KnnIndex; 2]>,
    /// The chunk-side pairs with a vector in the side's index: a content hash
    /// shared by a code chunk and a doc chunk is loaded on both.
    loaded: HashSet<(ContentHash, Side)>,
    /// The content hashes the cache had no vector for when last asked.
    absent: HashSet<ContentHash>,
    /// The cache's vector count when the indexes were calibrated.
    built_at: u64,
    /// The first slot of the cache not yet examined; `None` before the first
    /// sync completes.
    high_water: Option<u64>,
}

impl State {
    fn handle(&mut self, request: Request) {
        match request {
            Request::Plan {
                current,
                cache_vectors,
                reply,
            } => {
                let _ = reply.send(self.plan(current, cache_vectors));
            }
            Request::Adopt {
                vectors,
                build,
                reply,
            } => {
                let _ = reply.send(self.adopt(vectors, build));
            }
            Request::Settle {
                missing,
                high_water,
                reply,
            } => {
                for (hash, _) in missing {
                    if !self.is_loaded(&hash) {
                        self.absent.insert(hash);
                    }
                }
                self.high_water = Some(self.high_water.map_or(high_water, |h| h.max(high_water)));
                let _ = reply.send(());
            }
            Request::Lacking { hashes, reply } => {
                let lacking = hashes
                    .into_iter()
                    .filter(|hash| !self.is_loaded(hash))
                    .collect();
                let _ = reply.send(lacking);
            }
            Request::Search {
                side,
                query,
                threads,
                restrict,
                reply,
            } => {
                let _ = reply.send(self.search(side, &query, threads, restrict.as_deref()));
            }
        }
    }

    /// Whether `hash` has a vector in either side's index.
    fn is_loaded(&self, hash: &ContentHash) -> bool {
        [Side::Code, Side::Docs]
            .iter()
            .any(|&side| self.loaded.contains(&(*hash, side)))
    }

    /// Forgets what is gone from the worktree and says what to look up.
    fn plan(&mut self, current: Vec<(ContentHash, Side)>, cache_vectors: u64) -> Plan {
        if self.indexes.is_some() && cache_vectors >= self.built_at.saturating_mul(2) {
            *self = Self::default();
        }
        let present: HashSet<(ContentHash, Side)> = current.iter().copied().collect();
        let present_hashes: HashSet<ContentHash> = current.iter().map(|(hash, _)| *hash).collect();
        self.absent.retain(|hash| present_hashes.contains(hash));
        let gone: Vec<(ContentHash, Side)> = self
            .loaded
            .iter()
            .filter(|&pair| !present.contains(pair))
            .copied()
            .collect();
        for (hash, side) in gone {
            self.loaded.remove(&(hash, side));
            let indexes = self.indexes.as_mut().expect("loaded implies built");
            indexes[side as usize].remove(&hash);
        }
        let fresh = current
            .into_iter()
            .filter(|&(hash, side)| {
                !self.loaded.contains(&(hash, side)) && !self.absent.contains(&hash)
            })
            .collect();
        Plan {
            fresh,
            since: self.high_water,
        }
    }

    fn adopt(
        &mut self,
        vectors: Vec<(ContentHash, u64, Side)>,
        build: Option<Build>,
    ) -> Result<Adopted, knn::Error> {
        if self.indexes.is_none() {
            let Some(build) = build else {
                return Ok(Adopted::NeedsBuild(vectors));
            };
            let empty = || KnnIndex::build(build.calibration.clone(), &build.file, []);
            self.indexes = Some([empty()?, empty()?]);
            self.built_at = build.vectors;
        }
        let indexes = self.indexes.as_mut().expect("built above");
        for (hash, slot, side) in vectors {
            if self.loaded.contains(&(hash, side)) {
                continue;
            }
            indexes[side as usize].insert(hash, slot)?;
            self.loaded.insert((hash, side));
            self.absent.remove(&hash);
        }
        Ok(Adopted::Done)
    }

    fn search(
        &self,
        side: Side,
        query: &UnitVector,
        threads: usize,
        restrict: Option<&[ContentHash]>,
    ) -> Vec<ContentHash> {
        let Some(indexes) = &self.indexes else {
            return Vec::new();
        };
        let index = &indexes[side as usize];
        // Built here, after every update this thread has applied: a removal
        // renumbers rows.
        let mask = restrict.map(|hashes| index.mask(hashes));
        index
            .search(query, threads, mask.as_ref())
            .into_iter()
            .map(|(hash, _)| hash)
            .collect()
    }
}
