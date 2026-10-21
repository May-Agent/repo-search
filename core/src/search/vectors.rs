//! The worktree's in-memory vector indexes, owned by a thread of their own.
//!
//! Searches and updates are messages to that thread, so no future ever holds
//! an index: a query dropped half way leaves nothing outstanding, and the
//! blocking work (building, inserting, scanning) stays off the executor.
//!
//! The thread also keeps the books of what it loaded. It holds the
//! worktree's `(hash, side)` pairs as the caller's reconciler reported them
//! ([`Presence`]), so it never reads the index. A pair is *loaded* (it has a
//! vector in its side's index), its hash is *absent* (the cache had none
//! when last asked), or its hash is *fresh* (not asked about yet). The cache
//! is asked about a hash once, when it is first fresh; after that only
//! vectors stored *since* (slots from the high-water mark on) can turn an
//! absent hash into a loaded one.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::PathBuf;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use crate::UnitVector;
use crate::knn::{self, Calibration, KnnIndex, calibration_for};
use crate::qos::{QosClass, set_current_thread};
use crate::reconcile::Presence;
use crate::store::{ContentHash, Side, VectorCache};

use super::Error;
use super::output::VectorReads;

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
    /// Hashes the cache was never asked about.
    fresh: Vec<ContentHash>,
    /// Where to look for vectors stored since the last sync; `None` before
    /// the first one.
    since: Option<u64>,
}

enum Adopted {
    Done,
    /// There are no indexes yet; resend with a [`Build`].
    NeedsBuild(Vec<(ContentHash, u64)>),
}

enum Request {
    Follow {
        presence: Presence,
    },
    Plan {
        cache_vectors: u64,
        reply: oneshot::Sender<Plan>,
    },
    Adopt {
        vectors: Vec<(ContentHash, u64)>,
        build: Option<Build>,
        reply: oneshot::Sender<Result<Adopted, knn::Error>>,
    },
    Settle {
        missing: Vec<ContentHash>,
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
    /// Spawns the thread, holding `pairs` as the worktree's pairs.
    pub(super) fn spawn(pairs: Vec<(ContentHash, Side)>) -> io::Result<Self> {
        let (tx, mut rx) = mpsc::channel(QUEUE);
        thread::Builder::new()
            .name("repo-search-vectors-index".into())
            .spawn(move || {
                set_current_thread(QosClass::UserInteractive);
                let mut state = State::default();
                state.follow(Presence::Whole(pairs));
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

    /// Applies a pass's [`Presence`]: vectors of pairs that vanished leave
    /// the indexes at once, pairs that appeared are looked up by the next
    /// sync. Every later request sees it applied.
    pub(super) async fn follow<E: std::error::Error + 'static>(
        &self,
        presence: Presence,
    ) -> Result<(), Error<E>> {
        self.tx
            .send(Request::Follow { presence })
            .await
            .map_err(|_| Error::WorkerGone)
    }

    /// Brings the indexes in step with the cache: loads the vectors of
    /// fresh hashes and those stored since the last sync, and starts over
    /// once the cache has doubled since the calibration, so that a
    /// calibration from short chunks does not stick. Returns what it read.
    ///
    /// A sync cut short by a dropped future leaves the books consistent; the
    /// next one finishes the work.
    pub(super) async fn sync<E: std::error::Error + 'static>(
        &self,
        cache: &VectorCache,
        model: &str,
    ) -> Result<VectorReads, Error<E>> {
        let mut reads = VectorReads::default();
        let Some(info) = cache.model(model).await? else {
            return Ok(reads);
        };
        let plan: Plan = self
            .ask(|reply| Request::Plan {
                cache_vectors: info.vectors,
                reply,
            })
            .await?;

        let mut missing = Vec::new();
        for batch in plan.fresh.chunks(LOAD_BATCH) {
            let found = cache.get_many(model, batch).await?;
            reads.cache_lookups += batch.len() as u64;
            let found_hashes: HashSet<ContentHash> = found.iter().map(|v| v.content_hash).collect();
            missing.extend(batch.iter().filter(|h| !found_hashes.contains(h)).copied());
            self.adopt(
                cache,
                model,
                found
                    .into_iter()
                    .map(|v| (v.content_hash, v.slot))
                    .collect(),
            )
            .await?;
        }
        if let Some(from) = plan.since
            && info.next_slot > from
        {
            let stored = cache.since(model, from).await?;
            reads.cache_slots += stored.len() as u64;
            self.adopt(
                cache,
                model,
                stored
                    .into_iter()
                    .map(|v| (v.content_hash, v.slot))
                    .collect(),
            )
            .await?;
        }
        self.ask(|reply| Request::Settle {
            missing,
            high_water: info.next_slot,
            reply,
        })
        .await?;
        Ok(reads)
    }

    /// Adds vectors, given as `(hash, slot)`, to the index of every side
    /// the worktree holds the hash on; a pair already loaded is skipped, and
    /// so is a hash the worktree does not hold. The first adoption builds
    /// the indexes from the cache's calibration.
    pub(super) async fn adopt<E: std::error::Error + 'static>(
        &self,
        cache: &VectorCache,
        model: &str,
        vectors: Vec<(ContentHash, u64)>,
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
    /// The worktree's hashes, each with whether a row of each side (code,
    /// then docs) holds it.
    present: HashMap<ContentHash, [bool; 2]>,
    /// The chunk-side pairs with a vector in the side's index: a content hash
    /// shared by a code chunk and a doc chunk is loaded on both.
    loaded: HashSet<(ContentHash, Side)>,
    /// The present hashes the cache had no vector for when last asked.
    absent: HashSet<ContentHash>,
    /// The present hashes that are not absent and lack a vector on some
    /// side they are present on: the next sync asks the cache about them.
    fresh: HashSet<ContentHash>,
    /// The cache's vector count when the indexes were calibrated.
    built_at: u64,
    /// The first slot of the cache not yet examined; `None` before the first
    /// sync completes.
    high_water: Option<u64>,
}

impl State {
    fn handle(&mut self, request: Request) {
        match request {
            Request::Follow { presence } => self.follow(presence),
            Request::Plan {
                cache_vectors,
                reply,
            } => {
                let _ = reply.send(self.plan(cache_vectors));
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
                for hash in missing {
                    if self.present.contains_key(&hash) && !self.is_loaded(&hash) {
                        self.absent.insert(hash);
                        self.fresh.remove(&hash);
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

    /// Whether `hash` is present on a side where it has no vector loaded.
    fn lacks_a_side(&self, hash: &ContentHash) -> bool {
        self.present.get(hash).is_some_and(|sides| {
            [Side::Code, Side::Docs]
                .into_iter()
                .any(|side| sides[side as usize] && !self.loaded.contains(&(*hash, side)))
        })
    }

    /// Puts `hash` in `fresh` exactly when it is present, not absent and
    /// lacks a vector on a side it is present on.
    fn refresh(&mut self, hash: ContentHash) {
        if !self.absent.contains(&hash) && self.lacks_a_side(&hash) {
            self.fresh.insert(hash);
        } else {
            self.fresh.remove(&hash);
        }
    }

    /// Takes a vector out of its side's index.
    fn unload(&mut self, pair: (ContentHash, Side)) {
        if self.loaded.remove(&pair) {
            let indexes = self.indexes.as_mut().expect("loaded implies built");
            indexes[pair.1 as usize].remove(&pair.0);
        }
    }

    /// Applies a pass's presence.
    fn follow(&mut self, presence: Presence) {
        match presence {
            Presence::Whole(pairs) => {
                let mut present: HashMap<ContentHash, [bool; 2]> = HashMap::new();
                for (hash, side) in pairs {
                    present.entry(hash).or_default()[side as usize] = true;
                }
                let gone: Vec<(ContentHash, Side)> = self
                    .loaded
                    .iter()
                    .filter(|(hash, side)| {
                        !present.get(hash).is_some_and(|sides| sides[*side as usize])
                    })
                    .copied()
                    .collect();
                for pair in gone {
                    self.unload(pair);
                }
                self.absent.retain(|hash| present.contains_key(hash));
                self.present = present;
                self.fresh.clear();
                let hashes: Vec<ContentHash> = self.present.keys().copied().collect();
                for hash in hashes {
                    self.refresh(hash);
                }
            }
            Presence::Changed { appeared, vanished } => {
                for (hash, side) in vanished {
                    self.unload((hash, side));
                    let sides = self
                        .present
                        .get_mut(&hash)
                        .expect("a vanished pair was present");
                    sides[side as usize] = false;
                    if *sides == [false, false] {
                        self.present.remove(&hash);
                        self.absent.remove(&hash);
                    }
                    self.refresh(hash);
                }
                for (hash, side) in appeared {
                    self.present.entry(hash).or_default()[side as usize] = true;
                    self.refresh(hash);
                }
            }
        }
    }

    /// Says what to look up; once the cache has doubled since the
    /// calibration, the indexes start over and every present hash is fresh.
    fn plan(&mut self, cache_vectors: u64) -> Plan {
        if self.indexes.is_some() && cache_vectors >= self.built_at.saturating_mul(2) {
            self.indexes = None;
            self.loaded.clear();
            self.absent.clear();
            self.built_at = 0;
            self.high_water = None;
            self.fresh = self.present.keys().copied().collect();
        }
        Plan {
            fresh: self.fresh.iter().copied().collect(),
            since: self.high_water,
        }
    }

    fn adopt(
        &mut self,
        vectors: Vec<(ContentHash, u64)>,
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
        for (hash, slot) in vectors {
            let Some(sides) = self.present.get(&hash) else {
                continue;
            };
            for side in [Side::Code, Side::Docs] {
                if !sides[side as usize] || self.loaded.contains(&(hash, side)) {
                    continue;
                }
                indexes[side as usize].insert(hash, slot)?;
                self.loaded.insert((hash, side));
            }
            self.absent.remove(&hash);
            self.fresh.remove(&hash);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(n: u8) -> ContentHash {
        ContentHash::of_chunk("t", &n.to_string())
    }

    fn fresh(state: &mut State) -> HashSet<ContentHash> {
        state.plan(0).fresh.into_iter().collect()
    }

    fn settle(state: &mut State, missing: Vec<ContentHash>) {
        let (reply, _rx) = oneshot::channel();
        state.handle(Request::Settle {
            missing,
            high_water: 0,
            reply,
        });
    }

    #[test]
    fn only_hashes_new_to_the_books_are_looked_up() {
        let mut state = State::default();
        state.follow(Presence::Whole(vec![
            (hash(1), Side::Code),
            (hash(2), Side::Code),
            (hash(2), Side::Docs),
        ]));
        assert_eq!(fresh(&mut state), HashSet::from([hash(1), hash(2)]));

        // The cache had neither: they are absent and not asked again.
        settle(&mut state, vec![hash(1), hash(2)]);
        assert!(fresh(&mut state).is_empty());

        // An empty pass asks nothing; a new pair asks about its hash alone.
        state.follow(Presence::Changed {
            appeared: Vec::new(),
            vanished: Vec::new(),
        });
        assert!(fresh(&mut state).is_empty());
        state.follow(Presence::Changed {
            appeared: vec![(hash(3), Side::Docs)],
            vanished: vec![(hash(1), Side::Code)],
        });
        assert_eq!(fresh(&mut state), HashSet::from([hash(3)]));
        assert!(!state.present.contains_key(&hash(1)));
        assert!(!state.absent.contains(&hash(1)), "a gone hash is forgotten");

        // A hash present again after it vanished is new again.
        state.follow(Presence::Changed {
            appeared: vec![(hash(1), Side::Code)],
            vanished: vec![(hash(2), Side::Docs)],
        });
        assert_eq!(fresh(&mut state), HashSet::from([hash(1), hash(3)]));
        assert_eq!(state.present[&hash(2)], [true, false]);
        assert!(
            state.absent.contains(&hash(2)),
            "still held on the code side"
        );
    }

    #[test]
    fn a_whole_presence_replaces_the_pairs() {
        let mut state = State::default();
        state.follow(Presence::Whole(vec![
            (hash(1), Side::Code),
            (hash(2), Side::Code),
        ]));
        settle(&mut state, vec![hash(1), hash(2)]);
        state.follow(Presence::Whole(vec![
            (hash(2), Side::Code),
            (hash(3), Side::Code),
        ]));
        assert_eq!(fresh(&mut state), HashSet::from([hash(3)]));
        assert_eq!(state.absent, HashSet::from([hash(2)]));
        assert!(!state.present.contains_key(&hash(1)));
    }
}
