//! The books a [`super::Reconciler`] keeps between passes over one index:
//! every stored chunk row's hash by file, and which hashes have a vector,
//! so a pass reports coverage and its changes from what it changed instead
//! of from every stored chunk.

use std::collections::HashMap;

use super::Reads;
use crate::store::{self, CachedSlot, ContentHash, FileChange, Index, VectorCache};

/// The tally a row counts in: rows of files that are not ignored, then
/// rows of ignored files.
fn tally(ignored: bool) -> usize {
    usize::from(ignored)
}

/// The books of one index under one model.
///
/// A pass keeps them current from its own changes and from the cache's
/// slot log: slots are allocated densely and never freed, so the vectors
/// stored from `high_water` on are exactly those the books have not seen.
/// They describe one index handle and one cache; whoever changes the
/// index's rows outside the [`super::Reconciler`] that holds them, or
/// reopens either store, starts over with new books.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Books {
    /// The first slot of the model's log not yet read.
    high_water: u64,
    /// Every stored file with at least one chunk row: its ignored flag and
    /// its rows' hashes in index order.
    files: HashMap<String, BookedFile>,
    /// Every hash some stored row holds.
    hashes: HashMap<ContentHash, BookedHash>,
    /// Stored rows, by tally.
    rows: [u64; 2],
    /// Stored rows whose hash has a vector, by tally.
    with_vector: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BookedFile {
    ignored: bool,
    hashes: Vec<ContentHash>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BookedHash {
    /// Rows holding the hash, by tally.
    rows: [u32; 2],
    /// Whether the cache holds a vector for the hash.
    cached: bool,
}

impl Books {
    /// Reads the books of `index` under `model_id` whole: every stored
    /// chunk row, then the cache once per distinct hash. Returns them with
    /// what was read.
    ///
    /// # Errors
    ///
    /// When the index or the cache rejects a read.
    pub async fn load(
        index: &Index,
        cache: &VectorCache,
        model_id: &str,
    ) -> Result<(Self, Reads), store::Error> {
        // Read first: a vector stored after this is in the next slot read,
        // whatever the probe below saw.
        let high_water = cache
            .model(model_id)
            .await?
            .map_or(0, |info| info.next_slot);
        let rows = index.row_hashes().await?;
        let chunk_rows = rows.len() as u64;
        let mut files: HashMap<String, BookedFile> = HashMap::new();
        for row in rows {
            files
                .entry(row.path)
                .or_insert_with(|| BookedFile {
                    ignored: row.ignored,
                    hashes: Vec::new(),
                })
                .hashes
                .push(row.content_hash);
        }
        let mut books = Self {
            high_water,
            files: HashMap::with_capacity(files.len()),
            hashes: HashMap::new(),
            rows: [0; 2],
            with_vector: [0; 2],
        };
        let mut fresh = Vec::new();
        for (path, file) in files {
            books.add_file(path, file.ignored, file.hashes, &mut fresh);
        }
        if !fresh.is_empty() {
            for hash in cache.cached(model_id, &fresh).await? {
                books.mark_cached(&hash);
            }
        }
        Ok((
            books,
            Reads {
                chunk_rows,
                cache_probes: fresh.len() as u64,
                cache_slots: 0,
            },
        ))
    }

    /// The first slot of the model's log the books have not read.
    pub(super) fn high_water(&self) -> u64 {
        self.high_water
    }

    /// Chunk rows visible in the scope.
    pub(super) fn chunks_total(&self, include_ignored: bool) -> u64 {
        self.rows[0] + if include_ignored { self.rows[1] } else { 0 }
    }

    /// Chunk rows visible in the scope whose hash has a vector.
    pub(super) fn chunks_with_vector(&self, include_ignored: bool) -> u64 {
        self.with_vector[0]
            + if include_ignored {
                self.with_vector[1]
            } else {
                0
            }
    }

    /// Whether the cache holds a vector for `hash`, as far as the books know.
    pub(super) fn is_cached(&self, hash: &ContentHash) -> bool {
        self.hashes.get(hash).is_some_and(|entry| entry.cached)
    }

    /// Whether a row visible in the scope holds `hash`.
    pub(super) fn is_visible(&self, hash: &ContentHash, include_ignored: bool) -> bool {
        self.hashes
            .get(hash)
            .is_some_and(|entry| entry.rows[0] > 0 || (include_ignored && entry.rows[1] > 0))
    }

    /// Books one change the pass commits; hashes the books did not hold
    /// before are pushed to `fresh`, for the caller to probe.
    pub(super) fn record(&mut self, change: &FileChange, fresh: &mut Vec<ContentHash>) {
        match change {
            FileChange::Replace(state, records) => {
                self.drop_file(&state.path);
                self.add_file(
                    state.path.clone(),
                    state.ignored,
                    records.iter().map(|record| record.content_hash).collect(),
                    fresh,
                );
            }
            FileChange::Remove(path) => self.drop_file(path),
            FileChange::Unchunkable(record) => self.drop_file(&record.state.path),
            FileChange::Restamp(state) => self.restamp(&state.path, state.ignored),
            FileChange::RestampUnchunkable(_) => {}
        }
    }

    /// Marks `hash` as having a vector; a hash no row holds is ignored.
    pub(super) fn mark_cached(&mut self, hash: &ContentHash) {
        if let Some(entry) = self.hashes.get_mut(hash)
            && !entry.cached
        {
            entry.cached = true;
            self.with_vector[0] += u64::from(entry.rows[0]);
            self.with_vector[1] += u64::from(entry.rows[1]);
        }
    }

    /// Books the vectors the cache stored from `high_water` on.
    pub(super) fn advance(&mut self, slots: &[CachedSlot]) {
        for slot in slots {
            self.mark_cached(&slot.content_hash);
            self.high_water = self.high_water.max(slot.slot + 1);
        }
    }

    fn add_file(
        &mut self,
        path: String,
        ignored: bool,
        hashes: Vec<ContentHash>,
        fresh: &mut Vec<ContentHash>,
    ) {
        if hashes.is_empty() {
            return;
        }
        for &hash in &hashes {
            self.add_row(hash, ignored, fresh);
        }
        let previous = self.files.insert(path, BookedFile { ignored, hashes });
        assert!(previous.is_none(), "a file is booked once");
    }

    fn drop_file(&mut self, path: &str) {
        if let Some(file) = self.files.remove(path) {
            for hash in file.hashes {
                self.drop_row(hash, file.ignored);
            }
        }
    }

    /// Moves a file's rows to the tally of its new ignored flag. A file
    /// with no chunk row is not booked and has nothing to move.
    fn restamp(&mut self, path: &str, ignored: bool) {
        let Some(file) = self.files.get_mut(path) else {
            return;
        };
        if file.ignored == ignored {
            return;
        }
        let (from, to) = (tally(file.ignored), tally(ignored));
        file.ignored = ignored;
        for hash in &file.hashes {
            let entry = self
                .hashes
                .get_mut(hash)
                .expect("a booked row's hash is booked");
            entry.rows[from] -= 1;
            entry.rows[to] += 1;
            self.rows[from] -= 1;
            self.rows[to] += 1;
            if entry.cached {
                self.with_vector[from] -= 1;
                self.with_vector[to] += 1;
            }
        }
    }

    fn add_row(&mut self, hash: ContentHash, ignored: bool, fresh: &mut Vec<ContentHash>) {
        let entry = self.hashes.entry(hash).or_insert_with(|| {
            fresh.push(hash);
            BookedHash {
                rows: [0; 2],
                cached: false,
            }
        });
        entry.rows[tally(ignored)] += 1;
        self.rows[tally(ignored)] += 1;
        if entry.cached {
            self.with_vector[tally(ignored)] += 1;
        }
    }

    fn drop_row(&mut self, hash: ContentHash, ignored: bool) {
        let entry = self
            .hashes
            .get_mut(&hash)
            .expect("a booked row's hash is booked");
        entry.rows[tally(ignored)] -= 1;
        self.rows[tally(ignored)] -= 1;
        if entry.cached {
            self.with_vector[tally(ignored)] -= 1;
        }
        if entry.rows == [0, 0] {
            self.hashes.remove(&hash);
        }
    }
}
