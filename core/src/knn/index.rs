//! The in-memory index: int8 codes of the vectors present in one worktree.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::thread;

use numkong::{Dot, f16};

use crate::UnitVector;
use crate::store::ContentHash;

use super::calibration::Calibration;
use super::fp16::Fp16File;
use super::mask::RowMask;
use super::set_user_interactive;
use super::topk::TopK;
use super::{Error, FIRST_PASS, RESULTS};

/// Int8 codes of distinct content hashes, searchable by cosine similarity.
///
/// Rows are numbered densely; [`remove`](Self::remove) swaps the last row into
/// the hole, so row order is not stable across updates. Equal scores resolve to
/// the lower row.
pub struct KnnIndex {
    calibration: Calibration,
    /// `rows * dim` codes.
    codes: Vec<i8>,
    hashes: Vec<ContentHash>,
    slots: Vec<u64>,
    rows: HashMap<ContentHash, u32>,
    file: Fp16File,
}

impl KnnIndex {
    /// Builds an index over `rows` — `(content hash, fp16 slot)` pairs — whose
    /// vectors are read from the fp16 file at `file`.
    ///
    /// # Errors
    ///
    /// When the file cannot be mapped or a slot lies beyond its end.
    ///
    /// # Panics
    ///
    /// When `rows` repeats a content hash.
    pub fn build(
        calibration: Calibration,
        file: &Path,
        rows: impl IntoIterator<Item = (ContentHash, u64)>,
    ) -> Result<Self, Error> {
        let file = Fp16File::open(file, calibration.dim())?;
        let mut index = Self {
            calibration,
            codes: Vec::new(),
            hashes: Vec::new(),
            slots: Vec::new(),
            rows: HashMap::new(),
            file,
        };
        for (hash, slot) in rows {
            index.insert(hash, slot)?;
        }
        Ok(index)
    }

    /// The number of indexed rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the index holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Whether `hash` is indexed.
    #[must_use]
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.rows.contains_key(hash)
    }

    /// Adds the vector at `slot`, encoded from the fp16 file, which is
    /// remapped when it grew past the current map.
    ///
    /// # Errors
    ///
    /// When the file cannot be remapped or `slot` lies beyond its end.
    ///
    /// # Panics
    ///
    /// When `hash` is already indexed: rows are distinct content hashes.
    pub fn insert(&mut self, hash: ContentHash, slot: u64) -> Result<(), Error> {
        assert!(!self.rows.contains_key(&hash), "{hash} is already indexed");
        self.file.cover(slot)?;
        let dim = self.calibration.dim();
        let start = self.codes.len();
        self.codes.resize(start + dim, 0);
        self.calibration
            .encode(self.file.row(slot), &mut self.codes[start..]);
        let row = u32::try_from(self.hashes.len()).expect("fewer than 2^32 rows");
        self.hashes.push(hash);
        self.slots.push(slot);
        self.rows.insert(hash, row);
        Ok(())
    }

    /// Removes `hash`, returning whether it was indexed. The last row moves
    /// into its place.
    pub fn remove(&mut self, hash: &ContentHash) -> bool {
        let Some(row) = self.rows.remove(hash) else {
            return false;
        };
        let row = row as usize;
        let dim = self.calibration.dim();
        let last = self.hashes.len() - 1;
        if row != last {
            self.codes
                .copy_within(last * dim..(last + 1) * dim, row * dim);
        }
        self.codes.truncate(last * dim);
        self.hashes.swap_remove(row);
        self.slots.swap_remove(row);
        if row != last {
            self.rows.insert(self.hashes[row], row as u32);
        }
        true
    }

    /// The mask admitting the rows of `hashes`; a hash the index does not
    /// hold is ignored. Rows are renumbered by [`remove`](Self::remove), so
    /// build the mask after the last update and use it for one search.
    #[must_use]
    pub fn mask<'a>(&self, hashes: impl IntoIterator<Item = &'a ContentHash>) -> RowMask {
        let mut mask = RowMask::none(self.len());
        for hash in hashes {
            if let Some(&row) = self.rows.get(hash) {
                mask.insert(row as usize);
            }
        }
        mask
    }

    /// The [`RESULTS`] vectors most similar to `query`, best first, with their
    /// fp16 dot product against it. With a `mask`, only the rows it admits are
    /// scanned and returned; without one the whole index is.
    ///
    /// The query is quantized and every admitted code scanned on `threads`
    /// threads (the calling thread when 1), each keeping its [`FIRST_PASS`]
    /// best; the merged candidates are re-ranked with the fp16 rows. Equal
    /// scores resolve to the lower row, so the result does not depend on
    /// `threads`.
    ///
    /// # Panics
    ///
    /// When `threads` is 0, the query's dimension differs from the index's, or
    /// `mask` was built for a different number of rows.
    #[must_use]
    pub fn search(
        &self,
        query: &UnitVector,
        threads: usize,
        mask: Option<&RowMask>,
    ) -> Vec<(ContentHash, f32)> {
        assert!(threads > 0, "search needs at least one thread");
        if let Some(mask) = mask {
            assert_eq!(mask.rows(), self.len(), "the mask spans another row count");
        }
        let dim = self.calibration.dim();
        let mut weights = vec![0i8; dim];
        self.calibration.query_weights(query, &mut weights);

        let first_pass = self.first_pass(&weights, threads, mask);

        let query16: Vec<f16> = query.as_slice().iter().map(|&x| f16::from_f32(x)).collect();
        let mut best = TopK::new(RESULTS);
        for (row, _) in first_pass.into_best_first() {
            let fp16 = self.file.row(self.slots[row as usize]);
            best.push(f16::dot(fp16, &query16).expect("same dimension"), row);
        }
        best.into_best_first()
            .into_iter()
            .map(|(row, score)| (self.hashes[row as usize], score))
            .collect()
    }

    fn first_pass(&self, weights: &[i8], threads: usize, mask: Option<&RowMask>) -> TopK {
        let rows = self.hashes.len();
        if threads == 1 {
            return self.scan(weights, 0..rows, mask);
        }
        let per_thread = rows.div_ceil(threads);
        thread::scope(|scope| {
            (0..threads)
                .map(|t| {
                    let range = (t * per_thread).min(rows)..((t + 1) * per_thread).min(rows);
                    scope.spawn(move || {
                        set_user_interactive();
                        self.scan(weights, range, mask)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("scan thread panicked"))
                .reduce(TopK::merge)
                .expect("threads > 0")
        })
    }

    fn scan(&self, weights: &[i8], rows: Range<usize>, mask: Option<&RowMask>) -> TopK {
        let dim = self.calibration.dim();
        let mut top = TopK::new(FIRST_PASS);
        for row in rows {
            if mask.is_some_and(|m| !m.contains(row)) {
                continue;
            }
            let codes = &self.codes[row * dim..(row + 1) * dim];
            let score = i8::dot(codes, weights).expect("same dimension");
            top.push(score as f32, row as u32);
        }
        top
    }
}
