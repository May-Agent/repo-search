//! A per-query bitset over the rows of one [`KnnIndex`](super::KnnIndex).

/// The rows of an index a scan may return. Built for one query from the
/// index's current row numbering (see [`KnnIndex::mask`](super::KnnIndex::mask));
/// a removal reorders rows, so a mask is never kept across updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowMask {
    rows: usize,
    words: Vec<u64>,
}

impl RowMask {
    /// A mask over `rows` rows that admits none of them.
    #[must_use]
    pub fn none(rows: usize) -> Self {
        Self {
            rows,
            words: vec![0; rows.div_ceil(64)],
        }
    }

    /// The number of rows the mask spans.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Admits `row`.
    ///
    /// # Panics
    ///
    /// When `row` is outside the mask.
    pub fn insert(&mut self, row: usize) {
        assert!(row < self.rows, "row {row} is outside {} rows", self.rows);
        self.words[row / 64] |= 1 << (row % 64);
    }

    /// Whether `row` is admitted.
    #[must_use]
    pub fn contains(&self, row: usize) -> bool {
        self.words[row / 64] >> (row % 64) & 1 == 1
    }

    /// How many rows are admitted.
    #[must_use]
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}
