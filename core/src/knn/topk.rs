//! Bounded best-k selection with a deterministic tie-break.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::store::ContentHash;

/// A scored row. Ordered so that the best entry is the smallest: higher score
/// first, and on equal scores the lower content hash — a function of the
/// index's contents, never of the order its rows were loaded in.
#[derive(Clone, Copy, Debug)]
struct Entry {
    score: f32,
    hash: ContentHash,
    row: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.hash.cmp(&other.hash))
    }
}

/// The `k` best-scoring rows seen so far.
pub(super) struct TopK {
    k: usize,
    /// A max-heap under `Entry`'s order: the worst retained entry on top.
    heap: BinaryHeap<Entry>,
}

impl TopK {
    pub(super) fn new(k: usize) -> Self {
        assert!(k > 0, "top-k needs k > 0");
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// `hashes[row]` keys the tie-break; it is read only for a row that can
    /// enter the heap, so the scan's hot loop pays for it only on a real
    /// comparison.
    pub(super) fn push(&mut self, score: f32, row: u32, hashes: &[ContentHash]) {
        if self.heap.len() == self.k
            && score.total_cmp(&self.heap.peek().expect("k > 0").score) == Ordering::Less
        {
            return;
        }
        self.insert(Entry {
            score,
            hash: hashes[row as usize],
            row,
        });
    }

    fn insert(&mut self, entry: Entry) {
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if entry < *self.heap.peek().expect("k > 0") {
            self.heap.pop();
            self.heap.push(entry);
        }
    }

    pub(super) fn merge(mut self, other: Self) -> Self {
        for e in other.heap {
            self.insert(e);
        }
        self
    }

    /// `(row, score)`, best first.
    pub(super) fn into_best_first(self) -> Vec<(u32, f32)> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| (e.row, e.score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::TopK;
    use crate::store::ContentHash;

    /// A distinct content hash per row: `hashes[row]`.
    fn hashes(n: u32) -> Vec<ContentHash> {
        (0..n)
            .map(|row| ContentHash::of_chunk("t", &row.to_string()))
            .collect()
    }

    #[test]
    fn keeps_the_best_k_best_first() {
        let hashes = hashes(5);
        let mut top = TopK::new(3);
        for (row, score) in [(0, 1.0), (1, 5.0), (2, 3.0), (3, 4.0), (4, 2.0)] {
            top.push(score, row, &hashes);
        }
        assert_eq!(top.into_best_first(), [(1, 5.0), (3, 4.0), (2, 3.0)]);
    }

    #[test]
    fn equal_scores_prefer_the_lower_hash() {
        // The push order is not the hashes': the two kept are the two lowest
        // content hashes, whatever rows they sit on.
        let rows = [7, 3, 9, 1, 5];
        let hashes = hashes(10);
        let mut top = TopK::new(2);
        for row in rows {
            top.push(1.0, row, &hashes);
        }
        let mut want: Vec<(u32, f32)> = rows.iter().map(|&row| (row, 1.0)).collect();
        want.sort_by_key(|&(row, _)| hashes[row as usize]);
        want.truncate(2);
        assert_eq!(top.into_best_first(), want);
    }

    #[test]
    fn merge_matches_one_pass_over_all_rows() {
        let scores = [2.0, 9.0, 9.0, 1.0, 9.0, 4.0, 4.0, 0.0];
        let hashes = hashes(8);
        let mut whole = TopK::new(4);
        let mut left = TopK::new(4);
        let mut right = TopK::new(4);
        for (row, &score) in scores.iter().enumerate() {
            whole.push(score, row as u32, &hashes);
            if row < 3 { &mut left } else { &mut right }.push(score, row as u32, &hashes);
        }
        assert_eq!(left.merge(right).into_best_first(), whole.into_best_first());
    }
}
