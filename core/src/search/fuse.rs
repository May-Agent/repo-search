//! Reciprocal-rank fusion of the lexical and vector candidate lists.

use std::collections::HashMap;

use serde::Serialize;

use crate::store::ChunkId;

/// The `k` of `1 / (k + rank)`.
pub const RRF_K: f64 = 60.0;

/// A retrieval channel a chunk was found by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// FTS5 BM25 over chunk titles and bodies.
    Bm25,
    /// Cosine nearest neighbours.
    Vector,
}

/// A chunk with its fused score and the channels that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    /// The chunk.
    pub id: ChunkId,
    /// Sum of `1 / (k + rank)` over the channels that ranked it, ranks
    /// counted from 1.
    pub score: f64,
    /// The channels that ranked it, lexical first.
    pub channels: Vec<Channel>,
}

/// Fuses two best-first candidate lists. Best fused score first; equal scores
/// resolve to the better vector rank, a chunk the vector list lacks coming
/// after those it has, and then to the better lexical rank. That is the order
/// a stable sort gives the evaluation's fusion, which inserts the vector list
/// first, so the order is deterministic and the same as the one measured.
#[must_use]
pub fn fuse(bm25: &[ChunkId], vector: &[ChunkId]) -> Vec<Fused> {
    struct Entry {
        fused: Fused,
        /// Rank in the vector list, then in the lexical one; `usize::MAX`
        /// where absent.
        ranks: (usize, usize),
    }
    let mut by_id: HashMap<ChunkId, Entry> = HashMap::new();
    for (channel, list) in [(Channel::Bm25, bm25), (Channel::Vector, vector)] {
        for (i, &id) in list.iter().enumerate() {
            let entry = by_id.entry(id).or_insert_with(|| Entry {
                fused: Fused {
                    id,
                    score: 0.0,
                    channels: Vec::new(),
                },
                ranks: (usize::MAX, usize::MAX),
            });
            // A chunk repeated within one list keeps its best rank.
            if entry.fused.channels.contains(&channel) {
                continue;
            }
            entry.fused.score += 1.0 / (RRF_K + (i + 1) as f64);
            entry.fused.channels.push(channel);
            match channel {
                Channel::Bm25 => entry.ranks.1 = i,
                Channel::Vector => entry.ranks.0 = i,
            }
        }
    }
    let mut entries: Vec<Entry> = by_id.into_values().collect();
    entries.sort_by(|a, b| {
        b.fused
            .score
            .total_cmp(&a.fused.score)
            .then(a.ranks.cmp(&b.ranks))
    });
    entries.into_iter().map(|e| e.fused).collect()
}

#[cfg(test)]
mod tests {
    use super::{Channel, RRF_K, fuse};
    use crate::store::ChunkId;

    fn ids(v: &[i64]) -> Vec<ChunkId> {
        v.iter().copied().map(ChunkId).collect()
    }

    #[test]
    fn a_chunk_in_both_channels_outranks_single_channel_leaders() {
        let fused = fuse(&ids(&[1, 2, 3]), &ids(&[9, 3]));
        let order: Vec<i64> = fused.iter().map(|f| f.id.0).collect();
        // 3: 1/63 + 1/62; 1: 1/61; 9: 1/61.
        assert_eq!(order[0], 3);
        assert_eq!(fused[0].channels, [Channel::Bm25, Channel::Vector]);
        let expected = 1.0 / (RRF_K + 3.0) + 1.0 / (RRF_K + 2.0);
        assert!((fused[0].score - expected).abs() < 1e-12);
    }

    #[test]
    fn ties_resolve_to_the_vector_rank_then_the_lexical_rank() {
        // 7 leads the lexical list, 9 the vector list: equal scores, and the
        // higher id wins because the vector list comes first.
        let fused = fuse(&ids(&[7]), &ids(&[9]));
        let order: Vec<i64> = fused.iter().map(|f| f.id.0).collect();
        assert_eq!(order, [9, 7]);
        assert_eq!(fused[0].score, fused[1].score);

        // Rank 1 lexically plus rank 2 by vector equals the reverse: both
        // chunks tie, and the one the vector list ranks first wins.
        let fused = fuse(&ids(&[4, 3]), &ids(&[3, 4]));
        assert_eq!(fused[0].score, fused[1].score);
        assert_eq!(fused.iter().map(|f| f.id.0).collect::<Vec<_>>(), [3, 4]);

        // Lexical-only chunks tie in lexical order, after vector ones.
        let fused = fuse(&ids(&[8, 2]), &ids(&[]));
        assert_eq!(fused.iter().map(|f| f.id.0).collect::<Vec<_>>(), [8, 2]);
    }

    #[test]
    fn single_channel_hits_keep_their_channel_and_rank_order() {
        let fused = fuse(&ids(&[4, 2, 8]), &[]);
        assert_eq!(fused.iter().map(|f| f.id.0).collect::<Vec<_>>(), [4, 2, 8]);
        assert!(fused.iter().all(|f| f.channels == [Channel::Bm25]));
        let only_vector = fuse(&[], &ids(&[3, 1]));
        assert!(only_vector.iter().all(|f| f.channels == [Channel::Vector]));
        assert_eq!(only_vector[0].id.0, 3);
    }

    #[test]
    fn empty_inputs_fuse_to_nothing() {
        assert!(fuse(&[], &[]).is_empty());
    }

    #[test]
    fn a_repeated_id_counts_once_per_channel() {
        let fused = fuse(&ids(&[1, 1]), &[]);
        assert_eq!(fused.len(), 1);
        assert!((fused[0].score - 1.0 / (RRF_K + 1.0)).abs() < 1e-12);
    }
}
