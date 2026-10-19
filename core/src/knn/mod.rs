//! In-memory int8 KNN with an fp16 re-rank.
//!
//! The vector cache's flat fp16 file is the source of truth. A [`KnnIndex`]
//! holds a per-dimension int8 copy of the vectors present in one worktree and
//! scans all of it per query, keeping the [`FIRST_PASS`] best codes; those are
//! re-ranked against their fp16 rows read through a read-only mmap, and the
//! best [`RESULTS`] are returned.

mod calibration;
mod fp16;
mod index;
mod qos;
mod topk;

pub use calibration::{Calibration, calibration_for};
pub use index::KnnIndex;
#[cfg(target_os = "macos")]
pub use qos::performance_cores;

use std::io;

use crate::store;

/// Candidates each scan keeps and the merged first pass hands to the re-rank.
pub const FIRST_PASS: usize = 300;

/// Results [`KnnIndex::search`] returns after the re-rank.
pub const RESULTS: usize = 200;

/// Failures of building or updating an index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The vector cache failed.
    #[error("vector cache: {0}")]
    Store(#[from] store::Error),

    /// Mapping or inspecting the fp16 file failed.
    #[error("fp16 file: {0}")]
    Io(#[from] io::Error),

    /// The model has no cached vectors to calibrate from.
    #[error("model {0:?} has no cached vectors")]
    NoVectors(String),

    /// A slot lies beyond the end of the fp16 file; the cache wrote the
    /// database row without its vector.
    #[error("slot {slot} is past the end of the fp16 file ({rows} rows)")]
    SlotPastEnd {
        /// The requested slot.
        slot: u64,
        /// How many whole rows the file holds.
        rows: u64,
    },
}

#[cfg(test)]
mod tests;
