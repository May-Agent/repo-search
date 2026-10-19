//! Read-only mapping of a model's flat fp16 file.

use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::{MmapOptions, MmapRaw};
use numkong::f16;

use super::Error;

const _: () = assert!(
    cfg!(target_endian = "little"),
    "the fp16 file is little-endian and mapped in place"
);

/// The fp16 rows of one model: row `slot` is at byte offset `slot * dim * 2`.
///
/// The cache never rewrites a committed row and never shrinks the file, and
/// only committed slots (those the `vectors` table references) are read, so a
/// mapped committed row stays valid and unchanging. Bytes past `next_slot` may
/// be overwritten by other processes; the mapping is raw and no reference is
/// ever created over it as a whole, only over single committed rows.
pub(super) struct Fp16File {
    file: File,
    map: Option<MmapRaw>,
    dim: usize,
}

impl Fp16File {
    /// Maps whatever the file holds now.
    pub(super) fn open(path: &Path, dim: usize) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut this = Self {
            file,
            map: None,
            dim,
        };
        this.remap()?;
        Ok(this)
    }

    fn remap(&mut self) -> io::Result<()> {
        self.map = if self.file.metadata()?.len() == 0 {
            None
        } else {
            // A raw map is safe to create: no reference spans it. The cache
            // never truncates the file, so every page stays backed, and
            // `row` views only single committed rows.
            Some(MmapOptions::new().map_raw_read_only(&self.file)?)
        };
        Ok(())
    }

    fn rows(&self) -> u64 {
        self.map
            .as_ref()
            .map_or(0, |m| (m.len() / (self.dim * 2)) as u64)
    }

    /// Makes row `slot` readable, remapping when the file grew past the map.
    pub(super) fn cover(&mut self, slot: u64) -> Result<(), Error> {
        if slot >= self.rows() {
            self.remap()?;
        }
        if slot >= self.rows() {
            return Err(Error::SlotPastEnd {
                slot,
                rows: self.rows(),
            });
        }
        Ok(())
    }

    /// Row `slot`, which `cover` must have admitted.
    pub(super) fn row(&self, slot: u64) -> &[f16] {
        let map = self.map.as_ref().expect("row read before cover");
        let start = usize::try_from(slot).expect("slot fits in usize") * self.dim * 2;
        assert!(start + self.dim * 2 <= map.len(), "row past the map");
        // SAFETY: `cover` admitted `slot`, so the row lies inside the map; it
        // is a committed row, which is never rewritten, so the slice covers
        // only bytes that stay fixed for its lifetime, and no reference to any
        // other part of the map exists. `f16` is a transparent u16 and the host
        // is little-endian, as asserted above; the mapping is page-aligned and
        // `start` is even, so the pointer is 2-aligned; the length is `dim`
        // whole elements.
        unsafe { std::slice::from_raw_parts(map.as_ptr().add(start).cast::<f16>(), self.dim) }
    }
}
