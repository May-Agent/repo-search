//! Per-dimension int8 quantization.

use numkong::f16;

use crate::UnitVector;
use crate::store::{StoredCalibration, VectorCache};

use super::Error;
use super::fp16::Fp16File;

/// Rows read per `scan` page while calibrating.
const SCAN_PAGE: u32 = 4096;

/// Per-dimension affine map from fp32 components to unsigned 8-bit codes:
/// `code = round((x - lo) / scale)`, `scale = (hi - lo) / 255`.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Per-dimension minimum.
    pub lo: Box<[f32]>,
    /// Per-dimension step, `(hi - lo) / 255`; zero for a constant dimension.
    pub scale: Box<[f32]>,
}

/// Running per-dimension min/max.
struct Bounds {
    lo: Box<[f32]>,
    hi: Box<[f32]>,
    rows: u64,
}

impl Bounds {
    fn new(dim: usize) -> Self {
        Self {
            lo: vec![f32::INFINITY; dim].into(),
            hi: vec![f32::NEG_INFINITY; dim].into(),
            rows: 0,
        }
    }

    fn add(&mut self, row: &[f16]) {
        assert_eq!(row.len(), self.lo.len(), "row dimension");
        for ((x, lo), hi) in row.iter().zip(&mut self.lo).zip(&mut self.hi) {
            let x = x.to_f32();
            *lo = lo.min(x);
            *hi = hi.max(x);
        }
        self.rows += 1;
    }
}

impl Calibration {
    /// Calibrates over `rows`, which must not be empty and share one dimension.
    ///
    /// # Panics
    ///
    /// When `rows` is empty or the rows differ in dimension.
    #[must_use]
    pub fn from_rows<'a>(rows: impl IntoIterator<Item = &'a [f16]>) -> Self {
        let mut rows = rows.into_iter();
        let first = rows.next().expect("cannot calibrate without rows");
        let mut bounds = Bounds::new(first.len());
        bounds.add(first);
        rows.for_each(|row| bounds.add(row));
        Self::from_bounds(&bounds.lo, &bounds.hi)
    }

    /// The calibration of stored per-dimension bounds.
    ///
    /// # Panics
    ///
    /// When `lo` and `hi` differ in length.
    #[must_use]
    pub fn from_bounds(lo: &[f32], hi: &[f32]) -> Self {
        assert_eq!(lo.len(), hi.len(), "bounds dimension");
        Self {
            lo: lo.into(),
            scale: lo.iter().zip(hi).map(|(l, h)| (h - l) / 255.0).collect(),
        }
    }

    /// Components per vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.lo.len()
    }

    /// Encodes `row` into signed codes: `round((x - lo) / scale)` clamped to
    /// `0..=255`, minus 128. A constant dimension encodes to -128.
    ///
    /// # Panics
    ///
    /// When `row` or `out` is not [`dim`](Self::dim) long.
    pub fn encode(&self, row: &[f16], out: &mut [i8]) {
        assert_eq!(row.len(), self.dim(), "row dimension");
        assert_eq!(out.len(), self.dim(), "output dimension");
        for (((x, out), lo), scale) in row.iter().zip(out).zip(&self.lo).zip(&self.scale) {
            *out = if *scale == 0.0 {
                -128
            } else {
                (((x.to_f32() - lo) / scale).round().clamp(0.0, 255.0) - 128.0) as i8
            };
        }
    }

    /// Quantizes a query for the int8 dot with [`encode`](Self::encode)d rows:
    /// weights `q * scale` divided by their one scale `max|w| / 127`, rounded
    /// and clamped to `-127..=127`.
    ///
    /// # Panics
    ///
    /// When `q` or `out` is not [`dim`](Self::dim) long, or every weight is
    /// zero — impossible for a unit query against a calibration of real data.
    pub fn query_weights(&self, q: &UnitVector, out: &mut [i8]) {
        let q = q.as_slice();
        assert_eq!(q.len(), self.dim(), "query dimension");
        assert_eq!(out.len(), self.dim(), "output dimension");
        let weights: Vec<f32> = q.iter().zip(&self.scale).map(|(q, s)| q * s).collect();
        let step = weights.iter().fold(0f32, |m, w| m.max(w.abs())) / 127.0;
        assert!(
            step > 0.0,
            "query weights are all zero: the calibration is degenerate"
        );
        for (out, w) in out.iter_mut().zip(&weights) {
            *out = (w / step).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// The calibration to build an index of `model_id` with: the stored one while
/// the cache holds fewer than twice the vectors it was taken over, otherwise
/// bounds recomputed from every cached vector, stored, and used. A calibration
/// taken while the cache held only short chunks must not stick. Codes live in
/// memory only, so recalibrating invalidates nothing on disk.
///
/// The recomputation reads the whole fp16 file page by page, doing a
/// bounded amount of compute between awaits.
///
/// # Errors
///
/// When the model has no cached vectors, the cache or the fp16 file cannot be
/// read, or the new calibration cannot be stored.
pub async fn calibration_for(cache: &VectorCache, model_id: &str) -> Result<Calibration, Error> {
    let info = cache
        .model(model_id)
        .await?
        .ok_or_else(|| Error::NoVectors(model_id.to_owned()))?;
    if let Some(stored) = cache.calibration(model_id).await?
        && info.vectors < stored.count.saturating_mul(2)
    {
        return Ok(Calibration::from_bounds(&stored.lo, &stored.hi));
    }

    let mut file = Fp16File::open(&info.file, info.dim)?;
    let mut bounds = Bounds::new(info.dim);
    let mut cursor = None;
    loop {
        let page = cache.scan(model_id, cursor, SCAN_PAGE).await?;
        for entry in &page {
            file.cover(entry.slot)?;
            bounds.add(file.row(entry.slot));
        }
        if page.len() < SCAN_PAGE as usize {
            break;
        }
        cursor = page.last().map(|entry| entry.content_hash);
    }
    if bounds.rows == 0 {
        return Err(Error::NoVectors(model_id.to_owned()));
    }
    cache
        .set_calibration(
            model_id,
            StoredCalibration {
                lo: bounds.lo.clone(),
                hi: bounds.hi.clone(),
                count: bounds.rows,
            },
        )
        .await?;
    Ok(Calibration::from_bounds(&bounds.lo, &bounds.hi))
}
