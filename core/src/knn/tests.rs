use std::collections::HashSet;
use std::fs;
use std::path::Path;

use numkong::f16;

use super::{Calibration, Error, FIRST_PASS, KnnIndex, RESULTS};
use crate::UnitVector;
use crate::store::ContentHash;

fn fp16(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&x| f16::from_f32(x)).collect()
}

fn hash_of(i: usize) -> ContentHash {
    ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}"))
}

/// A seeded generator (splitmix64) of unit vectors.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self, dim: usize) -> UnitVector {
        let v = (0..dim)
            .map(|_| (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            .collect();
        UnitVector::normalize(v)
    }
}

/// Writes `vectors` as the flat fp16 file: row `i` at slot `i`.
fn write_rows(path: &Path, vectors: &[UnitVector]) {
    let bytes: Vec<u8> = vectors
        .iter()
        .flat_map(|v| {
            v.as_slice()
                .iter()
                .flat_map(|&x| f16::from_f32(x).0.to_le_bytes())
        })
        .collect();
    fs::write(path, bytes).unwrap();
}

fn calibrate(vectors: &[UnitVector]) -> Calibration {
    let rows: Vec<Vec<f16>> = vectors.iter().map(|v| fp16(v.as_slice())).collect();
    Calibration::from_rows(rows.iter().map(Vec::as_slice))
}

fn calibration_3d() -> Calibration {
    let a = fp16(&[0.0, 1.0, 5.0]);
    let b = fp16(&[2.0, 1.0, -1.0]);
    Calibration::from_rows([a.as_slice(), b.as_slice()])
}

#[test]
fn calibration_takes_per_dimension_bounds() {
    let c = calibration_3d();
    assert_eq!(&*c.lo, &[0.0, 1.0, -1.0]);
    assert_eq!(&*c.scale, &[2.0 / 255.0, 0.0, 6.0 / 255.0]);
}

#[test]
fn encode_follows_the_reference_formula() {
    let c = calibration_3d();
    let mut out = [0i8; 3];

    // 0.5 / (2/255) = 63.75 -> 64; the constant dimension is -128;
    // (0.75 + 1) / (6/255) = 74.375 -> 74.
    c.encode(&fp16(&[0.5, 7.0, 0.75]), &mut out);
    assert_eq!(out, [-64, -128, -54]);

    // Values outside the calibrated range clamp.
    c.encode(&fp16(&[3.0, 1.0, -5.0]), &mut out);
    assert_eq!(out, [127, -128, -128]);
}

#[test]
fn query_weights_fold_the_scale_and_quantize_to_one_step() {
    let c = calibration_3d();
    let mut out = [0i8; 3];

    // w = [0.6 * 2/255, 0, 0.8 * 6/255]; the largest maps to 127 and the
    // first to round(0.6 * 2 / (0.8 * 6) * 127) = round(31.75) = 32.
    c.query_weights(&UnitVector::normalize(vec![0.6, 0.0, 0.8]), &mut out);
    assert_eq!(out, [32, 0, 127]);

    c.query_weights(&UnitVector::normalize(vec![0.0, 1.0, -1.0]), &mut out);
    assert_eq!(out, [0, 0, -127]);
}

#[test]
#[should_panic(expected = "all zero")]
fn query_weights_reject_an_all_zero_fold() {
    let c = calibration_3d();
    // Only the constant dimension is non-zero, and its scale is 0.
    c.query_weights(&UnitVector::normalize(vec![0.0, 1.0, 0.0]), &mut [0; 3]);
}

#[test]
fn search_agrees_with_an_exact_scan_and_is_thread_count_independent() {
    const DIM: usize = 64;
    let mut rng = Rng(7);
    let vectors: Vec<UnitVector> = (0..2000).map(|_| rng.unit(DIM)).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.f16");
    write_rows(&file, &vectors);
    let rows = (0..vectors.len()).map(|i| (hash_of(i), i as u64));
    let index = KnnIndex::build(calibrate(&vectors), &file, rows).unwrap();
    assert_eq!(index.len(), 2000);

    let mut overlap = 0.0;
    let queries = 10;
    for _ in 0..queries {
        let query = rng.unit(DIM);
        let one = index.search(&query, 1, None);
        let four = index.search(&query, 4, None);
        assert_eq!(one, four);
        assert_eq!(one.len(), RESULTS);
        assert!(one.windows(2).all(|w| w[0].1 >= w[1].1), "best first");

        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot = v
                    .as_slice()
                    .iter()
                    .zip(query.as_slice())
                    .map(|(a, b)| a * b)
                    .sum();
                (i, dot)
            })
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        let truth: HashSet<ContentHash> =
            exact[..RESULTS].iter().map(|&(i, _)| hash_of(i)).collect();
        overlap += one.iter().filter(|(h, _)| truth.contains(h)).count() as f64 / RESULTS as f64;
        assert_eq!(
            one[0].0,
            hash_of(exact[0].0),
            "the nearest vector comes first"
        );
    }
    let overlap = overlap / f64::from(queries);
    assert!(overlap >= 0.95, "top-{RESULTS} overlap {overlap}");
}

#[test]
fn more_threads_than_rows_still_works() {
    let mut rng = Rng(1);
    let vectors: Vec<UnitVector> = (0..5).map(|_| rng.unit(16)).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.f16");
    write_rows(&file, &vectors);
    let rows = (0..5).map(|i| (hash_of(i), i as u64));
    let index = KnnIndex::build(calibrate(&vectors), &file, rows).unwrap();
    let query = rng.unit(16);
    assert_eq!(index.search(&query, 1, None), index.search(&query, 8, None));
    assert_eq!(index.search(&query, 1, None).len(), 5);
}

#[test]
fn first_pass_is_wider_than_the_result() {
    const { assert!(FIRST_PASS > RESULTS) };
}

#[test]
fn insert_remove_and_remap_keep_the_index_consistent() {
    const DIM: usize = 32;
    let mut rng = Rng(99);
    let vectors: Vec<UnitVector> = (0..40).map(|_| rng.unit(DIM)).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.f16");
    write_rows(&file, &vectors[..30]);
    let calibration = calibrate(&vectors);

    let mut index = KnnIndex::build(
        calibration.clone(),
        &file,
        (0..20).map(|i| (hash_of(i), i as u64)),
    )
    .unwrap();

    // A slot the file does not hold yet is an error, not a stray read.
    let err = index.insert(hash_of(35), 35).unwrap_err();
    assert!(
        matches!(err, Error::SlotPastEnd { slot: 35, rows: 30 }),
        "{err:?}"
    );

    // The file grows; the next insert past the old map remaps.
    write_rows(&file, &vectors);
    for i in 20..40 {
        index.insert(hash_of(i), i as u64).unwrap();
    }
    assert_eq!(index.len(), 40);

    // Remove from the front, middle and back (the last row swaps into holes).
    for i in [0, 17, 39, 5] {
        assert!(index.remove(&hash_of(i)));
        assert!(!index.contains(&hash_of(i)));
    }
    assert!(!index.remove(&hash_of(0)));
    assert_eq!(index.len(), 36);

    let remaining: Vec<usize> = (0..40).filter(|i| ![0, 17, 39, 5].contains(i)).collect();
    let fresh = KnnIndex::build(
        calibration,
        &file,
        remaining.iter().map(|&i| (hash_of(i), i as u64)),
    )
    .unwrap();
    for _ in 0..5 {
        let query = rng.unit(DIM);
        assert_eq!(index.search(&query, 1, None), fresh.search(&query, 1, None));
    }
}

#[test]
#[should_panic(expected = "already indexed")]
fn inserting_an_indexed_hash_panics() {
    let mut rng = Rng(3);
    let vectors: Vec<UnitVector> = (0..3).map(|_| rng.unit(8)).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.f16");
    write_rows(&file, &vectors);
    let mut index = KnnIndex::build(calibrate(&vectors), &file, [(hash_of(0), 0)]).unwrap();
    let _ = index.insert(hash_of(0), 1);
}

/// An index over `vectors`, row `i` holding `hash_of(i)`.
fn index_of(vectors: &[UnitVector], dir: &Path) -> KnnIndex {
    let file = dir.join("m.f16");
    write_rows(&file, vectors);
    let rows = (0..vectors.len()).map(|i| (hash_of(i), i as u64));
    KnnIndex::build(calibrate(vectors), &file, rows).unwrap()
}

fn dot(a: &UnitVector, b: &UnitVector) -> f32 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(a, b)| a * b)
        .sum()
}

#[test]
fn a_full_mask_changes_nothing_and_an_empty_one_finds_nothing() {
    let mut rng = Rng(11);
    let vectors: Vec<UnitVector> = (0..500).map(|_| rng.unit(32)).collect();
    let dir = tempfile::tempdir().unwrap();
    let index = index_of(&vectors, dir.path());
    let query = rng.unit(32);

    let all: Vec<ContentHash> = (0..500).map(hash_of).collect();
    let full = index.mask(&all);
    assert_eq!(full.count(), 500);
    assert_eq!(
        index.search(&query, 3, Some(&full)),
        index.search(&query, 3, None)
    );
    let none = index.mask(&[]);
    assert!(index.search(&query, 3, Some(&none)).is_empty());
    // A hash the index does not hold admits nothing.
    assert_eq!(index.mask(&[hash_of(9999)]).count(), 0);
}

#[test]
fn masked_search_returns_the_best_rows_inside_the_mask_whatever_the_thread_count() {
    const DIM: usize = 64;
    let mut rng = Rng(21);
    let vectors: Vec<UnitVector> = (0..3000).map(|_| rng.unit(DIM)).collect();
    let dir = tempfile::tempdir().unwrap();
    let index = index_of(&vectors, dir.path());
    let query = rng.unit(DIM);

    // The 300 rows farthest from the query: none can be in the unmasked top.
    let mut by_score: Vec<usize> = (0..vectors.len()).collect();
    by_score.sort_by(|&a, &b| dot(&vectors[a], &query).total_cmp(&dot(&vectors[b], &query)));
    let farthest: Vec<usize> = by_score[..300].to_vec();
    let hashes: Vec<ContentHash> = farthest.iter().map(|&i| hash_of(i)).collect();
    let mask = index.mask(&hashes);
    assert_eq!(mask.count(), 300);

    let unmasked = index.search(&query, 1, None);
    assert!(unmasked.iter().all(|(h, _)| !hashes.contains(h)));

    let one = index.search(&query, 1, Some(&mask));
    let four = index.search(&query, 4, Some(&mask));
    assert_eq!(one, four);
    assert_eq!(one.len(), RESULTS);
    assert!(one.iter().all(|(h, _)| hashes.contains(h)));
    assert!(one.windows(2).all(|w| w[0].1 >= w[1].1), "best first");

    let mut exact: Vec<usize> = farthest.clone();
    exact.sort_by(|&a, &b| dot(&vectors[b], &query).total_cmp(&dot(&vectors[a], &query)));
    let truth: HashSet<ContentHash> = exact[..RESULTS].iter().map(|&i| hash_of(i)).collect();
    let overlap = one.iter().filter(|(h, _)| truth.contains(h)).count() as f64 / RESULTS as f64;
    assert!(overlap >= 0.95, "masked top-{RESULTS} overlap {overlap}");
    assert_eq!(
        one[0].0,
        hash_of(exact[0]),
        "the nearest admitted row first"
    );
}

#[test]
fn masked_ties_resolve_to_the_lower_row_for_any_thread_count() {
    // Every row holds the same vector, so every score is equal.
    let mut rng = Rng(5);
    let v = rng.unit(16);
    let vectors: Vec<UnitVector> = (0..40).map(|_| v.clone()).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.f16");
    write_rows(&file, &vectors);
    let calibration = calibrate(&[v.clone(), rng.unit(16)]);
    let index =
        KnnIndex::build(calibration, &file, (0..40).map(|i| (hash_of(i), i as u64))).unwrap();

    let admitted = [30usize, 4, 17, 9, 38, 22];
    let hashes: Vec<ContentHash> = admitted.iter().map(|&i| hash_of(i)).collect();
    let mask = index.mask(&hashes);
    let want: Vec<ContentHash> = [4usize, 9, 17, 22, 30, 38]
        .iter()
        .map(|&i| hash_of(i))
        .collect();
    for threads in [1, 2, 4, 7] {
        let got: Vec<ContentHash> = index
            .search(&v, threads, Some(&mask))
            .into_iter()
            .map(|(h, _)| h)
            .collect();
        assert_eq!(got, want, "{threads} threads");
    }
}

#[test]
fn a_mask_built_after_a_removal_follows_the_new_row_numbers() {
    let mut rng = Rng(8);
    let vectors: Vec<UnitVector> = (0..20).map(|_| rng.unit(16)).collect();
    let dir = tempfile::tempdir().unwrap();
    let mut index = index_of(&vectors, dir.path());
    // Row 19 moves into row 2's place.
    assert!(index.remove(&hash_of(2)));
    let mask = index.mask(&[hash_of(19), hash_of(2)]);
    assert_eq!(mask.count(), 1);
    let hits = index.search(&vectors[19], 2, Some(&mask));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, hash_of(19));
}

#[test]
#[should_panic(expected = "another row count")]
fn a_mask_of_another_size_panics() {
    let mut rng = Rng(2);
    let vectors: Vec<UnitVector> = (0..4).map(|_| rng.unit(8)).collect();
    let dir = tempfile::tempdir().unwrap();
    let index = index_of(&vectors, dir.path());
    let mask = super::RowMask::none(3);
    let _ = index.search(&vectors[0], 1, Some(&mask));
}
