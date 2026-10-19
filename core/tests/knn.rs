use futures_executor::block_on;
use repo_search_core::UnitVector;
use repo_search_core::knn::{KnnIndex, calibration_for};
use repo_search_core::store::{ContentHash, VectorCache};

fn hash_of(i: usize) -> ContentHash {
    ContentHash::of_chunk("fn v", &format!("fn v{i}() {{}}"))
}

fn vector_of(i: usize) -> UnitVector {
    UnitVector::normalize(vec![
        (i % 5) as f32 + 1.0,
        (i % 3) as f32 - 1.0,
        i as f32 * 0.1 + 0.5,
    ])
}

fn put(cache: &VectorCache, range: std::ops::Range<usize>) {
    let batch = range.map(|i| (hash_of(i), vector_of(i))).collect();
    block_on(cache.put_many("m", batch)).unwrap();
}

#[test]
fn calibration_is_reused_until_the_cache_doubles() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    assert!(block_on(calibration_for(&cache, "m")).is_err());

    put(&cache, 0..10);
    let first = block_on(calibration_for(&cache, "m")).unwrap();
    let stored = block_on(cache.calibration("m")).unwrap().unwrap();
    assert_eq!(stored.count, 10);
    assert!(stored.lo.iter().zip(&stored.hi).all(|(lo, hi)| lo < hi));

    // 19 < 2 * 10: the stored calibration still applies, even though the new
    // vectors lie outside it.
    put(&cache, 10..19);
    assert_eq!(block_on(calibration_for(&cache, "m")).unwrap(), first);
    assert_eq!(block_on(cache.calibration("m")).unwrap().unwrap().count, 10);

    // 20 = 2 * 10: recalibrated from every vector and stored.
    put(&cache, 19..20);
    let second = block_on(calibration_for(&cache, "m")).unwrap();
    assert_ne!(second, first);
    assert_eq!(block_on(cache.calibration("m")).unwrap().unwrap().count, 20);
}

#[test]
fn an_index_built_from_the_cache_finds_the_stored_vectors() {
    let common = tempfile::tempdir().unwrap();
    let cache = block_on(VectorCache::open(common.path())).unwrap();
    put(&cache, 0..50);

    let calibration = block_on(calibration_for(&cache, "m")).unwrap();
    let info = block_on(cache.model("m")).unwrap().unwrap();
    let rows = block_on(cache.scan("m", None, 1000)).unwrap();
    let index = KnnIndex::build(
        calibration,
        &info.file,
        rows.iter().map(|r| (r.content_hash, r.slot)),
    )
    .unwrap();

    let hits = index.search(&vector_of(7), 2);
    assert_eq!(hits.len(), 50);
    assert!(hits[0].1 > 0.999, "{hits:?}");
}
