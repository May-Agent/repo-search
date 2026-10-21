//! Per-bucket median embed latency on the Neural Engine instance.
//!
//! `cargo run -p repo-search-coreml --release --example bench -- <model_dir> [iterations]`

use std::env;
use std::time::Instant;

use repo_search_coreml::{ComputeUnits, CoreMlEmbedder};

fn main() {
    let mut args = env::args().skip(1);
    let dir = args.next().expect("usage: bench <model_dir> [iterations]");
    let iters: usize = args.next().map_or(40, |s| s.parse().expect("iterations"));
    let mut embedder = CoreMlEmbedder::open(&dir, "bench", ComputeUnits::CpuAndNeuralEngine)
        .expect("open the model directory");
    for len in [128_usize, 256, 512] {
        // A full-length input routes to the `len_*` function of that bucket.
        let ids = vec![42_u32; len];
        for _ in 0..8 {
            embedder.embed_tokens(&ids).unwrap();
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            embedder.embed_tokens(&ids).unwrap();
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];
        println!("len_{len}: median {:.2} ms", median.as_secs_f64() * 1e3);
    }
}
