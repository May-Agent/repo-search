//! Model-dependent checks, gated on `REPO_SEARCH_MODEL_DIR` naming an
//! extracted model directory (`*.mlmodelc`, `tokenizer.json`, `manifest.json`).
//! They fail loudly when it is unset — the model is the fixture.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use repo_search_core::{BatchEmbedder, Embedder, QueryTask, UnitVector};
use repo_search_coreml::{ComputeUnits, CoreMlEmbedder, Error, gpu_headroom};
use serde::Deserialize;

/// The model archive's sha256: the model id vectors are cached under.
const MODEL_ID: &str = "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c";

const MIN_COSINE: f32 = 0.9999;

fn model_dir() -> PathBuf {
    PathBuf::from(
        env::var("REPO_SEARCH_MODEL_DIR")
            .expect("REPO_SEARCH_MODEL_DIR must name an extracted eg2-coreml model directory"),
    )
}

#[derive(Deserialize)]
struct Reference {
    samples: Vec<Sample>,
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Sample {
    file: String,
    tokens: usize,
    input_ids: Vec<u32>,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct Document {
    title: String,
    text: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct Query {
    query: String,
    /// Tokens before truncation; the reference emits the untruncated count.
    tokens: usize,
    embedding: Vec<f32>,
}

/// The Python Core ML reference, run on demand in the convert env.
fn reference(model_dir: &Path) -> Reference {
    let python = Path::new(env!("CARGO_MANIFEST_DIR")).join("../convert/.venv/bin/python");
    assert!(
        python.is_file(),
        "convert env missing ({}): `uv sync` in convert/, then \
         `uv pip install --python convert/.venv/bin/python --no-deps <fixed coremltools wheel>`",
        python.display()
    );
    let out = Command::new(&python)
        .args(["-I", "-m", "repo_search_convert.reference", "--model-dir"])
        .arg(model_dir)
        .output()
        .expect("spawn the reference python");
    assert!(
        out.status.success(),
        "reference failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("reference output is JSON")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "needs REPO_SEARCH_MODEL_DIR; run with --ignored"]
fn embeddings_match_python_reference() {
    let dir = model_dir();
    let reference = reference(&dir);
    let mut embedder =
        CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let mut worst = (f32::INFINITY, String::new());
    let mut check = |label: String, got: &UnitVector, want: &[f32]| {
        let c = cosine(got.as_slice(), want);
        if c < worst.0 {
            worst = (c, label.clone());
        }
        assert!(c >= MIN_COSINE, "{label}: cosine {c} < {MIN_COSINE}");
    };
    for s in &reference.samples {
        let v = embedder.embed_tokens(&s.input_ids).unwrap();
        check(
            format!("{} ({} tokens)", s.file, s.tokens),
            &v,
            &s.embedding,
        );
    }
    for d in &reference.documents {
        let v = embedder.embed_document(&d.title, &d.text).unwrap();
        check(format!("document {}", d.title), &v, &d.embedding);
    }
    for q in &reference.queries {
        let v = embedder
            .embed_query(QueryTask::CodeRetrieval, &q.query)
            .unwrap();
        let shown: String = q.query.chars().take(40).collect();
        check(
            format!("query {shown:?} ({} tokens)", q.tokens),
            &v,
            &q.embedding,
        );
    }
    // The query set must include one over the 512-token cap, so the
    // truncation path is what the parity above compares.
    assert!(
        reference.queries.iter().any(|q| q.tokens > 512),
        "reference has no over-cap query"
    );
    eprintln!("worst cosine {:.6} ({})", worst.0, worst.1);
}

#[test]
#[ignore = "needs REPO_SEARCH_MODEL_DIR; run with --ignored"]
fn ane_and_gpu_instances_run_concurrently() {
    let dir = model_dir();
    const TEXT: &str = "fn main() { index_everything(); }";
    const ITERATIONS: usize = 30;
    match gpu_headroom() {
        Ok(()) => {
            // Both threads finish `open` first, then release together so
            // their embedding loops actually overlap.
            let barrier = std::sync::Barrier::new(2);
            let run = |units: ComputeUnits| {
                let mut e = CoreMlEmbedder::open(&dir, MODEL_ID, units).unwrap();
                barrier.wait();
                let mut vs = Vec::with_capacity(ITERATIONS);
                for _ in 0..ITERATIONS {
                    vs.push(e.embed_document("main.rs", TEXT).unwrap());
                }
                vs
            };
            let (ane, gpu) = std::thread::scope(|s| {
                let ane = s.spawn(|| run(ComputeUnits::CpuAndNeuralEngine));
                let gpu = s.spawn(|| run(ComputeUnits::CpuAndGpu));
                (
                    ane.join().expect("ANE thread"),
                    gpu.join().expect("GPU thread"),
                )
            });
            for v in ane.iter().chain(&gpu) {
                let norm: f32 = v.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
            }
            // Same input on both devices: the vectors must agree.
            let c = cosine(ane[0].as_slice(), gpu[0].as_slice());
            assert!(c > 0.99, "ANE vs GPU cosine {c}");
        }
        Err(blocked) => {
            // Without headroom the guard must refuse construction.
            let err = match CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndGpu) {
                Err(e) => e,
                Ok(_) => panic!("GPU embedder must not open without headroom"),
            };
            assert!(
                matches!(err, Error::GpuBlocked(ref b) if *b == blocked),
                "{err}"
            );
        }
    }
}

#[test]
#[ignore = "needs REPO_SEARCH_MODEL_DIR; run with --ignored"]
fn document_over_cap_errors_and_query_truncates() {
    let dir = model_dir();
    let mut embedder =
        CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let err = match embedder.embed_tokens(&vec![5; 600]) {
        Err(e) => e,
        Ok(_) => panic!("600 tokens exceed the largest bucket"),
    };
    assert!(matches!(err, Error::TooLong { tokens: 600, .. }), "{err}");
    // A document whose tokenized form exceeds the cap fails the same way: it
    // is a chunker bug, not an input to truncate.
    let long_doc = "token ".repeat(600);
    assert!(matches!(
        embedder.embed_document("long.rs", &long_doc),
        Err(Error::TooLong { .. })
    ));
    // A query is truncated at the cap instead of failing.
    embedder
        .embed_query(QueryTask::CodeRetrieval, &long_doc)
        .expect("query is truncated at the cap");
}

/// Eight documents of one bucket.
fn batch_documents() -> Vec<(String, String)> {
    (0..8)
        .map(|i| {
            (
                format!("file{i}.rs"),
                format!("fn function_{i}() {{ index_everything({i}); }}"),
            )
        })
        .collect()
}

#[test]
#[ignore = "needs REPO_SEARCH_MODEL_DIR; run with --ignored"]
fn a_batch_of_eight_equals_eight_single_embeds() {
    let dir = model_dir();
    let docs = batch_documents();
    let docs: Vec<(&str, &str)> = docs.iter().map(|(t, x)| (t.as_str(), x.as_str())).collect();
    let mut ane = CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let singles: Vec<UnitVector> = docs
        .iter()
        .map(|(title, text)| ane.embed_document(title, text).unwrap())
        .collect();
    let batch = ane.embed_documents(&docs).unwrap();
    assert_eq!(batch.len(), docs.len());
    for (i, (b, s)) in batch.iter().zip(&singles).enumerate() {
        let c = cosine(b.as_slice(), s.as_slice());
        assert!(c >= MIN_COSINE, "document {i}: cosine {c} < {MIN_COSINE}");
    }
    match gpu_headroom() {
        Ok(()) => {
            let mut gpu = CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndGpu).unwrap();
            let batch = gpu.embed_documents(&docs).unwrap();
            assert_eq!(batch.len(), docs.len());
            for (i, (b, s)) in batch.iter().zip(&singles).enumerate() {
                let c = cosine(b.as_slice(), s.as_slice());
                assert!(c > 0.99, "document {i}: GPU batch vs ANE cosine {c}");
            }
        }
        Err(blocked) => {
            let err = match CoreMlEmbedder::open(&dir, MODEL_ID, ComputeUnits::CpuAndGpu) {
                Err(e) => e,
                Ok(_) => panic!("GPU embedder must not open without headroom"),
            };
            assert!(
                matches!(err, Error::GpuBlocked(ref b) if *b == blocked),
                "{err}"
            );
        }
    }
}
