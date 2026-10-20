//! The 512-token cap every chunk's embedded document is held to, counted
//! with the model's own tokenizer and document prompt.
//!
//! Both tests are `#[ignore]`d: they need the extracted model
//! (`REPO_SEARCH_MODEL_DIR`) and, for the corpus test, real corpora
//! (`REPO_SEARCH_CORPUS`). The release gate fetches them and runs the
//! tests with `--ignored`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use repo_search_core::TokenCounter;
use repo_search_core::chunk::{
    Chunk, Chunker, ConfigChunker, DocChunker, FileChunker, JavaScript, Language,
};
use repo_search_coreml::DocumentCounter;

const MAX_TOKENS: usize = 512;

/// The model's `DocumentCounter` from `REPO_SEARCH_MODEL_DIR`.
fn counter() -> Arc<DocumentCounter> {
    let dir = std::env::var("REPO_SEARCH_MODEL_DIR").expect(
        "REPO_SEARCH_MODEL_DIR is unset; point it at the extracted model \
         directory (manifest.json, tokenizer.json)",
    );
    Arc::new(
        DocumentCounter::open(&dir)
            .unwrap_or_else(|e| panic!("loading the model tokenizer from {dir}: {e}")),
    )
}

/// A single-line IIFE bundle — the usual shape of a bundled minified
/// file — becomes many capped chunks, every document under the cap. The
/// generated statements tokenize at a few bytes each, so the split is
/// wide enough to widen the `(part n/m)` suffix past three digits.
#[test]
#[ignore = "needs the model tokenizer (REPO_SEARCH_MODEL_DIR)"]
fn an_iife_bundle_becomes_many_capped_chunks() {
    let mut src = String::from("!function(){");
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..120_000 {
        // xorshift64*: deterministic pseudo-random hex, which merges badly.
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        src.push_str(&format!(
            "v{i}={:x};",
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        ));
    }
    src.push_str("}();");
    assert!(src.len() > 2 << 20);

    let counter = counter();
    let chunks = Chunker::new(JavaScript)
        .chunk_file("bundle.min.js", &src, &*counter)
        .unwrap();
    assert!(chunks.len() > 1000, "{} chunks", chunks.len());
    for chunk in &chunks {
        let mut text = String::new();
        chunk.write_text(&src, &mut text);
        assert!(
            counter.fits(&chunk.title, &[&text], MAX_TOKENS),
            "{} is over the cap",
            chunk.title,
        );
    }
}

/// Chunks `path`'s file when it is a supported language.
fn chunks_of(language: Language, rel: &str, src: &str, tokens: &dyn TokenCounter) -> Vec<Chunk> {
    let mut chunker: Box<dyn FileChunker> = match language {
        Language::C => Box::new(Chunker::new(repo_search_core::chunk::C)),
        Language::Cpp => Box::new(Chunker::new(repo_search_core::chunk::Cpp)),
        Language::Go => Box::new(Chunker::new(repo_search_core::chunk::Go)),
        Language::JavaScript => Box::new(Chunker::new(JavaScript)),
        Language::Python => Box::new(Chunker::new(repo_search_core::chunk::Python)),
        Language::Rust => Box::new(Chunker::new(repo_search_core::chunk::Rust)),
        Language::Swift => Box::new(Chunker::new(repo_search_core::chunk::Swift)),
        Language::TypeScript => Box::new(Chunker::new(repo_search_core::chunk::TypeScript)),
        Language::Tsx => Box::new(Chunker::new(repo_search_core::chunk::Tsx)),
        Language::Doc(format) => Box::new(DocChunker::new(format)),
        Language::Config(format) => Box::new(ConfigChunker::new(format)),
    };
    chunker.chunk_file(rel, src, tokens).unwrap()
}

/// Every file of `root`, recursively, skipping dot-entries and build output.
fn files(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_str().unwrap_or("");
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if name != "target" {
                files(&path, out);
            }
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// No chunk of the corpora exceeds 512 tokens of the embedding tokenizer.
/// `REPO_SEARCH_CORPUS` holds one or more corpus roots, separated like
/// `PATH`.
#[test]
#[ignore = "needs the corpora (REPO_SEARCH_CORPUS) and the model (REPO_SEARCH_MODEL_DIR); the release gate runs it"]
fn corpus_chunks_stay_under_the_cap() {
    let corpus = std::env::var("REPO_SEARCH_CORPUS").expect(
        "REPO_SEARCH_CORPUS is unset; point it at the corpus roots, \
         separated like PATH",
    );
    let counter = counter();

    let mut work: Vec<(PathBuf, PathBuf)> = Vec::new();
    for root in std::env::split_paths(&corpus) {
        assert!(root.is_dir(), "{root:?} is not a corpus checkout");
        let mut paths = Vec::new();
        files(&root, &mut paths);
        paths.sort();
        for path in paths {
            if Language::of_path(&path).is_some() {
                work.push((root.clone(), path));
            }
        }
    }
    assert!(!work.is_empty(), "no supported files under {corpus}");

    // ~100k tokenizer encodes: shard the file list across cores, strided
    // so the dense data files do not pile onto one worker.
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(work.len());
    let shards: Vec<Vec<&(PathBuf, PathBuf)>> = (0..workers)
        .map(|w| work.iter().skip(w).step_by(workers).collect())
        .collect();
    let totals: Vec<(u32, u32, Vec<PathBuf>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .iter()
            .map(|shard| {
                let counter = Arc::clone(&counter);
                scope.spawn(move || {
                    let (mut files, mut chunks_total) = (0u32, 0u32);
                    let mut non_utf8 = Vec::new();
                    for (root, path) in shard.iter().copied() {
                        let language = Language::of_path(path).unwrap();
                        let rel = path.strip_prefix(root).unwrap().to_str().unwrap();
                        let Ok(src) = std::fs::read_to_string(path) else {
                            non_utf8.push(path.to_path_buf());
                            continue;
                        };
                        let chunks = chunks_of(language, rel, &src, &*counter);
                        files += 1;
                        for chunk in &chunks {
                            let mut text = String::new();
                            chunk.write_text(&src, &mut text);
                            assert!(
                                counter.fits(&chunk.title, &[&text], MAX_TOKENS),
                                "{rel} chunk {title:?} is over the cap",
                                title = chunk.title,
                            );
                        }
                        chunks_total += chunks.len() as u32;
                    }
                    (files, chunks_total, non_utf8)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (files_total, chunks_total): (u32, u32) =
        totals.iter().fold((0, 0), |(f, c), t| (f + t.0, c + t.1));
    let non_utf8: Vec<&PathBuf> = totals.iter().flat_map(|t| t.2.iter()).collect();
    eprintln!(
        "chunked {chunks_total} chunks of {files_total} files; \
         {} files were not UTF-8 and were skipped:",
        non_utf8.len(),
    );
    for path in &non_utf8 {
        eprintln!("  not UTF-8: {}", path.display());
    }
    assert!(files_total > 0);
    assert!(chunks_total > 0);
}
