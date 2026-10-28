//! The 512-token cap every chunk's embedded document is held to, counted
//! with the daemon's [`PromptCounter`] — the model's own tokenizer and
//! document prompt.
//!
//! The cap tests and the parity test are `#[ignore]`d: they need the
//! extracted model (`REPO_SEARCH_MODEL_DIR`) and, for the corpus, real
//! corpora (`REPO_SEARCH_CORPUS`). The release gate fetches them and runs
//! the tests with `--ignored`.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use repo_search::daemon::counter::PromptCounter;
use repo_search::model_store::Sha256Digest;
use repo_search_core::TokenCounter;
use repo_search_core::chunk::{
    Chunk, Chunker, ConfigChunker, DocChunker, FileChunker, JavaScript, Language,
};
use repo_search_core::model::{DOCUMENT, MAX_TOKENS, TOKENIZER};

/// The daemon's `PromptCounter` from `REPO_SEARCH_MODEL_DIR`.
fn counter() -> Arc<PromptCounter> {
    let dir = std::env::var("REPO_SEARCH_MODEL_DIR").expect(
        "REPO_SEARCH_MODEL_DIR is unset; point it at the extracted model \
         directory (manifest.json, tokenizer.json)",
    );
    let tokenizer = Path::new(&dir).join("tokenizer.json");
    Arc::new(
        PromptCounter::open(&tokenizer)
            .unwrap_or_else(|e| panic!("loading the model tokenizer from {dir}: {e}")),
    )
}

/// The archive's own `tokenizer.json` hashes to the upstream
/// [`TOKENIZER`] pin: the daemon and the server can never tokenize
/// differently.
#[test]
#[ignore = "needs the extracted model (REPO_SEARCH_MODEL_DIR)"]
fn the_bundled_tokenizer_matches_the_pin() {
    let dir = std::env::var("REPO_SEARCH_MODEL_DIR").expect("REPO_SEARCH_MODEL_DIR is unset");
    let bytes = std::fs::read(Path::new(&dir).join("tokenizer.json")).unwrap();
    assert_eq!(
        Sha256Digest::of(&bytes).hex(),
        TOKENIZER.sha256,
        "the archive's tokenizer.json does not match the upstream pin"
    );
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

/// A query prompt that encodes over the cap is truncated for the
/// request: the `task:`/`query:` framing survives and the token count
/// comes down to `MAX_TOKENS`.
#[test]
#[ignore = "needs the model tokenizer (REPO_SEARCH_MODEL_DIR)"]
fn a_query_prompt_over_the_cap_is_truncated() {
    let counter = counter();
    let query = "how do I find this ".repeat(2000);
    let full = repo_search_core::model::QueryTask::CodeRetrieval.prompt(&query);
    let input = counter.query_input(repo_search_core::model::QueryTask::CodeRetrieval, &query);
    assert!(counter.count(&full) > MAX_TOKENS);
    assert!(input.starts_with("task: code retrieval | query: "));
    assert!(counter.count(&input) <= MAX_TOKENS);
    assert!(input.len() < full.len());
    // The cut still holds when the prompt is re-encoded.
    assert!(counter.count(&input) <= MAX_TOKENS);
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

/// The supported files of every `REPO_SEARCH_CORPUS` root as
/// `(root, path, text)`, sorted by path.
fn corpus() -> Vec<(PathBuf, PathBuf, String)> {
    let corpus = std::env::var("REPO_SEARCH_CORPUS").expect(
        "REPO_SEARCH_CORPUS is unset; point it at the corpus roots, \
         separated like PATH",
    );
    let mut work = Vec::new();
    for root in std::env::split_paths(&corpus) {
        assert!(root.is_dir(), "{root:?} is not a corpus checkout");
        let mut paths = Vec::new();
        files(&root, &mut paths);
        paths.sort();
        for path in paths {
            if Language::of_path(&path).is_some()
                && let Ok(src) = std::fs::read_to_string(&path)
            {
                work.push((root.clone(), path, src));
            }
        }
    }
    assert!(!work.is_empty(), "no supported files under {corpus}");
    work
}

/// `PromptCounter` and coreml's `DocumentCounter` answer `count`, `fits`,
/// `fit_end` and `document_tokens` identically over the whole corpus: the
/// daemon's own counter sees the same tokens the server's will.
#[test]
#[ignore = "needs the corpora (REPO_SEARCH_CORPUS) and the model (REPO_SEARCH_MODEL_DIR); the release gate runs it"]
fn prompt_counter_matches_document_counter() {
    let dir = std::env::var("REPO_SEARCH_MODEL_DIR").expect("REPO_SEARCH_MODEL_DIR is unset");
    let prompt = PromptCounter::open(Path::new(&dir).join("tokenizer.json"))
        .expect("the prompt counter loads");
    let document =
        repo_search_coreml::DocumentCounter::open(&dir).expect("the document counter loads");

    for (root, path, src) in corpus() {
        let rel = path.strip_prefix(&root).unwrap().to_str().unwrap();
        let doc = DOCUMENT.format(rel, &src);
        assert_eq!(prompt.count(&doc), document.count(&doc), "count {rel}");
        for max in [128, 256, 512, 1024] {
            assert_eq!(
                prompt.fits(rel, &[&src], max),
                document.fits(rel, &[&src], max),
                "fits {rel} at {max}"
            );
            assert_eq!(
                prompt.fit_end(rel, &[&src], max),
                document.fit_end(rel, &[&src], max),
                "fit_end {rel} at {max}"
            );
        }
        assert_eq!(
            prompt.document_tokens(rel, &src),
            document.document_tokens(rel, &src),
            "document_tokens {rel}"
        );
    }
}

/// No chunk of the corpora exceeds 512 tokens of the embedding tokenizer.
/// `REPO_SEARCH_CORPUS` holds one or more corpus roots, separated like
/// `PATH`.
#[test]
#[ignore = "needs the corpora (REPO_SEARCH_CORPUS) and the model (REPO_SEARCH_MODEL_DIR); the release gate runs it"]
fn corpus_chunks_stay_under_the_cap() {
    let counter = counter();
    let work = corpus();

    // ~100k tokenizer encodes: shard the file list across cores, strided
    // so the dense data files do not pile onto one worker.
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(work.len());
    let shards: Vec<Vec<&(PathBuf, PathBuf, String)>> = (0..workers)
        .map(|w| work.iter().skip(w).step_by(workers).collect())
        .collect();
    let totals: Vec<(u32, u32)> = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .iter()
            .map(|shard| {
                let counter = Arc::clone(&counter);
                scope.spawn(move || {
                    let (mut files, mut chunks_total) = (0u32, 0u32);
                    for (root, path, src) in shard.iter().copied() {
                        let language = Language::of_path(path).unwrap();
                        let rel = path.strip_prefix(root).unwrap().to_str().unwrap();
                        let chunks = chunks_of(language, rel, src, &*counter);
                        files += 1;
                        for chunk in &chunks {
                            let mut text = String::new();
                            chunk.write_text(src, &mut text);
                            assert!(
                                counter.fits(&chunk.title, &[&text], MAX_TOKENS),
                                "{rel} chunk {title:?} is over the cap",
                                title = chunk.title,
                            );
                        }
                        chunks_total += chunks.len() as u32;
                    }
                    (files, chunks_total)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (files_total, chunks_total): (u32, u32) =
        totals.iter().fold((0, 0), |(f, c), t| (f + t.0, c + t.1));
    eprintln!("chunked {chunks_total} chunks of {files_total} files");
    assert!(files_total > 0);
    assert!(chunks_total > 0);
}
