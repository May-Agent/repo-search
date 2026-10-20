//! The 512-token cap every chunk's embedded document is held to.
#![cfg(feature = "lang-javascript")]

mod counters;

use counters::CharTokens;
use repo_search_core::TokenCounter;
use repo_search_core::chunk::{Chunker, FileChunker, JavaScript};

const MAX_TOKENS: usize = 512;

/// A generated minified-JavaScript line of 256 KB becomes many chunks, each
/// under the cap — not one multi-megabyte chunk.
#[test]
fn a_minified_line_becomes_many_capped_chunks() {
    let src = format!("var x = \"{}\";\n", "a".repeat(256 << 10));
    let counter = CharTokens(1);
    let chunks = Chunker::new(JavaScript)
        .chunk_file("bundle.min.js", &src, &counter)
        .unwrap();
    assert!(chunks.len() > 400, "{} chunks", chunks.len());
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
