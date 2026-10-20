//! Tests of the documentation rules. Expected values follow the reference
//! prototype `chunk_md.py` except where noted.

use super::DocChunker;
use crate::TokenCounter;
use crate::chunk::{Chunk, ChunkKind, DocFormat, FileChunker, MemberKind};
use crate::counters::{CapOff, CharTokens};

fn chunks(format: DocFormat, path: &str, src: &str) -> Vec<(String, String)> {
    chunks_full(format, path, src)
        .into_iter()
        .map(|(title, _, text)| (title, text))
        .collect()
}

fn chunks_full(
    format: DocFormat,
    path: &str,
    src: &str,
) -> Vec<(String, std::ops::RangeInclusive<u32>, String)> {
    let all = DocChunker::new(format)
        .chunk_file(path, src, &CapOff)
        .unwrap();
    all.iter()
        .map(|chunk| (chunk.title.clone(), chunk.lines.clone(), text(src, chunk)))
        .collect()
}

fn text(src: &str, chunk: &Chunk) -> String {
    let mut out = String::new();
    chunk.write_text(src, &mut out);
    out
}

fn md(src: &str) -> Vec<(String, String)> {
    chunks(DocFormat::Markdown, "docs/layout.md", src)
}

fn titles(chunks: &[(String, String)]) -> Vec<&str> {
    chunks.iter().map(|(title, _)| title.as_str()).collect()
}

#[test]
fn nested_headings_make_the_title_path() {
    let src = "# Layout\nintro\n\n## Stacks\nstack text\n\n### Alignment\nalign text\n\n## Grids\ngrid text\n\n# Next\nnext text\n";
    let all = md(src);
    assert_eq!(
        titles(&all),
        [
            "docs/layout.md › Layout",
            "docs/layout.md › Layout › Stacks",
            "docs/layout.md › Layout › Stacks › Alignment",
            "docs/layout.md › Layout › Grids",
            "docs/layout.md › Next",
        ]
    );
    assert_eq!(all[2].1, "### Alignment\nalign text");
    assert_eq!(all[0].1, "# Layout\nintro");
}

#[test]
fn skipped_levels_still_nest() {
    let all = md("# A\ntext\n\n### C\ntext\n\n## B\ntext\n");
    assert_eq!(
        titles(&all),
        [
            "docs/layout.md › A",
            "docs/layout.md › A › C",
            "docs/layout.md › A › B"
        ]
    );
}

#[test]
fn preamble_before_the_first_heading_is_titled_with_the_path() {
    let all = md("Some intro.\n\n# H\nbody\n");
    assert_eq!(titles(&all), ["docs/layout.md", "docs/layout.md › H"]);
    assert_eq!(all[0].1, "Some intro.");
}

#[test]
fn a_heading_without_body_yields_no_chunk() {
    let all = md("# Part\n\n## Section\nbody\n");
    assert_eq!(titles(&all), ["docs/layout.md › Part › Section"]);
}

#[test]
fn hashes_in_code_fences_are_not_headings() {
    let src =
        "# Real\n```sh\n# comment\n## also comment\n```\n\n~~~\n# tilde\n~~~\n\n## After\nx\n";
    let all = md(src);
    assert_eq!(
        titles(&all),
        ["docs/layout.md › Real", "docs/layout.md › Real › After"]
    );
    assert!(all[0].1.contains("# comment"));
    assert!(all[0].1.contains("# tilde"));
}

#[test]
fn hashes_in_indented_code_and_quotes_are_not_headings() {
    let src = "# Real\n\n    # indented code\n\n> # quoted\n\n- # in a list\n\ntext\n";
    let all = md(src);
    assert_eq!(titles(&all), ["docs/layout.md › Real"]);
}

#[test]
fn front_matter_is_dropped() {
    let src = "---\ntitle: Layout\ntags: [a, b]\n---\n\n# Layout\nbody\n";
    let all = md(src);
    assert_eq!(titles(&all), ["docs/layout.md › Layout"]);
    assert!(!all[0].1.contains("tags"));

    let toml = "+++\ntitle = \"x\"\n+++\n# H\nbody\n";
    assert_eq!(md(toml)[0].1, "# H\nbody");

    let only = md("---\ntitle: x\n---\n");
    assert!(only.is_empty());
}

#[test]
fn front_matter_keeps_line_numbers() {
    let src = "---\ntitle: x\n---\n# H\nbody\n";
    let all = chunks_full(DocFormat::Markdown, "a.md", src);
    assert_eq!(all[0].1, 4..=5);
}

#[test]
fn setext_and_closed_atx_headings() {
    let src = "Title\n=====\ntext\n\nSub\n---\nmore\n\n## Closed ##\nend\n";
    let all = md(src);
    assert_eq!(
        titles(&all),
        [
            "docs/layout.md › Title",
            "docs/layout.md › Title › Sub",
            "docs/layout.md › Title › Closed",
        ]
    );
    assert_eq!(all[0].1, "Title\n=====\ntext");
}

#[test]
fn heading_titles_drop_inline_markup() {
    let all = md("# The `Env` [type](x) *here*\nbody\n");
    assert_eq!(titles(&all), ["docs/layout.md › The Env type here"]);
}

#[test]
fn mdx_is_markdown() {
    let all = chunks(DocFormat::Markdown, "a.mdx", "# Hi\nbody\n");
    assert_eq!(titles(&all), ["a.mdx › Hi"]);
}

#[test]
fn an_over_long_section_splits_on_paragraphs_keeping_the_heading() {
    let para = |c: char| format!("{}\n", c.to_string().repeat(1000));
    let src = format!(
        "# Big\n{}\n{}\n{}\n{}",
        para('a'),
        para('b'),
        para('c'),
        para('d')
    );
    let all = md(&src);
    assert_eq!(titles(&all), ["docs/layout.md › Big"; 2]);
    assert_eq!(
        all[0].1,
        format!("# Big\n{}\n\n{}", "a".repeat(1000), "b".repeat(1000))
    );
    assert_eq!(
        all[1].1,
        format!("# Big\n{}\n\n{}", "c".repeat(1000), "d".repeat(1000))
    );
    for (_, text) in &all {
        assert!(text.len() < 2400 + 10);
    }
}

#[test]
fn a_split_never_cuts_inside_a_fence() {
    let code = format!("```\n{}\n\n{}\n```", "x".repeat(900), "y".repeat(900));
    let src = format!("# Big\nlead\n\n{code}\n\ntail {}\n", "z".repeat(900));
    let all = md(&src);
    assert!(all.len() >= 2);
    let holder = all.iter().find(|(_, text)| text.contains('x')).unwrap();
    assert!(holder.1.contains('y'));
}

#[test]
fn an_over_long_paragraph_splits_on_lines() {
    let line = format!("{}\n", "w".repeat(99));
    let src = format!("# Big\n{}", line.repeat(60));
    let all = md(&src);
    assert!(all.len() >= 3);
    assert!(all.iter().all(|(title, _)| title == "docs/layout.md › Big"));
    for (_, text) in &all {
        assert!(text.starts_with("# Big\n"));
        assert!(text.len() <= 2400 + 10);
    }
}

#[test]
fn split_parts_have_their_own_line_ranges_and_two_members() {
    let para = |c: char| format!("{}\n", c.to_string().repeat(1500));
    let src = format!("intro\n\n# Big\n{}\n{}", para('a'), para('b'));
    let pieces = DocChunker::new(DocFormat::Markdown)
        .chunk_file("a.md", &src, &CapOff)
        .unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[1].kind, ChunkKind::Doc);
    assert_eq!(pieces[1].members.len(), 2);
    assert_eq!(pieces[1].members[0].kind, MemberKind::Doc);
    assert_eq!(pieces[1].members[0].symbol, "Big");
    assert_eq!(pieces[1].lines, 4..=4);
    assert_eq!(pieces[2].lines, 6..=6);
}

#[test]
fn symbol_is_the_heading_path() {
    let pieces = DocChunker::new(DocFormat::Markdown)
        .chunk_file("a.md", "# A\n## B\ntext\n", &CapOff)
        .unwrap();
    assert_eq!(pieces[0].members[0].symbol, "A › B");
}

#[test]
fn non_ascii_text_is_cut_on_boundaries() {
    let src = format!(
        "# Gr\u{fc}\u{df}e\n{}\n\n{}\n",
        "\u{65e5}".repeat(700),
        "\u{672c}".repeat(700)
    );
    let all = md(&src);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].0, "docs/layout.md › Gr\u{fc}\u{df}e");
}

#[test]
fn plain_text_is_one_section_split_on_paragraphs() {
    let small = chunks(DocFormat::Text, "notes.txt", "one\n\ntwo\n");
    assert_eq!(small, [("notes.txt".to_owned(), "one\n\ntwo".to_owned())]);
    let para = format!("{}\n", "p".repeat(1500));
    let big = chunks(DocFormat::Text, "notes.txt", &format!("{para}\n{para}"));
    assert_eq!(big.len(), 2);
    assert!(big.iter().all(|(title, _)| title == "notes.txt"));
    assert!(chunks(DocFormat::Text, "e.txt", "  \n\n").is_empty());
}

#[test]
fn rst_underlined_and_overlined_titles() {
    let src = "=====\nGuide\n=====\n\nintro\n\nSetup\n=====\n\nsetup text\n\nInstall\n-------\n\ninstall text\n\nUsage\n=====\n\nuse text\n";
    let all = chunks(DocFormat::Rst, "doc/guide.rst", src);
    assert_eq!(
        titles(&all),
        [
            "doc/guide.rst › Guide",
            "doc/guide.rst › Guide › Setup",
            "doc/guide.rst › Guide › Setup › Install",
            "doc/guide.rst › Guide › Usage",
        ]
    );
    assert_eq!(all[0].1, "=====\nGuide\n=====\n\nintro");
}

#[test]
fn rst_literal_blocks_hide_their_adornments() {
    let src = "Top\n===\n\nRun this::\n\n    Fake\n    ----\n\n    more\n\nAfter\n-----\n\nok\n\n.. code-block:: rst\n\n   Hidden\n   ======\n";
    let all = chunks(DocFormat::Rst, "a.rst", src);
    assert_eq!(titles(&all), ["a.rst › Top", "a.rst › Top › After"]);
    assert!(all[0].1.contains("Fake"));
    assert!(all[1].1.contains("Hidden"));
}

#[test]
fn rst_transitions_and_tables_are_not_titles() {
    let src = "Top\n===\n\npara\n\n----\n\nmore\n\n=====  =====\na      b\n=====  =====\n";
    let all = chunks(DocFormat::Rst, "a.rst", src);
    assert_eq!(titles(&all), ["a.rst › Top"]);
}

#[test]
fn asciidoc_titles_and_listing_blocks() {
    let src = "= Manual\n\nintro\n\n== Install\n\ntext\n\n----\n== not a title\n----\n\n=== Deep\n\ndeep text\n\n== Usage ==\n\nuse\n";
    let all = chunks(DocFormat::AsciiDoc, "m.adoc", src);
    assert_eq!(
        titles(&all),
        [
            "m.adoc › Manual",
            "m.adoc › Manual › Install",
            "m.adoc › Manual › Install › Deep",
            "m.adoc › Manual › Usage",
        ]
    );
    assert!(all[1].1.contains("== not a title"));
}

#[test]
fn a_mid_document_dash_pair_is_not_front_matter() {
    let all = md("Before\n\n---\nTitle Here\n---\n\nafter\n");
    assert_eq!(
        titles(&all),
        ["docs/layout.md", "docs/layout.md › Title Here"]
    );
    assert_eq!(all[0].1, "Before\n\n---");
    assert_eq!(all[1].1, "Title Here\n---\n\nafter");

    let all = md("# A\n\nintro\n\n---\nInstallation\n---\n\nsteps\n\n## B\n\nx\n");
    assert_eq!(
        titles(&all),
        [
            "docs/layout.md › A",
            "docs/layout.md › A › Installation",
            "docs/layout.md › A › B"
        ]
    );
}

#[test]
fn front_matter_after_a_byte_order_mark_is_dropped() {
    let all = md("\u{feff}---\nk: v\n---\n# A\nbody\n");
    assert_eq!(titles(&all), ["docs/layout.md › A"]);
}

#[test]
fn asciidoc_fences_hide_their_content() {
    let src = "= Doc\n\nintro\n\n== Sec\n\n```bash\n# install deps\nmake\n```\n\nafter\n";
    let all = chunks(DocFormat::AsciiDoc, "a.adoc", src);
    assert_eq!(titles(&all), ["a.adoc › Doc", "a.adoc › Doc › Sec"]);
    assert!(all[1].1.contains("# install deps"));
    assert!(all[1].1.ends_with("after"));
}

#[test]
fn split_parts_fit_the_limit_with_their_heading() {
    let heading = "## H";
    let paragraph = |c: char| format!("{}\n\n", c.to_string().repeat(1200));
    let src = format!(
        "{heading}\n\n{}{}{}",
        paragraph('a'),
        paragraph('b'),
        "c".repeat(1198)
    );
    let all = md(&src);
    assert_eq!(all.len(), 3);
    for (_, text) in &all {
        assert!(text.len() <= 2400, "{}", text.len());
    }
}

#[test]
fn a_bare_heading_before_a_sibling_or_higher_heading_is_kept() {
    let all = md("# A\n\n# B\n\nx\n");
    assert_eq!(titles(&all), ["docs/layout.md › A", "docs/layout.md › B"]);
    assert_eq!(all[0].1, "# A");

    // The setext heading `key: v` is bare and followed by a sibling.
    let mid = md("# A\n\nfirst\n\n---\nkey: v\n---\n\n## B\n\nsecond\n");
    assert_eq!(
        titles(&mid),
        [
            "docs/layout.md › A",
            "docs/layout.md › A › key: v",
            "docs/layout.md › A › B"
        ]
    );
    assert_eq!(mid[1].1, "key: v\n---");

    let mid2 = md("intro text here\n\n---\nk: v\n---\n\n# A\n\nbody\n");
    assert!(mid2.iter().any(|(_, text)| text.contains("k: v")));

    // The last heading of a file, and a bare one before a descendant.
    let last = md("# A\n\n## B\n\nx\n\n## C\n");
    assert_eq!(
        titles(&last),
        ["docs/layout.md › A › B", "docs/layout.md › A › C"]
    );
}

#[test]
fn bare_headings_are_kept_in_rst_and_asciidoc() {
    let rst = chunks(
        DocFormat::Rst,
        "a.rst",
        "One\n===\n\nTwo\n===\n\nbody\n\nSub\n---\n",
    );
    assert_eq!(
        titles(&rst),
        ["a.rst › One", "a.rst › Two", "a.rst › Two › Sub"]
    );
    let adoc = chunks(DocFormat::AsciiDoc, "a.adoc", "= A\n\n= B\n\nx\n");
    assert_eq!(titles(&adoc), ["a.adoc › A", "a.adoc › B"]);
}

#[test]
fn a_byte_order_mark_is_ignored_by_every_format() {
    let src = "\u{feff}# H\n\nb\n";
    let all = chunks_full(DocFormat::Markdown, "a.md", src);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, "a.md › H");
    assert_eq!(all[0].2, "# H\n\nb");
    // Member ranges are offsets into the original text.
    let chunk = &DocChunker::new(DocFormat::Markdown)
        .chunk_file("a.md", src, &CapOff)
        .unwrap()[0];
    assert_eq!(chunk.members[0].bytes.start, 3);
    assert_eq!(chunk.lines, 1..=3);

    let adoc = chunks(
        DocFormat::AsciiDoc,
        "a.adoc",
        "\u{feff}= Doc\n\nintro\n\n== S\n\nx\n",
    );
    assert_eq!(titles(&adoc), ["a.adoc › Doc", "a.adoc › Doc › S"]);

    let rst = chunks(DocFormat::Rst, "a.rst", "\u{feff}Title\n=====\n\nintro\n");
    assert_eq!(titles(&rst), ["a.rst › Title"]);
}

#[test]
fn cjk_sections_are_split_by_bytes() {
    let line = "名".repeat(100);
    let body: String = (0..30).map(|_| format!("{line}\n")).collect();
    for (format, src) in [
        (DocFormat::Markdown, format!("# H\n\n{body}")),
        (DocFormat::Text, body.clone()),
    ] {
        let all = chunks(format, "a", &src);
        assert!(all.len() > 1);
        for (_, text) in &all {
            assert!(text.len() <= 2400, "{}", text.len());
        }
    }
}

/// A single line whose document exceeds the token cap is cut inside the
/// line on char boundaries.
#[test]
fn a_single_line_over_the_cap_splits_on_char_boundaries() {
    let src = format!("# H\n\n{}\n", "名".repeat(2000));
    let all = DocChunker::new(DocFormat::Markdown)
        .chunk_file("a.md", &src, &CharTokens(1))
        .unwrap();
    assert!(all.len() > 2);
    let mut total = 0;
    for chunk in &all {
        let mut text = String::new();
        chunk.write_text(&src, &mut text);
        assert!(
            CharTokens(1).fits(&chunk.title, &[&text], 512),
            "{title} is over the cap",
            title = chunk.title
        );
        total += text.matches('名').count();
    }
    assert_eq!(total, 2000);
}
