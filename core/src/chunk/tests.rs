//! Unit tests for the chunker, one per rule of the reference semantics.
//! Expected values were produced by the reference chunker `chunk_ts.py`.

use super::{Chunk, ChunkKind, Chunker, FileChunker, Rust};
use crate::counters::CapOff;

const PATH: &str = "src/lib.rs";

fn chunks(src: &str) -> (Chunker<Rust>, Vec<Chunk>) {
    let mut chunker = Chunker::new(Rust);
    let out = chunker.chunk_file(PATH, src, &CapOff).unwrap();
    (chunker, out)
}

fn chunks_at(path: &str, src: &str) -> (Chunker<Rust>, Vec<Chunk>) {
    let mut chunker = Chunker::new(Rust);
    let out = chunker.chunk_file(path, src, &CapOff).unwrap();
    (chunker, out)
}

fn text(src: &str, chunk: &Chunk) -> String {
    let mut out = String::new();
    chunk.write_text(src, &mut out);
    out
}

fn kind(chunker: &Chunker<Rust>, chunk: &Chunk) -> String {
    chunker.chunk_kind_name(chunk.kind).into_owned()
}

/// An item carries the doc comments, attributes and comments directly above
/// it; the chunk starts at the first trivia line.
#[test]
fn item_carries_its_trivia() {
    let src = "/// doc\n#[derive(Debug)]\n// a note\nstruct S;\n";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    let c = &all[0];
    assert_eq!(kind(&chunker, c), "struct_item");
    assert_eq!(c.title, "src/lib.rs › struct S");
    assert_eq!(c.lines, 1..=4);
    assert_eq!(
        text(src, c),
        "/// doc\n#[derive(Debug)]\n// a note\nstruct S;"
    );
    assert_eq!(c.members.len(), 1);
    assert_eq!(chunker.member_kind_name(c.members[0].kind), "struct_item");
    assert_eq!(c.members[0].symbol, "S");
}

/// `//!` and `/*!` inner doc comments become their own `docs` chunk.
#[test]
fn module_docs_are_their_own_chunk() {
    // the following items are big enough that the docs units do not pack
    let src = format!(
        "//! line one\n//! line two\n\nfn f() {{ {} }}\n/*! block */\nfn g() {{ {} }}\n",
        "x".repeat(400),
        "x".repeat(400),
    );
    let (chunker, all) = chunks(&src);
    assert_eq!(all.len(), 4);
    assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
    assert_eq!(kind(&chunker, &all[0]), "docs");
    assert_eq!(all[0].title, "src/lib.rs › module docs");
    assert_eq!(all[0].lines, 1..=2);
    assert_eq!(text(&src, &all[0]), "//! line one\n//! line two\n");
    assert_eq!(all[0].members[0].symbol, "");
    assert_eq!(kind(&chunker, &all[1]), "function_item");
    // the file note is the first sentence of the file's inner docs — the
    // first `//!` line, since nodes join with a blank line between them
    assert_eq!(all[1].title, "src/lib.rs — line one › fn f");
    assert_eq!(all[2].kind, ChunkKind::ModuleDocs);
    assert_eq!(text(&src, &all[2]), "/*! block */");
    assert_eq!(all[3].title, "src/lib.rs — line one › fn g");
}

/// An inner doc inside a trivia run is emitted before the item; the rest of
/// the trivia stays attached to the item, whose byte range still spans the
/// removed doc's text.
#[test]
fn inner_doc_inside_trivia_splits_off() {
    let src = format!(
        "#[cfg(unix)]\n//! inner\nfn f() {{ {} }}\n",
        "x".repeat(400)
    );
    let (_, all) = chunks(&src);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
    assert_eq!(all[0].lines, 2..=2);
    assert_eq!(text(&src, &all[0]), "//! inner\n");
    assert_eq!(all[1].title, "src/lib.rs — inner › fn f");
    assert_eq!(all[1].lines, 1..=3);
    assert_eq!(
        text(&src, &all[1]),
        format!("#[cfg(unix)]\n//! inner\nfn f() {{ {} }}", "x".repeat(400))
    );
}

/// A skipped node drops pending trivia: the inner doc above `use` becomes a
/// docs chunk, and nothing attaches to the next item.
#[test]
fn skipped_node_flushes_pending_docs() {
    let src = format!("//! doc\nuse std::fmt;\nfn f() {{ {} }}\n", "x".repeat(400));
    let (_, all) = chunks(&src);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
    assert_eq!(text(&src, &all[0]), "//! doc\n");
    assert_eq!(all[1].lines, 3..=3);
    assert_eq!(
        text(&src, &all[1]),
        format!("fn f() {{ {} }}", "x".repeat(400))
    );
}

/// `use`, `extern crate`, `mod x;`, empty statements and inner attributes are
/// dropped without producing a chunk.
#[test]
fn skipped_kinds_produce_no_chunk() {
    let src = "use std::fmt;\nextern crate alloc;\nmod declarations;\n;\n#![allow(dead_code)]\nfn f() {}\n";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(kind(&chunker, &all[0]), "function_item");
    assert_eq!(all[0].title, "src/lib.rs › fn f");
    assert_eq!(all[0].lines, 6..=6);
    assert_eq!(text(src, &all[0]), "fn f() {}");
}

/// An inline `mod` with a body is an item, not skipped.
#[test]
fn inline_mod_is_an_item() {
    let src = "mod declarations;\nmod inline {\n    fn f() {}\n}\n";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(kind(&chunker, &all[0]), "mod_item");
    assert_eq!(all[0].title, "src/lib.rs › mod inline");
    assert_eq!(all[0].lines, 2..=4);
}

/// Labels: the keyword table, `impl` forms, macro definitions and
/// invocations. All items are small enough to pack, so the labels are the
/// packed chunk's member title segments.
#[test]
fn labels() {
    let src = "\
struct S { a: u8 }
enum E { V }
trait T {}
impl S {}
impl Display for S {}
const C: u8 = 0;
static ST: u8 = 0;
type Ty = u8;
union U { a: u8 }
macro_rules! mac { () => {} }
invoc!{}
";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(
        all[0].title,
        "src/lib.rs › struct S, enum E, trait T, impl S, impl Display for S, const C, static ST, type Ty, union U, macro_rules! mac, macro invoc"
    );
    let member_kinds: Vec<_> = all[0]
        .members
        .iter()
        .map(|m| chunker.member_kind_name(m.kind))
        .collect();
    assert_eq!(
        member_kinds,
        [
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "impl_item",
            "const_item",
            "static_item",
            "type_item",
            "union_item",
            "macro_definition",
            "macro_invocation",
        ]
    );
}

/// `m!();` at item level is an `expression_statement` labelled and symbolled
/// from its macro invocation, but defines no symbol of its own.
#[test]
fn macro_call_statement_label_and_symbol() {
    let src = "m!();\n";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(kind(&chunker, &all[0]), "expression_statement");
    assert_eq!(all[0].title, "src/lib.rs › macro m");
    // a macro call defines no name: the symbol is the enclosing path alone
    assert_eq!(all[0].members[0].symbol, "");
}

/// An oversized impl emits a header then its members; the members pack
/// within the impl's context, the header packs with its top-level
/// neighbours, and member symbols carry the impl's type name.
#[test]
fn oversized_impl_splits_into_members() {
    let src = include_str!("../../tests/fixtures/oversized_impl.rs");
    let (chunker, all) = chunks_at("tests/fixtures/oversized_impl.rs", src);
    assert_eq!(all.len(), 3);

    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(
        all[0].title,
        "tests/fixtures/oversized_impl.rs › struct Pad, impl Pad"
    );
    assert_eq!(all[0].lines, 1..=3);
    let kinds: Vec<_> = all[0]
        .members
        .iter()
        .map(|m| chunker.member_kind_name(m.kind))
        .collect();
    assert_eq!(kinds, ["struct_item", "impl_item"]);
    let symbols: Vec<&str> = all[0].members.iter().map(|m| m.symbol.as_str()).collect();
    assert_eq!(symbols, ["Pad", "Pad"]);

    assert_eq!(all[1].kind, ChunkKind::Packed);
    assert_eq!(
        all[1].title,
        "tests/fixtures/oversized_impl.rs › impl Pad › fn m00, fn m01, fn m02, fn m03, fn m04, fn m05, fn m06, fn m07, fn m08, fn m09, fn m10, fn m11, fn m12, fn m13, fn m14"
    );
    assert_eq!(all[1].lines, 4..=63);
    assert_eq!(all[1].members.len(), 15);
    assert_eq!(
        chunker.member_kind_name(all[1].members[0].kind),
        "function_item"
    );
    assert_eq!(all[1].members[0].symbol, "Pad::m00");
    assert_eq!(all[1].members[14].symbol, "Pad::m14");

    assert_eq!(all[2].lines, 64..=123);
    assert_eq!(all[2].members.len(), 15);
    assert_eq!(all[2].members[0].symbol, "Pad::m15");

    // a packed chunk's text is its members joined with "\n"
    let joined = all[1]
        .members
        .iter()
        .map(|m| &src[m.bytes.clone()])
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(text(src, &all[1]), joined);
    assert!(text(src, &all[1]).starts_with("fn m00(&self) -> usize {"));
}

/// An oversized single item is split on line boundaries with `(part i/n)`
/// titles, and a multi-byte line does not split a UTF-8 sequence.
#[test]
fn oversized_item_splits_on_lines() {
    let src = include_str!("../../tests/fixtures/oversized_fn.rs");
    let (chunker, all) = chunks_at("tests/fixtures/oversized_fn.rs", src);
    assert_eq!(all.len(), 2);

    assert_eq!(kind(&chunker, &all[0]), "function_item");
    assert_eq!(
        all[0].title,
        "tests/fixtures/oversized_fn.rs › fn big (part 1/2)"
    );
    assert_eq!(all[0].lines, 1..=25);
    assert_eq!(all[0].members.len(), 1);
    assert_eq!(all[0].members[0].symbol, "big");
    let part1 = text(src, &all[0]);
    assert_eq!(part1.len(), 2362);
    assert!(part1.ends_with("xxx\n"));

    assert_eq!(
        all[1].title,
        "tests/fixtures/oversized_fn.rs › fn big (part 2/2)"
    );
    assert_eq!(all[1].lines, 26..=35);
    // a part after the first opens with the item's signature + ` …\n`;
    // its first member is the signature's own byte range
    assert_eq!(all[1].members.len(), 2);
    assert_eq!(
        chunker.member_kind_name(all[1].members[0].kind),
        "signature"
    );
    let part2 = text(src, &all[1]);
    assert_eq!(part2.len(), 867 + "fn big() -> usize …\n".len());
    assert!(part2.starts_with("fn big() -> usize …\n"));
    // the multi-byte line (50 code points of é, 100 bytes) is whole in part 2
    assert!(part2.contains("ééé"));
    assert!(part2.ends_with('}'));

    // the parts partition the item's text exactly apart from the signature
    // prefix; the file's final newline lies outside the item
    assert_eq!(
        format!("{part1}{}", &part2["fn big() -> usize …\n".len()..]),
        src.strip_suffix('\n').unwrap()
    );
}

/// An oversized trait and an oversized inline mod split the same way as an
/// impl; module docs inside the mod pack with its members; a header that
/// starts a new group is emitted alone.
#[test]
fn oversized_trait_and_mod_split() {
    let src = include_str!("../../tests/fixtures/split_scopes.rs");
    let (chunker, all) = chunks_at("tests/fixtures/split_scopes.rs", src);
    assert_eq!(all.len(), 7);

    // the oversized preamble fn packs with the trait's header
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(
        all[0].title,
        "tests/fixtures/split_scopes.rs › fn preamble, trait Wide"
    );
    assert_eq!(all[0].lines, 1..=9);
    assert_eq!(
        chunker.member_kind_name(all[0].members[1].kind),
        "trait_item"
    );

    // trait members pack within the trait's context
    assert_eq!(all[1].kind, ChunkKind::Packed);
    assert_eq!(all[1].lines, 10..=31);
    assert_eq!(all[1].members.len(), 11);
    assert_eq!(
        chunker.member_kind_name(all[1].members[0].kind),
        "function_signature_item"
    );
    assert_eq!(all[1].members[0].symbol, "Wide::member_00");
    assert!(
        all[1]
            .title
            .starts_with("tests/fixtures/split_scopes.rs › trait Wide › fn member_00")
    );
    assert_eq!(all[3].lines, 54..=69);
    assert_eq!(all[3].members[7].symbol, "Wide::member_29");

    // the mod's header is the first unit of its own group and stays alone
    assert!(matches!(all[4].kind, ChunkKind::Header(_)));
    assert_eq!(kind(&chunker, &all[4]), "mod_item_header");
    assert_eq!(all[4].title, "tests/fixtures/split_scopes.rs › mod deep");
    assert_eq!(all[4].lines, 72..=72);
    assert_eq!(all[4].members[0].symbol, "deep");
    assert_eq!(text(src, &all[4]), "mod deep ");

    // the module docs inside the mod are a `docs` member of the packed group
    assert_eq!(all[5].kind, ChunkKind::Packed);
    assert_eq!(all[5].lines, 73..=134);
    assert_eq!(all[5].members.len(), 16);
    assert_eq!(
        chunker.member_kind_name(all[5].members[0].kind),
        "line_comment"
    );
    assert_eq!(all[5].members[0].symbol, "deep");
    assert_eq!(
        &src[all[5].members[0].bytes.clone()],
        "//! docs for the inline module\n"
    );
    assert_eq!(all[5].members[1].symbol, "deep::inner_00");
    assert!(
        all[5]
            .title
            .starts_with("tests/fixtures/split_scopes.rs › mod deep › module docs, fn inner_00")
    );

    assert_eq!(all[6].lines, 135..=194);
    assert_eq!(all[6].members.len(), 15);
}

/// Items under 400 bytes pack into one chunk; the text is the members' texts
/// joined with `\n` — bytes skipped between members (a dropped `use`) do not
/// appear.
#[test]
fn packing_skips_dropped_ranges() {
    let src = "const A: u8 = 0;\nuse std::fmt;\nconst B: u8 = 1;\n";
    let (_, all) = chunks(src);
    assert_eq!(all.len(), 1);
    let c = &all[0];
    assert_eq!(c.kind, ChunkKind::Packed);
    assert_eq!(c.title, "src/lib.rs › const A, const B");
    assert_eq!(c.lines, 1..=3);
    assert_eq!(c.members.len(), 2);
    assert_eq!(text(src, c), "const A: u8 = 0;\nconst B: u8 = 1;");
}

/// Small `docs` units pack like any other small unit.
#[test]
fn module_docs_pack_like_other_units() {
    let src = "//! docs\nfn f() {}\n";
    let (chunker, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(all[0].title, "src/lib.rs › module docs, fn f");
    assert_eq!(
        chunker.member_kind_name(all[0].members[0].kind),
        "line_comment"
    );
}

/// A packed group stops growing at 1200 bytes; the next small item starts a
/// new group. A large item ends a group.
#[test]
fn pack_size_boundaries() {
    // each item: a doc comment (~295 bytes) plus `fn` — ~305 bytes, so three
    // pack (915) and the fourth starts a new group
    let mut src = String::new();
    for i in 0..4 {
        src.push_str(&format!("/// {}\nfn k{i}() {{}}\n", "x".repeat(290)));
    }
    let (chunker, all) = chunks(&src);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(all[0].members.len(), 3);
    assert_eq!(all[0].title, "src/lib.rs › fn k0, fn k1, fn k2");
    assert_eq!(all[1].title, "src/lib.rs › fn k3");
    assert_eq!(kind(&chunker, &all[1]), "function_item");

    // a unit of 400 bytes or more is emitted alone and ends the group
    let mut src = String::from("const A: u8 = 0;\nconst B: u8 = 1;\n");
    src.push_str(&format!("fn big() {{ {} }}\n", "x".repeat(400)));
    src.push_str("const C: u8 = 2;\n");
    let (chunker, all) = chunks(&src);
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(all[0].members.len(), 2);
    assert_eq!(kind(&chunker, &all[1]), "function_item");
    assert_eq!(kind(&chunker, &all[2]), "const_item");
    assert_eq!(all[2].title, "src/lib.rs › const C");
}

/// Symbol paths: the `::`-joined container path plus the item's own name;
/// generics are stripped from an impl's type.
#[test]
fn symbol_paths() {
    let src = "\
fn f() {}
impl Env {
    fn insert(&self) {}
}
impl<T> Cache<T> for Store<T> {
    fn get(&self) {}
}
";
    let (_, all) = chunks(src);
    assert_eq!(all.len(), 1);
    assert_eq!(
        all[0].title,
        "src/lib.rs › fn f, impl Env, impl Cache<T> for Store<T>"
    );
    let symbols: Vec<&str> = all[0].members.iter().map(|m| m.symbol.as_str()).collect();
    assert_eq!(symbols, ["f", "Env", "Store"]);
}

/// The parser is reused across files on one chunker instance.
#[test]
fn parser_reuse() {
    let mut chunker = Chunker::new(Rust);
    let a = chunker.chunk_file("a.rs", "fn a() {}\n", &CapOff).unwrap();
    let b = chunker.chunk_file("b.rs", "fn b() {}\n", &CapOff).unwrap();
    assert_eq!(a[0].title, "a.rs › fn a");
    assert_eq!(b[0].title, "b.rs › fn b");
}

/// The size limit counts bytes: 1800 two-byte characters plus padding is
/// 3634 bytes, so the item splits on lines — under the 2400-byte budget,
/// never mind that it is under 2400 code points.
#[test]
fn line_split_counts_bytes_not_code_points() {
    let e = "é".repeat(600);
    let src = format!("fn m() {{\n    // {e}\n    // {e}\n    // {e}\n}}\n");
    let (chunker, all) = chunks(&src);
    assert_eq!(all.len(), 3);
    assert_eq!(kind(&chunker, &all[0]), "function_item");
    assert_eq!(all[0].title, "src/lib.rs › fn m (part 1/3)");
    assert_eq!(all[0].lines, 1..=2);
    assert_eq!(all[1].lines, 3..=3);
    assert_eq!(all[2].lines, 4..=5);
    // later parts open with the item's signature
    assert!(text(&src, &all[2]).starts_with("fn m() …\n"));
}

/// Lines split on every Python `splitlines` boundary; `\r\n` is one
/// boundary and the last line keeps its `\n`.
#[test]
fn python_lines_boundaries() {
    let src = "a\r\nb\rc\x0bd\x0ce\x1cf\x1dg\x1eh\u{85}i\u{2028}j\u{2029}k\n";
    let lines: Vec<&str> = super::units::python_lines(src)
        .into_iter()
        .map(|range| &src[range])
        .collect();
    assert_eq!(
        lines,
        [
            "a\r\n",
            "b\r",
            "c\x0b",
            "d\x0c",
            "e\x1c",
            "f\x1d",
            "g\x1e",
            "h\u{85}",
            "i\u{2028}",
            "j\u{2029}",
            "k\n"
        ]
    );
}

/// A CRLF file splits on `\r\n` boundaries and counts line ends by `\n`.
#[test]
fn crlf_file_line_split() {
    let mut src = String::from("fn c() {\r\n");
    for i in 0..60 {
        src.push_str(&format!(
            "    let v{i} = {i}; // padding padding padding padding\r\n"
        ));
    }
    src.push_str("}\r\n");
    let (_, all) = chunks(&src);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].title, "src/lib.rs › fn c (part 1/2)");
    assert_eq!(all[0].lines, 1..=45);
    assert_eq!(text(&src, &all[0]).len(), 2366);
    assert_eq!(all[1].title, "src/lib.rs › fn c (part 2/2)");
    assert_eq!(all[1].lines, 46..=62);
    let part2 = text(&src, &all[1]);
    assert!(part2.starts_with("fn c() …\n"));
    assert_eq!(part2.len(), 865 + "fn c() …\n".len());
}

/// An impl's symbol is its type's name with references, pointers and
/// generics stripped.
#[test]
fn impl_symbol_strips_reference_types() {
    let members: String = (0..80)
        .map(|i| format!("    fn r{i}(&self) -> usize {{ {i} + 1000000 }}\n"))
        .collect();
    let src = format!("impl<'a, T> Trait for &'a Foo<T> {{\n{members}}}\n");
    let (_, all) = chunks(&src);
    let symbols: Vec<&str> = all
        .iter()
        .flat_map(|c| c.members.iter().map(|m| m.symbol.as_str()))
        .collect();
    assert_eq!(symbols[0], "Foo");
    assert_eq!(symbols[1], "Foo::r0");
}

/// `first_sentence` strips comment markers, stops at a blank line, a
/// heading or a code fence, cuts after the first `.`/`!`/`?` followed by
/// whitespace or end, truncates at 160 chars on a word boundary and turns
/// `›` into `>`. Each `//!` node ends in a newline, so nodes join into a
/// text with blank lines between them — the note is the first line of a
/// `//!` run, matching the reference.
#[test]
fn first_sentence_rule() {
    use super::units::first_sentence;

    // sentence ends at the first terminator followed by whitespace
    assert_eq!(
        first_sentence(
            "//! One. Two.

//! Three.
"
        )
        .as_deref(),
        Some("One.")
    );
    // a terminator at end-of-text counts too
    assert_eq!(first_sentence("//! Last!").as_deref(), Some("Last!"));
    // a `.` inside a word does not cut
    assert_eq!(
        first_sentence(
            "//! a.b c.d
"
        )
        .as_deref(),
        Some("a.b c.d")
    );
    // a `//!` node's trailing newline makes a blank line between nodes:
    // the note is only the first doc line
    assert_eq!(
        first_sentence(
            "//! First.

//! Second.
"
        )
        .as_deref(),
        Some("First.")
    );
    // blank doc lines inside one node stop the note the same way
    assert_eq!(
        first_sentence(
            "//!
//! First.
//!
//! Second.
"
        )
        .as_deref(),
        Some("First.")
    );
    // a heading or a code fence stops the note
    assert_eq!(
        first_sentence("//! Words.\n//! # Section\n//! More.").as_deref(),
        Some("Words.")
    );
    assert_eq!(
        first_sentence("//! Words.\n//! ```rust\n//! More.").as_deref(),
        Some("Words.")
    );
    // a file whose docs open with a heading has no note
    assert_eq!(
        first_sentence(
            "//! # Title
//! Body.
"
        ),
        None
    );
    // block-comment markers and a trailing `*/` are stripped; a `/*!` node
    // has no inter-line blanks, so its first sentence spans lines
    assert_eq!(
        first_sentence(
            "/*! Block.
 * more
 */"
        )
        .as_deref(),
        Some("Block.")
    );
    // docs with no sentence terminator keep everything collected
    assert_eq!(
        first_sentence(
            "//! one
//! two"
        )
        .as_deref(),
        Some("one two")
    );
    // 160 chars, cut on a word boundary, ` …` appended
    let long = format!("//! {}", "word ".repeat(60));
    let note = first_sentence(&long).unwrap();
    let head = note.strip_suffix(" …").unwrap();
    assert!(head.chars().count() <= 160);
    assert!(head.chars().count() > 140);
    // `›` is not allowed in a title segment
    assert_eq!(
        first_sentence(
            "//! a › b.
"
        )
        .as_deref(),
        Some("a > b.")
    );
    // no docs, no note
    assert_eq!(first_sentence(""), None);
    assert_eq!(
        first_sentence(
            "//!
//!"
        ),
        None
    );
}

/// The first sentence of a file's module docs goes into every other chunk's
/// title after the path — `path — <sentence> › …`; the docs chunk itself
/// keeps the plain `path`.
#[test]
fn module_docs_sentence_joins_titles() {
    let src = format!(
        "//! First sentence. Rest.\n\nfn f() {{ {} }}\n\nfn g() {{ {} }}\n",
        "x".repeat(400),
        "y".repeat(400)
    );
    let (chunker, all) = chunks(&src);
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].title, "src/lib.rs › module docs");
    assert_eq!(all[1].title, "src/lib.rs — First sentence. › fn f");
    assert_eq!(all[2].title, "src/lib.rs — First sentence. › fn g");
    assert_eq!(kind(&chunker, &all[0]), "docs");
}

/// A line-split part after the first opens with the item's signature —
/// whitespace collapsed — plus ` …\n`; the part's own line range does not
/// count the prefix. The signature is a member of its own, so the stored
/// byte ranges rebuild the same text.
#[test]
fn split_parts_carry_the_signature() {
    let src = format!(
        "fn f(  a: u32,\n       b: u32\n) -> u32 {{\n{}\n}}\n",
        "x".repeat(5000)
    );
    let (chunker, all) = chunks(&src);
    assert!(all.len() > 2);
    // the collapsed signature
    let signature = "fn f( a: u32, b: u32 ) -> u32 …\n";
    for (index, chunk) in all.iter().enumerate().skip(1) {
        assert_eq!(chunker.chunk_kind_name(chunk.kind), "function_item");
        assert!(text(&src, chunk).starts_with(signature));
        assert_eq!(chunk.members.len(), 2);
        assert_eq!(chunker.member_kind_name(chunk.members[0].kind), "signature");
        assert_eq!(
            &src[chunk.members[0].bytes.clone()],
            "fn f(  a: u32,\n       b: u32\n) -> u32 "
        );
        let _ = index;
    }
    // the first part has no signature member
    assert_eq!(all[0].members.len(), 1);
}
