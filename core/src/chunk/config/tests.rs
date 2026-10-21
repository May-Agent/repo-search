//! Tests of the configuration rules.

use super::ConfigChunker;
use crate::chunk::{Chunk, ChunkKind, ConfigFormat, Error, FileChunker, MemberKind};
use crate::counters::CapOff;

fn run(format: ConfigFormat, src: &str) -> Vec<Chunk> {
    ConfigChunker::new(format)
        .chunk_file("conf", src, &CapOff)
        .unwrap()
}

fn text(src: &str, chunk: &Chunk) -> String {
    let mut out = String::new();
    chunk.write_text(src, &mut out);
    out
}

fn titles(chunks: &[Chunk]) -> Vec<&str> {
    chunks.iter().map(|chunk| chunk.title.as_str()).collect()
}

/// A key of about `size` bytes, so that two of them never pack together.
fn big(name: &str, size: usize) -> String {
    format!("{name} = \"{}\"\n", "v".repeat(size))
}

#[test]
fn toml_tables_and_root_keys() {
    let src = format!(
        "# the name\n{}{}\n# deps\n[dependencies]\n{}{}\n[package]\n{}",
        big("name", 500),
        big("edition", 500),
        big("a", 500),
        big("b", 500),
        big("c", 500),
    );
    let all = run(ConfigFormat::Toml, &src);
    assert_eq!(
        titles(&all),
        [
            "conf › name",
            "conf › edition",
            "conf › dependencies",
            "conf › package"
        ]
    );
    assert!(text(&src, &all[0]).starts_with("# the name\nname = "));
    assert_eq!(all[0].lines, 1..=2);
    assert!(text(&src, &all[2]).starts_with("# deps\n[dependencies]"));
    assert_eq!(all[2].members[0].symbol, "dependencies");
    assert_eq!(all[0].members[0].kind, MemberKind::Config);
    assert_eq!(all[0].kind, ChunkKind::Config);
}

#[test]
fn toml_small_neighbours_are_packed() {
    let src = "name = \"x\"\nversion = \"1\"\n\n[a]\nk = 1\n\n[b]\nk = 2\n";
    let all = run(ConfigFormat::Toml, src);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].kind, ChunkKind::Packed);
    assert_eq!(all[0].title, "conf › name, version, a, b");
    let symbols: Vec<_> = all[0].members.iter().map(|m| m.symbol.as_str()).collect();
    assert_eq!(symbols, ["name", "version", "a", "b"]);
}

#[test]
fn toml_arrays_of_tables_are_indexed() {
    let src = format!(
        "[[bin]]\n{}[[bin]]\n{}[bin.extra]\n{}",
        big("a", 450),
        big("b", 450),
        big("c", 450)
    );
    let all = run(ConfigFormat::Toml, &src);
    assert_eq!(
        titles(&all),
        ["conf › bin[0]", "conf › bin[1]", "conf › bin[1].extra"]
    );
}

#[test]
fn toml_dotted_keys_are_one_entry() {
    let src = format!("a.b = \"{}\"\na.c = 2\n", "x".repeat(500));
    let all = run(ConfigFormat::Toml, &src);
    assert_eq!(titles(&all), ["conf › a"]);
    assert!(text(&src, &all[0]).contains("a.c = 2"));
}

#[test]
fn toml_large_table_splits_into_keys_with_context() {
    let mut src = String::from("[deps]\n");
    for i in 0..8 {
        src.push_str(&big(&format!("k{i}"), 450));
    }
    let all = run(ConfigFormat::Toml, &src);
    assert_eq!(all[0].title, "conf › deps");
    assert_eq!(text(&src, &all[0]), "[deps]");
    assert_eq!(all.len(), 9);
    assert_eq!(all[1].title, "conf › deps › k0");
    assert_eq!(all[1].members[0].symbol, "deps.k0");
}

#[test]
fn toml_large_value_splits_on_lines() {
    let value: String = (0..60)
        .map(|i| format!("  \"item-{i:04}-{}\",\n", "x".repeat(40)))
        .collect();
    let src = format!("list = [\n{value}]\n");
    let all = run(ConfigFormat::Toml, &src);
    assert!(all.len() >= 2);
    assert!(all[0].title.starts_with("conf › list (part 1/"));
    let total: usize = all.iter().map(|chunk| text(&src, chunk).len()).sum();
    assert!(total >= value.len());
}

#[test]
fn toml_syntax_error_is_an_error() {
    let result = ConfigChunker::new(ConfigFormat::Toml).chunk_file("c.toml", "a = \n", &CapOff);
    assert!(matches!(result, Err(Error::Toml(_))));
}

#[test]
fn yaml_top_level_keys() {
    let src = format!(
        "# one\none: {}\ntwo: 2\nlist:\n  - a\n  - b\nthree: {}\n",
        "x".repeat(500),
        "y".repeat(500),
    );
    let all = run(ConfigFormat::Yaml, &src);
    assert_eq!(
        titles(&all),
        ["conf › one", "conf › two, list", "conf › three"]
    );
    assert!(text(&src, &all[0]).starts_with("# one\none: "));
    assert_eq!(all[1].members[1].symbol, "list");
    assert!(text(&src, &all[1]).ends_with("  - b"));
}

#[test]
fn yaml_large_mapping_splits_into_nested_keys() {
    let mut src = String::from("jobs:\n");
    for i in 0..6 {
        src.push_str(&format!("  job{i}:\n    run: \"{}\"\n", "x".repeat(450)));
    }
    let all = run(ConfigFormat::Yaml, &src);
    assert_eq!(all[0].title, "conf › jobs");
    assert_eq!(text(&src, &all[0]), "jobs:");
    assert_eq!(all[1].title, "conf › jobs › job0");
    assert_eq!(all[1].members[0].symbol, "jobs.job0");
    assert!(text(&src, &all[1]).starts_with("  job0:\n    run:"));
    assert_eq!(all.len(), 7);
}

#[test]
fn yaml_root_sequence_and_documents() {
    let seq = "- a: 1\n- b: 2\n";
    let all = run(ConfigFormat::Yaml, seq);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].title, "conf › [0], [1]");
    assert_eq!(text(seq, &all[0]), "- a: 1\n- b: 2");

    let docs = "kind: A\n---\nkind: B\n";
    let all = run(ConfigFormat::Yaml, docs);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].title, "conf › document 1, document 2");
}

#[test]
fn yaml_non_ascii_offsets_are_bytes() {
    let long = "\u{3046}".repeat(400);
    let src = format!("\u{3042}: \"\u{3044}\"\nk: {long}\nlast: z\n");
    let all = run(ConfigFormat::Yaml, &src);
    assert_eq!(titles(&all), ["conf › \u{3042}", "conf › k", "conf › last"]);
    assert_eq!(text(&src, &all[0]), "\u{3042}: \"\u{3044}\"");
    assert_eq!(text(&src, &all[1]), format!("k: {long}"));
    assert_eq!(text(&src, &all[2]), "last: z");
}

#[test]
fn yaml_syntax_error_is_an_error() {
    let result =
        ConfigChunker::new(ConfigFormat::Yaml).chunk_file("c.yaml", "a: [1, 2\nb: }\n", &CapOff);
    assert!(matches!(result, Err(Error::Yaml(_))));
}

#[test]
fn json_top_level_keys_with_comments() {
    let src = format!(
        "{{\n  // the name\n  \"name\": \"{}\",\n  \"scripts\": {{\"a\": 1}},\n  \"deps\": \"{}\"\n}}\n",
        "n".repeat(500),
        "d".repeat(500),
    );
    let all = run(ConfigFormat::Json, &src);
    assert_eq!(
        titles(&all),
        ["conf › name", "conf › scripts", "conf › deps"]
    );
    assert!(text(&src, &all[0]).starts_with("  // the name\n  \"name\""));
    assert_eq!(all[0].lines, 2..=3);
}

#[test]
fn json_large_object_and_array_split() {
    let mut src = String::from("{\"items\": [\n");
    for i in 0..6 {
        src.push_str(&format!(
            "  {{\"id\": {i}, \"text\": \"{}\"}},\n",
            "t".repeat(450)
        ));
    }
    src.push_str("]}\n");
    let all = run(ConfigFormat::Json, &src);
    assert_eq!(all[0].title, "conf › items");
    assert_eq!(all[1].title, "conf › items › [0]");
    assert_eq!(all[1].members[0].symbol, "items.0");
    assert_eq!(all.len(), 7);
}

#[test]
fn json_root_array_and_empty_files() {
    let all = run(ConfigFormat::Json, "[{\"a\": 1}, 2]");
    assert_eq!(titles(&all), ["conf › [0], [1]"]);
    assert!(run(ConfigFormat::Json, "").is_empty());
    assert!(run(ConfigFormat::Yaml, "# only a comment\n").is_empty());
    assert!(run(ConfigFormat::Toml, "").is_empty());
}

#[test]
fn json_syntax_error_is_an_error() {
    let result = ConfigChunker::new(ConfigFormat::Json).chunk_file("c.json", "{\"a\": }", &CapOff);
    assert!(matches!(result, Err(Error::Json(_))));
}

#[test]
fn toml_out_of_order_dotted_keys_never_duplicate_text() {
    let src = "a.b = 1\nc = 2\na.d = 3\n";
    let all = run(ConfigFormat::Toml, src);
    let joined: String = all
        .iter()
        .map(|chunk| text(src, chunk))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(joined.matches("c = 2").count(), 1);
    assert_eq!(joined.matches("a.b = 1").count(), 1);
    assert_eq!(joined.matches("a.d = 3").count(), 1);
}

#[test]
fn yaml_empty_documents_have_no_entry() {
    let src = "---\n---\na: 1\n";
    let all = run(ConfigFormat::Yaml, src);
    assert_eq!(titles(&all), ["conf › a"]);
    assert_eq!(text(src, &all[0]), "a: 1");

    let src = "a: 1\n---\n# c\n---\nb: 2\n";
    let all = run(ConfigFormat::Yaml, src);
    assert_eq!(titles(&all), ["conf › document 1, document 3"]);
    let joined = text(src, &all[0]);
    assert!(
        !joined.contains("---") && !joined.contains("# c"),
        "{joined}"
    );
}

#[test]
fn yaml_complex_keys_are_labelled_with_their_source() {
    let src = "? [a, b]\n: 1\nplain: 2\n";
    let all = run(ConfigFormat::Yaml, src);
    assert_eq!(titles(&all), ["conf › [a, b], plain"]);
}

#[test]
fn a_byte_order_mark_is_ignored_by_every_format() {
    let src = "\u{feff}a: 1\nb: 名\n";
    let all = run(ConfigFormat::Yaml, src);
    assert_eq!(titles(&all), ["conf › a, b"]);
    assert_eq!(text(src, &all[0]), "a: 1\nb: 名");
    assert_eq!(all[0].members[0].bytes.start, 3);

    let toml = "\u{feff}[t]\nk = 1\n";
    let all = run(ConfigFormat::Toml, toml);
    assert_eq!(titles(&all), ["conf › t"]);
    assert_eq!(text(toml, &all[0]), "[t]\nk = 1");

    let json = "\u{feff}{\"a\": 1}";
    let all = run(ConfigFormat::Json, json);
    assert_eq!(titles(&all), ["conf › a"]);
    assert_eq!(text(json, &all[0]), "\"a\": 1");
}

#[test]
fn yaml_block_collection_keys_get_single_line_labels() {
    let src = "? - x\n  - y\n: 2\nz: 1\n";
    let all = run(ConfigFormat::Yaml, src);
    assert_eq!(titles(&all), ["conf › - x - y, z"]);
}

#[test]
fn yaml_entries_include_the_key_marker_and_the_colon() {
    let src = "? [a, b]\n: 1\nplain:\nlast: z\n? k\n";
    let all = run(ConfigFormat::Yaml, src);
    let members: Vec<&str> = all[0]
        .members
        .iter()
        .map(|member| &src[member.bytes.clone()])
        .collect();
    assert_eq!(members, ["? [a, b]\n: 1", "plain:", "last: z", "? k"]);
}
