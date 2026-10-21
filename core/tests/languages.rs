//! Per-language chunker tests, one per rule adapted to each grammar.
//! Sources are generated inline; oversized containers get enough members
//! to cross the 2400-byte split limit.
#![allow(dead_code, unused_imports)]

mod counters;

use counters::CapOff;
use repo_search_core::chunk::{Chunk, ChunkKind, Chunker, FileChunker, Grammar};

const MAX: usize = 2400;

fn chunks<G: Grammar>(grammar: G, src: &str) -> (Chunker<G>, Vec<Chunk>) {
    let mut chunker = Chunker::new(grammar);
    let out = chunker.chunk_file(PATH, src, &CapOff).unwrap();
    (chunker, out)
}

fn text(src: &str, chunk: &Chunk) -> String {
    let mut out = String::new();
    chunk.write_text(src, &mut out);
    out
}

/// A chunk's source lines: its text minus a line-split part's signature
/// prefix — the last member's slice.
fn source_lines<'a>(src: &'a str, chunk: &Chunk) -> &'a str {
    &src[chunk
        .members
        .last()
        .expect("a chunk has a member")
        .bytes
        .clone()]
}

fn kind<G: Grammar>(chunker: &Chunker<G>, chunk: &Chunk) -> String {
    chunker.chunk_kind_name(chunk.kind).into_owned()
}

fn member_kinds<G: Grammar>(chunker: &Chunker<G>, chunk: &Chunk) -> Vec<String> {
    chunk
        .members
        .iter()
        .map(|m| chunker.member_kind_name(m.kind).into_owned())
        .collect()
}

fn member_symbols(chunk: &Chunk) -> Vec<&str> {
    chunk.members.iter().map(|m| m.symbol.as_str()).collect()
}

const PATH: &str = "src/main.x";

/// `count` numbered members of ~`size` bytes each, e.g. `func m07() { // pad… }`.
fn members(count: usize, f: impl Fn(usize) -> String) -> String {
    (0..count).map(&f).collect::<Vec<_>>().join("")
}

fn padded(pattern: &str, size: usize) -> String {
    let mut line = pattern.to_string();
    while line.len() < size {
        line.push('x');
    }
    line
}

/// Extension detection: `.h` is C++ when that language is built and C
/// otherwise; `.C`/`.H` name C++ case-sensitively; `.mts`/`.cts` are
/// TypeScript and `.pyi` is Python.
#[test]
#[allow(unused_variables)]
fn extensions_map_to_languages() {
    use repo_search_core::chunk::Language;
    use std::path::Path;
    let of = |path: &str| Language::of_path(Path::new(path));
    #[cfg(feature = "lang-cpp")]
    {
        assert_eq!(of("x.h"), Some(Language::Cpp));
        assert_eq!(of("x.C"), Some(Language::Cpp));
        assert_eq!(of("x.H"), Some(Language::Cpp));
    }
    #[cfg(all(feature = "lang-c", not(feature = "lang-cpp")))]
    assert_eq!(of("x.h"), Some(Language::C));
    #[cfg(feature = "lang-typescript")]
    {
        assert_eq!(of("x.mts"), Some(Language::TypeScript));
        assert_eq!(of("x.cts"), Some(Language::TypeScript));
    }
    #[cfg(feature = "lang-python")]
    assert_eq!(of("x.pyi"), Some(Language::Python));
}

#[cfg(feature = "lang-swift")]
mod swift {
    use super::*;
    use repo_search_core::chunk::Swift;

    fn chunks_of(src: &str) -> (Chunker<Swift>, Vec<Chunk>) {
        chunks(Swift, src)
    }

    /// `///` docs and `@attributes` ride with the item; `import` is dropped.
    #[test]
    fn item_carries_trivia() {
        let src = "import Foundation\n\n/// A doc.\n@MainActor\nstruct S {\n    var x: Int\n}\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        let c = &all[0];
        assert_eq!(kind(&chunker, c), "class_declaration");
        assert_eq!(c.title, "src/main.x › struct S");
        assert_eq!(c.lines, 3..=7);
        assert_eq!(
            text(src, c),
            "/// A doc.\n@MainActor\nstruct S {\n    var x: Int\n}"
        );
        assert_eq!(member_symbols(c), ["S"]);
    }

    /// `var`/`let` keywords come from `value_binding_pattern`.
    #[test]
    fn labels() {
        let src = "var a = 1\nlet b = 2\ntypealias A = Int\nenum E { case one }\nfunc f() {}\n";
        let (_, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(
            all[0].title,
            "src/main.x › var a, let b, typealias A, enum E, func f"
        );
    }

    /// An oversized class splits into a header plus member chunks under the
    /// class context; small members pack.
    #[test]
    fn oversized_class_splits_into_members() {
        let body = members(40, |i| {
            format!("    func m{i:02}() {{ {} }}\n", padded("x", 50))
        });
        let src = format!("class Big {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        assert!(all[0].title.starts_with("src/main.x › class Big"));
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "class_declaration_header")
            .expect("a split class emits a header chunk");
        assert_eq!(text(&src, header), "class Big ");
        let member_chunks: Vec<_> = all
            .iter()
            .filter(|c| c.title.contains(" › class Big › "))
            .collect();
        assert!(!member_chunks.is_empty());
        assert_eq!(member_chunks[0].members[0].symbol, "Big::m00");
        for chunk in &all {
            assert!(text(&src, chunk).len() <= MAX + 400);
        }
    }

    /// An oversized protocol splits the same way.
    #[test]
    fn oversized_protocol_and_extension_split() {
        let body = members(40, |i| format!("    func p{i:02}() -> Int\n"));
        let src = format!("protocol P {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › protocol P"));

        let body = members(40, |i| {
            format!("    func e{i:02}() {{ {} }}\n", padded("x", 50))
        });
        let src = format!("extension Foo {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "class_declaration_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › extension Foo");
        assert_eq!(text(&src, header), "extension Foo ");
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "Foo::e00"))
        );
    }

    /// An oversized function splits on line boundaries; multi-byte lines
    /// never split a UTF-8 sequence.
    #[test]
    fn oversized_item_splits_on_lines() {
        let line = format!("    let s = \"{}\"\n", padded("éé", 60));
        let body = line.repeat(50);
        let src = format!("func big() {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.len() > 1);
        assert_eq!(all[0].title, "src/main.x › func big (part 1/2)");
        // parts after the first carry the signature prefix; their own
        // lines rejoin to the source
        let joined: String = all.iter().map(|c| source_lines(&src, c)).collect();
        assert_eq!(joined, src.strip_suffix('\n').unwrap());
        assert!(text(&src, &all[1]).starts_with("func big() …\n"));
    }

    /// Non-ASCII identifiers and directives chunk safely.
    #[test]
    fn non_ascii_is_safe() {
        let src = "#if swift(>=5)\nlet こんにちは = \"wörld\"\n#endif\n";
        let (_, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        // trivia attaches: #if…#endif directives ride with the property
        assert_eq!(all[0].title, "src/main.x › let こんにちは");
        let t = text(src, &all[0]);
        assert!(t.starts_with("#if swift(>=5)"));
        assert!(t.ends_with("let こんにちは = \"wörld\""));
    }

    /// `#else`/`#elseif`/`#endif` close a guarded range: they are dropped,
    /// never attached as trivia to the next item.
    #[test]
    fn endif_and_else_are_dropped() {
        let src = "#if A\nlet x = 1\n#else\nlet y = 2\n#endif\nfunc f() {}\n";
        let (_, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        let texts: Vec<&str> = all[0]
            .members
            .iter()
            .map(|m| &src[m.bytes.clone()])
            .collect();
        assert_eq!(texts, ["#if A\nlet x = 1", "let y = 2", "func f() {}"]);
    }

    /// A `#Macro` invocation labels with the macro's name.
    #[test]
    fn macro_labels_with_name() {
        let src = "#Preview { Text(\"hi\") }\n";
        let (_, all) = chunks_of(src);
        assert_eq!(all[0].title, "src/main.x › #Preview");
    }
}

#[cfg(feature = "lang-typescript")]
mod typescript {
    use super::*;
    use repo_search_core::chunk::TypeScript;

    fn chunks_of(src: &str) -> (Chunker<TypeScript>, Vec<Chunk>) {
        chunks(TypeScript, src)
    }

    /// Comments and decorators attach; imports and bare re-exports drop;
    /// `export`/`declare` wrappers report the inner declaration.
    #[test]
    fn item_carries_trivia_and_wrappers_unwrap() {
        let src = "import { x } from 'm';\nexport { x };\n\n// doc\nexport function f() {}\n@Dec\nexport class C {}\ndeclare const y: number;\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].kind, ChunkKind::Packed);
        let c = &all[0];
        assert_eq!(
            member_kinds(&chunker, c),
            [
                "function_declaration",
                "class_declaration",
                "lexical_declaration"
            ]
        );
        let texts: Vec<String> = c
            .members
            .iter()
            .map(|m| src[m.bytes.clone()].to_string())
            .collect();
        assert_eq!(texts[0], "// doc\nexport function f() {}");
        assert_eq!(texts[1], "@Dec\nexport class C {}");
        assert_eq!(texts[2], "declare const y: number;");
        assert_eq!(c.title, "src/main.x › function f, class C, const y");
    }

    /// An oversized class splits into header and members under its context.
    #[test]
    fn oversized_class_splits_into_members() {
        let body = members(40, |i| format!("    m{i:02}() {{ {} }}\n", padded("x", 55)));
        let src = format!("export class Big {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "class_declaration_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › class Big");
        assert_eq!(text(&src, header), "export class Big ");
        let member = all
            .iter()
            .find(|c| c.members.iter().any(|m| m.symbol == "Big::m00"))
            .unwrap();
        assert!(member.title.contains(" › class Big › "));
    }

    /// A member decorator is a `class_body` sibling: it rides as trivia,
    /// so it and the member's doc comment stay inside the member's span.
    #[test]
    fn member_decorators_attach() {
        let body = members(40, |i| {
            format!(
                "    /** doc */\n    @dec\n    m{i:02}() {{ {} }}\n",
                padded("x", 45)
            )
        });
        let src = format!("class C {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        let member = all
            .iter()
            .flat_map(|c| c.members.iter())
            .find(|m| m.symbol == "C::m00")
            .expect("the decorated method is a member");
        let expected = format!("/** doc */\n    @dec\n    m00() {{ {} }}", padded("x", 45));
        assert_eq!(&src[member.bytes.clone()], expected);
    }

    /// Interface, enum and namespace containers split too.
    #[test]
    fn interface_enum_and_namespace_split() {
        let body = members(45, |i| format!("    prop{i:02}: string;\n"));
        let src = format!("interface I {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › interface I"));

        let body = members(320, |i| format!("    E{i:04} = {i},\n"));
        let src = format!("enum E {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › enum E"));

        let body = members(35, |i| {
            format!("    export function n{i:02}() {{ {} }}\n", padded("x", 50))
        });
        let src = format!("namespace NS {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title.contains(" › namespace NS › ")));
    }

    /// An oversized function splits on lines.
    #[test]
    fn oversized_item_splits_on_lines() {
        let line = format!("    const s = \"{}\";\n", padded("xy", 60));
        let src = format!("function big() {{\n{}}}\n", line.repeat(50));
        let (_, all) = chunks_of(&src);
        assert_eq!(all[0].title, "src/main.x › function big (part 1/2)");
        // parts after the first carry the signature prefix; their own
        // lines rejoin to the source
        let joined: String = all.iter().map(|c| source_lines(&src, c)).collect();
        assert_eq!(joined, src.strip_suffix('\n').unwrap());
        assert!(text(&src, &all[1]).starts_with("function big() …\n"));
    }

    /// Non-ASCII source never splits a UTF-8 sequence.
    #[test]
    fn non_ascii_is_safe() {
        let src = "// 注释\nconst café = \"héllo\";\n";
        let (_, all) = chunks_of(src);
        assert_eq!(all[0].title, "src/main.x › const café");
        assert_eq!(text(src, &all[0]), "// 注释\nconst café = \"héllo\";");
    }
}

#[cfg(feature = "lang-typescript")]
mod tsx {
    use super::*;
    use repo_search_core::chunk::Tsx;

    /// JSX parses under the TSX grammar.
    #[test]
    fn jsx_elements_chunk() {
        let src = "export const App = () => <div className=\"x\">{items.map(i => <li key={i}>{i}</li>)}</div>;\n";
        let (chunker, all) = chunks(Tsx, src);
        assert_eq!(all.len(), 1);
        assert_eq!(kind(&chunker, &all[0]), "lexical_declaration");
        assert_eq!(all[0].title, "src/main.x › const App");
    }
}

#[cfg(feature = "lang-javascript")]
mod javascript {
    use super::*;
    use repo_search_core::chunk::JavaScript;

    fn chunks_of(src: &str) -> (Chunker<JavaScript>, Vec<Chunk>) {
        chunks(JavaScript, src)
    }

    /// A shebang rides with the first item; `export default` unwraps.
    #[test]
    fn shebang_and_export_default() {
        let src = "#!/usr/bin/env node\nexport default function main() {}\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(kind(&chunker, &all[0]), "function_declaration");
        assert_eq!(all[0].title, "src/main.x › function main");
        assert_eq!(
            text(src, &all[0]),
            "#!/usr/bin/env node\nexport default function main() {}"
        );
    }

    /// `export default <expr>` is an item labelled `export default`.
    #[test]
    fn export_default_expression() {
        let src = "const a = {};\nexport default a;\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(
            member_kinds(&chunker, &all[0]),
            ["lexical_declaration", "export_statement"]
        );
        assert_eq!(all[0].title, "src/main.x › const a, export default");
    }

    /// An oversized class splits under the JS grammar too.
    #[test]
    fn oversized_class_splits() {
        let body = members(40, |i| format!("    m{i:02}() {{ {} }}\n", padded("x", 55)));
        let src = format!("class Big {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › class Big"));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "Big::m00"))
        );
    }
}

#[cfg(feature = "lang-python")]
mod python {
    use super::*;
    use repo_search_core::chunk::{ChunkKind, Python};

    fn chunks_of(src: &str) -> (Chunker<Python>, Vec<Chunk>) {
        chunks(Python, src)
    }

    /// The module docstring is a `docs` chunk; imports drop; decorators
    /// stay inside the item's span and the kind is the inner definition's.
    #[test]
    fn docstring_is_module_docs() {
        let src = format!(
            "\"\"\"Module doc.\"\"\"\nimport os\nfrom x import y\n\n# a note\n@decorator\ndef f():\n    x = \"{}\"\n",
            padded("x", 500)
        );
        let (chunker, all) = chunks_of(&src);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
        assert_eq!(kind(&chunker, &all[0]), "docs");
        assert_eq!(all[0].title, "src/main.x › module docs");
        assert_eq!(text(&src, &all[0]), "\"\"\"Module doc.\"\"\"");
        assert_eq!(kind(&chunker, &all[1]), "function_definition");
        assert_eq!(all[1].title, "src/main.x — Module doc. › def f");
        assert_eq!(all[1].lines, 5..=8);
        assert!(text(&src, &all[1]).starts_with("# a note\n@decorator\ndef f():"));
        assert_eq!(member_symbols(&all[1]), ["f"]);
    }

    /// A shebang, coding or license comment does not keep the docstring
    /// from opening the module; it still becomes the `docs` chunk.
    #[test]
    fn docstring_after_comments() {
        let src = format!(
            "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\"\"\"Module doc.\"\"\"\nimport os\nx = \"{}\"\n",
            padded("x", 500)
        );
        let (chunker, all) = chunks_of(&src);
        assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
        assert_eq!(kind(&chunker, &all[0]), "docs");
        assert_eq!(text(&src, &all[0]), "\"\"\"Module doc.\"\"\"");
        // the comment lines were trivia, not items
        assert!(
            all.iter()
                .flat_map(|c| c.members.iter())
                .all(|m| !src[m.bytes.clone()].contains("#!"))
        );
        assert_eq!(all[1].title, "src/main.x — Module doc. › x");
    }

    /// Assignments report the `assignment` kind and the target's name.
    #[test]
    fn assignments_and_labels() {
        let src = "X = 1\ny: int = 2\nasync def af():\n    pass\nif __name__ == \"__main__\":\n    main()\n";
        let (chunker, all) = chunks_of(src);
        // X and y pack; af and the if-block are their own items.
        let kinds: Vec<String> = all.iter().flat_map(|c| member_kinds(&chunker, c)).collect();
        assert!(kinds.contains(&"assignment".to_string()));
        let titles: Vec<&str> = all.iter().map(|c| c.title.as_str()).collect();
        assert!(
            titles
                .iter()
                .any(|t| t.contains("X, y") || t.contains("y, X"))
        );
        assert!(titles.iter().any(|t| t.contains("async def af")));
        assert!(
            titles
                .iter()
                .any(|t| t.contains("if __name__ == \"__main__\""))
        );
        assert_eq!(member_symbols(&all[0])[0], "X");
    }

    /// An oversized class splits; its docstring becomes a docs chunk under
    /// the class context.
    #[test]
    fn oversized_class_splits_and_docstring_in_context() {
        let body = members(40, |i| {
            format!(
                "    def m{i:02}(self):\n        return {i} + {}\n",
                padded("x", 40)
            )
        });
        let src = format!("class Big:\n    \"\"\"Class doc.\"\"\"\n{body}");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "class_definition_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › class Big");
        assert_eq!(text(&src, header), "class Big:\n    ");
        let docs = all
            .iter()
            .find(|c| c.title.contains("module docs"))
            .unwrap();
        assert!(
            docs.title
                .starts_with("src/main.x › class Big › module docs,")
        );
        let m = &docs.members[0];
        assert_eq!(&src[m.bytes.clone()], "\"\"\"Class doc.\"\"\"");
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "Big::m00"))
        );
    }

    /// An oversized function splits on lines; non-ASCII never splits a
    /// UTF-8 sequence.
    #[test]
    fn oversized_item_splits_on_lines() {
        let line = format!("    s = \"{}\"\n", padded("éé", 60));
        let src = format!("def big():\n{}", line.repeat(50));
        let (_, all) = chunks_of(&src);
        assert_eq!(all[0].title, "src/main.x › def big (part 1/2)");
        // parts after the first carry the signature prefix; their own
        // lines rejoin to the source
        let joined: String = all.iter().map(|c| source_lines(&src, c)).collect();
        assert_eq!(joined, src.strip_suffix('\n').unwrap());
        assert!(text(&src, &all[1]).starts_with("def big(): …\n"));
    }

    /// A non-ASCII docstring and identifiers chunk safely.
    #[test]
    fn non_ascii_is_safe() {
        let src = format!(
            "\"\"\"Döcùmént 注释。\"\"\"\nvalor = \"{}\"\n",
            padded("é", 500)
        );
        let (chunker, all) = chunks_of(&src);
        assert_eq!(all[0].kind, ChunkKind::ModuleDocs);
        assert_eq!(kind(&chunker, &all[0]), "docs");
        assert_eq!(all[1].title, "src/main.x — Döcùmént 注释。 › valor");
    }
}

#[cfg(feature = "lang-c")]
mod c {
    use super::*;
    use repo_search_core::chunk::C;

    fn chunks_of(src: &str) -> (Chunker<C>, Vec<Chunk>) {
        chunks(C, src)
    }

    /// `/* */` and `//` comments attach; `#include` drops; `#define` is an
    /// item.
    #[test]
    fn item_carries_trivia_and_defines() {
        let src = "#include <stdio.h>\n\n// doc\n#define MAX 10\n\n/* block */\nint f(void) { return 0; }\n";
        let (chunker, all) = chunks_of(src);
        let kinds: Vec<String> = all.iter().flat_map(|c| member_kinds(&chunker, c)).collect();
        assert!(kinds.contains(&"preproc_def".to_string()));
        assert!(kinds.contains(&"function_definition".to_string()));
        let titles: Vec<&str> = all.iter().map(|c| c.title.as_str()).collect();
        assert!(titles.iter().any(|t| t.contains("#define MAX")));
        assert!(titles.iter().any(|t| t.contains("fn f")));
        let f = all
            .iter()
            .flat_map(|c| c.members.iter())
            .find(|m| m.symbol == "f")
            .unwrap();
        assert_eq!(
            &src[f.bytes.clone()],
            "/* block */\nint f(void) { return 0; }"
        );
    }

    /// An oversized struct splits into a header plus member fields under
    /// the struct context.
    #[test]
    fn oversized_struct_splits_into_members() {
        let body = members(80, |i| {
            format!("    int f{i:03}; /* {} */\n", padded("x", 15))
        });
        let src = format!("struct Big {{\n{body}}};\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "struct_specifier_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › struct Big");
        assert_eq!(text(&src, header), "struct Big ");
        assert!(all.iter().any(|c| c.title.contains(" › struct Big › ")));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "Big::f000"))
        );
    }

    /// An oversized enum splits into enumerators.
    #[test]
    fn oversized_enum_splits() {
        let body = members(320, |i| format!("    E{i:04} = {i},\n"));
        let src = format!("enum E {{\n{body}}};\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › enum E"));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "E::E0000"))
        );
    }

    /// An oversized `#if` block splits on its directive line; guarded
    /// items carry the `#if` context.
    #[test]
    fn oversized_preproc_splits() {
        let body = members(40, |i| {
            format!("int v{i:02} = {i}; /* {} */\n", padded("x", 40))
        });
        let src = format!("#if CONFIG_X\n{body}#else\nint z = 1;\n#endif\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "preproc_if_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › #if CONFIG_X");
        assert_eq!(text(&src, header), "#if CONFIG_X");
        assert!(all.iter().any(|c| c.title.contains(" › #if CONFIG_X › ")));
        assert!(
            all.iter()
                .any(|c| c.title == "src/main.x › #if CONFIG_X › #else")
        );
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "v00"))
        );
    }

    /// A `#if`/`#elif` condition or `#ifdef`/`#elifdef` name is its
    /// parent's `condition`/`name` field, inside the directive header —
    /// never an item of its own.
    #[test]
    fn if_condition_is_not_an_item() {
        // `conditional_expression` is a valid `#if` operand and was not in
        // the old kind skip list.
        let body = members(40, |i| {
            format!("int v{i:02} = {i}; /* {} */\n", padded("x", 40))
        });
        let src = format!("#if FOO ? BAR : BAZ\n{body}#endif\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "preproc_if_header")
            .unwrap();
        assert_eq!(text(&src, header), "#if FOO ? BAR : BAZ");
        let kinds: Vec<String> = all.iter().flat_map(|c| member_kinds(&chunker, c)).collect();
        assert!(!kinds.contains(&"conditional_expression".to_string()));
        assert!(!kinds.contains(&"identifier".to_string()));
        // an `#elif` operand is covered too
        let body = members(40, |i| {
            format!("int w{i:02} = {i}; /* {} */\n", padded("x", 40))
        });
        let src = format!("#if A\n{body}#elif B ? C : D\nint z = 1;\n#endif\n");
        let (chunker, all) = chunks_of(&src);
        let kinds: Vec<String> = all.iter().flat_map(|c| member_kinds(&chunker, c)).collect();
        assert!(!kinds.contains(&"conditional_expression".to_string()));
    }

    /// No `preproc_*` names a symbol: an include guard is not a segment of
    /// the guarded items' symbol paths.
    #[test]
    fn preproc_items_have_no_symbol() {
        let src = "#define MAX 10\n";
        let (_, all) = chunks_of(src);
        assert_eq!(member_symbols(&all[0]), [""]);

        let body = members(40, |i| {
            format!("int g{i:02} = {i}; /* {} */\n", padded("x", 40))
        });
        let src = format!("#ifndef GUARD\n{body}#endif\n");
        let (_, all) = chunks_of(&src);
        assert!(
            all.iter()
                .flat_map(|c| c.members.iter().map(|m| m.symbol.as_str()))
                .all(|s| !s.starts_with("GUARD"))
        );
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "g00"))
        );
    }

    /// A small `#ifdef` stays one item including the directive lines.
    #[test]
    fn small_preproc_stays_whole() {
        let src = "#ifdef DEBUG\nint d = 1;\n#endif\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(kind(&chunker, &all[0]), "preproc_ifdef");
        assert_eq!(all[0].title, "src/main.x › #ifdef DEBUG");
        assert_eq!(text(src, &all[0]), src.strip_suffix('\n').unwrap());
    }

    /// An oversized function splits on lines; non-ASCII is safe.
    #[test]
    fn oversized_item_and_non_ascii() {
        let line = format!("    int x = {}; /* éé */\n", padded("1", 50));
        let src = format!("int big(void) {{\n{}}}\n", line.repeat(50));
        let (_, all) = chunks_of(&src);
        assert_eq!(all[0].title, "src/main.x › fn big (part 1/2)");
        // parts after the first carry the signature prefix; their own
        // lines rejoin to the source
        let joined: String = all.iter().map(|c| source_lines(&src, c)).collect();
        assert_eq!(joined, src.strip_suffix('\n').unwrap());
        assert!(text(&src, &all[1]).starts_with("int big(void) …\n"));
    }
}

#[cfg(feature = "lang-cpp")]
mod cpp {
    use super::*;
    use repo_search_core::chunk::Cpp;

    fn chunks_of(src: &str) -> (Chunker<Cpp>, Vec<Chunk>) {
        chunks(Cpp, src)
    }

    /// `using` drops; `template<>` reports the inner class; comments
    /// attach.
    #[test]
    fn wrappers_and_skips() {
        let src = "using namespace std;\n// doc\ntemplate<class T>\nclass Box { T v; };\nnamespace fs = std::filesystem;\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(kind(&chunker, &all[0]), "class_specifier");
        assert_eq!(all[0].title, "src/main.x › class Box");
        assert_eq!(
            text(src, &all[0]),
            "// doc\ntemplate<class T>\nclass Box { T v; };"
        );
    }

    /// A comment between `template<…>` and its declaration is skipped
    /// when the inner declaration is chosen as the item's subject.
    #[test]
    fn template_subject_skips_comments() {
        let src = "template <typename T> // why\nclass Box { T v; };\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(kind(&chunker, &all[0]), "class_specifier");
        assert_eq!(all[0].title, "src/main.x › class Box");
        assert_eq!(
            text(src, &all[0]),
            "template <typename T> // why\nclass Box { T v; };"
        );
    }

    /// An oversized class splits; `public:` attaches to the member that
    /// follows it.
    #[test]
    fn oversized_class_splits_with_access_specifiers() {
        let body = members(40, |i| {
            format!("    void m{i:02}(); // {}\n", padded("x", 45))
        });
        let src = format!("class Big {{\npublic:\n{body}}};\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › class Big"));
        let first_member = all
            .iter()
            .flat_map(|c| c.members.iter())
            .find(|m| m.symbol == "Big::m00")
            .unwrap();
        let member_text = &src[first_member.bytes.clone()];
        assert!(member_text.starts_with("public:"));
    }

    /// An oversized namespace splits into member declarations under the
    /// qualified name's context.
    #[test]
    fn oversized_namespace_splits() {
        let body = members(40, |i| {
            format!("int n{i:02}() {{ return {i} + {}; }}\n", padded("x", 35))
        });
        let src = format!("namespace a::b {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "namespace_definition_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › namespace a::b");
        assert!(all.iter().any(|c| c.title.contains(" › namespace a::b › ")));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "a::b::n00"))
        );
    }

    /// An oversized `extern "C"` block splits into guarded declarations.
    #[test]
    fn oversized_linkage_splits() {
        let body = members(40, |i| {
            format!("int ext{i:02}(int p); // {}\n", padded("x", 45))
        });
        let src = format!("extern \"C\" {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "linkage_specification_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › extern \"C\"");
        assert!(all.iter().any(|c| c.title.contains(" › extern \"C\" › ")));
    }
}

#[cfg(feature = "lang-go")]
mod go {
    use super::*;
    use repo_search_core::chunk::Go;

    fn chunks_of(src: &str) -> (Chunker<Go>, Vec<Chunk>) {
        chunks(Go, src)
    }

    /// `package`/`import` drop; doc comments attach; a method's symbol is
    /// `Receiver::Name`.
    #[test]
    fn item_carries_trivia_and_methods() {
        let src = "package main\nimport \"fmt\"\n\n// Doc.\ntype T struct{}\n\n// M docs.\nfunc (t *T) M() {}\n";
        let (chunker, all) = chunks_of(src);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].title, "src/main.x › type T, func (T) M");
        assert_eq!(
            member_kinds(&chunker, &all[0]),
            ["type_declaration", "method_declaration"]
        );
        assert_eq!(member_symbols(&all[0]), ["T", "T::M"]);
        let m = &all[0].members[1];
        assert_eq!(&src[m.bytes.clone()], "// M docs.\nfunc (t *T) M() {}");
    }

    /// An oversized struct splits into fields under the type context.
    #[test]
    fn oversized_struct_splits() {
        let body = members(150, |i| format!("    F{i:03} int // {}\n", padded("x", 20)));
        let src = format!("type Big struct {{\n{body}}}\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "type_declaration_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › type Big");
        assert_eq!(text(&src, header), "type Big struct ");
        assert!(all.iter().any(|c| c.title.contains(" › type Big › ")));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "Big::F000"))
        );
    }

    /// An oversized interface splits into method elements; the header
    /// covers `interface {` including leading comments.
    #[test]
    fn oversized_interface_splits() {
        let body = members(60, |i| {
            format!(
                "    M{i:03}(ctx context.Context) error // {}\n",
                padded("x", 8)
            )
        });
        let src = format!("type I interface {{\n{body}}}\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title == "src/main.x › type I"));
        assert!(all.iter().any(|c| c.title.contains(" › type I › ")));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "I::M000"))
        );
    }

    /// An oversized `const (…)` group splits into specs.
    #[test]
    fn oversized_const_group_splits() {
        let body = members(320, |i| format!("    C{i:04} = {i}\n"));
        let src = format!("const (\n{body})\n");
        let (chunker, all) = chunks_of(&src);
        let header = all
            .iter()
            .find(|c| kind(&chunker, c) == "const_declaration_header")
            .unwrap();
        assert_eq!(header.title, "src/main.x › const");
        assert_eq!(text(&src, header), "const (");
        assert!(all.iter().any(|c| c.title.contains(" › const › ")));
        assert!(
            all.iter()
                .any(|c| c.members.iter().any(|m| m.symbol == "C0000"))
        );
    }

    /// A `const (…)`/`type (…)` group header ends after `(`: the first
    /// spec's leading comment stays with the spec, and no two members
    /// share a byte.
    #[test]
    fn group_header_ends_at_paren() {
        let assert_group = |src: &str, header_kind: &str, header_text: &str, first_symbol: &str| {
            let (chunker, all) = chunks_of(src);
            let header = all
                .iter()
                .find(|c| kind(&chunker, c) == header_kind)
                .expect("the group emits a header chunk");
            assert_eq!(text(src, header), header_text);
            let first = all
                .iter()
                .flat_map(|c| c.members.iter())
                .find(|m| m.symbol == first_symbol)
                .expect("the first spec is a member");
            assert!(src[first.bytes.clone()].starts_with("// first doc"));
            let mut ranges: Vec<(usize, usize)> = all
                .iter()
                .flat_map(|c| c.members.iter().map(|m| (m.bytes.start, m.bytes.end)))
                .collect();
            ranges.sort_unstable();
            assert!(ranges.windows(2).all(|w| w[0].1 <= w[1].0));
        };
        let body = members(320, |i| format!("    C{i:04} = {i}\n"));
        assert_group(
            &format!("const (\n    // first doc\n{body})\n"),
            "const_declaration_header",
            "const (",
            "C0000",
        );
        let body = members(320, |i| format!("    T{i:04} int\n"));
        assert_group(
            &format!("type (\n    // first doc\n{body})\n"),
            "type_declaration_header",
            "type (",
            "T0000",
        );
    }

    /// `var (…)` groups split through their `var_spec_list`.
    #[test]
    fn oversized_var_group_splits() {
        let body = members(320, |i| format!("    v{i:04} = {i}\n"));
        let src = format!("var (\n{body})\n");
        let (_, all) = chunks_of(&src);
        assert!(all.iter().any(|c| c.title.contains(" › var › ")));
    }

    /// An oversized function splits on lines; non-ASCII is safe.
    #[test]
    fn oversized_item_and_non_ascii() {
        let line = format!("    s := \"{}\" // 注\n", padded("x", 55));
        let src = format!("func big() {{\n{}}}\n", line.repeat(50));
        let (_, all) = chunks_of(&src);
        assert_eq!(all[0].title, "src/main.x › func big (part 1/2)");
        // parts after the first carry the signature prefix; their own
        // lines rejoin to the source
        let joined: String = all.iter().map(|c| source_lines(&src, c)).collect();
        assert_eq!(joined, src.strip_suffix('\n').unwrap());
        assert!(text(&src, &all[1]).starts_with("func big() …\n"));
    }
}
