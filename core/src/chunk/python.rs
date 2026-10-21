//! The Python grammar, on `tree-sitter-python`.
//!
//! A docstring — a string literal as a scope's first statement — is the
//! module-docs chunk; decorators are children of the `decorated_definition`
//! wrapping the function or class they annotate.

use tree_sitter::{Language, Node};

use super::Grammar;
use super::units::text::{field_text, node_text};

/// The Python language.
pub struct Python;

impl Grammar for Python {
    fn language(&self) -> Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn is_leading(&self, node: Node) -> bool {
        if node.kind() == "comment" {
            return true;
        }
        // A docstring: a bare string literal statement that opens its
        // scope; it rides as trivia so `is_inner_doc` can split it off.
        // Leading comments — a shebang, a coding line, a license header —
        // do not keep it from opening the scope.
        if node.kind() == "expression_statement" {
            let mut prev = node.prev_named_sibling();
            while let Some(sibling) = prev {
                if sibling.kind() != "comment" {
                    return false;
                }
                prev = sibling.prev_named_sibling();
            }
            let first = node.named_child(0);
            return matches!(
                first.map(|child| child.kind()),
                Some("string" | "concatenated_string")
            );
        }
        false
    }

    fn is_skipped(&self, node: Node) -> bool {
        matches!(
            node.kind(),
            "import_statement" | "import_from_statement" | "future_import_statement"
        )
    }

    fn is_inner_doc(&self, text: &str) -> bool {
        is_docstring(text)
    }

    fn inner_doc_text<'a>(&self, text: &'a str) -> &'a str {
        // `"""docstring"""` -> `docstring`: the literal's prefix letters and
        // quotes are not part of the sentence.
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < 3
            && matches!(
                bytes.get(i).map(u8::to_ascii_lowercase),
                Some(b'r' | b'b' | b'f' | b'u' | b't')
            )
        {
            i += 1;
        }
        let text = &text[i..];
        for quote in ["\"\"\"", "'''", "\"", "'"] {
            if let Some(rest) = text.strip_prefix(quote) {
                return rest.strip_suffix(quote).unwrap_or(rest);
            }
        }
        text
    }

    fn subject<'a>(&self, node: Node<'a>) -> Node<'a> {
        match node.kind() {
            "decorated_definition" => node.child_by_field_name("definition").unwrap_or(node),
            // `x = 1` is an `expression_statement` wrapping `assignment`;
            // report the assignment's kind and name.
            "expression_statement" => match node.named_child(0).map(|child| child.kind()) {
                Some("assignment" | "augmented_assignment") => node.named_child(0).unwrap(),
                _ => node,
            },
            _ => node,
        }
    }

    fn label(&self, src: &str, node: Node) -> String {
        let word = match node.kind() {
            "function_definition" => {
                // `async def` opens with the `async` keyword.
                let mut cursor = node.walk();
                let keyword = node
                    .children(&mut cursor)
                    .find(|child| !child.is_named())
                    .map(|child| node_text(src, child))
                    .unwrap_or("def");
                return named(
                    src,
                    if keyword == "async" {
                        "async def"
                    } else {
                        "def"
                    },
                    node,
                );
            }
            "class_definition" => "class",
            "assignment" | "augmented_assignment" => {
                return field_text(src, node, "left")
                    .unwrap_or("assignment")
                    .to_string();
            }
            "type_alias_statement" => {
                return match field_text(src, node, "left") {
                    Some(name) => format!("type {name}"),
                    None => "type".to_string(),
                };
            }
            "if_statement" | "elif_clause" => {
                return match field_text(src, node, "condition") {
                    Some(condition) => format!("if {condition}"),
                    None => "if".to_string(),
                };
            }
            "while_statement" => {
                return match field_text(src, node, "condition") {
                    Some(condition) => format!("while {condition}"),
                    None => "while".to_string(),
                };
            }
            "for_statement" => {
                return match field_text(src, node, "left") {
                    Some(left) => format!("for {left}"),
                    None => "for".to_string(),
                };
            }
            "match_statement" => {
                return match field_text(src, node, "subject") {
                    Some(subject) => format!("match {subject}"),
                    None => "match".to_string(),
                };
            }
            "with_statement" => "with",
            "try_statement" => "try",
            "global_statement" => "global",
            "nonlocal_statement" => "nonlocal",
            _ => return node.kind().to_string(),
        };
        named(src, word, node)
    }

    fn split<'a>(&self, node: Node<'a>) -> Option<(usize, Node<'a>)> {
        let body = match node.kind() {
            "class_definition" => node.child_by_field_name("body")?,
            _ => return None,
        };
        Some((body.start_byte(), body))
    }

    fn symbol(&self, src: &str, node: Node) -> Option<String> {
        match node.kind() {
            "assignment" | "augmented_assignment" | "type_alias_statement" => {
                field_text(src, node, "left").map(str::to_string)
            }
            _ => field_text(src, node, "name").map(str::to_string),
        }
    }
}

/// `word name` when `node` has a `name` field, `word` alone otherwise.
fn named(src: &str, word: &str, node: Node) -> String {
    match field_text(src, node, "name") {
        Some(name) => format!("{word} {name}"),
        None => word.to_string(),
    }
}

/// `true` when `text` is a string literal: a quote possibly preceded by
/// prefix letters (`r`, `b`, `f`, `u`, `t` and their pairs).
fn is_docstring(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < 3
        && matches!(
            bytes.get(i).map(u8::to_ascii_lowercase),
            Some(b'r' | b'b' | b'f' | b'u' | b't')
        )
    {
        i += 1;
    }
    matches!(bytes.get(i), Some(b'"' | b'\''))
}
