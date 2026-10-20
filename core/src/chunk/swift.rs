//! The Swift grammar, on `tree-sitter-swift`.
//!
//! Attributes and modifiers (`@objc`, `public`, `@MainActor`) are children
//! of the declarations they annotate, so they stay inside the item's span.
//! The opening `#if` of a conditional-compilation group and
//! `#error`/`#warning` diagnostics attach as trivia to the item that
//! follows them; `#elseif`, `#else` and `#endif` close the guarded range
//! and are dropped.

use tree_sitter::{Language, Node};

use super::Grammar;
use super::units::text::{field_text, first_token, node_text};

/// The Swift language.
pub struct Swift;

impl Grammar for Swift {
    fn language(&self) -> Language {
        tree_sitter_swift::LANGUAGE.into()
    }

    fn is_leading(&self, node: Node) -> bool {
        match node.kind() {
            "comment" | "multiline_comment" | "shebang_line" | "diagnostic" => true,
            // Only the opening `#if` rides as trivia; `is_skipped` drops
            // the `#elseif`/`#else`/`#endif` directives that close a
            // guarded range.
            "directive" => node.child(0).is_some_and(|token| token.kind() == "#if"),
            _ => false,
        }
    }

    fn is_skipped(&self, node: Node) -> bool {
        // `directive` covers `#if` too, but `is_leading` runs first.
        matches!(node.kind(), "import_declaration" | "directive")
    }

    fn is_inner_doc(&self, _text: &str) -> bool {
        false
    }

    fn label(&self, src: &str, node: Node) -> String {
        let word = match node.kind() {
            // `class_declaration` also covers struct, enum, extension and
            // actor; the `declaration_kind` field is the keyword.
            "class_declaration" => {
                return named(
                    src,
                    node.child_by_field_name("declaration_kind")
                        .map(|kind| node_text(src, kind))
                        .unwrap_or("class"),
                    node,
                );
            }
            "protocol_declaration" => "protocol",
            "function_declaration" | "protocol_function_declaration" => "func",
            "init_declaration" => return "init".to_string(),
            "deinit_declaration" => return "deinit".to_string(),
            "subscript_declaration" => return "subscript".to_string(),
            // The `var`/`let` keyword sits inside `value_binding_pattern`,
            // after any modifiers and attributes.
            "property_declaration" | "protocol_property_declaration" => {
                let mut cursor = node.walk();
                let keyword = node
                    .named_children(&mut cursor)
                    .find(|child| child.kind() == "value_binding_pattern")
                    .and_then(|binding| first_token(src, binding))
                    .unwrap_or("var");
                return named(src, keyword, node);
            }
            "typealias_declaration" => "typealias",
            "associatedtype_declaration" => "associatedtype",
            "enum_entry" => "case",
            "macro_declaration" => "macro",
            // `#Preview`: the macro's `simple_identifier` is its name.
            "macro_invocation" => {
                let mut cursor = node.walk();
                let name = node
                    .named_children(&mut cursor)
                    .find(|child| child.kind() == "simple_identifier")
                    .map(|child| node_text(src, child))
                    .expect("a macro invocation names its macro");
                return format!("#{name}");
            }
            "operator_declaration" => "operator",
            "precedence_group_declaration" => "precedencegroup",
            _ => return node.kind().to_string(),
        };
        named(src, word, node)
    }

    fn split<'a>(&self, node: Node<'a>) -> Option<(usize, Node<'a>)> {
        let body = match node.kind() {
            "class_declaration" | "protocol_declaration" => node.child_by_field_name("body")?,
            _ => return None,
        };
        Some((body.start_byte(), body))
    }

    fn symbol(&self, src: &str, node: Node) -> Option<String> {
        match node.kind() {
            "init_declaration" => Some("init".to_string()),
            "deinit_declaration" => Some("deinit".to_string()),
            "subscript_declaration" => Some("subscript".to_string()),
            "macro_invocation" => None,
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
