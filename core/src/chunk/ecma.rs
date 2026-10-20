//! The TypeScript, TSX and JavaScript grammars, on `tree-sitter-typescript`
//! and `tree-sitter-javascript`; one rule set for the shared node shapes.
//!
//! Decorators are children of the declarations they annotate (or of the
//! `export_statement` wrapping them), so they stay inside the item's span.
//! `export`/`declare` wrappers report their inner declaration's kind,
//! label and symbol while the span covers the wrapper.

use tree_sitter::{Language, Node};

use super::Grammar;
use super::units::text::{field_text, first_token};

/// Forwards the shared ECMA rules to the free functions below.
macro_rules! forward {
    () => {
        fn is_leading(&self, node: Node) -> bool {
            is_leading(node)
        }
        fn is_skipped(&self, node: Node) -> bool {
            is_skipped(node)
        }
        fn is_inner_doc(&self, _text: &str) -> bool {
            false
        }
        fn subject<'a>(&self, node: Node<'a>) -> Node<'a> {
            subject(node)
        }
        fn label(&self, src: &str, node: Node) -> String {
            label(src, node)
        }
        fn split<'a>(&self, node: Node<'a>) -> Option<(usize, Node<'a>)> {
            split(node)
        }
        fn symbol(&self, src: &str, node: Node) -> Option<String> {
            symbol(src, node)
        }
    };
}

/// TypeScript without JSX (`.ts`).
#[cfg(feature = "lang-typescript")]
pub struct TypeScript;

/// TypeScript with JSX (`.tsx`).
#[cfg(feature = "lang-typescript")]
pub struct Tsx;

/// JavaScript, JSX included (`.js`, `.jsx`, `.mjs`, `.cjs`).
#[cfg(feature = "lang-javascript")]
pub struct JavaScript;

#[cfg(feature = "lang-typescript")]
impl Grammar for TypeScript {
    fn language(&self) -> Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }
    forward!();
}

#[cfg(feature = "lang-typescript")]
impl Grammar for Tsx {
    fn language(&self) -> Language {
        tree_sitter_typescript::LANGUAGE_TSX.into()
    }
    forward!();
}

#[cfg(feature = "lang-javascript")]
impl Grammar for JavaScript {
    fn language(&self) -> Language {
        tree_sitter_javascript::LANGUAGE.into()
    }
    forward!();
}

fn is_leading(node: Node) -> bool {
    // A member decorator is a `class_body` sibling of the member it
    // annotates, so it rides as trivia; a class decorator is a fielded
    // child of its declaration and never reaches the walk.
    matches!(
        node.kind(),
        "comment" | "html_comment" | "hash_bang_line" | "decorator"
    )
}

fn is_skipped(node: Node) -> bool {
    match node.kind() {
        "import_statement" | "import_alias" | "empty_statement" | "namespace_export" => true,
        // `export {a}` and `export *` name no item of this file; a
        // declaration or a `default` value does.
        "export_statement" => {
            node.child_by_field_name("declaration").is_none()
                && node.child_by_field_name("value").is_none()
        }
        _ => false,
    }
}

/// `true` for node kinds that declare a name inside a `declare` wrapper.
fn is_decl(kind: &str) -> bool {
    kind.ends_with("_declaration")
        || kind.ends_with("_signature")
        || matches!(
            kind,
            "internal_module" | "module" | "type_alias_declaration"
        )
}

fn subject(mut node: Node) -> Node {
    loop {
        let inner = match node.kind() {
            "export_statement" => node.child_by_field_name("declaration"),
            "ambient_declaration" => {
                let mut cursor = node.walk();
                node.named_children(&mut cursor)
                    .find(|child| is_decl(child.kind()))
            }
            // A `namespace`/`module` declaration parses as
            // `expression_statement > internal_module`; report the module.
            "expression_statement" => node.named_child(0).filter(|child| is_decl(child.kind())),
            _ => None,
        };
        let Some(inner) = inner else {
            return node;
        };
        node = inner;
    }
}

fn label(src: &str, node: Node) -> String {
    let word = match node.kind() {
        "function_declaration" | "function_signature" => "function",
        "generator_function_declaration" => "function*",
        "class_declaration" => "class",
        "abstract_class_declaration" => "abstract class",
        "interface_declaration" => "interface",
        "enum_declaration" => "enum",
        "internal_module" => "namespace",
        "module" => "module",
        "type_alias_declaration" => "type",
        "method_definition" | "method_signature" | "abstract_method_signature" => "method",
        "field_definition" | "public_field_definition" => "field",
        // `const`/`let`/`var`/`using` is the first unnamed child; the name
        // is the first declarator's.
        "lexical_declaration" | "variable_declaration" | "using_declaration" => {
            let keyword = first_token(src, node).unwrap_or("var");
            let mut cursor = node.walk();
            let declarator = node
                .named_children(&mut cursor)
                .find(|child| child.kind() == "variable_declarator");
            return match declarator.and_then(|d| field_text(src, d, "name")) {
                Some(name) => format!("{keyword} {name}"),
                None => keyword.to_string(),
            };
        }
        "variable_declarator" => {
            return field_text(src, node, "name")
                .unwrap_or("variable")
                .to_string();
        }
        "property_signature" | "enum_assignment" => {
            return field_text(src, node, "name")
                .unwrap_or(node.kind())
                .to_string();
        }
        "export_statement" => return "export default".to_string(),
        "ambient_declaration" => return "declare".to_string(),
        "index_signature" => return "index signature".to_string(),
        "call_signature" => return "call signature".to_string(),
        "construct_signature" => return "construct signature".to_string(),
        _ => return node.kind().to_string(),
    };
    match name_text(src, node) {
        Some(name) => format!("{word} {name}"),
        None => word.to_string(),
    }
}

fn split(node: Node) -> Option<(usize, Node)> {
    let body = match node.kind() {
        "class_declaration"
        | "abstract_class_declaration"
        | "interface_declaration"
        | "enum_declaration"
        | "internal_module"
        | "module" => node.child_by_field_name("body")?,
        // `declare global {}` carries its `statement_block` without a field.
        "ambient_declaration" => {
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .find(|child| child.kind() == "statement_block")?
        }
        _ => return None,
    };
    Some((body.start_byte(), body))
}

fn symbol(src: &str, node: Node) -> Option<String> {
    match node.kind() {
        "export_statement" | "ambient_declaration" => None,
        "lexical_declaration" | "variable_declaration" | "using_declaration" => {
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .find(|child| child.kind() == "variable_declarator")
                .and_then(|declarator| field_text(src, declarator, "name"))
                .map(str::to_string)
        }
        _ => name_text(src, node).map(str::to_string),
    }
}

/// The item's `name` field, or `property` for a class field.
fn name_text<'a>(src: &'a str, node: Node) -> Option<&'a str> {
    field_text(src, node, "name").or_else(|| field_text(src, node, "property"))
}
