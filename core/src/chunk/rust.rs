//! The Rust grammar, on `tree-sitter-rust`.

use tree_sitter::{Language, Node};

use super::Grammar;

/// The Rust language.
pub struct Rust;

impl Rust {
    /// The item's `name` field, or its `macro` field for a macro invocation.
    fn item_name<'a>(&self, node: Node<'a>) -> Option<Node<'a>> {
        node.child_by_field_name("name").or_else(|| {
            if node.kind() == "macro_invocation" {
                node.child_by_field_name("macro")
            } else {
                None
            }
        })
    }
}

impl Grammar for Rust {
    fn language(&self) -> Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn is_leading(&self, node: Node) -> bool {
        matches!(
            node.kind(),
            "line_comment" | "block_comment" | "attribute_item"
        )
    }

    fn is_skipped(&self, node: Node) -> bool {
        match node.kind() {
            "use_declaration"
            | "extern_crate_declaration"
            | "empty_statement"
            | "inner_attribute_item" => true,
            "mod_item" => node.child_by_field_name("body").is_none(),
            _ => false,
        }
    }

    fn is_inner_doc(&self, text: &str) -> bool {
        text.starts_with("//!") || text.starts_with("/*!")
    }

    fn label(&self, src: &str, node: Node) -> String {
        // An expression statement whose first named child is a macro
        // invocation is labelled from the invocation; its kind stays
        // `expression_statement`.
        if let Some(inner) = macro_inner(node) {
            return self.label(src, inner);
        }
        if node.kind() == "impl_item" {
            let trait_ = node.child_by_field_name("trait");
            let type_ = node.child_by_field_name("type");
            let type_str = type_.map_or("?", |t| &src[t.byte_range()]);
            return match trait_ {
                Some(t) => format!("impl {} for {type_str}", &src[t.byte_range()]),
                None => format!("impl {type_str}"),
            };
        }
        let word = match node.kind() {
            "function_item" | "function_signature_item" => "fn",
            "struct_item" => "struct",
            "enum_item" => "enum",
            "trait_item" => "trait",
            "mod_item" => "mod",
            "macro_definition" => "macro_rules!",
            "const_item" => "const",
            "static_item" => "static",
            "type_item" | "associated_type" => "type",
            "union_item" => "union",
            "macro_invocation" => "macro",
            kind => kind,
        };
        match self.item_name(node) {
            Some(name) => format!("{word} {}", &src[name.byte_range()]),
            None => word.to_string(),
        }
    }

    fn split_body<'a>(&self, node: Node<'a>) -> Option<Node<'a>> {
        match node.kind() {
            "impl_item" | "trait_item" | "mod_item" => node.child_by_field_name("body"),
            _ => None,
        }
    }

    fn symbol(&self, src: &str, node: Node) -> Option<String> {
        match node.kind() {
            // A macro call defines no name.
            "macro_invocation" => None,
            "expression_statement" if macro_inner(node).is_some() => None,
            "impl_item" => node.child_by_field_name("type").map(|t| type_name(src, t)),
            _ => node
                .child_by_field_name("name")
                .map(|name| src[name.byte_range()].to_string()),
        }
    }
}

/// The first named child of an `expression_statement` when it is a macro
/// invocation: `m!();` is labelled and symbolled as the macro it calls.
fn macro_inner<'a>(node: Node<'a>) -> Option<Node<'a>> {
    if node.kind() != "expression_statement" {
        return None;
    }
    let inner = node.named_child(0)?;
    if inner.kind() != "macro_invocation" {
        return None;
    }
    Some(inner)
}

/// The name of an impl's type with references, pointers and generics
/// stripped: `Vec` for `Vec<T>`, `Foo` for `&'a Foo<T>`, `foo::Bar` for
/// `foo::Bar<T>`.
fn type_name(src: &str, node: Node) -> String {
    match node.kind() {
        "generic_type" | "reference_type" | "pointer_type" => type_name(
            src,
            node.child_by_field_name("type")
                .expect("generic, reference and pointer types always have a `type` field"),
        ),
        _ => src[node.byte_range()].to_string(),
    }
}
