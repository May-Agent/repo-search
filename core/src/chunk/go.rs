//! The Go grammar, on `tree-sitter-go`.
//!
//! Doc comments and build directives (`//go:build`, `//go:generate`) are
//! comments riding as trivia; `package` and `import` declarations are
//! skipped like `use`. `type`/`const`/`var` groups, `struct` types and
//! `interface` types are the containers.

use tree_sitter::{Language, Node};

use super::Grammar;
use super::units::text::{field_text, node_text};

/// The Go language.
pub struct Go;

impl Grammar for Go {
    fn language(&self) -> Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn is_leading(&self, node: Node) -> bool {
        node.kind() == "comment"
    }

    fn is_skipped(&self, node: Node) -> bool {
        matches!(
            node.kind(),
            "package_clause" | "import_declaration" | "empty_statement"
        )
    }

    fn is_inner_doc(&self, _text: &str) -> bool {
        false
    }

    fn label(&self, src: &str, node: Node) -> String {
        match node.kind() {
            "function_declaration" => return named(src, "func", node),
            "method_declaration" => {
                return match (receiver_type(src, node), field_text(src, node, "name")) {
                    (Some(receiver), Some(name)) => format!("func ({receiver}) {name}"),
                    (_, Some(name)) => format!("func {name}"),
                    _ => "func".to_string(),
                };
            }
            "type_declaration" | "var_declaration" | "const_declaration" => {
                let keyword = node_text(
                    src,
                    node.child(0).expect("a declaration opens with its keyword"),
                );
                let specs = specs(node);
                return match specs.first() {
                    Some(&spec) if specs.len() == 1 => {
                        let name = field_text(src, spec, "name").unwrap_or(keyword);
                        format!("{keyword} {name}")
                    }
                    _ => keyword.to_string(),
                };
            }
            "type_spec" | "type_alias" | "var_spec" | "const_spec" | "method_elem" => {
                return field_text(src, node, "name")
                    .unwrap_or(node.kind())
                    .to_string();
            }
            "field_declaration" => {
                return field_text(src, node, "name")
                    .or_else(|| field_text(src, node, "type"))
                    .unwrap_or("field")
                    .to_string();
            }
            "type_elem" => return node_text(src, node).to_string(),
            "short_var_declaration" => {
                return field_text(src, node, "left").unwrap_or(":=").to_string();
            }
            "func_literal" => return "func literal".to_string(),
            _ => {}
        }
        node.kind().to_string()
    }

    fn split<'a>(&self, node: Node<'a>) -> Option<(usize, Node<'a>)> {
        match node.kind() {
            "type_declaration" => {
                let specs = specs(node);
                if specs.len() > 1 {
                    // A `type (…)` group: the members are the specs.
                    Some((group_open(node), node))
                } else {
                    spec_body(*specs.first()?)
                }
            }
            "type_spec" | "type_alias" => spec_body(node),
            "var_declaration" => {
                let mut cursor = node.walk();
                let list = node
                    .named_children(&mut cursor)
                    .find(|child| child.kind() == "var_spec_list")?;
                Some((list.start_byte(), list))
            }
            "const_declaration" => {
                let specs = specs(node);
                if specs.len() < 2 {
                    return None;
                }
                Some((group_open(node), node))
            }
            _ => None,
        }
    }

    fn symbol(&self, src: &str, node: Node) -> Option<String> {
        match node.kind() {
            "method_declaration" => {
                let name = field_text(src, node, "name")?;
                match receiver_type(src, node) {
                    Some(receiver) => Some(format!("{receiver}::{name}")),
                    None => Some(name.to_string()),
                }
            }
            "type_declaration" | "var_declaration" | "const_declaration" => {
                let specs = specs(node);
                match specs.first() {
                    Some(&spec) if specs.len() == 1 => {
                        field_text(src, spec, "name").map(str::to_string)
                    }
                    _ => None,
                }
            }
            "field_declaration" => field_text(src, node, "name")
                .or_else(|| field_text(src, node, "type"))
                .map(str::to_string),
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

/// The byte after a multi-spec declaration's `(`: the header ends there so
/// the first spec's leading comment is never inside it. A `type`/`const`
/// group with more than one spec is always parenthesized.
fn group_open(node: Node) -> usize {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == "(")
        .expect("a multi-spec declaration is parenthesized")
        .end_byte()
}

/// The spec children of a `type`/`var`/`const` declaration.
fn specs<'a>(node: Node<'a>) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|child| {
            matches!(
                child.kind(),
                "type_spec" | "type_alias" | "var_spec" | "const_spec"
            )
        })
        .collect()
}

/// The members of a spec's `struct`/`interface` type; `None` for other
/// types. An interface has no list node, so the type itself is the body.
fn spec_body(node: Node) -> Option<(usize, Node)> {
    let ty = node.child_by_field_name("type")?;
    match ty.kind() {
        "struct_type" => {
            let body = ty.named_child(0)?;
            Some((body.start_byte(), body))
        }
        "interface_type" => {
            let mut cursor = ty.walk();
            let member = ty.named_children(&mut cursor).next()?;
            Some((member.start_byte(), ty))
        }
        _ => None,
    }
}

/// The receiver's type name: the identifier under `*`/`()`/`[]` wrappers.
fn receiver_type<'a>(src: &'a str, node: Node<'a>) -> Option<&'a str> {
    let receiver = node.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    let parameter = receiver
        .named_children(&mut cursor)
        .find(|child| child.kind() == "parameter_declaration")?;
    let mut ty = parameter.child_by_field_name("type")?;
    loop {
        match ty.kind() {
            "pointer_type" | "parenthesized_type" | "generic_type" => {
                ty = ty.named_child(0)?;
            }
            _ => return Some(node_text(src, ty)),
        }
    }
}
