//! The C and C++ grammars, on `tree-sitter-c` and `tree-sitter-cpp`; one
//! rule set for the shared node shapes.
//!
//! `[[attr]]` specifiers, GNU `__attribute__` and MSVC `__declspec` are
//! children of the declarations they annotate; an `access_specifier`
//! (`public:`) attaches to the member that follows it. Conditional
//! `preproc_*` blocks split like namespaces: the directive line is the
//! header and the guarded items are members, so their titles carry the
//! `#if`/`#elif`/`#else` context.

use tree_sitter::{Language, Node};

use super::Grammar;
use super::units::text::{field_text, node_text};

/// The C language.
#[cfg(feature = "lang-c")]
pub struct C;

/// The C++ language.
#[cfg(feature = "lang-cpp")]
pub struct Cpp;

/// Forwards the shared C-family rules to the free functions below.
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

#[cfg(feature = "lang-c")]
impl Grammar for C {
    fn language(&self) -> Language {
        tree_sitter_c::LANGUAGE.into()
    }
    forward!();
}

#[cfg(feature = "lang-cpp")]
impl Grammar for Cpp {
    fn language(&self) -> Language {
        tree_sitter_cpp::LANGUAGE.into()
    }
    forward!();
}

fn is_leading(node: Node) -> bool {
    if matches!(node.kind(), "comment" | "access_specifier") {
        return true;
    }
    // The `:` completing an `access_specifier` (`public:`) is an unnamed
    // sibling; it rides along so both attach to the next member.
    node.kind() == ":"
        && node
            .prev_sibling()
            .is_some_and(|prev| prev.kind() == "access_specifier")
}

fn is_skipped(node: Node) -> bool {
    match node.kind() {
        "preproc_include" | "using_declaration" | "namespace_alias_definition" => true,
        // A `#if`/`#elif` condition or an `#ifdef`/`#elifdef` name sits in
        // its parent's `condition`/`name` field, inside the header the
        // split already covers; the guarded items are never fielded so.
        _ => ["condition", "name"].iter().any(|field| {
            node.parent()
                .and_then(|parent| parent.child_by_field_name(field))
                .is_some_and(|operand| operand == node)
        }),
    }
}

/// `template<...>` wrappers report the inner declaration.
fn subject(mut node: Node) -> Node {
    while node.kind() == "template_declaration" {
        let parameters = node.child_by_field_name("parameters");
        let mut cursor = node.walk();
        let inner = node.named_children(&mut cursor).find(|child| {
            Some(*child) != parameters && !matches!(child.kind(), "comment" | "requires_clause")
        });
        let Some(inner) = inner else {
            return node;
        };
        node = inner;
    }
    node
}

fn label(src: &str, node: Node) -> String {
    match node.kind() {
        "function_definition" => {
            return match declarator_name(src, node) {
                Some(name) => format!("fn {name}"),
                None => "fn".to_string(),
            };
        }
        "declaration" | "field_declaration" | "type_specifier" => {
            if let Some(specifier) = specifier_label(src, node) {
                return specifier;
            }
            if let Some(name) = declarator_name(src, node) {
                return name.to_string();
            }
        }
        "type_definition" => {
            return match declarator_name(src, node) {
                Some(name) => format!("typedef {name}"),
                None => "typedef".to_string(),
            };
        }
        "struct_specifier" | "union_specifier" | "enum_specifier" | "class_specifier" => {
            return specifier_label(src, node).unwrap_or_else(|| node.kind().to_string());
        }
        "namespace_definition" => return named(src, "namespace", node),
        "linkage_specification" => {
            return match field_text(src, node, "value") {
                Some(value) => format!("extern {value}"),
                None => "extern".to_string(),
            };
        }
        "alias_declaration" => return named(src, "using", node),
        "concept_definition" => return named(src, "concept", node),
        "friend_declaration" => return "friend".to_string(),
        "static_assert_declaration" => return "static_assert".to_string(),
        "enumerator" => {
            return field_text(src, node, "name")
                .unwrap_or("enumerator")
                .to_string();
        }
        "preproc_def" | "preproc_function_def" => return named(src, "#define", node),
        "preproc_call" => {
            return match (
                field_text(src, node, "directive"),
                field_text(src, node, "argument"),
            ) {
                (Some(directive), Some(argument)) => format!("{directive} {argument}"),
                (Some(directive), None) => directive.to_string(),
                _ => first_line(src, node).to_string(),
            };
        }
        "preproc_if" | "preproc_ifdef" | "preproc_elif" | "preproc_elifdef" | "preproc_else" => {
            return first_line(src, node).to_string();
        }
        _ => {}
    }
    match declarator_name(src, node) {
        Some(name) => name.to_string(),
        None => node.kind().to_string(),
    }
}

fn split(node: Node) -> Option<(usize, Node)> {
    match node.kind() {
        // A declaration's container is the struct/union/enum/class
        // specifier inside it.
        "declaration" | "field_declaration" | "type_definition" | "type_specifier"
        | "struct_specifier" | "union_specifier" | "enum_specifier" | "class_specifier" => {
            let body = specifier_body(node)?;
            Some((body.start_byte(), body))
        }
        "namespace_definition" => {
            let body = node.child_by_field_name("body")?;
            Some((body.start_byte(), body))
        }
        "linkage_specification" => {
            let body = node.child_by_field_name("body")?;
            if body.kind() != "declaration_list" {
                return None;
            }
            Some((body.start_byte(), body))
        }
        "preproc_if" | "preproc_ifdef" | "preproc_elif" | "preproc_elifdef" | "preproc_else" => {
            Some((directive_end(node), node))
        }
        _ => None,
    }
}

fn symbol(src: &str, node: Node) -> Option<String> {
    match node.kind() {
        "declaration" | "field_declaration" | "type_specifier" => {
            let name = specifier_name(src, node).or_else(|| declarator_name(src, node));
            name.map(str::to_string)
        }
        "type_definition" | "function_definition" => declarator_name(src, node).map(str::to_string),
        "linkage_specification" => None,
        // No `preproc_*` names a symbol: an include guard must not become
        // a segment of the guarded items' symbol paths.
        kind if kind.starts_with("preproc_") => None,
        _ => field_text(src, node, "name").map(str::to_string),
    }
}

/// `word name` when `node` has a `name` field, `word` alone otherwise.
fn named(src: &str, word: &str, node: Node) -> String {
    match field_text(src, node, "name") {
        Some(name) => format!("{word} {name}"),
        None => word.to_string(),
    }
}

/// The `struct`/`union`/`enum`/`class` keyword plus the specifier's `name`
/// for a declaration or specifier node; `None` when the node names no
/// specifier.
fn specifier_label<'a>(src: &'a str, node: Node<'a>) -> Option<String> {
    let specifier = specifier(node)?;
    let keyword = specifier_keyword(src, specifier);
    field_text(src, specifier, "name").map(|name| format!("{keyword} {name}"))
}

/// The `name` of the struct/union/enum/class specifier inside a
/// declaration, or of the specifier node itself.
fn specifier_name<'a>(src: &'a str, node: Node<'a>) -> Option<&'a str> {
    specifier(node).and_then(|specifier| field_text(src, specifier, "name"))
}

/// The struct/union/enum/class specifier of a declaration (`type` field)
/// or the node itself when it is a specifier.
fn specifier<'a>(node: Node<'a>) -> Option<Node<'a>> {
    let candidate = if is_specifier(node.kind()) {
        node
    } else {
        node.child_by_field_name("type")?
    };
    is_specifier(candidate.kind()).then_some(candidate)
}

fn is_specifier(kind: &str) -> bool {
    matches!(
        kind,
        "struct_specifier" | "union_specifier" | "enum_specifier" | "class_specifier"
    )
}

/// The specifier's `body` field (`field_declaration_list` or
/// `enumerator_list`); `None` for forward declarations.
fn specifier_body<'a>(node: Node<'a>) -> Option<Node<'a>> {
    specifier(node).and_then(|specifier| specifier.child_by_field_name("body"))
}

/// The keyword that opens a specifier: `struct`, `union`, `enum`, `class`
/// or `enum class`.
fn specifier_keyword<'a>(src: &'a str, specifier: Node) -> &'a str {
    let mut cursor = specifier.walk();
    let mut children = specifier
        .children(&mut cursor)
        .filter(|child| !child.is_named());
    let first = children
        .next()
        .map(|child| node_text(src, child))
        .unwrap_or("struct");
    if first == "enum"
        && children
            .next()
            .is_some_and(|child| node_text(src, child) == "class")
    {
        return "enum class";
    }
    first
}

/// The identifier a declarator chain names: digs through `declarator`
/// fields and wrappers to the terminal name.
fn declarator_name<'a>(src: &'a str, node: Node<'a>) -> Option<&'a str> {
    let mut node = node.child_by_field_name("declarator")?;
    loop {
        if let Some(inner) = node.child_by_field_name("declarator") {
            node = inner;
            continue;
        }
        match node.kind() {
            "identifier"
            | "field_identifier"
            | "type_identifier"
            | "destructor_name"
            | "operator_name"
            | "operator_cast"
            | "qualified_identifier" => return Some(node_text(src, node)),
            "parenthesized_declarator" => node = node.named_child(0)?,
            _ => return None,
        }
    }
}

/// Where a `preproc_*` directive line ends: after the `condition`/`name`
/// field, or after the `#else` token.
fn directive_end(node: Node) -> usize {
    if let Some(inner) = node
        .child_by_field_name("condition")
        .or_else(|| node.child_by_field_name("name"))
    {
        return inner.end_byte();
    }
    node.child(0)
        .map_or(node.start_byte(), |child| child.end_byte())
}

/// `node`'s text up to its first newline.
fn first_line<'a>(src: &'a str, node: Node) -> &'a str {
    let text = node_text(src, node);
    text.split_once('\n').map_or(text, |(line, _)| line)
}
