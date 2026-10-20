//! Chunking of a repository's files.
//!
//! A [`FileChunker`] turns one file (path and source) into [`Chunk`]s; the
//! file's [`Language`], detected from its extension, picks the implementation:
//!
//! - [`Chunker`] walks a tree-sitter syntax tree: one chunk per item together
//!   with the comments and attributes directly above it, module docs on their
//!   own, oversized impl / trait / mod bodies split into members, oversized
//!   items split on line boundaries, and small same-scope neighbours packed
//!   into one chunk. It is generic over [`Grammar`]; a language is a new
//!   `Grammar` impl and nothing else.
//! - [`DocChunker`] cuts documentation into heading sections.
//! - [`ConfigChunker`] cuts TOML, YAML and JSON by top-level table or key.

#[cfg(any(feature = "lang-c", feature = "lang-cpp"))]
mod c;
mod config;
mod doc;
#[cfg(any(feature = "lang-javascript", feature = "lang-typescript"))]
mod ecma;
#[cfg(feature = "lang-go")]
mod go;
mod language;
#[cfg(feature = "lang-python")]
mod python;
#[cfg(feature = "lang-rust")]
mod rust;
#[cfg(feature = "lang-swift")]
mod swift;
mod units;

#[cfg(feature = "lang-c")]
pub use c::C;
#[cfg(feature = "lang-cpp")]
pub use c::Cpp;
pub use config::ConfigChunker;
pub use doc::DocChunker;
#[cfg(feature = "lang-javascript")]
pub use ecma::JavaScript;
#[cfg(feature = "lang-typescript")]
pub use ecma::{Tsx, TypeScript};
#[cfg(feature = "lang-go")]
pub use go::Go;
pub use language::{ConfigFormat, DocFormat, Language};
#[cfg(feature = "lang-python")]
pub use python::Python;
#[cfg(feature = "lang-rust")]
pub use rust::Rust;
#[cfg(feature = "lang-swift")]
pub use swift::Swift;

use std::borrow::Cow;
use std::ops::{Range, RangeInclusive};

use smallvec::SmallVec;
use tree_sitter::{Language as TsLanguage, Node, Parser};

use units::{CtxFrame, Emitter, MAX_BYTES, Span, ctx_symbol, join_symbol};

/// Everything language-specific the chunking algorithm needs.
pub trait Grammar {
    /// The tree-sitter language of this grammar.
    fn language(&self) -> TsLanguage;

    /// `true` for node kinds collected as trivia in front of an item:
    /// comments and attributes.
    fn is_leading(&self, node: Node) -> bool;

    /// `true` for nodes that produce no chunk and break the trivia run in
    /// front of an item, such as import declarations.
    fn is_skipped(&self, node: Node) -> bool;

    /// `true` when a leading-trivia node's text is an inner (module) doc
    /// comment.
    fn is_inner_doc(&self, text: &str) -> bool;

    /// The label a chunk's title ends with for this item, such as `fn insert`
    /// or `impl Trait for Environment`.
    fn label(&self, src: &str, node: Node) -> String;

    /// The node an item reports: an inner declaration for wrapper items
    /// such as `export const`/`declare`/`@decorated`/`template<>` blocks;
    /// `node` itself by default. The emitted byte span stays the outer
    /// node's, so the wrapper text is part of the chunk.
    fn subject<'a>(&self, node: Node<'a>) -> Node<'a> {
        node
    }

    /// An oversized container's split: the byte offset where the header
    /// unit ends and the body whose children become member items. `None`
    /// for kinds that are emitted whole.
    fn split<'a>(&self, node: Node<'a>) -> Option<(usize, Node<'a>)>;

    /// The segment this item contributes to a symbol path: its own name
    /// (`Environment` for `impl Environment`, `insert` for `fn insert`), or
    /// `None` when it has none.
    fn symbol(&self, src: &str, node: Node) -> Option<String>;
}

/// The stable tree-sitter kind id of a node (`Node::kind_id()`); kind names
/// are looked up through the chunker's `Language`, so ids stay valid when a
/// language is loaded dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeKind(u16);

impl NodeKind {
    /// The kind of `node`.
    pub fn of(node: Node<'_>) -> Self {
        Self(node.kind_id())
    }

    /// The raw tree-sitter kind id.
    pub fn id(self) -> u16 {
        self.0
    }
}

/// The kind of a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// A single item, or one line-split part of an oversized item.
    Item(NodeKind),
    /// The header of a container that was split into members.
    Header(NodeKind),
    /// Module docs (`//!`, `/*!`).
    ModuleDocs,
    /// A packed group of small same-scope items or configuration entries.
    Packed,
    /// A documentation section, or one part of an oversized one.
    Doc,
    /// A configuration table or key, a split container's header, or one
    /// line-split part of an oversized value.
    Config,
}

/// The kind of one [`Member`]: what `--kind` filters on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberKind {
    /// A syntax node of the chunker's tree-sitter language.
    Node(NodeKind),
    /// Documentation text.
    Doc,
    /// A configuration entry.
    Config,
}

/// One syntactic item's run of bytes inside a chunk.
#[derive(Debug)]
pub struct Member {
    /// The member's byte range in the file.
    pub bytes: Range<usize>,
    /// The kind of the member.
    pub kind: MemberKind,
    /// The `::`-joined path of the enclosing containers plus the item's name
    /// (`Environment::insert` for `fn insert` inside `impl Environment`); for
    /// documentation the `›`-joined heading path, for configuration the
    /// `.`-joined key path.
    pub symbol: String,
}

impl ChunkKind {
    /// The kind string of a kind that names no tree-sitter node: `docs` for
    /// module docs, `packed`, `doc` and `config`; `None` for the node-named
    /// `Item` and `Header`.
    pub const fn static_name(self) -> Option<&'static str> {
        match self {
            Self::Item(_) | Self::Header(_) => None,
            Self::ModuleDocs => Some("docs"),
            Self::Packed => Some("packed"),
            Self::Doc => Some("doc"),
            Self::Config => Some("config"),
        }
    }
}

impl MemberKind {
    /// The kind string of a member that is no tree-sitter node: `doc` or
    /// `config`; `None` for [`MemberKind::Node`].
    pub fn static_name(self) -> Option<&'static str> {
        match self {
            Self::Node(_) => None,
            Self::Doc => Some("doc"),
            Self::Config => Some("config"),
        }
    }
}

/// A chunk of a source file: a title, a kind, a line range, and the byte
/// ranges of its members.
#[derive(Debug)]
pub struct Chunk {
    /// `path › context › label`-style title.
    pub title: String,
    /// The kind of the item, a split container's header, module docs, a
    /// documentation section, a configuration entry, or a packed group.
    pub kind: ChunkKind,
    /// First and last line of the chunk, 1-based.
    pub lines: RangeInclusive<u32>,
    /// One member for a single item, a header, module docs or a line-split
    /// part; one per packed item, in order, for a packed chunk.
    pub members: SmallVec<[Member; 1]>,
}

impl Chunk {
    /// Appends the chunk's text — its members' slices joined with `"\n"` —
    /// to `out`.
    ///
    /// # Panics
    ///
    /// If `src` is not the file this chunk was produced from.
    pub fn write_text(&self, src: &str, out: &mut String) {
        let mut first = true;
        for member in &self.members {
            if !first {
                out.push('\n');
            }
            first = false;
            out.push_str(&src[member.bytes.clone()]);
        }
    }
}

/// Chunks source files of one language, reusing the parser across files.
pub struct Chunker<G: Grammar> {
    grammar: G,
    parser: Parser,
    language: TsLanguage,
}

impl<G: Grammar> Chunker<G> {
    /// A chunker for `grammar`'s language.
    ///
    /// # Panics
    ///
    /// If the grammar's tree-sitter language is ABI-incompatible with the
    /// linked runtime: a build misconfiguration, not a runtime condition.
    pub fn new(grammar: G) -> Self {
        let language = grammar.language();
        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .expect("grammar language must match the tree-sitter runtime ABI");
        Self {
            grammar,
            parser,
            language,
        }
    }

    /// The tree-sitter kind name of `kind`, such as `function_item`.
    ///
    /// # Panics
    ///
    /// If `kind` did not come from this chunker's language.
    pub fn kind_name(&self, kind: NodeKind) -> &str {
        self.language
            .node_kind_for_id(kind.0)
            .expect("a NodeKind is always valid for the chunker's language")
    }
}

/// Chunking failed because the file is not valid in its format.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A TOML file that does not parse.
    #[error("invalid TOML: {0}")]
    Toml(#[from] toml_edit::TomlError),
    /// A YAML file that does not parse.
    #[error("invalid YAML: {0}")]
    Yaml(#[from] saphyr_parser::ScanError),
    /// A JSON file that does not parse.
    #[error("invalid JSON: {0}")]
    Json(#[from] jsonc_parser::errors::ParseError),
}

/// Chunks whole files of one language: path and source in, chunks out. The
/// implementation is picked by the file's [`Language`].
pub trait FileChunker {
    /// Chunks `src`, the text of the file at `path` relative to the
    /// repository root. `src` is the file's content; chunks never copy it.
    ///
    /// # Errors
    ///
    /// When `src` is not valid in the chunker's format.
    fn chunk_file(&mut self, path: &str, src: &str) -> Result<Vec<Chunk>, Error>;

    /// The kind string of a chunk this chunker produced.
    ///
    /// # Panics
    ///
    /// If `kind` did not come from this chunker.
    fn chunk_kind_name(&self, kind: ChunkKind) -> Cow<'_, str>;

    /// The kind string of a member of a chunk this chunker produced.
    ///
    /// # Panics
    ///
    /// If `kind` did not come from this chunker.
    fn member_kind_name(&self, kind: MemberKind) -> Cow<'_, str>;
}

impl<G: Grammar> FileChunker for Chunker<G> {
    fn chunk_file(&mut self, path: &str, src: &str) -> Result<Vec<Chunk>, Error> {
        let tree = self
            .parser
            .parse(src, None)
            .expect("tree-sitter parsing cannot fail: no timeout is set");
        let mut walker = Walker {
            grammar: &self.grammar,
            src,
            out: Emitter::new(path, src),
        };
        walker.walk_children(tree.root_node(), &mut Vec::new());
        Ok(walker.out.finish())
    }

    /// The node kind, `<kind>_header` for a split container's header, `docs`
    /// for module docs, `packed` for a group.
    fn chunk_kind_name(&self, kind: ChunkKind) -> Cow<'_, str> {
        match kind {
            ChunkKind::Item(kind) => Cow::Borrowed(self.kind_name(kind)),
            ChunkKind::Header(kind) => {
                let mut name = String::from(self.kind_name(kind));
                name.push_str("_header");
                Cow::Owned(name)
            }
            ChunkKind::ModuleDocs | ChunkKind::Packed => Cow::Borrowed(
                kind.static_name()
                    .expect("module docs and packed groups have static names"),
            ),
            ChunkKind::Doc | ChunkKind::Config => {
                panic!("a tree-sitter chunker produces no {kind:?} chunks")
            }
        }
    }

    fn member_kind_name(&self, kind: MemberKind) -> Cow<'_, str> {
        match kind {
            MemberKind::Node(kind) => Cow::Borrowed(self.kind_name(kind)),
            MemberKind::Doc | MemberKind::Config => {
                panic!("a tree-sitter chunker produces no {kind:?} members")
            }
        }
    }
}

struct Walker<'a, G> {
    grammar: &'a G,
    src: &'a str,
    out: Emitter<'a>,
}

impl<G: Grammar> Walker<'_, G> {
    /// Walks `parent`'s children — named and unnamed — attaching leading
    /// trivia to the item that follows it.
    fn walk_children(&mut self, parent: Node, ctx: &mut Vec<CtxFrame>) {
        let mut pending: Vec<Node> = Vec::new();
        let mut cursor = parent.walk();
        for node in parent.children(&mut cursor) {
            if self.grammar.is_leading(node) {
                pending.push(node);
                continue;
            }
            if self.grammar.is_skipped(node) || !node.is_named() {
                self.flush_docs(&pending, ctx);
                pending.clear();
                continue;
            }
            // Inner docs inside the pending run are emitted first; the
            // remaining trivia stays attached to the item.
            let mut inner = Vec::new();
            let mut rest = Vec::new();
            for p in pending.drain(..) {
                if self.grammar.is_inner_doc(&self.src[p.byte_range()]) {
                    inner.push(p);
                } else {
                    rest.push(p);
                }
            }
            self.flush_docs(&inner, ctx);
            let first = rest.first().unwrap_or(&node);
            self.item(node, first.start_byte(), first.start_position().row, ctx);
        }
        self.flush_docs(&pending, ctx);
    }

    /// Emits any inner docs in `pending` as one `module docs` chunk spanning
    /// the first doc's start to the last doc's end.
    fn flush_docs(&mut self, pending: &[Node], ctx: &[CtxFrame]) {
        let mut first = None;
        let mut end = 0;
        for node in pending {
            if self.grammar.is_inner_doc(&self.src[node.byte_range()]) {
                first.get_or_insert(*node);
                end = node.end_byte();
            }
        }
        if let Some(first) = first {
            self.out.emit(
                Span {
                    bytes: first.start_byte()..end,
                    row: first.start_position().row,
                },
                ctx,
                "module docs",
                ChunkKind::ModuleDocs,
                MemberKind::Node(NodeKind::of(first)),
                ctx_symbol(ctx),
            );
        }
    }

    /// Emits `node` (with `start` possibly earlier, at its leading trivia),
    /// splitting an oversized container into a header plus member items.
    fn item(&mut self, node: Node, start: usize, start_row: usize, ctx: &mut Vec<CtxFrame>) {
        let subject = self.grammar.subject(node);
        let label = self.grammar.label(self.src, subject);
        let split = (node.end_byte() - start > MAX_BYTES)
            .then(|| self.grammar.split(subject))
            .flatten();
        let kind = NodeKind::of(subject);
        if let Some((header_end, body)) = split {
            let symbol = self.grammar.symbol(self.src, subject);
            if header_end > start {
                let header_symbol = join_symbol(ctx, symbol.as_deref(), "::");
                self.out.emit(
                    Span {
                        bytes: start..header_end,
                        row: start_row,
                    },
                    ctx,
                    &label,
                    ChunkKind::Header(kind),
                    MemberKind::Node(kind),
                    header_symbol,
                );
            }
            let scope = self.out.new_scope();
            ctx.push(CtxFrame {
                label,
                symbol,
                scope,
            });
            self.walk_children(body, ctx);
            ctx.pop();
            return;
        }
        let symbol = join_symbol(ctx, self.grammar.symbol(self.src, subject).as_deref(), "::");
        self.out.emit(
            Span {
                bytes: start..node.end_byte(),
                row: start_row,
            },
            ctx,
            &label,
            ChunkKind::Item(kind),
            MemberKind::Node(kind),
            symbol,
        );
    }
}

#[cfg(all(test, feature = "lang-rust"))]
mod tests;
