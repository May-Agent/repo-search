//! Syntax-aware chunking: one chunk per item together with the comments and
//! attributes directly above it, module docs on their own, oversized impl /
//! trait / mod bodies split into members, oversized items split on line
//! boundaries, and small same-scope neighbours packed into one chunk.
//!
//! The algorithm is generic over [`Grammar`]; a language is a new `Grammar`
//! impl and nothing else.

mod rust;

pub use rust::Rust;

use std::borrow::Cow;
use std::ops::{Range, RangeInclusive};

use smallvec::SmallVec;
use tree_sitter::{Language, Node, Parser};

/// Largest emitted unit before an item is split, in bytes.
const MAX_BYTES: usize = 2400;
/// Largest byte size of a unit that can join a packed group.
const SMALL_BYTES: usize = 400;
/// Largest combined byte size of a packed group.
const PACK_BYTES: usize = 1200;

/// Everything language-specific the chunking algorithm needs.
pub trait Grammar {
    /// The tree-sitter language of this grammar.
    fn language(&self) -> Language;

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

    /// The body of a container that is split into members when it exceeds
    /// the size limit; `None` for kinds that are emitted whole.
    fn split_body<'a>(&self, node: Node<'a>) -> Option<Node<'a>>;

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
    /// A packed group of small same-scope items.
    Packed,
}

/// One syntactic item's run of bytes inside a chunk.
#[derive(Debug)]
pub struct Member {
    /// The member's byte range in the file.
    pub bytes: Range<usize>,
    /// The tree-sitter kind of the member's node.
    pub kind: NodeKind,
    /// The `::`-joined path of the enclosing containers plus the item's name
    /// (`Environment::insert` for `fn insert` inside `impl Environment`).
    pub symbol: String,
}

/// A chunk of a source file: a title, a kind, a line range, and the byte
/// ranges of its members.
#[derive(Debug)]
pub struct Chunk {
    /// `path › context › label`-style title.
    pub title: String,
    /// The tree-sitter kind of the item, a split container's header, module
    /// docs, or a packed group.
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
    language: Language,
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

    /// The kind string of a chunk: the node kind, `<kind>_header` for a
    /// split container's header, `docs` for module docs, `packed` for a
    /// group.
    pub fn chunk_kind_name(&self, kind: ChunkKind) -> Cow<'_, str> {
        match kind {
            ChunkKind::Item(kind) => Cow::Borrowed(self.kind_name(kind)),
            ChunkKind::Header(kind) => {
                let mut name = String::from(self.kind_name(kind));
                name.push_str("_header");
                Cow::Owned(name)
            }
            ChunkKind::ModuleDocs => Cow::Borrowed("docs"),
            ChunkKind::Packed => Cow::Borrowed("packed"),
        }
    }

    /// Chunks `src`, the text of the file at `path` relative to the
    /// repository root. `src` is the file's content; chunks never copy it.
    pub fn chunk_file(&mut self, path: &str, src: &str) -> Vec<Chunk> {
        let tree = self
            .parser
            .parse(src, None)
            .expect("tree-sitter parsing cannot fail: no timeout is set");
        let mut walker = Walker {
            grammar: &self.grammar,
            src,
            path,
            units: Vec::new(),
            next_scope: 0,
        };
        walker.walk_children(tree.root_node(), &mut Vec::new());
        pack(walker.units)
    }
}

/// A context frame: the label of an enclosing split container and the symbol
/// segment it contributes.
struct CtxFrame {
    label: String,
    symbol: Option<String>,
    /// Identifies this split container frame; units pack only within one
    /// scope. Two sibling containers with the same label are always
    /// separated by the second one's header unit, so a scope id is
    /// equivalent to comparing label paths.
    scope: u32,
}

/// An emitted unit, before packing.
struct Unit {
    /// The innermost enclosing split container's scope id (0 at file level);
    /// units pack only within one scope.
    scope: u32,
    lines: RangeInclusive<u32>,
    title: String,
    /// The chunk kind this unit produces when emitted alone.
    kind: ChunkKind,
    /// The kind of this unit's member entry.
    member: NodeKind,
    bytes: Range<usize>,
    symbol: String,
}

/// A byte range together with the 0-based line it starts on.
struct Span {
    bytes: Range<usize>,
    row: usize,
}

struct Walker<'a, G> {
    grammar: &'a G,
    src: &'a str,
    path: &'a str,
    units: Vec<Unit>,
    /// The scope id given to the next split container frame.
    next_scope: u32,
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
            self.emit(
                Span {
                    bytes: first.start_byte()..end,
                    row: first.start_position().row,
                },
                ctx,
                "module docs",
                ChunkKind::ModuleDocs,
                NodeKind::of(first),
                ctx_symbol(ctx),
            );
        }
    }

    /// Emits `node` (with `start` possibly earlier, at its leading trivia),
    /// splitting an oversized container into a header plus member items.
    fn item(&mut self, node: Node, start: usize, start_row: usize, ctx: &mut Vec<CtxFrame>) {
        let label = self.grammar.label(self.src, node);
        let split = (node.end_byte() - start > MAX_BYTES)
            .then(|| self.grammar.split_body(node))
            .flatten();
        let kind = NodeKind::of(node);
        if let Some(body) = split {
            let symbol = self.grammar.symbol(self.src, node);
            if body.start_byte() > start {
                let header_symbol = join_symbol(ctx, symbol.as_deref());
                self.emit(
                    Span {
                        bytes: start..body.start_byte(),
                        row: start_row,
                    },
                    ctx,
                    &label,
                    ChunkKind::Header(kind),
                    kind,
                    header_symbol,
                );
            }
            self.next_scope += 1;
            ctx.push(CtxFrame {
                label,
                symbol,
                scope: self.next_scope,
            });
            self.walk_children(body, ctx);
            ctx.pop();
            return;
        }
        let symbol = join_symbol(ctx, self.grammar.symbol(self.src, node).as_deref());
        self.emit(
            Span {
                bytes: start..node.end_byte(),
                row: start_row,
            },
            ctx,
            &label,
            ChunkKind::Item(kind),
            kind,
            symbol,
        );
    }

    /// Emits `span` as one unit, line-splitting it when it exceeds the size
    /// limit.
    fn emit(
        &mut self,
        span: Span,
        ctx: &[CtxFrame],
        label: &str,
        kind: ChunkKind,
        member: NodeKind,
        symbol: String,
    ) {
        let Span { bytes, row } = span;
        let start = bytes.start;
        let body = &self.src[bytes.clone()];
        let start_line = row as u32 + 1;
        let mut title = String::with_capacity(self.path.len() + label.len() + 32);
        title.push_str(self.path);
        for frame in ctx {
            title.push_str(" › ");
            title.push_str(&frame.label);
        }
        title.push_str(" › ");
        title.push_str(label);

        let parts: Vec<Range<usize>> = if body.len() > MAX_BYTES {
            line_split(body, MAX_BYTES)
        } else {
            std::iter::once(0..body.len()).collect()
        };
        let part_total = parts.len();
        let mut line = start_line;
        for (index, part) in parts.into_iter().enumerate() {
            let piece = &body[part.clone()];
            let newlines = piece.as_bytes().iter().filter(|&&b| b == b'\n').count() as u32;
            let end_line = line + newlines.saturating_sub(u32::from(piece.ends_with('\n')));
            let part_title = if part_total == 1 {
                title.clone()
            } else {
                format!("{title} (part {}/{part_total})", index + 1)
            };
            self.units.push(Unit {
                scope: ctx.last().map_or(0, |frame| frame.scope),
                lines: line..=end_line,
                title: part_title,
                kind,
                member,
                bytes: (start + part.start)..(start + part.end),
                symbol: symbol.clone(),
            });
            line += newlines;
        }
    }
}

/// The `::`-joined path of `ctx`'s symbol segments plus `item`.
fn join_symbol(ctx: &[CtxFrame], item: Option<&str>) -> String {
    let mut symbol = String::new();
    let mut push = |segment: &str| {
        if !symbol.is_empty() {
            symbol.push_str("::");
        }
        symbol.push_str(segment);
    };
    for frame in ctx {
        if let Some(segment) = &frame.symbol {
            push(segment);
        }
    }
    if let Some(item) = item {
        push(item);
    }
    symbol
}

/// The `::`-joined path of `ctx`'s symbol segments alone.
fn ctx_symbol(ctx: &[CtxFrame]) -> String {
    join_symbol(ctx, None)
}

/// Byte ranges of `text`'s pieces after splitting on Python `splitlines`
/// boundaries, accumulating while `size + line <= limit` where sizes count
/// Unicode code points.
fn line_split(text: &str, limit: usize) -> Vec<Range<usize>> {
    let mut parts = Vec::new();
    let mut cur_start = 0;
    let mut size = 0;
    let mut pos = 0;
    for line in python_lines(text) {
        let len = code_points(&text[line.clone()]);
        if size > 0 && size + len > limit {
            parts.push(cur_start..line.start);
            cur_start = line.start;
            size = 0;
        }
        size += len;
        pos = line.end;
    }
    if pos > cur_start {
        parts.push(cur_start..pos);
    }
    parts
}

/// The number of Unicode code points: the non-continuation UTF-8 bytes.
fn code_points(text: &str) -> usize {
    text.as_bytes()
        .iter()
        .filter(|&&b| b & 0xC0 != 0x80)
        .count()
}

/// Byte ranges of `text`'s lines including their terminators, on Python
/// `str.splitlines(keepends=True)` boundaries: `\n`, `\r\n`, `\r`, `\x0b`,
/// `\x0c`, `\x1c`, `\x1d`, `\x1e`, `\u{85}`, `\u{2028}` and `\u{2029}`.
fn python_lines(text: &str) -> Vec<Range<usize>> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let end = match bytes[i] {
            b'\n' => i + 1,
            b'\r' if bytes.get(i + 1) == Some(&b'\n') => i + 2,
            b'\r' | 0x0B | 0x0C | 0x1C | 0x1D | 0x1E => i + 1,
            0xC2 if bytes.get(i + 1) == Some(&0x85) => i + 2,
            0xE2 if bytes.get(i + 1) == Some(&0x80) => match bytes.get(i + 2) {
                Some(0xA8 | 0xA9) => i + 3,
                _ => {
                    i += 1;
                    continue;
                }
            },
            _ => {
                i += 1;
                continue;
            }
        };
        lines.push(start..end);
        start = end;
        i = end;
    }
    if start < bytes.len() {
        lines.push(start..bytes.len());
    }
    lines
}

/// Packs the emitted units: a unit under [`SMALL_BYTES`] joins the current
/// group when the scope matches and the group stays under [`PACK_BYTES`];
/// larger groups become one `packed` chunk.
fn pack(units: Vec<Unit>) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut group: Vec<Unit> = Vec::new();
    let mut gscope = 0;
    let mut gbytes = 0;
    for unit in units {
        let size = unit.bytes.len();
        if size < SMALL_BYTES
            && !group.is_empty()
            && unit.scope == gscope
            && gbytes + size <= PACK_BYTES
        {
            gbytes += size;
            group.push(unit);
            continue;
        }
        flush_group(&mut out, &mut group);
        gbytes = 0;
        if size < SMALL_BYTES {
            gscope = unit.scope;
            gbytes = size;
            group.push(unit);
        } else {
            out.push(single(unit));
        }
    }
    flush_group(&mut out, &mut group);
    out
}

fn flush_group(out: &mut Vec<Chunk>, group: &mut Vec<Unit>) {
    if group.len() == 1 {
        out.push(single(group.remove(0)));
        return;
    }
    if group.len() > 1 {
        let title = packed_title(group);
        let lines = *group[0].lines.start()..=*group[group.len() - 1].lines.end();
        let members = group
            .iter()
            .map(|unit| Member {
                bytes: unit.bytes.clone(),
                kind: unit.member,
                symbol: unit.symbol.clone(),
            })
            .collect();
        out.push(Chunk {
            title,
            kind: ChunkKind::Packed,
            lines,
            members,
        });
        group.clear();
    }
}

/// `path › a, b, c`: the first title's prefix plus the members' last title
/// segments joined with `, `.
fn packed_title(group: &[Unit]) -> String {
    let mut title = String::new();
    for (index, unit) in group.iter().enumerate() {
        let (prefix, last) = unit
            .title
            .rsplit_once(" › ")
            .expect("a chunk title always contains ' › '");
        if index == 0 {
            title.push_str(prefix);
        }
        title.push_str(if index == 0 { " › " } else { ", " });
        title.push_str(last);
    }
    title
}

fn single(unit: Unit) -> Chunk {
    let mut members = SmallVec::new();
    members.push(Member {
        bytes: unit.bytes,
        kind: unit.member,
        symbol: unit.symbol,
    });
    Chunk {
        title: unit.title,
        kind: unit.kind,
        lines: unit.lines,
        members,
    }
}

#[cfg(test)]
mod tests;
