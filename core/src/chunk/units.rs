//! The size rules and unit machinery shared by every file chunker: units are
//! emitted under context titles, oversized ones are split on line boundaries,
//! and small same-scope neighbours are packed into one chunk.

use std::ops::{Range, RangeInclusive};

use smallvec::SmallVec;

use super::{Chunk, ChunkKind, Member, MemberKind};

/// Largest emitted unit before an item is split, in bytes.
pub(super) const MAX_BYTES: usize = 2400;
/// Largest byte size of a unit that can join a packed group.
const SMALL_BYTES: usize = 400;
/// Largest combined byte size of a packed group.
const PACK_BYTES: usize = 1200;

/// A context frame: the label of an enclosing split container and the symbol
/// segment it contributes.
pub(super) struct CtxFrame {
    pub label: String,
    pub symbol: Option<String>,
    /// Identifies this split container frame; units pack only within one
    /// scope. Two sibling containers with the same label are always
    /// separated by the second one's header unit, so a scope id is
    /// equivalent to comparing label paths.
    pub scope: u32,
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
    member: MemberKind,
    bytes: Range<usize>,
    symbol: String,
}

/// A byte range together with the 0-based line it starts on.
pub(super) struct Span {
    pub bytes: Range<usize>,
    pub row: usize,
}

/// The `separator`-joined path of `ctx`'s symbol segments plus `item`.
pub(super) fn join_symbol(ctx: &[CtxFrame], item: Option<&str>, separator: &str) -> String {
    let mut symbol = String::new();
    let mut push = |segment: &str| {
        if !symbol.is_empty() {
            symbol.push_str(separator);
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
pub(super) fn ctx_symbol(ctx: &[CtxFrame]) -> String {
    join_symbol(ctx, None, "::")
}

/// Byte ranges of `text`'s pieces after splitting on Python `splitlines`
/// boundaries, accumulating while `size + line <= limit` where sizes count
/// Unicode code points.
pub(super) fn line_split(text: &str, limit: usize) -> Vec<Range<usize>> {
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

/// Byte ranges of `text`'s pieces after splitting on Python `splitlines`
/// boundaries, accumulating while the byte size stays within `limit`. A line
/// over the limit on its own is cut on char boundaries; a piece is at least
/// one char, so a limit below a char's width is exceeded by that char.
pub(super) fn line_split_bytes(text: &str, limit: usize) -> Vec<Range<usize>> {
    let mut parts = Vec::new();
    let mut cur_start = 0;
    let mut pos = 0;
    for line in python_lines(text) {
        if line.len() > limit {
            if pos > cur_start {
                parts.push(cur_start..pos);
            }
            let mut start = line.start;
            while line.end - start > limit {
                let mut cut = start + limit;
                while !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                if cut == start {
                    cut = start + 1;
                    while !text.is_char_boundary(cut) {
                        cut += 1;
                    }
                }
                parts.push(start..cut);
                start = cut;
            }
            cur_start = start;
            pos = line.end;
            continue;
        }
        if pos > cur_start && line.end - cur_start > limit {
            parts.push(cur_start..pos);
            cur_start = line.start;
        }
        pos = line.end;
    }
    if pos > cur_start {
        parts.push(cur_start..pos);
    }
    parts
}

/// Runs `chunk` on `src` without a leading UTF-8 byte order mark and shifts
/// the members' byte ranges back, so they are offsets into `src` itself. The
/// mark sits on the first line, so line numbers are unchanged.
pub(super) fn without_bom(
    src: &str,
    chunk: impl FnOnce(&str) -> Result<Vec<Chunk>, super::Error>,
) -> Result<Vec<Chunk>, super::Error> {
    const BOM: char = '\u{feff}';
    let Some(rest) = src.strip_prefix(BOM) else {
        return chunk(src);
    };
    let mut chunks = chunk(rest)?;
    for member in chunks.iter_mut().flat_map(|chunk| chunk.members.iter_mut()) {
        member.bytes = member.bytes.start + BOM.len_utf8()..member.bytes.end + BOM.len_utf8();
    }
    Ok(chunks)
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
pub(super) fn python_lines(text: &str) -> Vec<Range<usize>> {
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

/// Collects the units of one file and packs them into chunks.
pub(super) struct Emitter<'a> {
    src: &'a str,
    path: &'a str,
    units: Vec<Unit>,
    /// The scope id given to the next split container frame.
    next_scope: u32,
}

impl<'a> Emitter<'a> {
    pub fn new(path: &'a str, src: &'a str) -> Self {
        Self {
            src,
            path,
            units: Vec::new(),
            next_scope: 0,
        }
    }

    /// A fresh scope id for a split container's frame.
    pub fn new_scope(&mut self) -> u32 {
        self.next_scope += 1;
        self.next_scope
    }

    /// Emits `span` as one unit under `path › ctx… › label`, line-splitting
    /// it when it exceeds the size limit.
    pub fn emit(
        &mut self,
        span: Span,
        ctx: &[CtxFrame],
        label: &str,
        kind: ChunkKind,
        member: MemberKind,
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

    /// Packs the emitted units into chunks.
    pub fn finish(self) -> Vec<Chunk> {
        pack(self.units)
    }
}

/// Byte offsets of a text's line starts, for byte → line lookups.
pub(super) struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(src: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(
            src.bytes()
                .enumerate()
                .filter(|&(_, b)| b == b'\n')
                .map(|(i, _)| i + 1),
        );
        Self { starts }
    }

    /// The 0-based line containing byte `offset`.
    pub fn row(&self, offset: usize) -> usize {
        self.starts.partition_point(|&start| start <= offset) - 1
    }

    /// The byte offset of the start of the line containing `offset`.
    pub fn line_start(&self, offset: usize) -> usize {
        self.starts[self.row(offset)]
    }
}

/// `range` with surrounding whitespace removed.
pub(super) fn trim_range(src: &str, range: Range<usize>) -> Range<usize> {
    let text = &src[range.clone()];
    let start = range.start + (text.len() - text.trim_start().len());
    let end = range.end - (text.len() - text.trim_end().len());
    start.min(end)..end
}

/// Node-text helpers shared by the tree-sitter grammar implementations;
/// each is compiled only when a grammar using it exists.
#[cfg(any(
    feature = "lang-c",
    feature = "lang-cpp",
    feature = "lang-go",
    feature = "lang-javascript",
    feature = "lang-python",
    feature = "lang-swift",
    feature = "lang-typescript"
))]
pub(super) mod text {
    use tree_sitter::Node;

    /// The text of `node` in `src`.
    pub fn node_text<'a>(src: &'a str, node: Node) -> &'a str {
        &src[node.byte_range()]
    }

    /// The text of `node`'s `field` child, or `None` when it has none.
    pub fn field_text<'a>(src: &'a str, node: Node, field: &str) -> Option<&'a str> {
        node.child_by_field_name(field)
            .map(|child| node_text(src, child))
    }

    /// The text of `node`'s first unnamed child: the keyword that opens the
    /// item, such as `var`, `func` or `class`.
    #[cfg(any(
        feature = "lang-javascript",
        feature = "lang-swift",
        feature = "lang-typescript"
    ))]
    pub fn first_token<'a>(src: &'a str, node: Node) -> Option<&'a str> {
        let mut cursor = node.walk();
        node.children(&mut cursor)
            .find(|child| !child.is_named())
            .map(|child| node_text(src, child))
    }
}
