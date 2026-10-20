//! The size rules and unit machinery shared by every file chunker: units are
//! emitted under context titles, oversized ones are split on line boundaries,
//! and small same-scope neighbours are packed into one chunk.

use std::ops::{Range, RangeInclusive};

use smallvec::SmallVec;

use crate::TokenCounter;

use super::{Chunk, ChunkKind, Member, MemberKind};

/// Largest emitted unit before an item is split, in bytes.
pub(super) const MAX_BYTES: usize = 2400;
/// Largest token count of a chunk's document — `title: <title> | text:
/// <text>` — under the model's [`TokenCounter`]. The check is on the final
/// document: the title including the worst-case `(part n/m)` suffix, the
/// signature prefix a later part carries, and packed groups.
pub(super) const MAX_TOKENS: usize = 512;
/// The `…` + newline that ends a split part's signature prefix.
pub(crate) const SIGNATURE_SUFFIX: &str = " …\n";
/// The longest first-sentence note a file's module docs put into titles.
const NOTE_CHARS: usize = 160;
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
    /// The unit's members: the piece alone, or the item's signature member
    /// plus the piece for a line-split part after the first.
    members: SmallVec<[Member; 2]>,
    /// Byte size of the unit's emitted text, signature prefix included.
    size: usize,
}

/// A byte range together with the 0-based line it starts on.
pub(super) struct Span {
    pub bytes: Range<usize>,
    pub row: usize,
}

/// What an emitted unit's member is made of.
pub(super) struct Item {
    /// The kind of this item's member entry.
    pub kind: MemberKind,
    /// Its symbol path.
    pub symbol: String,
    /// The byte range of the item's signature, emitted as a `signature`
    /// member on every part after the first — whitespace-collapsed and cut
    /// to the longest prefix that fits its quarter-cap budget.
    pub signature: Option<Range<usize>>,
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

/// Whitespace in Python's sense (`str.isspace()`): Unicode `White_Space`
/// plus the FS/GS/RS/US separators `\u{1c}`..`\u{1f}`, which Rust's
/// `char::is_whitespace` does not count.
pub(crate) fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `text` trimmed of Python whitespace on both ends.
fn python_trim(text: &str) -> &str {
    text.trim_matches(is_python_space)
}

/// `" ".join(text.split())`, the reference's signature whitespace collapse:
/// runs of Python whitespace become one ASCII space.
pub(crate) fn collapse(text: &str) -> String {
    let mut out = String::new();
    for word in text.split(is_python_space) {
        if word.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// Appends the first `cut` chars of `collapse(text)` to `out`, reading only
/// as much of `text` as that takes: a signature can be megabytes long while
/// its prefix is a few hundred chars.
fn collapse_into(text: &str, cut: usize, out: &mut String) {
    let mut chars = 0;
    let mut gap = false;
    for c in text.chars() {
        if is_python_space(c) {
            gap = chars > 0;
            continue;
        }
        if chars == cut {
            return;
        }
        if gap {
            out.push(' ');
            chars += 1;
            gap = false;
            if chars == cut {
                return;
            }
        }
        out.push(c);
        chars += 1;
    }
}

/// `line` minus its leading whitespace and at most one doc-comment marker
/// (`///`, `//!`, `/**`, `/*!`, `*/` or `*`), in the reference's
/// `first_sentence` alternation order.
fn doc_marker_strip(line: &str) -> &str {
    let line = line.trim_start_matches(is_python_space);
    for marker in ["///", "//!", "/**", "/*!", "*/", "*"] {
        if let Some(rest) = line.strip_prefix(marker) {
            return rest;
        }
    }
    line
}

/// The end of the first sentence of `text`: the first `.`, `!` or `?` after
/// at least one char and followed by whitespace or the end — the reference's
/// `(.+?[.!?])(\s|$)`. `None` when there is none.
fn sentence_end(text: &str) -> Option<usize> {
    for (i, c) in text.char_indices().skip(1) {
        if !matches!(c, '.' | '!' | '?') {
            continue;
        }
        let rest = &text[i + c.len_utf8()..];
        if rest.chars().next().is_none_or(is_python_space) {
            return Some(i + c.len_utf8());
        }
    }
    None
}

/// The first sentence of the inner-docs `texts` — comment markers stripped,
/// stopping at a blank line, a heading or a code fence, cut after the first
/// `.`/`!`/`?` followed by whitespace or the end; at most [`NOTE_CHARS`]
/// characters, cut on a word boundary and given ` …`; `›` becomes `>`.
/// `None` when there is no sentence. The reference's `first_sentence`,
/// applied to the file's module docs for every other chunk's title.
pub(super) fn first_sentence(docs: &str) -> Option<String> {
    let mut words: Vec<&str> = Vec::new();
    'outer: for line in python_lines(docs) {
        let line = python_trim(doc_marker_strip(&docs[line]));
        let line = python_trim(line.strip_suffix("*/").unwrap_or(line));
        if line.is_empty() {
            if !words.is_empty() {
                break 'outer;
            }
            continue;
        }
        if line.starts_with('#') || line.starts_with("```") {
            break 'outer;
        }
        words.push(line);
    }
    let mut note = words.join(" ");
    if note.is_empty() {
        return None;
    }
    if let Some(end) = sentence_end(&note) {
        note.truncate(end);
    }
    if code_points(&note) > NOTE_CHARS {
        let end = note
            .char_indices()
            .nth(NOTE_CHARS)
            .map_or(note.len(), |(i, _)| i);
        let head = note[..end]
            .rsplit_once(' ')
            .map_or(&note[..end], |(h, _)| h);
        note = format!("{head} …");
    }
    Some(note.replace('\u{203a}', ">"))
}

/// The token budget every chunk's embedded document fits, counted with the
/// model's [`TokenCounter`].
pub(super) struct Cap<'a> {
    tokens: &'a dyn TokenCounter,
    /// The worst title a part's document can carry: the `(part n/m)`
    /// suffix a split can add is already in it.
    worst_title: String,
}

impl<'a> Cap<'a> {
    /// A cap on the documents built over `worst_title`; `worst_title`
    /// already carries the `(part n/m)` suffix a split can add.
    pub(super) fn new(tokens: &'a dyn TokenCounter, worst_title: &str) -> Self {
        Self {
            tokens,
            worst_title: worst_title.to_owned(),
        }
    }

    /// Whether `prefix + text` fits the cap under this cap's title.
    pub(super) fn fits(&self, prefix: &str, text: &str) -> bool {
        self.tokens
            .fits(&self.worst_title, &[prefix, text], MAX_TOKENS)
    }
}

/// `body`'s pieces as byte ranges, after the reference's `fit_tokens`:
/// every part is the longest prefix of the remaining pieces that stays
/// under `max_bytes` and whose document — the cap's title, `first_prefix`
/// on the first part or `rest_prefix` on the rest, then the part's text —
/// fits the cap, found by binary search over the piece count so only the
/// probes are encoded. A line that does not fit on its own is halved on
/// char boundaries until its pieces do.
pub(super) fn token_split(
    cap: &Cap,
    first_prefix: &str,
    rest_prefix: &str,
    body: &str,
    max_bytes: usize,
) -> Vec<Range<usize>> {
    let mut pieces = Vec::new();
    for line in python_lines(body) {
        cut_line(body, line, cap, rest_prefix, &mut pieces);
    }
    let mut parts = Vec::new();
    let mut pos = 0;
    while pos < pieces.len() {
        let prefix = if parts.is_empty() {
            first_prefix
        } else {
            rest_prefix
        };
        let start = pieces[pos].start;
        // The byte limit alone bounds how many pieces the part can hold.
        let mut bound = pos + 1;
        let mut bytes = pieces[pos].len();
        while bound < pieces.len() && bytes + pieces[bound].len() <= max_bytes {
            bytes += pieces[bound].len();
            bound += 1;
        }
        // The greedy rule takes the first piece unconditionally; `lo` is a
        // piece count that fits (or is that piece), `hi` the first count
        // that cannot — one past the byte bound.
        let (mut lo, mut hi) = (1usize, bound - pos + 1);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if cap.fits(prefix, &body[start..pieces[pos + mid - 1].end]) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        parts.push(start..pieces[pos + lo - 1].end);
        pos += lo;
    }
    parts
}

/// The longest char prefix of the collapsed signature `sig` whose line —
/// `<prefix> …\n` — fits a quarter of [`MAX_TOKENS`], found by binary
/// search over the char count: the reference's `signature_prefix` budget.
fn signature_cut(sig: &str, tokens: &dyn TokenCounter) -> usize {
    const BUDGET: usize = MAX_TOKENS / 4;
    let fits = |chars: usize| {
        let end = sig.char_indices().nth(chars).map_or(sig.len(), |(i, _)| i);
        tokens.count(&format!("{}{SIGNATURE_SUFFIX}", &sig[..end])) <= BUDGET
    };
    let total = code_points(sig);
    if fits(total) {
        return total;
    }
    // ` …\n` alone always fits the budget, the whole signature does not.
    let (mut lo, mut hi) = (0, total);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Pushes `range`'s pieces into `out`: `range` itself when it fits the cap
/// under `prefix` or holds fewer than two chars; otherwise the halves of a
/// split at the middle char, recursively.
fn cut_line(body: &str, range: Range<usize>, cap: &Cap, prefix: &str, out: &mut Vec<Range<usize>>) {
    let text = &body[range.clone()];
    if cap.fits(prefix, text) || code_points(text) < 2 {
        out.push(range);
        return;
    }
    let chars = code_points(text);
    // For ASCII text the middle char is the middle byte.
    let mid = if chars == text.len() {
        chars / 2
    } else {
        text.char_indices()
            .nth(chars / 2)
            .map_or(text.len(), |(i, _)| i)
    };
    cut_line(body, range.start..range.start + mid, cap, prefix, out);
    cut_line(body, range.start + mid..range.end, cap, prefix, out);
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

/// How a member's slice becomes text when a chunk's text is written or
/// rebuilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberEmit {
    /// The slice verbatim, joined by `\n` with the other members.
    Text,
    /// A line-split part's signature: the slice whitespace-collapsed, cut
    /// to `cut` characters, plus ` …\n`.
    Signature {
        /// Characters of the collapsed signature the prefix carries.
        cut: usize,
    },
}

/// Appends the text `members` emit — their slices joined by `\n`, a
/// signature member emitting its slice whitespace-collapsed (cut to `cut`
/// chars) plus ` …\n` — to `out`. `None` when a range leaves `src` or
/// splits a character.
pub(crate) fn write_members(
    src: &str,
    members: impl IntoIterator<Item = (Range<usize>, MemberEmit)>,
    out: &mut String,
) -> Option<()> {
    let mut newline = false;
    for (bytes, emit) in members {
        let slice = src.get(bytes)?;
        if newline {
            out.push('\n');
        }
        match emit {
            MemberEmit::Signature { cut } => {
                collapse_into(slice, cut, out);
                out.push_str(SIGNATURE_SUFFIX);
                newline = false;
            }
            MemberEmit::Text => {
                out.push_str(slice);
                newline = true;
            }
        }
    }
    Some(())
}

/// Packs the emitted units: a unit under [`SMALL_BYTES`] joins the current
/// group when the scope matches, the group stays under [`PACK_BYTES`] and
/// the packed document still fits the token cap; larger groups become one
/// `packed` chunk.
fn pack(units: Vec<Unit>, src: &str, tokens: &dyn TokenCounter) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut group: Vec<Unit> = Vec::new();
    let mut gscope = 0;
    let mut gbytes = 0;
    for unit in units {
        if unit.size < SMALL_BYTES
            && !group.is_empty()
            && unit.scope == gscope
            && gbytes + unit.size <= PACK_BYTES
            && pack_fits(&group, &unit, src, tokens)
        {
            gbytes += unit.size;
            group.push(unit);
            continue;
        }
        flush_group(&mut out, &mut group);
        gbytes = 0;
        if unit.size < SMALL_BYTES {
            gscope = unit.scope;
            gbytes = unit.size;
            group.push(unit);
        } else {
            out.push(single(unit));
        }
    }
    flush_group(&mut out, &mut group);
    out
}

/// Whether `unit` can join `group` without the packed chunk's document —
/// the packed title and the members' joined text — exceeding the token cap.
fn pack_fits(group: &[Unit], unit: &Unit, src: &str, tokens: &dyn TokenCounter) -> bool {
    let candidates = group.iter().chain(std::iter::once(unit));
    let title = packed_title(candidates.clone());
    let mut text = String::new();
    for (index, unit) in candidates.enumerate() {
        if index > 0 {
            text.push('\n');
        }
        write_members(
            src,
            unit.members.iter().map(|m| (m.bytes.clone(), m.emit())),
            &mut text,
        );
    }
    tokens.fits(&title, &[&text], MAX_TOKENS)
}

fn flush_group(out: &mut Vec<Chunk>, group: &mut Vec<Unit>) {
    if group.len() == 1 {
        out.push(single(group.remove(0)));
        return;
    }
    if group.len() > 1 {
        let title = packed_title(group.iter());
        let lines = *group[0].lines.start()..=*group[group.len() - 1].lines.end();
        let members = group
            .iter()
            .flat_map(|unit| unit.members.iter().cloned())
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
fn packed_title<'a>(units: impl Iterator<Item = &'a Unit>) -> String {
    let mut title = String::new();
    for (index, unit) in units.enumerate() {
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
    Chunk {
        title: unit.title,
        kind: unit.kind,
        lines: unit.lines,
        members: unit.members,
    }
}

/// Collects the units of one file and packs them into chunks.
pub(super) struct Emitter<'a> {
    src: &'a str,
    path: &'a str,
    /// The first sentence of the file's module docs, in every other unit's
    /// title as `path — <note>`.
    note: Option<&'a str>,
    tokens: &'a dyn TokenCounter,
    units: Vec<Unit>,
    /// The scope id given to the next split container frame.
    next_scope: u32,
}

impl<'a> Emitter<'a> {
    pub fn new(
        path: &'a str,
        src: &'a str,
        tokens: &'a dyn TokenCounter,
        note: Option<&'a str>,
    ) -> Self {
        Self {
            src,
            path,
            note,
            tokens,
            units: Vec::new(),
            next_scope: 0,
        }
    }

    /// A fresh scope id for a split container's frame.
    pub fn new_scope(&mut self) -> u32 {
        self.next_scope += 1;
        self.next_scope
    }

    /// Emits `span` as units under `path — <note> › ctx… › label`, splitting
    /// on line boundaries when a part's document exceeds the size limits.
    /// `item`'s `signature`, when given, is the byte range of the item's
    /// signature: every part after the first carries it as a `signature`
    /// member, whose whitespace-collapsed text plus ` …\n` prefixes the
    /// part's text.
    pub fn emit(&mut self, span: Span, ctx: &[CtxFrame], label: &str, kind: ChunkKind, item: Item) {
        let Item {
            kind: member,
            symbol,
            signature,
        } = item;
        let Span { bytes, row } = span;
        let start = bytes.start;
        let body = &self.src[bytes.clone()];
        let start_line = row as u32 + 1;
        let mut title = String::with_capacity(self.path.len() + label.len() + 32);
        // The module-docs chunk keeps the plain `path`; every other unit's
        // title opens with `path — <note>` when the file has module docs.
        title.push_str(self.path);
        if let Some(note) = self.note.filter(|_| kind != ChunkKind::ModuleDocs) {
            title.push_str(" — ");
            title.push_str(note);
        }
        for frame in ctx {
            title.push_str(" › ");
            title.push_str(&frame.label);
        }
        title.push_str(" › ");
        title.push_str(label);

        let signature_text = signature
            .as_ref()
            .map(|sig| collapse(&self.src[sig.clone()]));
        // The signature line every part after the first carries: the
        // collapsed signature cut to the longest char prefix whose
        // `<sig> …\n` fits a quarter of the cap, so every later part keeps
        // a signature and room for its own text.
        let (sig_cut, rest_prefix) = match &signature_text {
            Some(sig) => {
                let cut = signature_cut(sig, self.tokens);
                let end = sig.char_indices().nth(cut).map_or(sig.len(), |(i, _)| i);
                (cut, format!("{}{SIGNATURE_SUFFIX}", &sig[..end]))
            }
            None => (0, String::new()),
        };
        let mut digits = 3u32;
        let parts = loop {
            // The `(part n/m)` suffix is counted at its widest for the
            // digit count a split can reach; a split that still reaches
            // 10^digits parts is redone one digit wider.
            let nines = "9".repeat(digits as usize);
            let cap = Cap::new(self.tokens, &format!("{title} (part {nines}/{nines})"));
            let parts = token_split(&cap, "", &rest_prefix, body, MAX_BYTES);
            if parts.len() < 10usize.pow(digits) {
                break parts;
            }
            digits += 1;
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
            let mut members = SmallVec::new();
            let mut size = piece.len();
            if index > 0
                && let Some(sig) = &signature
            {
                members.push(Member {
                    bytes: sig.clone(),
                    kind: MemberKind::Signature { cut: sig_cut },
                    symbol: symbol.clone(),
                });
                size += rest_prefix.len();
            }
            members.push(Member {
                bytes: (start + part.start)..(start + part.end),
                kind: member,
                symbol: symbol.clone(),
            });
            self.units.push(Unit {
                scope: ctx.last().map_or(0, |frame| frame.scope),
                lines: line..=end_line,
                title: part_title,
                kind,
                members,
                size,
            });
            line += newlines;
        }
    }

    /// Packs the emitted units into chunks.
    pub fn finish(self) -> Vec<Chunk> {
        pack(self.units, self.src, self.tokens)
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
#[cfg(test)]
mod tests {
    use super::{collapse, collapse_into};

    /// `collapse_into` is `collapse` cut to `cut` chars, for every cut.
    #[test]
    fn collapse_into_is_collapse_cut() {
        for text in [
            "",
            "   ",
            "fn  a(\t b:\n\u{1c}u8 ) ",
            "  lead and   trail  ",
            "é  ü\u{2028}x",
            "one",
        ] {
            let whole = collapse(text);
            for cut in 0..=whole.chars().count() + 2 {
                let want: String = whole.chars().take(cut).collect();
                let mut got = String::new();
                collapse_into(text, cut, &mut got);
                assert_eq!(got, want, "{text:?} cut {cut}");
            }
        }
    }
}
