//! Configuration chunking: one chunk per top-level table or key of a TOML,
//! YAML or JSON file, titled with its key path. A table or mapping over the
//! size limit is split into its entries, which carry the enclosing keys in
//! their titles; a value over the limit is split on line boundaries; small
//! neighbours are packed, as in the item chunker.
//!
//! A format only has to produce the [`Entry`] tree with byte spans; the
//! traversal, titles and size rules are shared.

mod json;
mod toml;
mod yaml;

use std::borrow::Cow;
use std::ops::Range;

use super::units::{CtxFrame, Emitter, LineIndex, MAX_BYTES, Span, trim_range, without_bom};
use super::{Chunk, ChunkKind, ConfigFormat, Error, FileChunker, MemberKind};

/// Separates key path segments in a symbol.
const SYMBOL_SEPARATOR: &str = ".";

/// A table, key or array element and the entries nested in it.
struct Entry {
    /// What the title shows: the key, `[2]` for an array element.
    label: String,
    /// What the symbol path shows: the key, `2` for an array element.
    segment: String,
    /// From the entry's first byte to its last, comments excluded.
    bytes: Range<usize>,
    /// Entries nested in this one, in document order; the entry is split
    /// into them when it exceeds the size limit.
    children: Vec<Entry>,
}

impl Entry {
    fn key(label: String, bytes: Range<usize>, children: Vec<Entry>) -> Self {
        Self {
            segment: label.clone(),
            label,
            bytes,
            children,
        }
    }

    fn element(index: usize, bytes: Range<usize>, children: Vec<Entry>) -> Self {
        Self {
            label: format!("[{index}]"),
            segment: index.to_string(),
            bytes,
            children,
        }
    }
}

/// Chunks configuration files of one [`ConfigFormat`].
pub struct ConfigChunker {
    format: ConfigFormat,
}

impl ConfigChunker {
    /// A chunker for `format`.
    #[must_use]
    pub fn new(format: ConfigFormat) -> Self {
        Self { format }
    }
}

impl FileChunker for ConfigChunker {
    fn chunk_file(&mut self, path: &str, src: &str) -> Result<Vec<Chunk>, Error> {
        without_bom(src, |src| {
            let (entries, comment) = match self.format {
                ConfigFormat::Toml => (toml::entries(src)?, "#"),
                ConfigFormat::Yaml => (yaml::entries(src)?, "#"),
                ConfigFormat::Json => (json::entries(src)?, "//"),
            };
            let mut walker = Walker {
                src,
                comment,
                lines: LineIndex::new(src),
                out: Emitter::new(path, src),
            };
            let mut floor = 0;
            walker.walk(&entries, &mut Vec::new(), &mut floor);
            Ok(walker.out.finish())
        })
    }

    fn chunk_kind_name(&self, kind: ChunkKind) -> Cow<'_, str> {
        Cow::Borrowed(
            kind.static_name()
                .expect("a configuration chunker produces only `config` and `packed` chunks"),
        )
    }

    fn member_kind_name(&self, kind: MemberKind) -> Cow<'_, str> {
        Cow::Borrowed(
            kind.static_name()
                .expect("a configuration chunker produces only `config` members"),
        )
    }
}

struct Walker<'a> {
    src: &'a str,
    /// The line comment marker; comment lines directly above an entry belong
    /// to it.
    comment: &'static str,
    lines: LineIndex,
    out: Emitter<'a>,
}

impl Walker<'_> {
    /// Emits `entries`, siblings under `ctx`. `floor` is the end of the
    /// previous entry: leading comments never reach below it.
    fn walk(&mut self, entries: &[Entry], ctx: &mut Vec<CtxFrame>, floor: &mut usize) {
        for entry in entries {
            let bytes = trim_range(self.src, entry.bytes.clone());
            if bytes.is_empty() {
                continue;
            }
            let start = self.entry_start(bytes.start, *floor);
            *floor = bytes.end;
            let segments: Vec<&str> = ctx
                .iter()
                .filter_map(|frame| frame.symbol.as_deref())
                .collect();
            let symbol = if segments.is_empty() {
                entry.segment.clone()
            } else {
                format!(
                    "{}{SYMBOL_SEPARATOR}{}",
                    segments.join(SYMBOL_SEPARATOR),
                    entry.segment
                )
            };
            let split = (bytes.end - start > MAX_BYTES && !entry.children.is_empty())
                .then(|| trim_range(self.src, entry.children[0].bytes.clone()).start);
            let Some(first_child) = split else {
                self.emit(start..bytes.end, ctx, entry, symbol);
                continue;
            };
            let first_child = self.entry_start(first_child, bytes.start);
            if first_child > start {
                let header = trim_range(self.src, start..first_child);
                self.emit(header, ctx, entry, symbol.clone());
            }
            let scope = self.out.new_scope();
            ctx.push(CtxFrame {
                label: entry.label.clone(),
                symbol: Some(entry.segment.clone()),
                scope,
            });
            let mut child_floor = bytes.start;
            self.walk(&entry.children, ctx, &mut child_floor);
            ctx.pop();
        }
    }

    fn emit(&mut self, bytes: Range<usize>, ctx: &[CtxFrame], entry: &Entry, symbol: String) {
        let row = self.lines.row(bytes.start);
        self.out.emit(
            Span { bytes, row },
            ctx,
            &entry.label,
            ChunkKind::Config,
            MemberKind::Config,
            symbol,
        );
    }

    /// The start of the entry beginning at `start`: the start of the comment
    /// lines directly above it, or of its own line when only whitespace
    /// precedes it there. Never before `floor`.
    fn entry_start(&self, start: usize, floor: usize) -> usize {
        let mut begin = self.lines.line_start(start);
        if !self.src[begin..start].trim().is_empty() || begin < floor {
            return start;
        }
        while begin > floor {
            let previous = self.lines.line_start(begin - 1);
            if previous < floor
                || !self.src[previous..begin]
                    .trim_start()
                    .starts_with(self.comment)
            {
                break;
            }
            begin = previous;
        }
        begin
    }
}

#[cfg(test)]
mod tests;
