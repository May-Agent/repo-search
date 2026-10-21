//! Documentation chunking: one chunk per heading section, titled
//! `path › H1 › H2 › …`, with fenced and literal blocks kept whole inside
//! their section; a section over the size limit is split on paragraph
//! boundaries, every part keeping the heading.
//!
//! A format only has to find the headings and the byte ranges that must not
//! be split ([`Outline`]); the sectioning and splitting are shared.

mod asciidoc;
mod markdown;
mod rst;

use std::borrow::Cow;
use std::ops::Range;

use smallvec::SmallVec;

use super::units::{Cap, LineIndex, MAX_BYTES, token_split, trim_range, without_bom};
use super::{Chunk, ChunkKind, DocFormat, Error, FileChunker, Member, MemberKind};
use crate::TokenCounter;

/// Separates the path and the headings in a title and a symbol.
const SEPARATOR: &str = " › ";

/// A heading found in a document.
struct Heading {
    /// 1 for the outermost level.
    level: u8,
    title: String,
    /// The heading's own lines, including an underline or overline.
    bytes: Range<usize>,
}

/// What a format tells the sectioning about a document.
struct Outline {
    /// Where the content starts, after front matter.
    body_start: usize,
    /// Headings in document order.
    headings: Vec<Heading>,
    /// Sorted, disjoint ranges with no paragraph breaks inside: code blocks.
    protected: Vec<Range<usize>>,
}

/// Chunks documentation files of one [`DocFormat`].
pub struct DocChunker {
    format: DocFormat,
}

impl DocChunker {
    /// A chunker for `format`.
    #[must_use]
    pub fn new(format: DocFormat) -> Self {
        Self { format }
    }
}

impl FileChunker for DocChunker {
    fn chunk_file(
        &mut self,
        path: &str,
        src: &str,
        tokens: &dyn TokenCounter,
    ) -> Result<Vec<Chunk>, Error> {
        without_bom(src, |src| {
            let outline = match self.format {
                DocFormat::Markdown => markdown::outline(src),
                DocFormat::Rst => rst::outline(src),
                DocFormat::AsciiDoc => asciidoc::outline(src),
                DocFormat::Text => Outline {
                    body_start: 0,
                    headings: Vec::new(),
                    protected: Vec::new(),
                },
            };
            Ok(Sectioner::new(path, src, outline, tokens).chunks())
        })
    }

    fn chunk_kind_name(&self, kind: ChunkKind) -> Cow<'_, str> {
        match kind {
            ChunkKind::Doc => {
                Cow::Borrowed(kind.static_name().expect("a doc chunk has a static name"))
            }
            _ => panic!("a documentation chunker produces only `doc` chunks"),
        }
    }

    fn member_kind_name(&self, kind: MemberKind) -> Cow<'_, str> {
        match kind {
            MemberKind::Doc => {
                Cow::Borrowed(kind.static_name().expect("a doc member has a static name"))
            }
            _ => panic!("a documentation chunker produces only `doc` members"),
        }
    }
}

struct Sectioner<'a> {
    path: &'a str,
    src: &'a str,
    lines: LineIndex,
    outline: Outline,
    tokens: &'a dyn TokenCounter,
    chunks: Vec<Chunk>,
}

impl<'a> Sectioner<'a> {
    fn new(path: &'a str, src: &'a str, outline: Outline, tokens: &'a dyn TokenCounter) -> Self {
        Self {
            path,
            src,
            lines: LineIndex::new(src),
            outline,
            tokens,
            chunks: Vec::new(),
        }
    }

    fn chunks(mut self) -> Vec<Chunk> {
        let headings = std::mem::take(&mut self.outline.headings);
        // (level, title) of the enclosing headings.
        let mut stack: Vec<(u8, &str)> = Vec::new();
        let mut path = String::new();
        let first_start = headings
            .first()
            .map_or(self.src.len(), |heading| heading.bytes.start);
        self.section(
            &path,
            None,
            self.outline.body_start.min(first_start)..first_start,
            false,
        );
        for (index, heading) in headings.iter().enumerate() {
            stack.retain(|&(level, _)| level < heading.level);
            stack.push((heading.level, &heading.title));
            path.clear();
            for (_, title) in stack.iter().filter(|(_, title)| !title.is_empty()) {
                if !path.is_empty() {
                    path.push_str(SEPARATOR);
                }
                path.push_str(title);
            }
            let end = headings
                .get(index + 1)
                .map_or(self.src.len(), |next| next.bytes.start);
            // A heading with no body of its own is covered by its first
            // descendant's title path; any other one is a chunk of its own.
            let covered = headings
                .get(index + 1)
                .is_some_and(|next| next.level > heading.level);
            self.section(
                &path,
                Some(heading),
                heading.bytes.end..end,
                !covered && !heading.title.is_empty(),
            );
        }
        self.chunks
    }

    /// Emits the section under `heading_path`: its heading lines (none for
    /// the text before the first heading) and body. A section without a body
    /// yields a chunk of its heading alone only when `keep_bare` is set.
    fn section(
        &mut self,
        heading_path: &str,
        heading: Option<&Heading>,
        body: Range<usize>,
        keep_bare: bool,
    ) {
        let body = trim_range(self.src, body);
        let heading = heading.map(|heading| trim_range(self.src, heading.bytes.clone()));
        let title = if heading_path.is_empty() {
            self.path.to_owned()
        } else {
            format!("{}{SEPARATOR}{heading_path}", self.path)
        };
        let cap = Cap::new(self.tokens, &title);
        if body.is_empty() {
            if let (true, Some(heading)) = (keep_bare, heading) {
                for part in token_split(&cap, "", "", &self.src[heading.clone()], MAX_BYTES) {
                    let part = heading.start + part.start..heading.start + part.end;
                    self.push(heading_path, &title, [part]);
                }
            }
            return;
        }
        let whole = heading.as_ref().map_or(body.start, |h| h.start)..body.end;
        if whole.len() <= MAX_BYTES && cap.fits("", &self.src[whole.clone()]) {
            self.push(heading_path, &title, [whole]);
            return;
        }
        // Every part is emitted with the heading and a newline before it.
        let prefix = heading
            .as_ref()
            .map_or(String::new(), |h| format!("{}\n", &self.src[h.clone()]));
        let budget = MAX_BYTES
            .saturating_sub(heading.as_ref().map_or(0, |h| h.len() + 1))
            .max(1);
        for part in self.parts(&cap, &prefix, body, budget) {
            match &heading {
                Some(heading) => self.push(heading_path, &title, [heading.clone(), part]),
                None => self.push(heading_path, &title, [part]),
            }
        }
    }

    /// Splits `body` into parts of at most `budget` bytes on paragraph
    /// boundaries; a paragraph over the limit is split on line boundaries,
    /// and no part's document under `prefix` may exceed the token cap.
    fn parts(
        &self,
        cap: &Cap,
        prefix: &str,
        body: Range<usize>,
        budget: usize,
    ) -> Vec<Range<usize>> {
        let mut parts: Vec<Range<usize>> = Vec::new();
        for piece in self.pieces(body, budget, cap, prefix) {
            match parts.last_mut() {
                Some(part)
                    if piece.end - part.start <= budget
                        && cap.fits(prefix, &self.src[part.start..piece.end]) =>
                {
                    part.end = piece.end;
                }
                _ => parts.push(piece),
            }
        }
        parts
    }

    /// The paragraphs of `body` (blank-line separated, never inside a
    /// protected range), each split on line boundaries when over the limit.
    fn pieces(
        &self,
        body: Range<usize>,
        budget: usize,
        cap: &Cap,
        prefix: &str,
    ) -> Vec<Range<usize>> {
        let mut pieces = Vec::new();
        let mut block: Option<Range<usize>> = None;
        let mut offset = body.start;
        for line in self.src[body.clone()].split_inclusive('\n') {
            let range = offset..offset + line.len();
            offset = range.end;
            if line.trim().is_empty() && !self.is_protected(range.start) {
                pieces.extend(block.take());
                continue;
            }
            block = Some(block.map_or(range.clone(), |open| open.start..range.end));
        }
        pieces.extend(block);
        pieces
            .into_iter()
            .flat_map(|block| self.fit_block(cap, prefix, block, budget))
            .collect()
    }

    /// `block` itself, or its line-split parts when its document exceeds the
    /// byte budget or the token cap.
    fn fit_block(
        &self,
        cap: &Cap,
        prefix: &str,
        block: Range<usize>,
        budget: usize,
    ) -> Vec<Range<usize>> {
        let block = trim_range(self.src, block);
        token_split(cap, prefix, prefix, &self.src[block.clone()], budget)
            .into_iter()
            .map(|part| trim_range(self.src, block.start + part.start..block.start + part.end))
            .filter(|part| !part.is_empty())
            .collect()
    }

    fn is_protected(&self, offset: usize) -> bool {
        let after = self
            .outline
            .protected
            .partition_point(|range| range.end <= offset);
        self.outline
            .protected
            .get(after)
            .is_some_and(|range| range.start <= offset)
    }

    /// Appends a chunk of `title` made of `members`, in order.
    fn push<const N: usize>(
        &mut self,
        heading_path: &str,
        title: &str,
        members: [Range<usize>; N],
    ) {
        // The lines of the text itself: a split part does not claim the
        // heading's lines.
        let text = &members[N - 1];
        let lines = self.lines.row(text.start) as u32 + 1..=self.lines.row(text.end - 1) as u32 + 1;
        let members: SmallVec<[Member; 2]> = members
            .into_iter()
            .map(|bytes| Member {
                bytes,
                kind: MemberKind::Doc,
                symbol: heading_path.to_owned(),
            })
            .collect();
        self.chunks.push(Chunk {
            title: title.to_owned(),
            kind: ChunkKind::Doc,
            lines,
            members,
        });
    }
}

#[cfg(test)]
mod tests;
