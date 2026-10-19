//! reStructuredText outlines: section titles are a text line with an
//! adornment underline (and optionally the same overline); the level of a
//! title is the order in which its adornment style first appeared. Literal
//! blocks (after `::`) and code directives hide their content.

use std::ops::Range;

use super::{Heading, Outline};

/// Directives whose indented content is literal text.
const LITERAL_DIRECTIVES: [&str; 6] = [
    "code-block",
    "code",
    "sourcecode",
    "parsed-literal",
    "literalinclude",
    "highlight",
];

/// An adornment style: its character and whether the title has an overline.
#[derive(PartialEq, Eq)]
struct Style {
    character: u8,
    overline: bool,
}

struct Line<'a> {
    text: &'a str,
    start: usize,
}

impl Line<'_> {
    fn end(&self) -> usize {
        self.start + self.text.len()
    }

    fn indent(&self) -> usize {
        self.text.len() - self.text.trim_start().len()
    }

    fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

pub(super) fn outline(src: &str) -> Outline {
    let mut offset = 0;
    let lines: Vec<Line> = src
        .split_inclusive('\n')
        .map(|text| {
            let line = Line {
                text,
                start: offset,
            };
            offset += text.len();
            line
        })
        .collect();
    let mut outline = Outline {
        body_start: 0,
        headings: Vec::new(),
        protected: Vec::new(),
    };
    let mut styles: Vec<Style> = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        if let Some(block) = literal_block(&lines, index) {
            outline.protected.push(block.range);
            index = block.next;
            continue;
        }
        if let Some(title) = title_at(&lines, index) {
            let level = styles
                .iter()
                .position(|known| *known == title.style)
                .unwrap_or_else(|| {
                    styles.push(title.style);
                    styles.len() - 1
                });
            outline.headings.push(Heading {
                level: u8::try_from(level + 1)
                    .expect("an rst document has fewer than 256 title styles"),
                title: title.text.trim().to_owned(),
                bytes: title.bytes,
            });
            index = title.next;
            continue;
        }
        index += 1;
    }
    outline
}

/// The adornment character of a line made of one repeated punctuation
/// character, at least three long.
fn adornment(line: &Line) -> Option<u8> {
    let text = line.text.trim_end();
    let first = *text.as_bytes().first()?;
    (text.len() >= 3
        && !first.is_ascii_alphanumeric()
        && first.is_ascii_punctuation()
        && text.bytes().all(|b| b == first))
    .then_some(first)
}

/// A section title found in the lines.
struct Title<'a> {
    style: Style,
    text: &'a str,
    /// The title's lines including adornments.
    bytes: Range<usize>,
    /// The index of the line after the title.
    next: usize,
}

/// The section title starting at line `index`.
fn title_at<'a>(lines: &[Line<'a>], index: usize) -> Option<Title<'a>> {
    let line = &lines[index];
    if line.is_blank() {
        return None;
    }
    // A title with an overline: adornment, text, same adornment.
    if let Some(character) = adornment(line)
        && let (Some(title), Some(under)) = (lines.get(index + 1), lines.get(index + 2))
        && !title.is_blank()
        && title.indent() == 0
        && adornment(under) == Some(character)
    {
        return Some(Title {
            style: Style {
                character,
                overline: true,
            },
            text: title.text,
            bytes: line.start..under.end(),
            next: index + 3,
        });
    }
    // A title with an underline, outside of any paragraph.
    let under = lines.get(index + 1)?;
    let after_blank = index == 0 || lines[index - 1].is_blank();
    let character = adornment(under)?;
    (line.indent() == 0 && adornment(line).is_none() && after_blank).then_some(Title {
        style: Style {
            character,
            overline: false,
        },
        text: line.text,
        bytes: line.start..under.end(),
        next: index + 2,
    })
}

struct LiteralBlock {
    range: Range<usize>,
    /// The index of the first line after the block.
    next: usize,
}

/// The literal block introduced by line `index`: a paragraph ending in `::`
/// or a code directive, followed by its more-indented content.
fn literal_block(lines: &[Line], index: usize) -> Option<LiteralBlock> {
    let line = &lines[index];
    let text = line.text.trim();
    let introduces = text.ends_with("::")
        || text.strip_prefix("..").is_some_and(|rest| {
            rest.trim_start()
                .split_once("::")
                .is_some_and(|(name, _)| LITERAL_DIRECTIVES.contains(&name.trim()))
        });
    if !introduces {
        return None;
    }
    let base = line.indent();
    let mut end = index + 1;
    let mut last_content = None;
    while let Some(next) = lines.get(end) {
        if next.is_blank() {
            end += 1;
        } else if next.indent() > base {
            last_content = Some(end);
            end += 1;
        } else {
            break;
        }
    }
    let last = last_content?;
    Some(LiteralBlock {
        range: lines[index + 1].start..lines[last].end(),
        next: last + 1,
    })
}
