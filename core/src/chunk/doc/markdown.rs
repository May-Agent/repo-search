//! Markdown and MDX outlines, on `pulldown-cmark`.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use super::{Heading, Outline};

pub(super) fn outline(src: &str) -> Outline {
    // Front matter is only a block at the very start of the file; the
    // parser would take any `---` pair for one, so it is cut off here and the
    // metadata options stay off.
    let body_start = front_matter_end(src);
    let options = Options::ENABLE_TABLES | Options::ENABLE_HEADING_ATTRIBUTES;
    let mut outline = Outline {
        body_start,
        headings: Vec::new(),
        protected: Vec::new(),
    };
    // Open container tags; only a heading outside every container (block
    // quote, list item) cuts a section.
    let mut depth = 0usize;
    // The heading being read: its level, byte range start and title.
    let mut open: Option<(u8, usize, String)> = None;
    let rest = src
        .get(body_start..)
        .expect("front matter ends on a line boundary");
    for (event, range) in Parser::new_ext(rest, options).into_offset_iter() {
        let range = range.start + body_start..range.end + body_start;
        match event {
            Event::Start(Tag::CodeBlock(_)) => {
                outline.protected.push(range);
                depth += 1;
            }
            Event::Start(Tag::Heading { level, .. }) => {
                if depth == 0 {
                    open = Some((level_number(level), range.start, String::new()));
                }
                depth += 1;
            }
            Event::End(TagEnd::Heading(_)) => {
                depth -= 1;
                if depth == 0
                    && let Some((level, start, title)) = open.take()
                {
                    outline.headings.push(Heading {
                        level,
                        title: title.trim().to_owned(),
                        bytes: start..range.end,
                    });
                }
            }
            Event::Start(_) => depth += 1,
            Event::End(_) => depth -= 1,
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, _, title)) = &mut open {
                    title.push_str(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some((_, _, title)) = &mut open {
                    title.push(' ');
                }
            }
            _ => {}
        }
    }
    outline
}

/// The end of a `---` or `+++` front matter block at the start of `src`
/// or 0 when there is none. A block
/// without a closing delimiter line is not front matter.
fn front_matter_end(src: &str) -> usize {
    let mut lines = src.split_inclusive('\n');
    let Some(first) = lines.next() else {
        return 0;
    };
    let delimiter = first.trim_end();
    if delimiter != "---" && delimiter != "+++" {
        return 0;
    }
    let mut offset = first.len();
    for line in lines {
        offset += line.len();
        if line.trim_end() == delimiter {
            return offset;
        }
    }
    0
}

fn level_number(level: HeadingLevel) -> u8 {
    level as u8
}
