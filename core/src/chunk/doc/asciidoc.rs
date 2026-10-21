//! AsciiDoc outlines: `=` (or Markdown-style `#`) section titles, with
//! listing, literal, passthrough and comment blocks hiding their content.

use super::{Heading, Outline};

/// Block delimiter characters whose content is kept verbatim.
const VERBATIM_DELIMITERS: &[u8] = b"-.+/";

/// The Markdown-style code fence, a code block in AsciiDoc too.
const FENCE: &str = "```";

pub(super) fn outline(src: &str) -> Outline {
    let mut outline = Outline {
        body_start: 0,
        headings: Vec::new(),
        protected: Vec::new(),
    };
    // The open verbatim block: its delimiter line and where it started.
    let mut open: Option<(&str, usize)> = None;
    let mut offset = 0;
    for raw in src.split_inclusive('\n') {
        let start = offset;
        offset += raw.len();
        let line = raw.trim_end();
        if let Some((delimiter, from)) = open {
            if line == delimiter || (delimiter.starts_with(FENCE) && line == FENCE) {
                outline.protected.push(from..start);
                open = None;
            }
            continue;
        }
        if is_verbatim_delimiter(line) || line.starts_with(FENCE) {
            open = Some((line, offset));
            continue;
        }
        if let Some((level, title)) = title(line) {
            outline.headings.push(Heading {
                level,
                title: title.to_owned(),
                bytes: start..start + line.len(),
            });
        }
    }
    if let Some((_, from)) = open {
        outline.protected.push(from..src.len());
    }
    outline
}

/// A line of four or more of one verbatim delimiter character.
fn is_verbatim_delimiter(line: &str) -> bool {
    let Some(&first) = line.as_bytes().first() else {
        return false;
    };
    line.len() >= 4 && VERBATIM_DELIMITERS.contains(&first) && line.bytes().all(|b| b == first)
}

/// The level and text of a `== Title` or `## Title` line, up to six marker
/// characters followed by a space; a closing marker run is dropped.
fn title(line: &str) -> Option<(u8, &str)> {
    let marker = *line
        .as_bytes()
        .first()
        .filter(|&&b| b == b'=' || b == b'#')?;
    let level = line.bytes().take_while(|&b| b == marker).count();
    let rest = line.get(level..)?.strip_prefix(' ')?;
    if level > 6 {
        return None;
    }
    let text = rest.trim().trim_end_matches(marker as char).trim_end();
    (!text.is_empty()).then_some((level as u8, text))
}
