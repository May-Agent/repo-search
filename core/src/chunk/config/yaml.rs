//! YAML entries, on `saphyr-parser`'s event spans: the keys of the root
//! mapping, the elements of the root sequence, or one entry per document
//! when the stream holds several.

use std::iter::Peekable;
use std::ops::Range;

use saphyr_parser::{Event, Parser, ScalarStyle, Span};

use super::Entry;
use crate::chunk::Error;

type Events<'input> = Peekable<std::vec::IntoIter<(Event<'input>, Range<usize>)>>;

pub(super) fn entries(src: &str) -> Result<Vec<Entry>, Error> {
    let offsets = ByteOffsets::new(src);
    let mut events = Vec::new();
    for event in Parser::new_from_str(src) {
        let (event, span) = event?;
        events.push((event, offsets.range(&span)));
    }
    let mut events: Events = events.into_iter().peekable();
    let mut documents = Vec::new();
    while let Some((event, _)) = events.next() {
        if matches!(event, Event::DocumentStart(_)) {
            let node = node(src, &mut events)?;
            // The document's end event.
            events.next();
            documents.push(node);
        }
    }
    // An empty document (`---` followed by another marker or a comment)
    // has no entry; the others keep their position in the stream.
    let mut documents: Vec<(usize, Node)> = documents
        .into_iter()
        .enumerate()
        .filter(|(_, node)| !node.empty && !node.bytes.is_empty())
        .collect();
    Ok(match documents.len() {
        0 => Vec::new(),
        1 => {
            let (_, node) = documents.remove(0);
            if node.children.is_empty() {
                vec![Entry::key("value".to_owned(), node.bytes, Vec::new())]
            } else {
                node.children
            }
        }
        _ => documents
            .into_iter()
            .map(|(index, node)| {
                Entry::key(format!("document {}", index + 1), node.bytes, node.children)
            })
            .collect(),
    })
}

/// A parsed value: its byte span and, for collections, its entries.
struct Node {
    bytes: Range<usize>,
    children: Vec<Entry>,
    /// The scalar text, for use as a mapping key.
    text: Option<String>,
    /// A plain scalar with no text: the parser's stand-in for an absent node.
    empty: bool,
}

fn node(src: &str, events: &mut Events) -> Result<Node, Error> {
    let (event, span) = events
        .next()
        .expect("the parser ends every collection and document it starts");
    match event {
        Event::MappingStart(..) => {
            let mut bytes = span.clone();
            let mut children = Vec::new();
            while !matches!(events.peek(), Some((Event::MappingEnd, _)) | None) {
                let key = node(src, events)?;
                let value = node(src, events)?;
                // A complex key (`? [a, b]`) or an alias is labelled with its
                // source text; whitespace runs collapse to one space.
                let label = key.text.unwrap_or_else(|| {
                    src.get(key.bytes.clone())
                        .expect("a span lies on char boundaries")
                        .to_owned()
                });
                let label = label.split_whitespace().collect::<Vec<_>>().join(" ");
                // The entry runs from the `?` of an explicit key (or the key
                // itself) to the end of the value, or of the `:` after a key
                // without one.
                let start = explicit_start(src, key.bytes.start);
                let end = if value.empty {
                    colon_end(src, key.bytes.end, start < key.bytes.start)
                } else {
                    value.bytes.end.max(key.bytes.end)
                };
                let entry = start..end;
                bytes.end = bytes.end.max(entry.end);
                children.push(Entry::key(label, entry, value.children));
            }
            close(events, &mut bytes);
            Ok(Node {
                bytes,
                children,
                text: None,
                empty: false,
            })
        }
        Event::SequenceStart(..) => {
            let mut bytes = span.clone();
            let mut children = Vec::new();
            while !matches!(events.peek(), Some((Event::SequenceEnd, _)) | None) {
                let element = node(src, events)?;
                bytes.end = bytes.end.max(element.bytes.end);
                let start = dash_start(src, element.bytes.start);
                children.push(Entry::element(
                    children.len(),
                    start..element.bytes.end,
                    element.children,
                ));
            }
            close(events, &mut bytes);
            Ok(Node {
                bytes,
                children,
                text: None,
                empty: false,
            })
        }
        Event::Scalar(text, style, ..) => Ok(Node {
            bytes: span,
            children: Vec::new(),
            empty: text.is_empty() && matches!(style, ScalarStyle::Plain),
            text: Some(text.into_owned()),
        }),
        _ => Ok(Node {
            bytes: span,
            children: Vec::new(),
            text: None,
            empty: false,
        }),
    }
}

/// The offset of the `-` that introduces the sequence element starting at
/// `start`, or `start` itself when none is on its line.
fn dash_start(src: &str, start: usize) -> usize {
    let before = src
        .get(..start)
        .expect("a span lies on char boundaries")
        .trim_end_matches(' ');
    before.strip_suffix('-').map_or(start, str::len)
}

/// The offset of the `?` that introduces the explicit key starting at
/// `start`, or `start` itself when the key is implicit.
fn explicit_start(src: &str, start: usize) -> usize {
    let before = src.get(..start).expect("a span lies on char boundaries");
    let Some(marked) = before
        .strip_suffix(' ')
        .map(|spaced| spaced.trim_end_matches(' '))
        .and_then(|spaced| spaced.strip_suffix('?'))
    else {
        return start;
    };
    match marked.chars().next_back() {
        None => marked.len(),
        Some(c) if c.is_whitespace() || matches!(c, '{' | '[' | ',') => marked.len(),
        Some(_) => start,
    }
}

/// The end of the `:` that follows the key ending at `end` when the value is
/// absent, or `end` when there is no such `:`. An explicit key's `:` may sit
/// on a later line.
fn colon_end(src: &str, end: usize, explicit: bool) -> usize {
    let after = src.get(end..).expect("a span lies on char boundaries");
    let skipped = if explicit {
        after.trim_start()
    } else {
        after.trim_start_matches([' ', '\t'])
    };
    if skipped.starts_with(':') {
        src.len() - skipped.len() + 1
    } else {
        end
    }
}

/// Consumes a collection's end event; a flow collection's closing bracket
/// has a span, a block collection's end is empty and ignored.
fn close(events: &mut Events, bytes: &mut Range<usize>) {
    if let Some((_, span)) = events.next()
        && !span.is_empty()
    {
        bytes.end = bytes.end.max(span.end);
    }
}

/// Converts the parser's character offsets into byte offsets.
struct ByteOffsets {
    /// Byte offset of every character, then the text's length; empty for an
    /// ASCII text, where characters and bytes coincide.
    table: Vec<usize>,
}

impl ByteOffsets {
    fn new(src: &str) -> Self {
        let table = if src.is_ascii() {
            Vec::new()
        } else {
            src.char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(src.len()))
                .collect()
        };
        Self { table }
    }

    fn byte(&self, character: usize) -> usize {
        if self.table.is_empty() {
            character
        } else {
            self.table[character]
        }
    }

    fn range(&self, span: &Span) -> Range<usize> {
        self.byte(span.start.index())..self.byte(span.end.index())
    }
}
