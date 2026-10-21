//! TOML entries, on `toml_edit`'s spans: a headed table (or one element of an
//! array of tables) is an entry holding its direct keys; keys before the
//! first header are entries of their own.

use std::ops::Range;

use toml_edit::{Document, Item, Key, Table};

use super::Entry;
use crate::chunk::Error;

pub(super) fn entries(src: &str) -> Result<Vec<Entry>, Error> {
    let document = Document::parse(src)?;
    let mut entries = Vec::new();
    let root = document.as_table();
    entries.extend(keys(src, root));
    sections(src, root, "", &mut entries);
    entries.sort_by_key(|entry| entry.bytes.start);
    Ok(entries)
}

/// The entries of the key/value pairs directly inside `table`; sub-tables
/// with their own header are sections, not keys. The pairs of one dotted key
/// (`a.b = 1`, `a.c = 2`) form one entry as long as they are adjacent; when
/// other keys sit between them, each run is an entry of its own, so no text
/// belongs to two entries.
fn keys(src: &str, table: &Table) -> Vec<Entry> {
    let mut pairs = Vec::new();
    for (name, item) in table.iter() {
        match item {
            Item::Value(value) => {
                if let (Some(key), Some(value)) =
                    (table.key(name).and_then(Key::span), value.span())
                {
                    pairs.push((name, key.start..value.end));
                }
            }
            Item::Table(dotted) if dotted.is_dotted() => {
                dotted_pairs(src, dotted, name, &mut pairs)
            }
            _ => {}
        }
    }
    pairs.sort_by_key(|(_, bytes)| bytes.start);
    let mut keys: Vec<Entry> = Vec::new();
    let mut previous = None;
    for (name, bytes) in pairs {
        match keys.last_mut() {
            Some(entry) if previous == Some(name) => entry.bytes.end = bytes.end,
            _ => keys.push(Entry::key(name.to_owned(), bytes, Vec::new())),
        }
        previous = Some(name);
    }
    keys
}

/// The first non-blank byte of the line holding `at`.
fn first_byte_of_line(src: &str, at: usize) -> usize {
    let before = src.get(..at).expect("a key span is on a char boundary");
    let line = before.rfind('\n').map_or(0, |newline| newline + 1);
    let indent = before
        .get(line..)
        .expect("a line start is on a char boundary")
        .bytes()
        .take_while(|b| matches!(b, b' ' | b'\t'))
        .count();
    line + indent
}

/// The pairs below the dotted key `top`, each from the first byte of its
/// line (a dotted key always starts one) to the end of its value.
fn dotted_pairs<'t>(
    src: &str,
    table: &'t Table,
    top: &'t str,
    pairs: &mut Vec<(&'t str, Range<usize>)>,
) {
    for (name, item) in table.iter() {
        match item {
            Item::Value(value) => {
                if let (Some(key), Some(value)) =
                    (table.key(name).and_then(Key::span), value.span())
                {
                    let start = first_byte_of_line(src, key.start);
                    pairs.push((top, start..value.end));
                }
            }
            Item::Table(inner) if inner.is_dotted() => dotted_pairs(src, inner, top, pairs),
            _ => {}
        }
    }
}

/// Appends an entry per headed table below `table`, whose own path is `path`.
fn sections(src: &str, table: &Table, path: &str, out: &mut Vec<Entry>) {
    for (name, item) in table.iter() {
        let path = if path.is_empty() {
            name.to_owned()
        } else {
            format!("{path}.{name}")
        };
        match item {
            Item::Table(inner) if !inner.is_dotted() => {
                section(src, inner, &path, out);
                sections(src, inner, &path, out);
            }
            Item::ArrayOfTables(array) => {
                for (index, inner) in array.iter().enumerate() {
                    let path = format!("{path}[{index}]");
                    section(src, inner, &path, out);
                    sections(src, inner, &path, out);
                }
            }
            _ => {}
        }
    }
}

/// The entry of a headed table; an implicit table has no header and no
/// entry.
fn section(src: &str, table: &Table, path: &str, out: &mut Vec<Entry>) {
    let Some(header) = table.span().filter(|_| !table.is_implicit()) else {
        return;
    };
    let children = keys(src, table);
    let end = children
        .iter()
        .map(|child| child.bytes.end)
        .max()
        .map_or(header.end, |end| end.max(header.end));
    out.push(Entry::key(path.to_owned(), header.start..end, children));
}
