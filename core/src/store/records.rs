//! Plain records the store persists, plus the hashing and FTS-body helpers
//! that derive some of their fields.

use std::ops::{Range, RangeInclusive};

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

/// A 32-byte BLAKE3 digest identifying file or chunk content.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(blake3::Hash);

impl ContentHash {
    /// Hashes a whole file's bytes.
    #[must_use]
    pub fn of_file(bytes: &[u8]) -> Self {
        Self(blake3::hash(bytes))
    }

    /// Hashes a chunk under its title: `title`, a `0x00` separator, then
    /// `text` — so a chunk whose title changes is re-embedded.
    #[must_use]
    pub fn of_chunk(title: &str, text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(title.as_bytes());
        hasher.update(&[0x00]);
        hasher.update(text.as_bytes());
        Self(hasher.finalize())
    }

    /// The raw 32 digest bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }
}

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash({self})")
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_hex())
    }
}

impl ToSql for ContentHash {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(ValueRef::Blob(self.as_bytes())))
    }
}

impl FromSql for ContentHash {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let blob = value.as_blob()?;
        let bytes: [u8; 32] = blob.try_into().map_err(|_| FromSqlError::InvalidBlobSize {
            expected_size: 32,
            blob_size: blob.len(),
        })?;
        Ok(Self(blake3::Hash::from(bytes)))
    }
}

/// One indexed file's stat fingerprint, what reconcile diffs the walk against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    /// Worktree-relative path, `/`-separated.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Modification time in nanoseconds.
    pub mtime_ns: i64,
    /// Inode number.
    pub inode: u64,
    /// BLAKE3 of the file's bytes.
    pub content_hash: ContentHash,
}

/// A chunk of a file: one or more byte ranges under one title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRecord {
    /// 1-based line range, inclusive.
    pub lines: RangeInclusive<u32>,
    /// The chunk's title, e.g. `path › impl Environment › fn insert`.
    pub title: String,
    /// The chunk's syntactic kind.
    pub kind: String,
    /// One entry per packed item; a plain chunk has one member.
    pub members: Vec<MemberRecord>,
    /// BLAKE3 of `of_chunk(title, text)`; the vector cache's key.
    pub content_hash: ContentHash,
    /// What FTS indexes, built by [`fts_body`]; the chunk text itself is never
    /// stored.
    pub fts_body: String,
}

/// One byte range of a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberRecord {
    /// Byte range in the file.
    pub bytes: Range<u64>,
    /// The item's syntactic kind, for `--kind` filtering.
    pub kind: String,
    /// The item's symbol path, for `--symbol` filtering.
    pub symbol: String,
}

/// The FTS body of a chunk: its text, a newline, then its camelCase-split
/// identifiers — so `HTTPServer` also matches `http` and `server`.
#[must_use]
pub fn fts_body(text: &str) -> String {
    let mut body = String::with_capacity(text.len() + 32);
    body.push_str(text);
    body.push('\n');
    body.push_str(&split_identifiers(text));
    body
}

/// `ComputedSignal` -> `Computed Signal`, as the evaluation's
/// `split_identifiers` does: every `[A-Za-z][A-Za-z0-9]+` word that contains a
/// camel boundary, with a space inserted at each boundary of
/// `(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])`, joined by spaces.
/// Implemented over bytes; the rule is ASCII-only.
#[must_use]
pub fn split_identifiers(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let word = &bytes[start..i];
        if word.len() < 2 || !(1..word.len()).any(|j| is_boundary(word, j)) {
            continue;
        }
        if !out.is_empty() {
            out.push(b' ');
        }
        let mut piece = 0;
        for (j, _) in word.iter().enumerate().skip(1) {
            if is_boundary(word, j) {
                out.extend_from_slice(&word[piece..j]);
                out.push(b' ');
                piece = j;
            }
        }
        out.extend_from_slice(&word[piece..]);
    }
    // Every byte pushed is an ASCII alphanumeric or a space.
    String::from_utf8(out).expect("identifier bytes and separators are ASCII")
}

/// Whether a camel boundary sits between `word[j - 1]` and `word[j]`: a
/// lowercase or digit followed by an uppercase, or an uppercase followed by an
/// uppercase that a lowercase follows (the last capital of an acronym).
fn is_boundary(word: &[u8], j: usize) -> bool {
    let prev = word[j - 1];
    let cur = word[j];
    ((prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase())
        || (prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && word.get(j + 1).is_some_and(u8::is_ascii_lowercase))
}
