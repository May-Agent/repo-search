-- Per-worktree index. Chunk text is never stored: `files` + `chunks` +
-- `chunk_members` describe where text lives, `fts` only indexes it.

CREATE TABLE files (
    path         TEXT PRIMARY KEY,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    inode        INTEGER NOT NULL,
    content_hash BLOB NOT NULL
) WITHOUT ROWID;

-- Files that fail to chunk, fingerprinted like `files` so a pass can skip
-- them unread. `reason` is the text form of `UnchunkableReason`.
CREATE TABLE unchunkable (
    path         TEXT PRIMARY KEY,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    inode        INTEGER NOT NULL,
    content_hash BLOB NOT NULL,
    reason       TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    title        TEXT NOT NULL,
    kind         TEXT NOT NULL,
    content_hash BLOB NOT NULL
);
CREATE INDEX chunks_by_path ON chunks (path);
CREATE INDEX chunks_by_hash ON chunks (content_hash);

CREATE TABLE chunk_members (
    chunk_id   INTEGER NOT NULL,
    ord        INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte   INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    cut        INTEGER NOT NULL,
    symbol     TEXT NOT NULL,
    PRIMARY KEY (chunk_id, ord)
) WITHOUT ROWID;
CREATE INDEX chunk_members_by_kind ON chunk_members (kind);
CREATE INDEX chunk_members_by_symbol ON chunk_members (symbol);

-- Contentless FTS5 over title + body; rowid = chunks.id.
CREATE VIRTUAL TABLE fts USING fts5 (
    title,
    body,
    content = '',
    contentless_delete = 1,
    tokenize = 'unicode61'
);
