-- Per-worktree index. Chunk text is never stored: `files` + `chunks` +
-- `chunk_members` describe where text lives, `fts` only indexes it.

-- The chunker identity the rows were built under: the chunker fingerprint and
-- the model whose tokenizer caps every chunk, and the index phase (0 first
-- index, 1 maintained). At most one row.
CREATE TABLE meta (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    chunker     TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    phase       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE files (
    path         TEXT PRIMARY KEY,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    inode        INTEGER NOT NULL,
    content_hash BLOB NOT NULL,
    -- 1 for a file only a pass that includes ignored files reaches; its chunks
    -- stay stored but are hidden from searches that exclude ignored files.
    ignored      INTEGER NOT NULL CHECK (ignored IN (0, 1))
) WITHOUT ROWID;

-- Files that fail to chunk, fingerprinted like `files` so a pass can skip
-- them unread. `reason` is the text form of `UnchunkableReason`.
CREATE TABLE unchunkable (
    path         TEXT PRIMARY KEY,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    inode        INTEGER NOT NULL,
    content_hash BLOB NOT NULL,
    ignored      INTEGER NOT NULL CHECK (ignored IN (0, 1)),
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

-- Exact BM25 statistics of `fts`: its row count and each column's token
-- total. The table's own averages record keeps deleted rows, so every write
-- to `fts` adjusts this row in the same transaction and `exact_bm25` ranks by
-- it. Exactly one row.
CREATE TABLE bm25_stats (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    rows         INTEGER NOT NULL,
    title_tokens INTEGER NOT NULL,
    body_tokens  INTEGER NOT NULL
);
INSERT INTO bm25_stats (id, rows, title_tokens, body_tokens) VALUES (1, 0, 0, 0);
