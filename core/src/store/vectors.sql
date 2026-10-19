-- Repository-wide vector cache, shared by every worktree. `vector` is the
-- little-endian fp16 components of a `UnitVector`.

CREATE TABLE vectors (
    model_id     TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    vector       BLOB NOT NULL,
    PRIMARY KEY (model_id, content_hash)
) WITHOUT ROWID;
