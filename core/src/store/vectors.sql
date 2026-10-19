-- Repository-wide vector cache, shared by every worktree. The fp16 components
-- of a vector live in the model's flat file (`vectors.v1.<model hash>.f16`) at
-- row `slot`; this database maps content hashes to slots.

CREATE TABLE vectors (
    model_id     TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    slot         INTEGER NOT NULL,
    PRIMARY KEY (model_id, content_hash)
) WITHOUT ROWID;
-- `since` reads a model's newest slots; the primary key orders hashes, not slots.
CREATE INDEX vectors_by_slot ON vectors (model_id, slot);

-- `next_slot` is the first unallocated row of the model's file. The calibration
-- (per-dimension bounds the in-memory int8 codes are quantized with, as
-- little-endian f32, and the vector count it was taken at) is NULL until set.
CREATE TABLE models (
    model_id    TEXT NOT NULL PRIMARY KEY,
    dim         INTEGER NOT NULL,
    next_slot   INTEGER NOT NULL,
    calib_lo    BLOB,
    calib_hi    BLOB,
    calib_count INTEGER
) WITHOUT ROWID;
