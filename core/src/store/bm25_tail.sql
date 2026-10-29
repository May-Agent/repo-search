ORDER BY bm25(fts), c.content_hash, c.path, c.start_line, c.id
LIMIT :limit
