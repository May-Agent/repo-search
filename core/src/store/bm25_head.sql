SELECT c.id, bm25(fts) AS score
FROM fts JOIN chunks c ON c.id = fts.rowid
WHERE fts MATCH :query
