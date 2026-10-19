SELECT c.id FROM fts JOIN chunks c ON c.id = fts.rowid
WHERE fts MATCH :query
