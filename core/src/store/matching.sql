SELECT c.id FROM fts JOIN chunks c ON c.id = fts.rowid
WHERE fts MATCH :query
    AND c.id IN (SELECT value FROM json_each(:ids))
ORDER BY c.id
