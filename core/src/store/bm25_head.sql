SELECT c.id,
    exact_bm25(fts, (SELECT rows FROM bm25_stats), (SELECT title_tokens FROM bm25_stats),
               (SELECT body_tokens FROM bm25_stats)) AS score
FROM fts JOIN chunks c ON c.id = fts.rowid
WHERE fts MATCH :query
