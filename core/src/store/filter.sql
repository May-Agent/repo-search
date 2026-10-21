    AND (:include_ignored OR c.path NOT IN (SELECT path FROM files WHERE ignored = 1))
    AND (CASE WHEN :docs THEN c.kind = :doc_kind ELSE c.kind <> :doc_kind END)
    AND (:paths IS NULL OR c.path IN (SELECT value FROM json_each(:paths)))
    AND (:extensions IS NULL
         OR EXISTS (SELECT 1 FROM json_each(:extensions) e
                    WHERE c.path LIKE '%.' || e.value))
    AND (:kinds IS NULL
         OR c.kind IN (SELECT value FROM json_each(:kinds))
         OR EXISTS (SELECT 1 FROM chunk_members m
                    WHERE m.chunk_id = c.id
                      AND m.kind IN (SELECT value FROM json_each(:kinds))))
    AND (:symbol IS NULL
         OR EXISTS (SELECT 1 FROM chunk_members m
                    WHERE m.chunk_id = c.id AND m.symbol GLOB :symbol))