-- '[1,2,3]' is a placeholder for the query embedding
SELECT * FROM document_chunks ORDER BY embedding <=> '[1,2,3]'::vector;
