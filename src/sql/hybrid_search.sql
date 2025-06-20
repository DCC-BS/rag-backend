-- 'keyboard' is a placeholder for the query text
-- '[1,2,3]' is a placeholder for the query embedding

WITH bm25_ranked AS (
    SELECT id, RANK() OVER (ORDER BY score DESC) AS rank
    FROM (
      SELECT id, paradedb.score(id) AS score
      FROM document_chunks
      WHERE chunk_text @@@ 'keyboard'
      ORDER BY paradedb.score(id) DESC
      LIMIT 20
    ) AS bm25_score
),
semantic_search AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> '[1,2,3]') AS rank
    FROM document_chunks
    ORDER BY embedding <=> '[1,2,3]'
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, bm25_ranked.id) AS id,
    COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
    COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) AS score,
    document_chunks.chunk_text,
    document_chunks.embedding,
    document_chunks.page_number,
    document_chunks.created_at,
    document_chunks.document_id,
    documents.file_name,
    documents.document_path,
    documents.access_roles
FROM semantic_search
FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
JOIN document_chunks ON document_chunks.id = COALESCE(semantic_search.id, bm25_ranked.id)
JOIN documents ON documents.id = document_chunks.document_id
ORDER BY score DESC, chunk_text
LIMIT 5;
