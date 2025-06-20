CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);


CREATE INDEX search_idx ON document_chunks
USING bm25 (id, chunk_text, page_number, created_at)
WITH (key_field='id');
