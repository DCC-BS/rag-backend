-- Create the 'documents' table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    document_path VARCHAR(1024) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    access_roles TEXT[] NOT NULL
);

-- Create the 'document_chunks' table
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1024),
    page_number INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_documents_access_roles ON documents USING GIN (access_roles);
