-- ============================================================
-- LexAssist — Supabase vector store setup
-- Run ONCE in Supabase SQL Editor → New Query → Run
--
-- IMPORTANT: If you ran the old setup (384-dim all-MiniLM),
-- drop and recreate the documents table before running this.
-- Old embeddings are incompatible with voyage-law-2 (1024-dim).
--
-- To reset:  DROP TABLE IF EXISTS documents;
-- ============================================================

-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table (1024 dims for voyage-law-2)
CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT        NOT NULL,
    embedding  VECTOR(1024),           -- voyage-law-2 output dimension
    metadata   JSONB       DEFAULT '{}',
    file_hash  TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. IVFFlat index for approximate nearest-neighbour search
--    lists=100 works well up to ~1M rows; increase for larger corpora.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 4. match_documents RPC — called by rag.py
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding   VECTOR(1024),
    match_count       INT  DEFAULT 5,
    filter_doc_type   TEXT DEFAULT NULL
)
RETURNS TABLE (
    id         BIGINT,
    content    TEXT,
    metadata   JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE
        filter_doc_type IS NULL
        OR d.metadata->>'doc_type' = filter_doc_type
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
