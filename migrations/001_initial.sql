-- Migration: 001_initial
-- Creates the context_engine memory table with all required fields

-- Create extension if not exists (superuser only, run manually if needed)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,

    -- Unique document identifier (SHA256 hash of content by default)
    doc_id VARCHAR(64) UNIQUE NOT NULL,

    -- The actual memory content
    content TEXT NOT NULL,

    -- 768-dimensional embedding vector (nomic-embed-text)
    embedding VECTOR(768),

    -- Organization
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',  -- Project isolation
    category VARCHAR(50) NOT NULL DEFAULT 'general',   -- Memory category
    source VARCHAR(50),                               -- Origin (file, user, agent)
    filename VARCHAR(255),                            -- Source file if applicable

    -- Classification
    importance FLOAT DEFAULT 1.0,   -- 0.1-10.0 importance score
    tags TEXT[],                    -- Array of tags for filtering

    -- Usage tracking
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,

    -- Session/conversation linking
    session_key VARCHAR(64),
    conversation_chain INTEGER[],   -- Links to parent conversation turns

    -- Expiration
    expires_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Metadata as JSONB for flexibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_namespace
    ON memories (namespace);

CREATE INDEX IF NOT EXISTS idx_memories_category
    ON memories (category);

CREATE INDEX IF NOT EXISTS idx_memories_namespace_category
    ON memories (namespace, category);

CREATE INDEX IF NOT EXISTS idx_memories_session_key
    ON memories (session_key);

CREATE INDEX IF NOT EXISTS idx_memories_expires_at
    ON memories (expires_at) WHERE expires_at IS NOT NULL;

-- Vector similarity search index
-- Note: IVFFlat is faster for large datasets; use HNSW for better recall
CREATE INDEX IF NOT EXISTS idx_memories_embedding_cosine
    ON memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Composite index for namespace + active (not expired)
CREATE INDEX IF NOT EXISTS idx_memories_namespace_active
    ON memories (namespace, created_at DESC)
    WHERE expires_at IS NULL OR expires_at > NOW();

COMMENT ON TABLE memories IS 'Semantic memory storage for context engine';
COMMENT ON COLUMN memories.namespace IS 'Project/isolation namespace';
COMMENT ON COLUMN memories.embedding IS '768-dim vector from nomic-embed-text';
