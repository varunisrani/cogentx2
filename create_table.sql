-- Create agent_embeddings table
CREATE TABLE IF NOT EXISTS agent_embeddings (
    id SERIAL PRIMARY KEY,
    folder_name TEXT NOT NULL,
    agents_code TEXT,
    main_code TEXT,
    models_code TEXT,
    tools_code TEXT,
    mcp_json TEXT,
    purpose TEXT,
    metadata JSONB,
    embedding VECTOR(1536),  -- For OpenAI's text-embedding-3-small model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a function to search by embedding similarity
CREATE OR REPLACE FUNCTION search_agent_embeddings(
    query_embedding VECTOR(1536),
    similarity_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id INTEGER,
    folder_name TEXT,
    purpose TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.id,
        t.folder_name,
        t.purpose,
        t.metadata,
        1 - (t.embedding <=> query_embedding) as similarity
    FROM agent_embeddings t
    WHERE 1 - (t.embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Create an index to speed up similarity searches
CREATE INDEX IF NOT EXISTS agent_embeddings_embedding_idx 
ON agent_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); 