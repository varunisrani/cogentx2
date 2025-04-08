# Embeddings Processor

This tool processes agent templates, creates embeddings, and stores them in a Supabase vector database for similarity search.

## Features

- Process agent templates from time_agent, file_agent, and weather_agent
- Generate embeddings using OpenAI's text-embedding model
- Store embeddings and agent code in Supabase for similarity search
- Search for similar agent templates based on natural language queries

## Setup

1. **Create a .env file with your API keys:**

```
# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=sk-your-openai-api-key

# LLM API Key (for agent operations)
LLM_API_KEY=sk-your-llm-api-key

# Supabase Credentials (required for database operations)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your-supabase-service-key

# Optional: Embedding model to use
EMBEDDING_MODEL=text-embedding-3-small
```

2. **Set up the Supabase database:**

Make sure your Supabase database has the pgvector extension enabled and the necessary tables created. You can use the `create_table.sql` script to set up the required tables and functions.

## Usage

### Process All Agents

To process all agents (time_agent, file_agent, and weather_agent) and store them in Supabase:

```bash
python run_embeddings_processor.py batch
```

### Process a Single Agent

To process a single agent folder:

```bash
python run_embeddings_processor.py process time_agent
```

### Search for Similar Templates

To search for similar templates based on a natural language query:

```bash
python run_embeddings_processor.py search "agent that can work with time and dates"
```

## How It Works

1. The processor reads the agent code files (agent.py, main.py, models.py, tools.py, etc.)
2. It analyzes the code to determine the agent's purpose and capabilities
3. It creates a combined text representation of the agent
4. It generates an embedding vector using OpenAI's embedding model
5. It stores the agent code, metadata, and embedding in Supabase
6. When searching, it compares the query embedding with stored embeddings to find similar agents

## Files

- `embeddings_processor.py` - Main processor for creating embeddings for agent templates
- `batch_process_agents.py` - Script to process multiple agent folders at once
- `run_embeddings_processor.py` - Command-line interface for the embeddings processor
- `create_table.sql` - SQL queries to create the database tables and search functions

## Database Schema

The `agent_embeddings` table in Supabase has the following structure:

- `id` - Unique identifier
- `folder_name` - Name of the agent folder
- `agents_code` - Content of agent.py
- `main_code` - Content of main.py
- `models_code` - Content of models.py
- `tools_code` - Content of tools.py
- `mcp_json` - Content of mcp.json
- `purpose` - Determined purpose of the agent
- `metadata` - JSON object with additional metadata
- `embedding` - Vector embedding of the agent
- `created_at` - Timestamp of creation
