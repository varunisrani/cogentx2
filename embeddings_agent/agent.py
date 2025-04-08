"""
Embeddings Agent Module

This module provides the setup for the Embeddings agent using the MCP protocol.
"""

import logging
import sys
import traceback
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from embeddings_agent.models import Config, EmbeddingsDeps
from embeddings_agent.tools import display_mcp_tools, execute_mcp_tool, create_mcp_server

# System prompt for Embeddings agent
system_prompt = """
You are an embeddings expert with access to text embedding generation, vector database operations, and similarity search tools.

Your task is to help users with embeddings-related operations like:
1. Generating embeddings for text
2. Storing embeddings in vector databases
3. Searching for similar content using embeddings
4. Processing and chunking text for efficient embedding
5. Analyzing semantic similarity between texts

Only answer questions related to embeddings and vector operations. For other queries, explain that you are specialized in embeddings-related operations.
"""

async def setup_agent(config: Config) -> Agent:
    """Set up and initialize the Embeddings agent with MCP server."""
    try:
        # Create MCP server instance for Embeddings
        server = create_mcp_server(config.OPENAI_API_KEY, config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY, config.EMBEDDING_MODEL)
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for embeddings operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Embeddings MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def get_model(config: Config):
    """Get the LLM model for the agent."""
    try:
        provider = OpenAIProvider(api_key=config.LLM_API_KEY, base_url=config.BASE_URL)
        model = OpenAIModel(provider=provider, model=config.MODEL_CHOICE)
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        sys.exit(1)

def create_embeddings_agent(config: Config = None):
    """Create a standalone Embeddings agent with MCP tools."""
    if not config:
        from embeddings_agent.models import load_config
        config = load_config()
    
    # Create a client for API requests
    client = httpx.AsyncClient()
    
    # Initialize the Embeddings agent with tools
    embeddings_agent = Agent(
        get_model(config),
        system_prompt=system_prompt,
        deps_type=EmbeddingsDeps,
        retries=2
    )
    
    # Create and connect MCP server
    server = create_mcp_server(
        config.OPENAI_API_KEY, 
        config.SUPABASE_URL, 
        config.SUPABASE_SERVICE_KEY, 
        config.EMBEDDING_MODEL
    )
    embeddings_agent.mcp_servers = [server]
    
    return embeddings_agent
