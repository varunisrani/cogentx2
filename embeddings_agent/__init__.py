"""
Embeddings MCP Agent Package

This package provides a convenient interface to use embeddings-related functions
through the Pydantic AI framework and MCP protocol.
"""

from embeddings_agent.models import Config, EmbeddingsDeps, load_config
from embeddings_agent.agent import setup_agent, get_model, create_embeddings_agent
from embeddings_agent.tools import display_mcp_tools, run_embeddings_query, execute_mcp_tool, create_mcp_server
from embeddings_agent.main import main

__all__ = [
    'Config', 
    'EmbeddingsDeps', 
    'load_config',
    'setup_agent', 
    'get_model', 
    'create_embeddings_agent',
    'display_mcp_tools',
    'run_embeddings_query',
    'execute_mcp_tool',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0"
