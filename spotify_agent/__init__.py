"""
Spotify MCP Agent Package

This package provides a convenient interface to use Spotify API
through the Pydantic AI framework and MCP protocol.
"""

from spotify_agent.models import Config, load_config
from spotify_agent.agent import setup_agent, get_model
from spotify_agent.tools import display_mcp_tools, execute_mcp_tool, run_spotify_query, create_mcp_server
from spotify_agent.main import main

__all__ = [
    'Config',
    'load_config',
    'setup_agent',
    'get_model',
    'display_mcp_tools',
    'execute_mcp_tool',
    'run_spotify_query',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0" 