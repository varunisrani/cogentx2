"""
Serper-Spotify Combined Agent

This module combines the capabilities of both Serper search and Spotify music
into a single unified agent that can perform web searches and control Spotify.
"""

__version__ = "1.0.0"
__author__ = "Archon"

from .agent import setup_agent
from .models import Config, load_config
from .tools import (
    create_serper_mcp_server,
    create_spotify_mcp_server,
    display_mcp_tools,
    execute_mcp_tool,
    run_query
)

__all__ = [
    "setup_agent",
    "Config",
    "load_config",
    "create_serper_mcp_server",
    "create_spotify_mcp_server", 
    "display_mcp_tools",
    "execute_mcp_tool",
    "run_query"
] 