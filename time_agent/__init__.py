"""
Time MCP Agent Package

This package provides a convenient interface to use time-related functions
through the Pydantic AI framework and MCP protocol.
"""

from time_agent.models import Config, TimeDeps, load_config
from time_agent.agent import setup_agent, get_model
from time_agent.tools import create_mcp_server, run_time_query, execute_mcp_tool, display_mcp_tools
from time_agent.main import main

__all__ = [
    'Config',
    'TimeDeps',
    'load_config',
    'setup_agent',
    'get_model',
    'create_mcp_server',
    'run_time_query',
    'execute_mcp_tool',
    'display_mcp_tools',
    'main'
]

__version__ = "0.1.0"
