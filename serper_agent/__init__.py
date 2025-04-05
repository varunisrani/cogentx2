"""
Serper MCP Agent Package

This package provides a convenient interface to use Serper search API
through the Pydantic AI framework and MCP protocol.
"""

from serper_agent.models import Config, load_config
from serper_agent.agent import setup_agent, get_model
from serper_agent.tools import display_mcp_tools, execute_mcp_tool, run_serper_query, create_mcp_server
from serper_agent.main import main

__all__ = [
    'Config',
    'load_config',
    'setup_agent',
    'get_model',
    'display_mcp_tools',
    'execute_mcp_tool',
    'run_serper_query',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0" 