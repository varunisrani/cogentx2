"""
Filesystem MCP Agent Package

This package provides a convenient interface to interact with filesystem
through the Pydantic AI framework and MCP protocol.
"""

from file_agent.models import Config, FileDeps, load_config
from file_agent.agent import setup_agent, get_model, create_filesystem_agent
from file_agent.tools import display_mcp_tools, run_file_query, execute_mcp_tool, create_mcp_server
from file_agent.main import main

__all__ = [
    'Config', 
    'FileDeps', 
    'load_config',
    'setup_agent', 
    'get_model', 
    'create_filesystem_agent',
    'display_mcp_tools', 
    'run_file_query', 
    'execute_mcp_tool',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0"
