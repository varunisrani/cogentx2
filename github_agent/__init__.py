"""
GitHub MCP Agent Package

This package provides a convenient interface to use GitHub API
through the Pydantic AI framework and MCP protocol.
"""

from github_agent.models import Config, GitHubDeps, load_config
from github_agent.agent import setup_agent, get_model, create_github_agent
from github_agent.tools import display_mcp_tools, run_github_query, execute_mcp_tool, create_mcp_server
from github_agent.main import main

__all__ = [
    'Config', 
    'GitHubDeps', 
    'load_config',
    'setup_agent', 
    'get_model', 
    'create_github_agent',
    'display_mcp_tools', 
    'run_github_query', 
    'execute_mcp_tool',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0" 