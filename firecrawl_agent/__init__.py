"""
Firecrawl MCP Agent Package

This package provides a convenient interface to use the Firecrawl web crawling service
through the Pydantic AI framework and MCP protocol.
"""

from firecrawl_agent.models import Config, load_config
from firecrawl_agent.agent import setup_agent, get_model
from firecrawl_agent.tools import display_mcp_tools, execute_mcp_tool, run_firecrawl_query, create_mcp_server
from firecrawl_agent.main import main

__all__ = [
    'Config',
    'load_config',
    'setup_agent',
    'get_model',
    'display_mcp_tools',
    'execute_mcp_tool',
    'run_firecrawl_query',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0" 