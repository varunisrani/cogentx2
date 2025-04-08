"""
Weather MCP Agent Package

This package provides a convenient interface to use OpenWeather API
through the Pydantic AI framework and MCP protocol.
"""

from weather_agent.models import Config, WeatherDeps, load_config
from weather_agent.agent import setup_agent, get_model, create_weather_agent
from weather_agent.tools import display_mcp_tools, run_weather_query, execute_mcp_tool, create_mcp_server
from weather_agent.main import main

__all__ = [
    'Config', 
    'WeatherDeps', 
    'load_config',
    'setup_agent', 
    'get_model', 
    'create_weather_agent',
    'display_mcp_tools',
    'run_weather_query',
    'execute_mcp_tool',
    'create_mcp_server',
    'main'
]

__version__ = "0.1.0"
