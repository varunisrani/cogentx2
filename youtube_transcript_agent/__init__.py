"""
YouTube Transcript Agent

A module for extracting and analyzing YouTube video transcripts using the MCP system.

Features:
- Extract full transcripts from YouTube videos
- Search within transcripts for specific content
- Generate summaries of video content
- Identify key points and topics in videos

The agent uses pydantic_ai and a YouTube Transcript MCP server to enable AI-powered
interactions with YouTube video transcripts.
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"

from .models import Config, load_config
from .agent import setup_agent
from .tools import create_mcp_server, display_mcp_tools, execute_mcp_tool, run_youtube_query

__all__ = [
    "Config", 
    "load_config",
    "setup_agent",
    "create_mcp_server",
    "display_mcp_tools", 
    "execute_mcp_tool",
    "run_youtube_query"
] 