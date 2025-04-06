from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import json
import traceback

from models import Config
from tools import create_serper_mcp_server, create_spotify_mcp_server, display_mcp_tools
from tools import execute_mcp_tool, run_query

# Combined system prompt for Serper-Spotify agent
system_prompt = """
You are a powerful assistant with dual capabilities:

1. Web Search (Serper): You can search the web for information using the Serper API
   - Search for any information on the internet
   - Retrieve news articles, general knowledge, and up-to-date information
   - Find images and news about specific topics

2. Spotify Music: You can control and interact with Spotify
   - Search for songs, albums, and artists
   - Create and manage playlists
   - Control playback (play, pause, skip, etc.)
   - Get information about the user's library and recommendations

IMPORTANT USAGE NOTES:
- For Spotify search operations, a 'market' parameter (e.g., 'US') is required
- For web searches, try to be specific about what information you're looking for
- When the user asks a knowledge-based question, use web search
- When the user asks about music or wants to control Spotify, use the Spotify tools

When responding to the user, always be concise and helpful. If you don't know how to do something with 
the available tools, explain what you can do instead.
"""

def get_model(config: Config) -> OpenAIModel:
    """Initialize the OpenAI model with the provided configuration."""
    try:
        model = OpenAIModel(
            config.MODEL_CHOICE,
            provider=OpenAIProvider(
                base_url=config.BASE_URL,
                api_key=config.LLM_API_KEY
            )
        )
        logging.debug(f"Initialized model with choice: {config.MODEL_CHOICE}")
        return model
    except Exception as e:
        logging.error("Error initializing model: %s", e)
        sys.exit(1)

async def setup_agent(config: Config) -> Agent:
    """Set up and initialize the combined Serper-Spotify agent with both MCP servers."""
    try:
        # Create MCP server instances for both Serper and Spotify
        logging.info("Creating Serper MCP Server...")
        serper_server = create_serper_mcp_server(config.SERPER_API_KEY)
        
        logging.info("Creating Spotify MCP Server...")
        spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)
        
        # Create agent with both servers
        logging.info("Initializing agent with both Serper and Spotify MCP Servers...")
        agent = Agent(get_model(config), mcp_servers=[serper_server, spotify_server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility from both servers
        try:
            serper_tools = await display_mcp_tools(serper_server, "SERPER")
            logging.info(f"Found {len(serper_tools) if serper_tools else 0} MCP tools available for Serper operations")
            
            spotify_tools = await display_mcp_tools(spotify_server, "SPOTIFY")
            logging.info(f"Found {len(spotify_tools) if spotify_tools else 0} MCP tools available for Spotify operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with both Serper and Spotify MCP servers.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1) 