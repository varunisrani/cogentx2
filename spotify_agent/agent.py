from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import traceback

from models import Config
from tools import display_mcp_tools, execute_mcp_tool, create_mcp_server, run_spotify_query

# System prompt for Spotify agent
system_prompt = """
You are a helpful Spotify assistant that can help users interact with their Spotify account.

Your capabilities include:
- Searching for songs, albums, and artists
- Creating and managing playlists
- Controlling playback (play, pause, skip, etc.)
- Getting information about the user's library and recommendations

IMPORTANT: When using Spotify API tools, be aware of these requirements:
- For searching or getting top tracks, a 'market' parameter (e.g., 'US') is required
- Playlist operations need playlist IDs
- Most track operations require track IDs

When responding to the user, always be concise and helpful. If you don't know how to do something with the available tools, 
explain what you can do instead.
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
    """Set up and initialize the Spotify agent with MCP server."""
    try:
        # Create MCP server instance for Spotify
        logging.info("Creating Spotify MCP Server...")
        server = create_mcp_server(config.SPOTIFY_API_KEY)
        
        # Create agent with server
        logging.info("Initializing agent with Spotify MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for Spotify operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Spotify MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1) 