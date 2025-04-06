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

async def setup_agent(
    serper_api_key=None, 
    spotify_api_key=None, 
    agent_model=None, 
    system_prompt=None
):
    """Set up an agent with Serper (Google Search) and Spotify tools.
    
    Args:
        serper_api_key: API key for Serper
        spotify_api_key: API key for Spotify
        agent_model: The model to use for the agent
        system_prompt: The system prompt to use for the agent
        
    Returns:
        Agent: The configured agent
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize server variables to None
    serper_server = None
    spotify_server = None
    
    # Keep track of activated servers
    activated_servers = []
    
    # Try to create the Serper MCP server
    if serper_api_key:
        try:
            logger.info("Creating Serper MCP server...")
            serper_server = create_serper_mcp_server(serper_api_key)
            
            if serper_server:
                logger.info("Serper MCP server created successfully")
                activated_servers.append("Serper")
            else:
                logger.warning("Failed to create Serper MCP server")
        except Exception as e:
            logger.error(f"Error creating Serper MCP server: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
    else:
        logger.warning("No Serper API key provided - Serper search will not be available")
    
    # Try to create the Spotify MCP server
    if spotify_api_key:
        try:
            logger.info("Creating Spotify MCP server...")
            spotify_server = create_spotify_mcp_server(spotify_api_key)
            
            if spotify_server:
                logger.info("Spotify MCP server created successfully")
                activated_servers.append("Spotify")
            else:
                logger.warning("Failed to create Spotify MCP server")
        except Exception as e:
            logger.error(f"Error creating Spotify MCP server: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
    else:
        logger.warning("No Spotify API key provided - Spotify features will not be available")
    
    # Log the number of active servers
    logger.info(f"Active servers: {len(activated_servers)} ({', '.join(activated_servers) if activated_servers else 'None'})")
    
    # Check if at least one server is available
    if not activated_servers:
        logger.error("No MCP servers could be created. Please check your API keys and try again.")
        logger.error("To diagnose issues, run the test_mcp_revised.py script.")
        return None
    
    # Adjust the system prompt based on available servers
    if system_prompt is None:
        # Start with a base prompt
        base_prompt = """You are a helpful assistant with access to information from"""
        
        if "Serper" in activated_servers and "Spotify" in activated_servers:
            system_prompt = base_prompt + " Google Search and Spotify."
        elif "Serper" in activated_servers:
            system_prompt = base_prompt + " Google Search."
        elif "Spotify" in activated_servers:
            system_prompt = base_prompt + " Spotify."
        else:
            system_prompt = "You are a helpful assistant."
            
        system_prompt += """ Respond concisely unless the human requests detailed information. 
        Always structure your answer to be as helpful and easy to understand as possible.
        """
    
    # Configure the agent with the available servers
    agent_config = {}
    
    # Add Serper server if available
    if serper_server:
        agent_config["serper"] = serper_server
        
    # Add Spotify server if available
    if spotify_server:
        agent_config["spotify"] = spotify_server
    
    # Create the agent
    logger.info("Creating agent with available tools...")
    agent = Agent(
        mcp_servers=agent_config,
        model=agent_model,
        system_prompt=system_prompt
    )
    
    return agent 