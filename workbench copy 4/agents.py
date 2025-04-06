from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import json
import traceback

from models import Config
from tools import create_mcp_server, display_mcp_tools, execute_mcp_tool, run_serper_query, run_spotify_query

class AgentConfig:
    def __init__(self, config: Config):
        self.config = config

def get_openai_model(config: Config) -> OpenAIModel:
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

async def setup_integrated_agent(config: Config) -> Agent:
    """Set up and initialize an agent that integrates both Serper and Spotify capabilities."""
    try:
        logging.info("Creating integrated MCP Servers for Serper and Spotify...")
        serper_server = create_mcp_server(config.SERPER_API_KEY)
        spotify_server = create_mcp_server(config.SPOTIFY_API_KEY)
        
        system_prompt = """
        You are a versatile assistant that can help users with both web search and Spotify interactions.
        
        Your capabilities include:
        - Initiating web searches and information retrieval via Serper
        - Searching for songs, albums, and artists on Spotify
        - Managing playlists and controlling playback on Spotify
        
        When responding to the user, always provide concise, informative, and actionable responses.
        Explain any limitations clearly and suggest alternative actions when necessary.
        """

        logging.info("Initializing agent with integrated Serper and Spotify MCP Servers...")
        agent = Agent(get_openai_model(config), mcp_servers=[serper_server, spotify_server])
        agent.system_prompt = system_prompt

        # Display and capture MCP tools for each service
        try:
            serper_tools = await display_mcp_tools(serper_server)
            logging.info(f"Found {len(serper_tools) if serper_tools else 0} Serper MCP tools available")
            
            spotify_tools = await display_mcp_tools(spotify_server)
            logging.info(f"Found {len(spotify_tools) if spotify_tools else 0} Spotify MCP tools available")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)

        logging.debug("Integrated agent setup complete with Serper and Spotify MCP servers.")
        return agent

    except Exception as e:
        logging.error("Error setting up integrated agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def initialize_integrated_agent(config: Config):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(setup_integrated_agent(config))