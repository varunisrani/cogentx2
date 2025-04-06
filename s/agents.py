import logging
import sys
import asyncio
import traceback
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from models import Config
from tools import display_mcp_tools, create_mcp_server

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

async def setup_agent(config: Config, system_prompt: str, api_key: str) -> Agent:
    """Set up and initialize an agent with MCP server."""
    try:
        # Create MCP server instance
        logging.info(f"Creating MCP Server with API key: {api_key}...")
        server = create_mcp_server(api_key)

        # Create agent with server
        logging.info("Initializing agent with MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])

        # Set system prompt
        agent.system_prompt = system_prompt

        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)

        logging.debug("Agent setup complete with MCP server.")
        return agent

    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

# System prompt for Spotify agent
spotify_system_prompt = """
You are a helpful Spotify assistant that can help users interact with their Spotify account.

Your capabilities include:
- Searching for songs, albums, and artists
- Creating and managing playlists
- Controlling playback (play, pause, skip, etc.)
- Getting information about the user's library and recommendations

Always provide a helpful and concise response. When proceeding with an action, make sure you have all necessary information.
"""

# System prompt for GitHub agent
github_system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

Your capabilities include:
- Creating and managing repositories
- Working with issues and pull requests
- Searching for repositories and code
- Getting information about users and organizations

Always provide a helpful and concise response. When proceeding with an action, make sure you have all necessary information.
"""

# Combined system prompt for Spotify and GitHub functionalities
combined_system_prompt = """
You are a versatile assistant that can help users both with Spotify and GitHub.

Spotify capabilities include:
- Recommending songs and artists
- Managing playlists
- Providing information about the user's library

GitHub capabilities include:
- Creating and managing repositories
- Committing changes
- Providing information from repositories

Always provide a helpful and concise response. When proceeding with an action, make sure you have all necessary information.
"""

async def initialize_agent(agent_type: str, config: Config) -> Agent:
    """Initialize agent based on the agent type specified."""
    if agent_type == "combined":
        return await setup_agent(config, combined_system_prompt, config.SPOTIFY_API_KEY)
    elif agent_type == "spotify":
        return await setup_agent(config, spotify_system_prompt, config.SPOTIFY_API_KEY)
    elif agent_type == "github":
        return await setup_agent(config, github_system_prompt, config.GITHUB_PERSONAL_ACCESS_TOKEN)
    else:
        logging.error(f"Unrecognized agent type: {agent_type}")
        sys.exit(1)

async def setup_spotify_agent(config: Config) -> Agent:
    """Set up and initialize the Spotify agent with MCP server."""
    return await setup_agent(config, spotify_system_prompt, config.SPOTIFY_API_KEY)

async def setup_github_agent(config: Config) -> Agent:
    """Set up and initialize the GitHub agent with MCP server."""
    return await setup_agent(config, github_system_prompt, config.GITHUB_PERSONAL_ACCESS_TOKEN)