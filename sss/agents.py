import logging
import sys
import traceback
import asyncio
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from models import Config, GitHubDeps
from tools import recommend_songs_from_spotify, manage_github_repository, create_mcp_server

# System prompt for Spotify song recommendation
spotify_system_prompt = """
You are a Spotify assistant with the ability to recommend songs to users.

Your capabilities include:
- Recommending songs based on user preferences
- Searching for songs, albums, and artists
- Getting information about the user's library and recommendations

IMPORTANT: For song search or recommendations, ensure that the 'market' parameter (e.g., 'US') is specified.

Always provide concise and helpful recommendations. Explain what you can do if a requested action is not possible.
"""

# System prompt for GitHub repository management
github_system_prompt = """
You are a GitHub assistant with access to create and manage repositories on behalf of the user.

You can perform tasks such as:
- Creating new repositories
- Managing existing repositories
- Providing information about repositories

When responding, include the full repository URL in brackets and provide a concise response on the next line.

For example:
[Using https://github.com/[repo URL from the user]]
Your answer here...
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

async def setup_agent(config: Config, service_type: str) -> Agent:
    """Set up and initialize the agent with MCP server for given service type."""
    try:
        if service_type == 'Spotify':
            mcp_server_key = config.SPOTIFY_API_KEY
            system_prompt = spotify_system_prompt
        elif service_type == 'GitHub':
            mcp_server_key = config.GITHUB_PERSONAL_ACCESS_TOKEN
            system_prompt = github_system_prompt
        else:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Create MCP server instance
        logging.info(f"Creating {service_type} MCP Server...")
        server = create_mcp_server(mcp_server_key)

        # Create agent with server
        logging.info(f"Initializing agent with {service_type} MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])

        # Set system prompt
        agent.system_prompt = system_prompt

        # Load tools for operations visibility
        try:
            tools = await (recommend_songs_from_spotify if service_type == 'Spotify' else manage_github_repository)(server)
            logging.info(f"Loaded {len(tools) if tools else 0} tools for {service_type} operations.")
        except Exception as tool_err:
            logging.warning(f"Could not display tools: {str(tool_err)}")
            logging.debug("Tool load error details:", exc_info=True)

        logging.debug(f"Agent setup complete with {service_type} MCP server.")
        return agent

    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def setup_spotify_agent(config: Config) -> Agent:
    """Create a standalone Spotify agent with its tools for recommending songs."""
    return asyncio.run(setup_agent(config, 'Spotify'))

def setup_github_agent(config: Config) -> Agent:
    """Create a standalone GitHub agent with its tools for managing repositories."""
    return asyncio.run(setup_agent(config, 'GitHub'))