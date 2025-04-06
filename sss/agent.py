import logging
import sys
import traceback
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import Config
from tools import create_spotify_mcp_server, create_github_mcp_server, display_tool_usage

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

async def setup_agent(config: Config, service_type: str = None) -> Agent:
    """Set up and initialize the agent with MCP servers."""
    try:
        # Create MCP server instances
        logging.info("Creating MCP Servers...")

        servers = []

        # Determine which server to create based on service_type
        if service_type == 'spotify' and config.SPOTIFY_API_KEY:
            spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)
            servers.append(spotify_server)
            logging.info("Spotify MCP Server created")
        elif service_type == 'github' and config.GITHUB_PERSONAL_ACCESS_TOKEN:
            github_server = create_github_mcp_server(config.GITHUB_PERSONAL_ACCESS_TOKEN)
            servers.append(github_server)
            logging.info("GitHub MCP Server created")
        # If no service_type specified, try to create all available servers
        elif not service_type:
            if config.SPOTIFY_API_KEY:
                spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)
                servers.append(spotify_server)
                logging.info("Spotify MCP Server created")
            if config.GITHUB_PERSONAL_ACCESS_TOKEN:
                github_server = create_github_mcp_server(config.GITHUB_PERSONAL_ACCESS_TOKEN)
                servers.append(github_server)
                logging.info("GitHub MCP Server created")

        if not servers:
            raise ValueError("No MCP servers could be created. Check your API keys/tokens.")

        # Create agent with servers
        logging.info("Initializing agent with MCP Servers...")
        agent = Agent(get_model(config), mcp_servers=servers)

        # Set system prompt based on available servers
        if len(servers) == 1:
            if config.SPOTIFY_API_KEY:
                agent.system_prompt = spotify_system_prompt
            else:
                agent.system_prompt = github_system_prompt
        else:
            # Combined system prompt for both services
            agent.system_prompt = f"""
            You are an assistant that can help with both Spotify music recommendations and GitHub repository management.

            For Spotify:
            {spotify_system_prompt}

            For GitHub:
            {github_system_prompt}

            Please determine which service the user is asking about and respond accordingly.
            """

        # Display available tools for each server
        for server in servers:
            try:
                server_name = "SPOTIFY" if config.SPOTIFY_API_KEY else "GITHUB"
                tools = await display_tool_usage(server, server_name)
                logging.info(f"Found {len(tools) if tools else 0} MCP tools available for {server_name} operations")
            except Exception as tool_err:
                logging.warning(f"Could not display MCP tools: {str(tool_err)}")
                logging.debug("Tool display error details:", exc_info=True)

        logging.debug("Agent setup complete with MCP servers.")
        return agent

    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)
