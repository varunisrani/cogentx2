import logging
import sys
import traceback
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from models import Config, GitHubDeps
from tools import display_mcp_tools, execute_mcp_tool, create_mcp_server

# System prompt for GitHub agent
system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the repository with the provided tools before answering the user's question unless you have already.

When answering a question about the repo, always start your answer with the full repo URL in brackets and then give your answer on a newline. Like:

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

async def setup_agent(config: Config) -> Agent:
    """Set up and initialize the GitHub agent with MCP server."""
    try:
        # Create MCP server instance for GitHub
        server = create_mcp_server(config.GITHUB_PERSONAL_ACCESS_TOKEN)
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for GitHub operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with GitHub MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def create_github_agent(config: Config = None):
    """Create a standalone GitHub agent with MCP tools."""
    if not config:
        from models import load_config
        config = load_config()
    
    # Create a client for GitHub API requests
    client = httpx.AsyncClient()
    
    # Initialize the GitHub agent with tools
    github_agent = Agent(
        get_model(config),
        system_prompt=system_prompt,
        deps_type=GitHubDeps,
        retries=2
    )
    
    # Create and connect MCP server
    server = create_mcp_server(config.GITHUB_PERSONAL_ACCESS_TOKEN)
    github_agent.mcp_servers = [server]
    
    return github_agent 