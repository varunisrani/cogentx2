import logging
import sys
import traceback
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from models import Config, FileDeps
from tools import display_mcp_tools, execute_mcp_tool, create_mcp_server

# System prompt for filesystem agent
system_prompt = """
You are a filesystem operations expert with access to the filesystem through the MCP protocol to help users manage and interact with their files and directories.

Your job is to assist with filesystem operations and provide clear, accurate responses about the results.

Use the available MCP tools to:
- List files and directories
- Create, read, write, and delete files
- Search file contents
- Create and manage directories

When answering a query about filesystem operations, always:
1. Use absolute paths where necessary
2. Verify paths exist before operations
3. Handle errors gracefully
4. Provide clear feedback about operations performed

Example interactions:
Q: "List files in /path/to/dir"
A: "Here are the files in /path/to/dir:
[list of files...]"

Q: "Create a new file named test.txt"
A: "Created file test.txt at /path/to/test.txt"
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
    """Set up and initialize the filesystem agent with MCP server."""
    try:
        # Create MCP server instance for filesystem
        server = create_mcp_server(config.BASE_PATH)
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for filesystem operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with filesystem MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def create_filesystem_agent(config: Config = None):
    """Create a standalone filesystem agent with MCP tools."""
    if not config:
        from models import load_config
        config = load_config()
    
    # Create a client for filesystem operations
    client = httpx.AsyncClient()
    
    # Initialize the filesystem agent with tools
    filesystem_agent = Agent(
        get_model(config),
        system_prompt=system_prompt,
        deps_type=FileDeps,
        retries=2
    )
    
    # Create and connect MCP server
    server = create_mcp_server(config.BASE_PATH)
    filesystem_agent.mcp_servers = [server]
    
    return filesystem_agent
