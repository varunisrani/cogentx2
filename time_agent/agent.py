import logging
import sys
import traceback
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from models import Config

# System prompt for Time agent
system_prompt = """
You are a time expert with access to timezone conversion, time arithmetic, and time formatting tools.

Your task is to help users with time-related operations like:
1. Getting current time in any timezone
2. Converting between timezones  
3. Calculating time differences
4. Parsing natural language time expressions
5. Formatting dates/times in various formats

Only answer questions related to time operations. For other queries, explain that you are specialized in time-related operations.
"""

async def setup_agent(config: Config) -> Agent:
    """Set up and initialize the Time agent with MCP server."""
    try:
        # Create MCP server instance for Time operations
        server = MCPServerStdio(
            'python', 
            ['-m', 'mcp_server_time', f'--local-timezone={config.LOCAL_TIMEZONE}']
        )
        
        # Create agent with server
        agent = Agent(system_prompt=system_prompt, mcp_servers=[server])
        
        logging.debug("Time agent setup complete")
        return agent
            
    except Exception as e:
        logging.error(f"Error setting up agent: {str(e)}")
async def setup_agent(config: Config) -> Agent:
    """Set up and initialize the Time agent with MCP server."""
    try:
        # Create MCP server instance for Time
        server = create_mcp_server()
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for Time operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Time MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def create_time_agent(config: Config = None):
    """Create a standalone Time agent with MCP tools."""
    if not config:
        from models import load_config
        config = load_config()
    
    # Initialize the Time agent with tools
    time_agent = Agent(
        get_model(config),
        system_prompt=system_prompt,
        deps_type=TimeDeps,
        retries=2
    )
    
    # Create and connect MCP server
    server = create_mcp_server()
    time_agent.mcp_servers = [server]
    
    return time_agent
