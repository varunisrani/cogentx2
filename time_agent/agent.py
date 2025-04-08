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
        logging.error(f"Error details: {traceback.format_exc()}")
        sys.exit(1)
