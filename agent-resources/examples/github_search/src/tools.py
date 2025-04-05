from pydantic_ai.mcp import MCPServerStdio
import json
import logging

logger = logging.getLogger(__name__)

def create_github_mcp_server(config) -> MCPServerStdio:
    """Create and configure the GitHub MCP server"""
    logger.debug("Configuring GitHub MCP server...")
    
    # Configure environment variables for the MCP server
    env = {
        "GITHUB_PERSONAL_ACCESS_TOKEN": config.access_token
    }
    
    logger.debug("MCP Server Configuration: Token configured (hidden for security)")
    
    try:
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                '@modelcontextprotocol/server-github'
            ],
            env=env
        )
        
        logger.info("GitHub MCP server configured successfully")
        return server
        
    except Exception as e:
        logger.error(f"Failed to create MCP server: {str(e)}")
        raise 