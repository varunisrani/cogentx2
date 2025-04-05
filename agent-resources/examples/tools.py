from pydantic_ai.mcp import MCPServerStdio
import json
import logging

logger = logging.getLogger(__name__)

def create_serper_mcp_server(api_key: str) -> MCPServerStdio:
    """Create and configure the Serper MCP server"""
    logger.debug("Configuring Serper MCP server...")
    
    serper_config = {
        "serperApiKey": api_key
    }
    
    server = MCPServerStdio(
        'npx',
        [
            '-y',
            '@smithery/cli@latest',
            'run',
            '@marcopesani/mcp-server-serper',
            '--config',
            json.dumps(serper_config)
        ]
    )
    
    logger.debug("Serper MCP server configured successfully")
    return server 