from pydantic_ai.mcp import MCPServerStdio
import json
import logging

logger = logging.getLogger(__name__)

def create_firecrawl_mcp_server(config) -> MCPServerStdio:
    """Create and configure the FireCrawl MCP server"""
    logger.debug("Configuring FireCrawl MCP server...")
    
    # Configure environment variables for the MCP server
    env = {
        "FIRECRAWL_API_KEY": config.api_key,
        "FIRECRAWL_RETRY_MAX_ATTEMPTS": str(config.max_retries),
        "FIRECRAWL_RETRY_INITIAL_DELAY": str(config.initial_delay),
        "FIRECRAWL_RETRY_MAX_DELAY": str(config.max_delay),
        "FIRECRAWL_RETRY_BACKOFF_FACTOR": str(config.backoff_factor),
        "FIRECRAWL_CREDIT_WARNING_THRESHOLD": str(config.credit_warning),
        "FIRECRAWL_CREDIT_CRITICAL_THRESHOLD": str(config.credit_critical)
    }
    
    logger.debug(f"MCP Server Configuration: {json.dumps(env, indent=2)}")
    
    try:
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                'firecrawl-mcp'
            ],
            env=env
        )
        
        logger.info("FireCrawl MCP server configured successfully")
        return server
        
    except Exception as e:
        logger.error(f"Failed to create MCP server: {str(e)}")
        raise 