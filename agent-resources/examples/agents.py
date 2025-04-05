from pydantic_ai import Agent
import logging
from models import SearchResponse, SerperConfig
from tools import create_serper_mcp_server

logger = logging.getLogger(__name__)

class SerperAgent:
    def __init__(self, config: SerperConfig):
        self.config = config
        logger.debug("Initializing Serper Agent...")
        
        self.agent = Agent(
            f'openai:{config.model_name}',
            system_prompt=(
                "You are a search assistant that provides accurate and relevant "
                "information from web searches. Summarize the results clearly and concisely."
            ),
            result_type=SearchResponse
        )
        logger.debug("Agent initialized with model: %s", config.model_name)
        
        # Configure Serper MCP server
        self.mcp_server = create_serper_mcp_server(config.serper_api_key)
        logger.info("Serper agent initialized successfully")
        
    async def search(self, query: str) -> SearchResponse:
        """Perform a search using Serper and process results"""
        try:
            logger.info(f"Executing search query: {query}")
            logger.debug("Setting up MCP servers...")
            
            # Set up MCP servers for the agent
            self.agent.mcp_servers = [self.mcp_server]
            
            # Log the full request being sent
            request = (
                f"Search for: {query}\n"
                f"Provide results in a structured format with title, snippet, and link for each result.\n"
                f"Include a brief summary of the key findings."
            )
            logger.debug(f"Sending request to agent:\n{request}")
            
            result = await self.agent.run(request)
            logger.info("Search completed successfully")
            return result.data
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise 