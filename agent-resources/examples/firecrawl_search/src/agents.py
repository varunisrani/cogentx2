from pydantic_ai import Agent
import logging
import json
from .models import FireCrawlResponse, FireCrawlConfig
from .tools import create_firecrawl_mcp_server

logger = logging.getLogger(__name__)

class FireCrawlAgent:
    def __init__(self, config: FireCrawlConfig):
        self.config = config
        logger.debug("Initializing FireCrawl Agent...")
        
        try:
            self.agent = Agent(
                f'openai:{config.model_name}',
                system_prompt=(
                    """
                    You are an advanced web crawling agent that uses FireCrawl to search and analyze web content.
                    Your task is to crawl web pages and extract relevant information based on the user's query.
                    For each result, provide:
                    1. A clear title of the webpage
                    2. A relevant snippet of content
                    3. The URL of the page
                    4. When available, the timestamp of when the page was crawled
                    
                    Additionally, provide a concise summary of all findings, highlighting the most important
                    information discovered during the crawl. Focus on accuracy and relevance while maintaining
                    a structured and easy-to-read format.
                    """
                ),
                result_type=FireCrawlResponse
            )
            logger.debug(f"Agent initialized with model: {config.model_name}")
            
            # Configure FireCrawl MCP server
            logger.info("Setting up FireCrawl MCP server...")
            self.mcp_server = create_firecrawl_mcp_server(config)
            logger.info("FireCrawl agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FireCrawlAgent: {str(e)}")
            raise
        
    async def crawl(self, query: str) -> FireCrawlResponse:
        """Perform a web crawl using FireCrawl and process results"""
        try:
            logger.info(f"Executing crawl query: {query}")
            logger.debug("Setting up MCP servers...")
            
            # Set up MCP servers for the agent
            self.agent.mcp_servers = [self.mcp_server]
            logger.debug("MCP servers configured for agent")
            
            # Log the full request being sent
            request = (
                f"Crawl and analyze web content for: {query}\n"
                f"Provide results in a structured format with title, snippet, URL, and timestamp for each result.\n"
                f"Include a comprehensive summary of the key findings."
            )
            logger.debug(f"Sending request to agent:\n{request}")
            
            # Execute crawl
            logger.debug("Executing crawl through MCP server...")
            result = await self.agent.run(request)
            
            # Log the results
            if hasattr(result, 'data'):
                logger.debug(f"Raw crawl results: {json.dumps(result.data.dict(), indent=2)}")
                logger.info("Crawl completed successfully")
                return result.data
            else:
                logger.error("Crawl result does not have expected 'data' attribute")
                raise ValueError("Invalid crawl result format")
                
        except Exception as e:
            logger.error(f"Crawl failed: {str(e)}", exc_info=True)
            raise 