from pydantic_ai import Agent
import logging
import json
from datetime import datetime
from .models import GitHubSearchResponse, GitHubConfig
from .tools import create_github_mcp_server

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class GitHubAgent:
    def __init__(self, config: GitHubConfig):
        self.config = config
        logger.debug("Initializing GitHub Agent...")
        
        try:
            self.agent = Agent(
                f'openai:{config.model_name}',
                system_prompt=(
                    """
                    You are a GitHub search agent that helps users find repositories and issues.
                    Your task is to search GitHub and provide detailed information about:
                    
                    For Repositories:
                    1. Repository name and owner
                    2. Description and primary language
                    3. Stars and forks count
                    4. Last update timestamp
                    5. Repository URL
                    
                    For Issues:
                    1. Issue title and number
                    2. Current state (open/closed)
                    3. Creation date and labels
                    4. Issue description
                    5. Issue URL
                    
                    IMPORTANT: For every search, you MUST:
                    1. Always include a 'summary' field with a concise overview
                    2. Highlight the most relevant repositories and issues
                    3. Explain why the results are relevant to the query
                    4. If no results are found, provide suggestions for alternative search terms
                    
                    Format your response carefully to ensure all required fields are included
                    and properly formatted according to the GitHubSearchResponse schema.
                    """
                ),
                result_type=GitHubSearchResponse
            )
            logger.debug(f"Agent initialized with model: {config.model_name}")
            
            # Configure GitHub MCP server
            logger.info("Setting up GitHub MCP server...")
            self.mcp_server = create_github_mcp_server(config)
            logger.info("GitHub agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GitHubAgent: {str(e)}")
            raise
        
    async def search(self, query: str) -> GitHubSearchResponse:
        """Perform a GitHub search and process results"""
        try:
            logger.info(f"Executing GitHub search query: {query}")
            logger.debug("Setting up MCP servers...")
            
            # Set up MCP servers for the agent
            self.agent.mcp_servers = [self.mcp_server]
            logger.debug("MCP servers configured for agent")
            
            # Log the full request being sent
            request = (
                f"Search GitHub for: {query}\n"
                f"Provide results in a structured format with repository and issue information.\n"
                f"Include a comprehensive summary of the findings.\n"
                f"Limit results to {self.config.max_results} items per category."
            )
            logger.debug(f"Sending request to agent:\n{request}")
            
            # Execute search
            logger.debug("Executing search through MCP server...")
            result = await self.agent.run(request)
            
            # Log the results using the custom encoder
            if hasattr(result, 'data'):
                logger.debug(f"Raw search results: {json.dumps(result.data.dict(), indent=2, cls=DateTimeEncoder)}")
                logger.info("Search completed successfully")
                return result.data
            else:
                logger.error("Search result does not have expected 'data' attribute")
                raise ValueError("Invalid search result format")
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise 