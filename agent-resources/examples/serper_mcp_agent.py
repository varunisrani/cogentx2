from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import asyncio
import os
import json
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Model for search results"""
    title: str = Field(description="Title of the search result")
    snippet: str = Field(description="Snippet/description of the search result")
    link: str = Field(description="URL of the search result")

    def __str__(self):
        return f"Title: {self.title}\nSnippet: {self.snippet}\nURL: {self.link}\n"

class SearchResponse(BaseModel):
    """Model for the complete search response"""
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="List of search results")
    summary: str = Field(description="AI-generated summary of results")

    def __str__(self):
        results_str = "\n".join(str(r) for r in self.results)
        return f"Query: {self.query}\nSummary: {self.summary}\nResults:\n{results_str}"

class SerperConfig(BaseModel):
    """Configuration for the Serper agent"""
    serper_api_key: str = Field(description="API key for Serper")
    model_name: str = Field(default="gpt-4o-mini", description="Model to use for the agent")
    temperature: float = Field(default=0.7, description="Temperature for model responses")

def load_config() -> SerperConfig:
    """Load configuration from environment variables"""
    load_dotenv()
    
    config = SerperConfig(
        serper_api_key=os.getenv("SERPER_API_KEY"),
        model_name=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
    )
    logger.info("Configuration loaded successfully")
    logger.debug(f"Using model: {config.model_name}")
    return config

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
        serper_config = {
            "serperApiKey": config.serper_api_key
        }
        logger.debug("Configuring Serper MCP server...")
        
        self.mcp_server = MCPServerStdio(
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
            
            # Log the raw response data
            logger.debug(f"Raw response data: {json.dumps(result.data.dict(), indent=2)}")
            
            logger.info("Search completed successfully")
            return result.data
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise

async def main():
    # Load configuration
    config = load_config()
    
    # Initialize agent
    agent = SerperAgent(config)
    
    try:
        # Example search query
        query = "What are the latest developments in quantum computing?"
        logger.info(f"Starting search for query: {query}")
        result = await agent.search(query)
        
        # Print results in a formatted way
        print("\n" + "="*50)
        print("SEARCH RESULTS")
        print("="*50)
        print(f"\nSearch Query: {result.query}\n")
        print("Summary:")
        print("-"*50)
        print(result.summary)
        print("\nDetailed Results:")
        print("-"*50)
        for i, r in enumerate(result.results, 1):
            print(f"\n{i}. {r.title}")
            print(f"   {'-'*40}")
            print(f"   {r.snippet}")
            print(f"   URL: {r.link}")
            print()
            
    except KeyboardInterrupt:
        logger.info("Exiting gracefully...")
        print("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...") 