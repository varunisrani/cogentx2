from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import traceback

from models import Config
from tools import create_mcp_server, display_mcp_tools, execute_mcp_tool, run_firecrawl_query

def get_model(config: Config) -> OpenAIModel:
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
    try:
        # Create MCP server instance for Firecrawl
        server = create_mcp_server(
            config.FIRECRAWL_API_KEY,
            config.FIRECRAWL_RETRY_MAX_ATTEMPTS,
            config.FIRECRAWL_RETRY_INITIAL_DELAY,
            config.FIRECRAWL_RETRY_MAX_DELAY,
            config.FIRECRAWL_RETRY_BACKOFF_FACTOR,
            config.FIRECRAWL_CREDIT_WARNING_THRESHOLD,
            config.FIRECRAWL_CREDIT_CRITICAL_THRESHOLD
        )
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Display and capture MCP tools for visibility (now enabled)
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for Firecrawl operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Firecrawl MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

async def process_query(agent: Agent, user_query: str) -> tuple:
    """Process a user query and return the result with elapsed time"""
    try:
        start_time = asyncio.get_event_loop().time()
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        return result, elapsed_time
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        raise 