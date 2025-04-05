from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import json
import traceback

from models import Config
from tools import create_mcp_server, display_mcp_tools, execute_mcp_tool, run_serper_query

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
        # Create MCP server instance for Serper
        logging.info("Creating Serper MCP Server...")
        server = create_mcp_server(config.SERPER_API_KEY)
        
        # Create agent with server
        logging.info("Initializing agent with Serper MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for Serper operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Serper MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1) 