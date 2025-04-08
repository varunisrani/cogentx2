import logging
import sys
import traceback
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

from models import Config, WeatherDeps
from tools import display_mcp_tools, execute_mcp_tool, create_mcp_server

# System prompt for Weather agent
system_prompt = """
You are a weather expert with access to OpenWeather API to help users get accurate weather information and forecasts.

Your only job is to assist with weather-related queries and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look up the weather with the provided tools before answering the user's question.

When answering a question, always start with the location in brackets and then give your answer on a newline. Like:

[Weather for: San Francisco, CA]

Your answer here...

Make sure to include:
- Current temperature and conditions
- Humidity and wind speed when available 
- High/low temperatures for the day
- Brief forecast if requested
"""

def get_model(config: Config) -> OpenAIModel:
    """Initialize the OpenAI model with the provided configuration."""
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
    """Set up and initialize the Weather agent with MCP server."""
    try:
        # Create MCP server instance for Weather
        server = create_mcp_server(config.OPENWEATHER_API_KEY)
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for weather operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Weather MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

def create_weather_agent(config: Config = None):
    """Create a standalone Weather agent with MCP tools."""
    if not config:
        from models import load_config
        config = load_config()
    
    # Create a client for Weather API requests
    client = httpx.AsyncClient()
    
    # Initialize the Weather agent with tools
    weather_agent = Agent(
        get_model(config),
        system_prompt=system_prompt,
        deps_type=WeatherDeps,
        retries=2
    )
    
    # Create and connect MCP server
    server = create_mcp_server(config.OPENWEATHER_API_KEY)
    weather_agent.mcp_servers = [server]
    
    return weather_agent
