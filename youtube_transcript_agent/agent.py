from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import logging
import sys
import asyncio
import traceback

from models import Config
from tools import display_mcp_tools, execute_mcp_tool, create_mcp_server, run_youtube_query

# System prompt for YouTube transcript agent
system_prompt = """
You are a YouTube transcript assistant with the ability to extract and analyze transcripts from YouTube videos.

Your capabilities include:
- Getting transcripts from YouTube videos when provided with a URL
- Searching within transcripts for specific content
- Providing summaries of video content based on transcripts
- Analyzing the main topics and key points in videos

IMPORTANT: You need a valid YouTube video URL to extract transcripts. 
For best results, ensure the video has captions available.

Always provide concise and helpful responses. If you can't get a transcript, explain why and suggest alternatives.
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
    """Set up and initialize the YouTube transcript agent with MCP server."""
    try:
        # Create MCP server instance for YouTube transcript
        logging.info("Creating YouTube Transcript MCP Server...")
        server = create_mcp_server(config.YOUTUBE_API_KEY)
        
        # Create agent with server
        logging.info("Initializing agent with YouTube Transcript MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility
        try:
            tools = await display_mcp_tools(server)
            logging.info(f"Found {len(tools) if tools else 0} MCP tools available for YouTube transcript operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with YouTube Transcript MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1) 