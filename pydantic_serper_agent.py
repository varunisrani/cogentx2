#!/usr/bin/env python3
"""
Pydantic Web Search Agent
--------------------
A Pydantic AI agent that uses MCP for web search.
"""
from __future__ import annotations as _annotations

import asyncio
import os
import json
import sys
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, ValidationError
import argparse
import colorlog
from logging.handlers import RotatingFileHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_search_agent")

# Silence logfire warnings
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

# Try to import logfire for detailed agent logs
try:
    import logfire
    logfire.configure(send_to_logfire='never')
    has_logfire = True
    logger.info("Logfire imported successfully for detailed agent logging")
except ImportError:
    has_logfire = False
    logger.info("Logfire not available - will use basic logging")

# Load environment variables
load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Serper MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='serper_agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    return parser.parse_args()

# Configure logging with colors and better formatting
def setup_logging(args):
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for complete logs
    file_handler = RotatingFileHandler(
        args.log_file,
        maxBytes=args.max_log_size,
        backupCount=args.log_backups
    )
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress verbose logging from libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    return root_logger

class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SERPER_API_KEY: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("SERPER_API_KEY"):
            missing_vars.append("SERPER_API_KEY")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["SERPER_API_KEY"] = "***" if safe_config["SERPER_API_KEY"] else None
        logging.debug(f"Config values: {safe_config}")
        return config
    except ValidationError as e:
        logging.error("Configuration validation error:")
        for error in e.errors():
            logging.error(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {str(e)}")
        sys.exit(1)

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

async def display_mcp_tools(server: MCPServerStdio):
    """Display available Serper MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("SERPER MCP TOOLS AND FUNCTIONS")
        logging.info("="*50)
        
        # Try different methods to get tools based on MCP protocol
        tools = []
        
        try:
            # Method 1: Try getting tools through server's session
            if hasattr(server, 'session') and server.session:
                response = await server.session.list_tools()
                if hasattr(response, 'tools'):
                    tools = response.tools
                elif isinstance(response, dict):
                    tools = response.get('tools', [])
            
            # Method 2: Try direct tool listing if available
            elif hasattr(server, 'list_tools'):
                response = await server.list_tools()
                if hasattr(response, 'tools'):
                    tools = response.tools
                elif isinstance(response, dict):
                    tools = response.get('tools', [])
                    
            # Method 3: Try fallback method for older MCP versions
            elif hasattr(server, 'listTools'):
                response = server.listTools()
                if isinstance(response, dict):
                    tools = response.get('tools', [])
                    
        except Exception as tool_error:
            logging.debug(f"Error listing tools: {tool_error}", exc_info=True)
        
        if tools:
            # Group tools by category
            categories = {
                'Search': [],
                'Web Search': [],
                'News Search': [],
                'Image Search': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
                if 'search' in name.lower() and 'image' in name.lower():
                    categories['Image Search'].append(name)
                elif 'search' in name.lower() and 'news' in name.lower():
                    categories['News Search'].append(name)
                elif 'search' in name.lower() and ('web' in name.lower() or 'google' in name.lower()):
                    categories['Web Search'].append(name)
                elif 'search' in name.lower():
                    categories['Search'].append(name)
                else:
                    categories['Other'].append(name)
                
            # Display tools by category
            for category, tool_names in categories.items():
                category_tools = []
                for name in tool_names:
                    if name in tool_dict:
                        category_tools.append(tool_dict[name])
                        uncategorized.discard(name)
                
                if category_tools:
                    logging.info(f"\n{category}")
                    logging.info("="*50)
                    
                    for tool in category_tools:
                        logging.info(f"\n📌 {tool.get('name')}")
                        logging.info("   " + "-"*40)
                        
                        if tool.get('description'):
                            logging.info(f"   📝 Description: {tool.get('description')}")
                        
                        if tool.get('parameters'):
                            logging.info("   🔧 Parameters:")
                            params = tool['parameters'].get('properties', {})
                            required = tool['parameters'].get('required', [])
                            
                            for param_name, param_info in params.items():
                                is_required = param_name in required
                                param_type = param_info.get('type', 'unknown')
                                description = param_info.get('description', '')
                                
                                logging.info(f"   - {param_name}")
                                logging.info(f"     Type: {param_type}")
                                logging.info(f"     Required: {'✅' if is_required else '❌'}")
                                if description:
                                    logging.info(f"     Description: {description}")
            
            logging.info("\n" + "="*50)
            logging.info(f"Total Available Tools: {len(tools)}")
            logging.info("="*50)
            
            # Display example usage for Serper searches
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'Search for recent news about artificial intelligence'")
            logging.info("2. 'Find information about climate change from scientific sources'")
            logging.info("3. 'Show me images of the Golden Gate Bridge'")
            logging.info("4. 'What are the latest updates on the Mars rover?'")
            logging.info("5. 'Search for recipes with chicken and rice'")
            
        else:
            logging.warning("\nNo Serper MCP tools were discovered. This could mean either:")
            logging.warning("1. The Serper MCP server doesn't expose any tools")
            logging.warning("2. The tools discovery mechanism is not supported")
            logging.warning("3. The server connection is not properly initialized")
                
    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)

def create_mcp_server(serper_api_key):
    """Create MCP server instance for Serper"""
    config = {
        "serperApiKey": serper_api_key
    }
    
    return MCPServerStdio(
        'npx',
        [
            '-y',
            '@smithery/cli@latest',
            'run',
            '@marcopesani/mcp-server-serper',
            '--config',
            json.dumps(config)
        ]
    )

async def setup_agent(config: Config) -> Agent:
    try:
        # Create MCP server instance for Serper
        logging.info("Creating Serper MCP Server...")
        server = create_mcp_server(config.SERPER_API_KEY)
        
        # Create agent with server
        logging.info("Initializing agent with Serper MCP Server...")
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Display MCP tools (for visibility)
        try:
            await display_mcp_tools(server)
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with Serper MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        import traceback
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)

async def process_query(agent: Agent, user_query: str) -> tuple:
    """Process a user query and return the result with elapsed time and tool usage"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Execute the query and track execution
        logging.info(f"Executing query: '{user_query}'")
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Extract tool usage information if available
        tool_usage = []
        if hasattr(result, 'metadata') and result.metadata:
            try:
                # Try to extract tool usage information from metadata
                if isinstance(result.metadata, dict) and 'tools' in result.metadata:
                    tool_usage = result.metadata['tools']
                elif hasattr(result.metadata, 'tools'):
                    tool_usage = result.metadata.tools
                
                # Handle case where tools might be nested further
                if not tool_usage and isinstance(result.metadata, dict) and 'tool_calls' in result.metadata:
                    tool_usage = result.metadata['tool_calls']
            except Exception as tool_err:
                logging.debug(f"Could not extract tool usage: {tool_err}")
        
        # Log the tools used
        if tool_usage:
            logging.info(f"Tools used in this query: {json.dumps(tool_usage, indent=2)}")
            logging.info(f"Number of tools used: {len(tool_usage)}")
            
            # Log details about each tool used
            for i, tool in enumerate(tool_usage):
                tool_name = tool.get('name', 'Unknown Tool')
                logging.info(f"Tool {i+1}: {tool_name}")
        else:
            logging.info("No specific tools were recorded for this query")
            
        return result, elapsed_time, tool_usage
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        import traceback
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

# Function to display tool usage in a user-friendly way
def display_tool_usage(tool_usage):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return
    
    print("\n🔍 SERPER TOOLS USED:")
    print("-"*50)
    
    for i, tool in enumerate(tool_usage, 1):
        tool_name = tool.get('name', 'Unknown Tool')
        tool_params = tool.get('parameters', {})
        
        print(f"{i}. Tool: {tool_name}")
        if tool_params:
            print("   Parameters:")
            for param, value in tool_params.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"   - {param}: {value}")
        print()

async def main():
    # Parse command line arguments and set up logging
    args = parse_args()
    logger = setup_logging(args)
    
    logger.info("Starting Serper MCP Agent")
    
    # Check for npm and node.js
    try:
        import subprocess
        node_version = subprocess.check_output(['node', '--version']).decode().strip()
        npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
        logger.info(f"Node.js version: {node_version}, npm version: {npm_version}")
    except Exception as e:
        logger.warning(f"Could not detect Node.js/npm: {str(e)}. Make sure these are installed.")
        
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Setup agent
    logger.info("Setting up agent...")
    agent = await setup_agent(config)
    
    try:
        async with agent.run_mcp_servers():
            logger.info("Serper MCP Server started successfully")
            
            print("\n" + "="*50)
            print("🔍 SERPER SEARCH AGENT 🔍")
            print("="*50)
            print("Type 'exit', 'quit', or press Ctrl+C to exit.\n")
            
            while True:
                try:
                    # Get query from user
                    user_query = input("\n🔍 Enter your search query: ")
                    
                    # Check if user wants to exit
                    if user_query.lower() in ['exit', 'quit', '']:
                        print("Exiting Serper agent...")
                        break
                    
                    # Log the query
                    logger.info(f"Processing query: '{user_query}'")
                    print(f"\nSearching for: '{user_query}'")
                    print("This may take a moment...\n")
                    
                    # Run the query through the agent
                    try:
                        result, elapsed_time, tool_usage = await process_query(agent, user_query)
                        
                        # Log and display the result
                        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
                        
                        # Display the tools that were used
                        display_tool_usage(tool_usage)
                        
                        print("\n" + "="*50)
                        print("SEARCH RESULTS:")
                        print("="*50)
                        print(result.data)
                        print("="*50)
                        print(f"Search completed in {elapsed_time:.2f} seconds")
                    except Exception as query_error:
                        logger.error(f"Error processing query: {str(query_error)}")
                        print(f"\n❌ Error: {str(query_error)}")
                        print("Please try a different query or check the logs for details.")
                    
                except KeyboardInterrupt:
                    logger.info("User interrupted the process")
                    print("\nExiting due to keyboard interrupt...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    print(f"\n❌ Error: {str(e)}")
                    print("Please try again or check the logs for details.")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Serper agent shutting down")
        print("\nThank you for using the Serper search agent!")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1) 