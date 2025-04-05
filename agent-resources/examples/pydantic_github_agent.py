from __future__ import annotations as _annotations

import asyncio
import os
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Any, List, Dict
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import shutil
import time
import re
import json

import httpx
import logfire
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from devtools import debug
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, ValidationError
import argparse
import colorlog
from logging.handlers import RotatingFileHandler

load_dotenv()

def get_model():
    llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', os.getenv('OPENAI_API_KEY'))

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class GitHubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None

system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the repository with the provided tools before answering the user's question unless you have already.

When answering a question about the repo, always start your answer with the full repo URL in brackets and then give your answer on a newline. Like:

[Using https://github.com/[repo URL from the user]]

Your answer here...
"""

# Initialize the GitHub agent with the model from get_model function
github_agent = Agent(
    get_model(),
    system_prompt=system_prompt,
    deps_type=GitHubDeps,
    retries=2
)

@github_agent.tool
async def get_repo_info(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get repository information including size and description using GitHub API.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Repository information as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository info: {response.text}"
    
    data = response.json()
    size_mb = data['size'] / 1024
    
    return (
        f"Repository: {data['full_name']}\n"
        f"Description: {data['description']}\n"
        f"Size: {size_mb:.1f}MB\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Language: {data['language']}\n"
        f"Created: {data['created_at']}\n"
        f"Last Updated: {data['updated_at']}"
    )

@github_agent.tool
async def get_repo_structure(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get the directory structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Directory structure as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get repository structure: {response.text}"
    
    data = response.json()
    tree = data['tree']
    
    # Build directory structure
    structure = []
    for item in tree:
        if not any(excluded in item['path'] for excluded in ['.git/', 'node_modules/', '__pycache__/']):
            structure.append(f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}")
    
    return "\n".join(structure)

@github_agent.tool
async def get_file_content(ctx: RunContext[GitHubDeps], github_url: str, file_path: str) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        file_path: Path to the file within the repository.

    Returns:
        str: File content as a string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get file content: {response.text}"
    
    return response.text

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='GitHub MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='github_agent.log', help='Log file path')
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
    GITHUB_PERSONAL_ACCESS_TOKEN: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            missing_vars.append("GITHUB_PERSONAL_ACCESS_TOKEN")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_access_token
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            GITHUB_PERSONAL_ACCESS_TOKEN=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] = "***" if safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] else None
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
    """Display available GitHub MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("GITHUB MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category (these categories will be modified once we know GitHub's tools)
            categories = {
                'Repositories': [],
                'Issues': [],
                'Pull Requests': [],
                'Users': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
                if 'repo' in name.lower():
                    categories['Repositories'].append(name)
                elif 'issue' in name.lower():
                    categories['Issues'].append(name)
                elif 'pull' in name.lower() or 'pr' in name.lower():
                    categories['Pull Requests'].append(name)
                elif 'user' in name.lower():
                    categories['Users'].append(name)
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
            
            # Display example usage
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'List my GitHub repositories'")
            logging.info("2. 'Search for repositories about machine learning'")
            logging.info("3. 'Show open issues in repository X'")
            logging.info("4. 'Create a new issue in repository Y'")
            logging.info("5. 'Check the status of PR #123 in repository Z'")
            
        else:
            logging.warning("\nNo GitHub MCP tools were discovered. This could mean either:")
            logging.warning("1. The GitHub MCP server doesn't expose any tools")
            logging.warning("2. The tools discovery mechanism is not supported")
            logging.warning("3. The server connection is not properly initialized")
                
    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)

async def setup_agent(config: Config) -> Agent:
    try:
        # Create MCP server instance for GitHub
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                '@modelcontextprotocol/server-github'
            ],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": config.GITHUB_PERSONAL_ACCESS_TOKEN
            }
        )
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Display MCP tools for visibility
        try:
            await display_mcp_tools(server)
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with GitHub MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        sys.exit(1)

async def process_query(agent: Agent, user_query: str) -> tuple:
    """Process a user query and return the result with elapsed time and tool usage"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Execute the query and track execution
        logging.info(f"Processing query: '{user_query}'")
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
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

async def main():
    # Parse command line arguments and set up logging
    args = parse_args()
    logger = setup_logging(args)
    
    logger.info("Starting GitHub MCP Agent")
    config = load_config()
    agent = await setup_agent(config)
    
    try:
        async with agent.run_mcp_servers():
            logger.info("GitHub MCP Server started successfully")
            
            print("\n" + "="*50)
            print("📊 GITHUB MCP AGENT 📊")
            print("="*50)
            print("Type 'exit', 'quit', or press Ctrl+C to exit.\n")
            
            while True:
                try:
                    # Get query from user
                    user_query = input("\n🔍 Enter your query: ")
                    
                    # Check if user wants to exit
                    if user_query.lower() in ['exit', 'quit', '']:
                        print("Exiting GitHub agent...")
                        break
                    
                    # Process the query and get results
                    result, elapsed_time, tool_usage = await process_query(agent, user_query)
                    
                    # Display the result
                    print("\n" + "="*50)
                    print("RESULT:")
                    print("="*50)
                    print(result.data)
                    print("="*50)
                    print(f"Query completed in {elapsed_time:.2f} seconds")
                    
                except KeyboardInterrupt:
                    logger.info("User interrupted the process")
                    print("\nExiting due to keyboard interrupt...")
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)
                    print(f"\n❌ Error: {str(e)}")
                    print("Please try a different query or check the logs for details.")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("GitHub agent shutting down")
        print("\nThank you for using the GitHub agent!")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)