from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import asyncio
import os
import sys
from typing import Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    GITHUB_TOKEN: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("GITHUB_TOKEN"):
            missing_vars.append("GITHUB_TOKEN")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_personal_access_token
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.dict()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["GITHUB_TOKEN"] = "***" if safe_config["GITHUB_TOKEN"] else None
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
    """Display available MCP tools and their details"""
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
            # Group tools by category
            categories = {
                'Repository Management': ['create_repository', 'fork_repository', 'create_branch'],
                'File Operations': ['create_or_update_file', 'get_file_contents', 'push_files'],
                'Issues': ['create_issue', 'list_issues', 'update_issue', 'get_issue', 'add_issue_comment'],
                'Pull Requests': ['create_pull_request', 'get_pull_request', 'list_pull_requests', 
                                'create_pull_request_review', 'merge_pull_request', 'get_pull_request_files',
                                'get_pull_request_status', 'update_pull_request_branch', 
                                'get_pull_request_comments', 'get_pull_request_reviews'],
                'Search': ['search_repositories', 'search_code', 'search_issues', 'search_users'],
                'Commits': ['list_commits'],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
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
            
            # Display any uncategorized tools
            if uncategorized:
                logging.info("\nOther Tools")
                logging.info("="*50)
                for name in uncategorized:
                    tool = tool_dict[name]
                    logging.info(f"\n📌 {tool.get('name')}")
                    if tool.get('description'):
                        logging.info(f"   📝 Description: {tool.get('description')}")
            
            logging.info("\n" + "="*50)
            logging.info(f"Total Available Tools: {len(tools)}")
            logging.info("="*50)
            
            # Display example usage
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'List my GitHub repositories'")
            logging.info("2. 'Create a new repository named my-project'")
            logging.info("3. 'Search for Python repositories with more than 1000 stars'")
            logging.info("4. 'Create an issue in owner/repo titled Bug Report'")
            logging.info("5. 'List open pull requests in owner/repo'")
            
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
                "GITHUB_PERSONAL_ACCESS_TOKEN": config.GITHUB_TOKEN
            }
        )
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Try to display available tools
        await display_mcp_tools(server)
        
        logging.debug("Agent setup complete with GitHub MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        sys.exit(1)

async def main():
    config = load_config()
    agent = await setup_agent(config)
    
    try:
        async with agent.run_mcp_servers():
            logging.debug("Running GitHub MCP servers.")
            # Example GitHub query - you can modify this based on what you want to do
            result = await agent.run('create a new repository named my-project')
            logging.debug(f"Received result: {result.data}")
            print(result.data)
    except Exception as e:
        logging.error("Error during execution: %s", e)
        sys.exit(1)
    finally:
        try:
            user_input = input("Press enter to quit...")
        except KeyboardInterrupt:
            logging.info("Exiting gracefully...")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        sys.exit(1)