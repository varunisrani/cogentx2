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
import argparse
import colorlog
from logging.handlers import RotatingFileHandler

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Firecrawl MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='firecrawl_agent.log', help='Log file path')
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
    FIRECRAWL_API_KEY: str
    FIRECRAWL_RETRY_MAX_ATTEMPTS: str = "5"
    FIRECRAWL_RETRY_INITIAL_DELAY: str = "2000"
    FIRECRAWL_RETRY_MAX_DELAY: str = "30000"
    FIRECRAWL_RETRY_BACKOFF_FACTOR: str = "3"
    FIRECRAWL_CREDIT_WARNING_THRESHOLD: str = "2000"
    FIRECRAWL_CREDIT_CRITICAL_THRESHOLD: str = "500"

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("FIRECRAWL_API_KEY"):
            missing_vars.append("FIRECRAWL_API_KEY")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional

# Optional Firecrawl settings
FIRECRAWL_RETRY_MAX_ATTEMPTS=5
FIRECRAWL_RETRY_INITIAL_DELAY=2000
FIRECRAWL_RETRY_MAX_DELAY=30000
FIRECRAWL_RETRY_BACKOFF_FACTOR=3
FIRECRAWL_CREDIT_WARNING_THRESHOLD=2000
FIRECRAWL_CREDIT_CRITICAL_THRESHOLD=500
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            FIRECRAWL_API_KEY=os.getenv("FIRECRAWL_API_KEY"),
            FIRECRAWL_RETRY_MAX_ATTEMPTS=os.getenv("FIRECRAWL_RETRY_MAX_ATTEMPTS", "5"),
            FIRECRAWL_RETRY_INITIAL_DELAY=os.getenv("FIRECRAWL_RETRY_INITIAL_DELAY", "2000"),
            FIRECRAWL_RETRY_MAX_DELAY=os.getenv("FIRECRAWL_RETRY_MAX_DELAY", "30000"),
            FIRECRAWL_RETRY_BACKOFF_FACTOR=os.getenv("FIRECRAWL_RETRY_BACKOFF_FACTOR", "3"),
            FIRECRAWL_CREDIT_WARNING_THRESHOLD=os.getenv("FIRECRAWL_CREDIT_WARNING_THRESHOLD", "2000"),
            FIRECRAWL_CREDIT_CRITICAL_THRESHOLD=os.getenv("FIRECRAWL_CREDIT_CRITICAL_THRESHOLD", "500")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["FIRECRAWL_API_KEY"] = "***" if safe_config["FIRECRAWL_API_KEY"] else None
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
    """Display available Firecrawl MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("FIRECRAWL MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category (these categories will be modified once we know Firecrawl's tools)
            categories = {
                'Web Search': [],
                'Web Browsing': [],
                'Web Scraping': [],
                'Data Analysis': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
                if 'search' in name.lower():
                    categories['Web Search'].append(name)
                elif 'browse' in name.lower() or 'navigate' in name.lower():
                    categories['Web Browsing'].append(name)
                elif 'scrape' in name.lower() or 'extract' in name.lower():
                    categories['Web Scraping'].append(name)
                elif 'analyze' in name.lower() or 'process' in name.lower():
                    categories['Data Analysis'].append(name)
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
            
            # Display example usage (will be updated with actual examples once we know Firecrawl's capabilities)
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'Search for recent news about artificial intelligence'")
            logging.info("2. 'Find information about climate change from scientific sources'")
            logging.info("3. 'Extract product details from amazon.com/dp/B07PXGQC1Q'")
            logging.info("4. 'Browse to wikipedia.org and find information about quantum computing'")
            logging.info("5. 'Compare prices of iPhone 15 across major retailers'")
            
        else:
            logging.warning("\nNo Firecrawl MCP tools were discovered. This could mean either:")
            logging.warning("1. The Firecrawl MCP server doesn't expose any tools")
            logging.warning("2. The tools discovery mechanism is not supported")
            logging.warning("3. The server connection is not properly initialized")
                
    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)

async def setup_agent(config: Config) -> Agent:
    try:
        # Create MCP server instance for Firecrawl
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                'firecrawl-mcp'
            ],
            env={
                "FIRECRAWL_API_KEY": config.FIRECRAWL_API_KEY,
                "FIRECRAWL_RETRY_MAX_ATTEMPTS": config.FIRECRAWL_RETRY_MAX_ATTEMPTS,
                "FIRECRAWL_RETRY_INITIAL_DELAY": config.FIRECRAWL_RETRY_INITIAL_DELAY,
                "FIRECRAWL_RETRY_MAX_DELAY": config.FIRECRAWL_RETRY_MAX_DELAY,
                "FIRECRAWL_RETRY_BACKOFF_FACTOR": config.FIRECRAWL_RETRY_BACKOFF_FACTOR,
                "FIRECRAWL_CREDIT_WARNING_THRESHOLD": config.FIRECRAWL_CREDIT_WARNING_THRESHOLD,
                "FIRECRAWL_CREDIT_CRITICAL_THRESHOLD": config.FIRECRAWL_CREDIT_CRITICAL_THRESHOLD
            }
        )
        
        # Create agent with server
        agent = Agent(get_model(config), mcp_servers=[server])
        
        # Remove the call to display_mcp_tools as it's causing errors
        # await display_mcp_tools(server)
        
        logging.debug("Agent setup complete with Firecrawl MCP server.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        sys.exit(1)

async def main():
    # Parse command line arguments and set up logging
    args = parse_args()
    logger = setup_logging(args)
    
    logger.info("Starting Firecrawl MCP Agent")
    config = load_config()
    agent = await setup_agent(config)
    
    try:
        async with agent.run_mcp_servers():
            logger.info("Firecrawl MCP Server started successfully")
            
            print("\n" + "="*50)
            print("🔥 FIRECRAWL MCP AGENT 🔥")
            print("="*50)
            print("Type 'exit', 'quit', or press Ctrl+C to exit.\n")
            
            while True:
                try:
                    # Get query from user
                    user_query = input("\n🔍 Enter your query: ")
                    
                    # Check if user wants to exit
                    if user_query.lower() in ['exit', 'quit', '']:
                        print("Exiting Firecrawl agent...")
                        break
                    
                    # Log the query
                    logger.info(f"Processing query: '{user_query}'")
                    print(f"\nProcessing query: '{user_query}'")
                    print("This may take a moment...\n")
                    
                    # Run the query through the agent
                    start_time = asyncio.get_event_loop().time()
                    result = await agent.run(user_query)
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    
                    # Log and display the result
                    logger.info(f"Query completed in {elapsed_time:.2f} seconds")
                    
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
        logger.info("Firecrawl agent shutting down")
        print("\nThank you for using the Firecrawl agent!")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1) 