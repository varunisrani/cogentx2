import asyncio
import logging
import sys
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import json
import traceback

from models import load_config
from agent import setup_agent
from tools import run_firecrawl_query

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

# Function to display tool usage in a user-friendly way
def display_tool_usage(tool_usage):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return
    
    print("\n🔧 FIRECRAWL TOOLS USED:")
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
                    result, elapsed_time, tool_usage = await run_firecrawl_query(agent, user_query)
                    
                    # Log and display the result
                    logger.info(f"Query completed in {elapsed_time:.2f} seconds")
                    
                    # Display the tools that were used
                    display_tool_usage(tool_usage)
                    
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