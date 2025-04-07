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
from tools import run_spotify_query

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Spotify MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='spotify_agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    parser.add_argument('--streamlit', action='store_true', help='Running in Streamlit mode')
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
    
    print("\n🎵 SPOTIFY TOOLS USED:")
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

def display_startup_message():
    print("\n" + "="*50)
    print("🎵 SPOTIFY MCP AGENT 🎵")
    print("="*50)
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("\nTroubleshooting tips if issues occur:")
    print("1. Make sure your Spotify API key has the necessary permissions")
    print("2. Run 'npm install' to make sure all dependencies are installed")
    print("3. Check the log file for detailed error messages")
    print("="*50)

async def process_query(agent, user_query, logger):
    """Process a single query and return the result"""
    try:
        # Log the query
        logger.info(f"Processing query: '{user_query}'")
        print(f"\nProcessing Spotify query: '{user_query}'")
        print("This may take a moment...\n")
        
        # Run the query through the agent
        result, elapsed_time, tool_usage = await run_spotify_query(agent, user_query)
        
        # Log and display the result
        logger.info(f"Query completed in {elapsed_time:.2f} seconds")
        
        # Display the tools that were used
        display_tool_usage(tool_usage)
        
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(result.data)
        print("="*50)
        print(f"Query completed in {elapsed_time:.2f} seconds")
        
        # Return success
        return True
    except Exception as query_error:
        logger.error(f"Error processing query: {str(query_error)}")
        print(f"\n❌ Error: {str(query_error)}")
        print("Please try a different query or check the logs for details.")
        print("\nSuggestions:")
        print("1. Make sure your Spotify API key has the necessary permissions")
        print("2. Try a different query format")
        print("3. Check that you have an active Spotify account")
        
        # Return failure
        return False

async def main():
    # Parse command line arguments and set up logging
    args = parse_args()
    logger = setup_logging(args)
    
    try:
        logger.info("Starting Spotify MCP Agent")
        
        # Check for Node.js
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
                logger.info("Spotify MCP Server started successfully")
                
                display_startup_message()
                
                # In Streamlit mode, we just print the prompt and exit the main loop
                # The actual query processing will be handled by the process_query function
                # which will be called directly from the Streamlit terminal
                if args.streamlit:
                    print("\n🎵 Enter your Spotify query: ", end="", flush=True)
                    
                    # Wait for input in a way that doesn't block the event loop
                    while True:
                        # This is just to keep the server running
                        # The actual input will be handled by the Streamlit terminal
                        await asyncio.sleep(1)
                        
                        # Check if there's any input available
                        if sys.stdin in asyncio.get_event_loop()._ready:
                            user_query = input()
                            
                            # Check if user wants to exit
                            if user_query.lower() in ['exit', 'quit', '']:
                                print("Exiting Spotify agent...")
                                break
                            
                            # Process the query
                            await process_query(agent, user_query, logger)
                            
                            # Print the prompt again
                            print("\n🎵 Enter your Spotify query: ", end="", flush=True)
                else:
                    # Regular terminal mode
                    while True:
                        try:
                            # Get query from user
                            user_query = input("\n🎵 Enter your Spotify query: ")
                            
                            # Check if user wants to exit
                            if user_query.lower() in ['exit', 'quit', '']:
                                print("Exiting Spotify agent...")
                                break
                            
                            # Process the query
                            await process_query(agent, user_query, logger)
                            
                        except KeyboardInterrupt:
                            logger.info("User interrupted the process")
                            print("\nExiting due to keyboard interrupt...")
                            break
                        except Exception as e:
                            logger.error(f"Error in main loop: {str(e)}")
                            logger.error(f"Error details: {traceback.format_exc()}")
                            print(f"\n❌ Error in main loop: {str(e)}")
                            print("Please try again or check the logs for details.")
                
        except Exception as server_error:
            logger.error(f"Error running MCP server: {str(server_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            print(f"\n❌ Error running Spotify MCP server: {str(server_error)}")
            print("\nTroubleshooting steps:")
            print("1. Make sure you have Node.js installed (version 16+)")
            print("2. Run 'npm install' to install dependencies")
            print("3. Check that your Spotify API key is valid and has proper permissions")
            print("4. Check the log file for more detailed error information")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)
    finally:
        logger.info("Spotify agent shutting down")
        print("\nThank you for using the Spotify agent!")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
