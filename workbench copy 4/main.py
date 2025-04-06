import asyncio
import logging
import sys
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import subprocess

from models import load_config
from agent import setup_agent
from tools import run_serper_query, run_spotify_query

def parse_args():
    parser = argparse.ArgumentParser(description='Serper and Spotify Integrated Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    parser.add_argument('--api', choices=['serper', 'spotify', 'both'], required=True, help='Select the API(s) to query')
    return parser.parse_args()

def setup_logging(args):
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
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
    
    file_handler = RotatingFileHandler(
        args.log_file,
        maxBytes=args.max_log_size,
        backupCount=args.log_backups
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    return root_logger

def display_tool_usage(tool_usage, api):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return
    
    header = "🔍 TOOLS USED:"
    print(f"\n{header}")
    print("-"*50)
    
    for i, tool in enumerate(tool_usage, 1):
        tool_name = tool.get('name', 'Unknown Tool')
        tool_params = tool.get('parameters', {})
        
        print(f"{i}. Tool: {tool_name}")
        if tool_params:
            print("   Parameters:")
            for param, value in tool_params.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"   - {param}: {value}")
        print()

def display_startup_message(api):
    title = "🔍 SERPER AND SPOTIFY INTEGRATED AGENT 🔍"
    
    suggestions = (
        "\nTroubleshooting tips if issues occur:"
        "\n1. Ensure both Serper and Spotify API keys have necessary permissions"
        "\n2. Run 'npm install' if needed"
        "\n3. Check the log file for detailed error messages"
    ) if api == 'both' else ""
    
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print(suggestions)
    print("="*50)

async def main():
    args = parse_args()
    logger = setup_logging(args)
    
    api = args.api
    logger.info(f"Starting {api.capitalize()} Integrated Agent")
    
    try:
        node_version = subprocess.check_output(['node', '--version']).decode().strip()
        npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
        logger.info(f"Node.js version: {node_version}, npm version: {npm_version}")
    except Exception as e:
        logger.warning(f"Could not detect Node.js/npm: {e}. Ensure these are installed.")
    
    logger.info("Loading configuration...")
    config = load_config()
    
    logger.info("Setting up agent...")
    agent = await setup_agent(config)
    
    try:
        async with agent.run_mcp_servers():
            logger.info(f"{api.capitalize()} MCP Server started successfully")
            
            display_startup_message(api)
            
            while True:
                try:
                    user_query = input(f"\nEnter your query for {api}: ")
                    
                    if user_query.lower() in ['exit', 'quit', '']:
                        print(f"Exiting {api.capitalize()} agent...")
                        break
                    
                    logger.info(f"Processing query: '{user_query}'")
                    print(f"\nProcessing query: '{user_query}'")
                    print("This may take a moment...\n")
                    
                    query_runners = {
                        'serper': run_serper_query,
                        'spotify': run_spotify_query,
                        'both': run_serper_query  # Adjusted logic for illustration
                    }
                    
                    results = []
                    for service in api.split(','):
                        run_query = query_runners[service.strip()]
                        try:
                            result, elapsed_time, tool_usage = await run_query(agent, user_query)
                            logger.info(f"Query for {service} completed in {elapsed_time:.2f} seconds")
                            results.append((result, elapsed_time, tool_usage, service))
                        except Exception as query_error:
                            logger.error(f"Error processing {service} query: {query_error}")
                            print(f"\n❌ Error with {service} query: {query_error}")
                    
                    for result, elapsed_time, tool_usage, service in results:
                        display_tool_usage(tool_usage, service)
                        print("\n" + "="*50)
                        print(f"RESULTS FROM {service.upper()}:")
                        print("="*50)
                        print(result.data)
                        print("="*50)
                        print(f"{service.capitalize()} query completed in {elapsed_time:.2f} seconds")
                
                except KeyboardInterrupt:
                    logger.info("User interrupted the process")
                    print("\nExiting due to keyboard interrupt...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    print(f"\n❌ Error: {e}")
                    print("Please try again or check the logs for details.")
            
    except Exception as server_error:
        logger.error(f"Error running agent server: {server_error}", exc_info=True)
        print(f"\n❌ Error running {api.capitalize()} agent server: {server_error}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Node.js is installed (version 16+)")
        print("2. Run 'npm install' to install dependencies")
        print("3. Verify your API keys are valid with proper permissions")
        print("4. Check the log file for more details")
    finally:
        logger.info(f"{api.capitalize()} agent shutting down")
        print(f"\nThank you for using the {api.capitalize()} agent!")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)