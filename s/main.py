import asyncio
import logging
import sys
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import json
import traceback
import subprocess

from models import load_config
from agents import setup_spotify_agent, setup_github_agent, initialize_agent
from tools import recommend_songs_from_spotify, manage_github_repository

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Music and Repo Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    parser.add_argument('--service-type', type=str, choices=['spotify', 'github'], required=True, help='Type of service to run (spotify or github)')
    return parser.parse_args()

# Configure logging with colors and better formatting
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

def display_tool_usage(tool_usage, service):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return

    print(f"\n🛠️ {service.upper()} TOOLS USED:")
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

def display_startup_message(service):
    print("\n" + "="*50)
    print(f"🎵 {service.capitalize()} AGENT 🎵" if service == 'spotify' else f"📁 {service.capitalize()} AGENT 📁")
    print("="*50)
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("\nTroubleshooting tips if issues occur:")
    print(f"1. Ensure the {service} API key/token has the necessary permissions")
    print(f"2. Ensure all dependencies are installed for the {service} agent")
    print("3. Check the log file for detailed error messages")
    print("="*50)

async def main():
    args = parse_args()
    logger = setup_logging(args)
    service = args.service_type

    try:
        logger.info(f"Starting {service.capitalize()} Agent")

        try:
            node_version = subprocess.check_output(['node', '--version']).decode().strip()
            npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
            logger.info(f"Node.js version: {node_version}, npm version: {npm_version}")
        except Exception as e:
            logger.warning(f"Could not detect Node.js/npm: {str(e)}. Ensure these are installed.")

        logger.info("Loading configuration...")
        config = load_config()

        logger.info("Setting up agent...")
        agent = await (setup_spotify_agent(config) if service == 'spotify' else setup_github_agent(config))

        run_query = recommend_songs_from_spotify if service == 'spotify' else manage_github_repository

        try:
            async with agent.run_mcp_servers():
                logger.info(f"{service.capitalize()} Server started successfully")

                display_startup_message(service)

                while True:
                    try:
                        user_query = input(f"\n🔍 Enter your {service.capitalize()} query: ")

                        if user_query.lower() in ['exit', 'quit', '']:
                            print(f"Exiting {service} agent...")
                            break

                        logger.info(f"Processing query: '{user_query}'")
                        print(f"\nProcessing {service.capitalize()} query: '{user_query}'")
                        print("This may take a moment...\n")

                        try:
                            result, elapsed_time, tool_usage = await run_query(agent, user_query)

                            logger.info(f"Query completed in {elapsed_time:.2f} seconds")
                            display_tool_usage(tool_usage, service)

                            print("\n" + "="*50)
                            print("RESULTS:")
                            print("="*50)
                            print(result.data)
                            print("="*50)
                            print(f"Query completed in {elapsed_time:.2f} seconds")
                        except Exception as query_error:
                            logger.error(f"Error processing query: {str(query_error)}")
                            print(f"\n❌ Error: {str(query_error)}")
                            suggestions = {
                                'spotify': [
                                    "Make sure your Spotify API key has the necessary permissions",
                                    "Try a different query format",
                                    "Ensure you have an active Spotify account"
                                ],
                                'github': [
                                    "Make sure your GitHub token has the necessary permissions",
                                    "Try a different query format",
                                    "Ensure dependencies are correctly installed"
                                ]
                            }
                            print("\nSuggestions:")
                            for tip in suggestions[service]:
                                print(f"1. {tip}")

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
            logger.error(f"Error running server: {str(server_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            print(f"\n❌ Error running {service.capitalize()} server: {str(server_error)}")

            troubleshooting_steps = {
                'spotify': [
                    "Ensure Node.js is installed (version 16+)",
                    "Run 'npm install' to install dependencies",
                    "Check that your Spotify API key is valid and has proper permissions"
                ],
                'github': [
                    "Ensure Node.js is installed (version 16+)",
                    "Run 'bash setup_github_agent.sh' to install dependencies",
                    "Check that your GitHub token is valid and has proper permissions"
                ]
            }
            print("\nTroubleshooting steps:")
            for step in troubleshooting_steps[service]:
                print(f"1. {step}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)
    finally:
        logger.info(f"{service.capitalize()} agent shutting down")
        print(f"\nThank you for using the {service.capitalize()} agent!")

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