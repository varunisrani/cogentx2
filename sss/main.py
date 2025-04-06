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
from agent import setup_agent
from tools import recommend_songs_from_spotify, manage_github_repository, run_query

def parse_args():
    parser = argparse.ArgumentParser(description='Universal Spotify-GitHub Agent')
    parser.add_argument('--service', type=str, choices=['spotify', 'github'], required=True, help='Service for the agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
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

def display_tool_usage(tool_usage, service):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return

    print("\n🎵 SPOTIFY TOOLS USED:" if service == 'spotify' else "\n🔧 GITHUB TOOLS USED:")
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
    print("🎵 SPOTIFY RECOMMENDER AGENT 🎵" if service == 'spotify' else "🐙 GITHUB MANAGEMENT AGENT 🐙")
    print("="*50)
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("\nTroubleshooting tips if issues occur:")
    if service == 'spotify':
        print("1. Ensure your Spotify API credentials are correct")
        print("2. Install all dependencies via 'npm install'")
    else:
        print("1. Ensure your GitHub token is valid and has permissions")
        print("2. Run 'bash setup_github_agent.sh' to set up dependencies")
    print("3. Check the log file for detailed error messages")
    print("="*50)

def check_node(logger):
    try:
        node_version = subprocess.check_output(['node', '--version']).decode().strip()
        npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
        logger.info(f"Node.js version: {node_version}, npm version: {npm_version}")
    except Exception as e:
        logger.warning(f"Could not detect Node.js/npm: {str(e)}. Make sure these are installed.")

async def agent_main(service, run_query_fn):
    args = parse_args()
    logger = setup_logging(args)

    try:
        logger.info(f"Initializing {service.capitalize()} Agent")

        check_node(logger)

        logger.info("Loading configuration...")
        config = load_config()

        logger.info(f"Setting up {service} agent...")
        agent = await setup_agent(config, service)

        try:
            async with agent.run_mcp_servers():
                logger.info(f"{service.capitalize()} Server started successfully")

                display_startup_message(service)

                while True:
                    try:
                        user_query = input(f"\n{'🎵' if service == 'spotify' else '💻'} Enter your {service.capitalize()} task: ")

                        if user_query.lower() in ['exit', 'quit', '']:
                            print(f"Exiting {service.capitalize()} agent...")
                            break

                        logger.info(f"Processing task: '{user_query}'")
                        print(f"\nProcessing {service.capitalize()} task: '{user_query}'")
                        print("This may take a moment...\n")

                        try:
                            result, elapsed_time, tool_usage = await run_query_fn(agent, user_query)

                            logger.info(f"Task completed in {elapsed_time:.2f} seconds")

                            display_tool_usage(tool_usage, service)

                            print("\n" + "="*50)
                            print("RESULTS:")
                            print("="*50)
                            print(result.data)
                            print("="*50)
                            print(f"Task completed in {elapsed_time:.2f} seconds")
                        except Exception as query_error:
                            logger.error(f"Error processing task: {str(query_error)}")
                            print(f"\n❌ Error: {str(query_error)}")
                            print("Try a different task or check logs for details.")
                            if service == 'spotify':
                                print("1. Ensure Spotify API credentials are valid")
                                print("2. Try a different task format")
                                print("3. Verify active Spotify account")
                            else:
                                print("1. Verify GitHub token permissions")
                                print("2. Try a different task format")
                                print("3. Run 'bash setup_github_agent.sh' to reinstall dependencies")

                    except KeyboardInterrupt:
                        logger.info("User interrupted the process")
                        print("\nExiting due to keyboard interrupt...")
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        print(f"\n❌ Error in main loop: {str(e)}")
                        print("Retry or check logs for details.")

        except Exception as server_error:
            logger.error(f"Error starting agent server: {str(server_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            print(f"\n❌ Error starting {service.capitalize()} server: {str(server_error)}")
            print("\nTroubleshooting steps:")
            if service == 'spotify':
                print("1. Ensure Node.js is installed (version 16+)")
                print("2. Install dependencies with 'npm install'")
                print("3. Verify Spotify API credentials")
            else:
                print("1. Ensure Node.js is installed (version 16+)")
                print("2. Set up with 'bash setup_github_agent.sh'")
                print("3. Verify GitHub token validity and permissions")
            print("4. Check the log file for more detailed error information")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Fatal error: {str(e)}")
        print("Check logs for more details.")
        sys.exit(1)
    finally:
        logger.info(f"{service.capitalize()} agent shutting down")
        print(f"\nThank you for using the {service.capitalize()} agent!")

if __name__ == '__main__':
    try:
        args = parse_args()
        if args.service == 'spotify':
            asyncio.run(agent_main('spotify', recommend_songs_from_spotify))
        elif args.service == 'github':
            asyncio.run(agent_main('github', manage_github_repository))
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)