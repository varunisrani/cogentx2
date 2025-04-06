#!/usr/bin/env python3
"""
MCP Server Test Script for Serper-Spotify Agent

This script tests if MCP servers for Serper and Spotify can be properly initialized.
It will help diagnose issues with your configuration.

Usage:
    python test_mcp.py [--verbose]
"""

import asyncio
import argparse
import logging
import os
import sys
import traceback
import json
from dotenv import load_dotenv
import colorlog
from pydantic_ai.mcp import MCPServerStdio

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Test MCP Server Initialization for Serper-Spotify Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

# Configure logging with colors
def setup_logging(verbose=False):
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
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
    
    return root_logger

def create_spotify_mcp_server(spotify_api_key):
    """Create an MCP server for Spotify API"""
    try:
        logging.info(f"Creating Spotify MCP server with token: {'***' + spotify_api_key[-4:] if spotify_api_key and len(spotify_api_key) > 4 else 'None or placeholder'}")
        
        if not spotify_api_key or spotify_api_key == "your_spotify_api_key_here":
            logging.error("Cannot create Spotify MCP server: No valid Spotify API key provided")
            return None
        
        # Set up environment
        env = os.environ.copy()
        
        # On macOS, set MallocNanoZone to prevent memory issues
        if sys.platform == 'darwin':
            env['MallocNanoZone'] = '0'
        
        # Using the smithery approach for Spotify
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                '@smithery/cli@latest',
                'run',
                '@superseoworld/mcp-spotify',
                '--key',
                spotify_api_key
            ],
            env=env
        )
        
        logging.info("Spotify MCP server created")
        return server
    except Exception as e:
        logging.error(f"Error creating Spotify MCP server: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

def create_serper_mcp_server(serper_api_key):
    """Create an MCP server for Serper search API"""
    try:
        logging.info(f"Creating Serper MCP server with API key: {'***' + serper_api_key[-4:] if serper_api_key and len(serper_api_key) > 4 else 'None or placeholder'}")
        
        if not serper_api_key or serper_api_key == "your_serper_api_key_here":
            logging.error("Cannot create Serper MCP server: No valid Serper API key provided")
            return None
        
        # Set up environment
        env = os.environ.copy()
        
        # On macOS, set MallocNanoZone to prevent memory issues
        if sys.platform == 'darwin':
            env['MallocNanoZone'] = '0'
        
        # Create config with API key
        config = {
            "serperApiKey": serper_api_key
        }
        
        # Using the smithery approach for Serper
        server = MCPServerStdio(
            'npx',
            [
                '-y',
                '@smithery/cli@latest',
                'run',
                '@marcopesani/mcp-server-serper',
                '--config',
                json.dumps(config)
            ],
            env=env
        )
        
        logging.info("Serper MCP server created")
        return server
    except Exception as e:
        logging.error(f"Error creating Serper MCP server: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

async def test_mcp_server(server, server_name):
    """Test if an MCP server can be initialized and connected"""
    logging.info(f"Testing {server_name} MCP server...")
    
    if not server:
        logging.error(f"{server_name} server could not be created")
        return False
    
    try:
        # Try to start the server
        logging.info(f"Starting {server_name} server process...")
        
        # Try different methods to start the server based on MCP library version
        started = False
        
        # Method 1: Use ensure_running if available
        if hasattr(server, 'ensure_running') and callable(server.ensure_running):
            try:
                await server.ensure_running()
                logging.info(f"{server_name} server started using ensure_running()")
                started = True
            except Exception as run_err:
                logging.warning(f"Failed to start with ensure_running(): {str(run_err)}")
                # Continue to next method
        
        # Method 2: Use start if available
        if not started and hasattr(server, 'start') and callable(server.start):
            try:
                await server.start()
                logging.info(f"{server_name} server started using start()")
                started = True
            except Exception as start_err:
                logging.warning(f"Failed to start with start(): {str(start_err)}")
                # Continue to next method
        
        # Method 3: Start manually using subprocess if server is already running
        if not started and hasattr(server, 'process') and server.process:
            try:
                # Check if process is already running
                if server.process.poll() is None:
                    logging.info(f"{server_name} server process already running with PID: {server.process.pid}")
                    started = True
            except Exception as proc_err:
                logging.warning(f"Failed to check process: {str(proc_err)}")
        
        # Method 4: For some versions, the server might start automatically on creation
        if not started:
            logging.warning(f"No explicit method found to start {server_name} server")
            logging.warning("Proceeding with assumption that server may start automatically")
            started = True  # Assume it might work and try to connect
            
        if not started:
            logging.error(f"No method available to start {server_name} server")
            return False
            
        # Give the server some time to initialize
        await asyncio.sleep(2)
        
        # Try to list tools
        logging.info(f"Testing {server_name} server connection by listing tools...")
        
        try:
            tools = []
            
            # Method 1: Try getting tools through server's session
            if hasattr(server, 'session') and server.session:
                try:
                    response = await server.session.list_tools()
                    if hasattr(response, 'tools'):
                        tools = response.tools
                    elif isinstance(response, dict):
                        tools = response.get('tools', [])
                    logging.info(f"Listed tools through server.session.list_tools()")
                except Exception as e:
                    logging.warning(f"Error listing tools via session: {str(e)}")
            
            # Method 2: Try direct tool listing if available
            if not tools and hasattr(server, 'list_tools'):
                try:
                    response = await server.list_tools()
                    if hasattr(response, 'tools'):
                        tools = response.tools
                    elif isinstance(response, dict):
                        tools = response.get('tools', [])
                    logging.info(f"Listed tools through server.list_tools()")
                except Exception as e:
                    logging.warning(f"Error listing tools via list_tools: {str(e)}")
                    
            # Method 3: Try fallback method for older MCP versions
            if not tools and hasattr(server, 'listTools'):
                try:
                    response = server.listTools()
                    if isinstance(response, dict):
                        tools = response.get('tools', [])
                    logging.info(f"Listed tools through server.listTools()")
                except Exception as e:
                    logging.warning(f"Error listing tools via listTools: {str(e)}")
                
            if not tools:
                logging.error(f"No method available to list tools for {server_name} server")
                return False
                
            logging.info(f"Successfully listed {len(tools)} tools from {server_name} server")
            
            # Show some example tools
            if tools:
                logging.info(f"Example {server_name} tools:")
                for i, tool in enumerate(tools[:3]):  # Show up to 3 example tools
                    logging.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
                if len(tools) > 3:
                    logging.info(f"  ... and {len(tools) - 3} more")
                    
            return True
                
        except Exception as list_err:
            logging.error(f"Error listing tools from {server_name} server: {str(list_err)}")
            logging.debug(traceback.format_exc())
            return False
            
    except Exception as e:
        logging.error(f"Error testing {server_name} server: {str(e)}")
        logging.debug(traceback.format_exc())
        return False

def check_node_npm():
    """Check if Node.js and npm are installed and available"""
    logging.info("Checking Node.js and npm availability...")
    
    # Check for Node.js
    try:
        import subprocess
        
        node_process = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if node_process.returncode != 0:
            logging.error("Node.js is not installed or not in PATH")
            return False
            
        node_version = node_process.stdout.strip()
        logging.info(f"Node.js version: {node_version}")
        
        # Check minimum version (we need at least Node.js 16)
        major_version = int(node_version.replace('v', '').split('.')[0])
        if major_version < 16:
            logging.error(f"Node.js version {node_version} is too old (minimum required is v16)")
            logging.error("Please upgrade Node.js using nvm or by downloading from https://nodejs.org/")
            return False
        
        # Check for npm
        npm_process = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if npm_process.returncode != 0:
            logging.error("npm is not installed or not in PATH")
            return False
            
        npm_version = npm_process.stdout.strip()
        logging.info(f"npm version: {npm_version}")
        
        # Try a simple npx command to verify it works
        logging.info("Testing npx command...")
        npx_process = subprocess.run(['npx', '--version'], capture_output=True, text=True)
        if npx_process.returncode != 0:
            logging.error("npx is not working properly")
            return False
            
        logging.info(f"npx version: {npx_process.stdout.strip()}")
        logging.info("Node.js, npm, and npx are available and working correctly")
        return True
        
    except Exception as e:
        logging.error(f"Error checking Node.js and npm: {str(e)}")
        logging.debug(traceback.format_exc())
        return False

async def main():
    # Parse arguments and set up logging
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    print("\n" + "="*60)
    print(" 🧪 SERPER-SPOTIFY MCP SERVER TEST UTILITY 🧪 ")
    print("="*60)
    print("This utility will test if your MCP servers can be initialized and connected.")
    print("It will help diagnose issues with Serper and Spotify MCP servers.")
    print("="*60 + "\n")
    
    try:
        # Load environment variables
        load_dotenv()
        logging.info("Environment variables loaded from .env file")
        
        # Check Node.js and npm first
        if not check_node_npm():
            logging.error("Node.js or npm check failed. MCP servers require Node.js (v16+) and npm.")
            logging.error("Please install or upgrade Node.js and npm before proceeding.")
            return
            
        # Check API keys
        serper_api_key = os.getenv("SERPER_API_KEY")
        spotify_api_key = os.getenv("SPOTIFY_API_KEY")
        
        if not serper_api_key and not spotify_api_key:
            logging.error("No API keys found in .env file. Please add at least one of the following:")
            logging.error("  - SERPER_API_KEY=your_serper_api_key")
            logging.error("  - SPOTIFY_API_KEY=your_spotify_api_key")
            return
            
        # Test each server if an API key is provided
        results = {}
        
        if serper_api_key and serper_api_key != "your_serper_api_key_here":
            serper_server = create_serper_mcp_server(serper_api_key)
            serper_result = await test_mcp_server(serper_server, "Serper")
            results["Serper"] = serper_result
        else:
            logging.warning("Skipping Serper MCP server test - no valid API key provided")
            results["Serper"] = "Skipped"
            
        if spotify_api_key and spotify_api_key != "your_spotify_api_key_here":
            spotify_server = create_spotify_mcp_server(spotify_api_key)
            spotify_result = await test_mcp_server(spotify_server, "Spotify")
            results["Spotify"] = spotify_result
        else:
            logging.warning("Skipping Spotify MCP server test - no valid API key provided")
            results["Spotify"] = "Skipped"
            
        # Display summary
        print("\n" + "="*60)
        print(" 📋 MCP SERVER TEST RESULTS 📋 ")
        print("="*60)
        
        for server, result in results.items():
            status = "✅ PASSED" if result is True else "❌ FAILED" if result is False else "⏭️ SKIPPED"
            print(f"{server} MCP Server: {status}")
            
        print("\n" + "="*60)
        if all(result is True for result in results.values() if result != "Skipped"):
            print("🎉 All tested MCP servers are working correctly!")
            print("You can now use the main.py script to run the agent.")
        else:
            print("❌ Some MCP servers failed the test.")
            print("Please check the logs above for specific error messages.")
            print("\nTroubleshooting tips:")
            print("1. Make sure your API keys are valid and have the necessary permissions")
            print("2. Check your Node.js installation (version 16+)")
            print("3. Try running 'npm install -g @smithery/cli' to pre-install the Smithery CLI")
            print("4. Check your network connection")
            print("5. If on macOS, ensure MallocNanoZone=0 is set")
            print("6. For Serper: Make sure your API key is correct and has sufficient credits")
            print("7. For Spotify: Ensure your API token is valid and not expired")
        print("="*60 + "\n")
            
    except Exception as e:
        logging.error(f"Error during MCP server testing: {str(e)}")
        logging.debug(traceback.format_exc())
        
if __name__ == '__main__':
    asyncio.run(main()) 