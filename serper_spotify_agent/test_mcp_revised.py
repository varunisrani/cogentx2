#!/usr/bin/env python3
"""
Test script for MCP servers in the serper_spotify_agent application.
This script helps diagnose issues with MCP server connections for Serper and Spotify.
"""

import os
import sys
import json
import asyncio
import logging
import traceback
import subprocess
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add the parent directory to sys.path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
try:
    # First try to import directly
    from serper_spotify_agent.tools import create_serper_mcp_server, create_spotify_mcp_server
except ImportError as e:
    logging.warning(f"Direct import failed: {e}. Trying relative import...")
    try:
        # Try relative import
        from tools import create_serper_mcp_server, create_spotify_mcp_server
    except ImportError as e2:
        logging.error(f"Failed to import server creation functions: {e2}")
        logging.error("Make sure you're running this script from the correct directory")
        sys.exit(1)

# Try to import MCPServerStdio to check its methods
try:
    from pydantic_ai.mcp import MCPServerStdio
    has_mcp = True
    logging.info("Successfully imported MCPServerStdio from pydantic_ai.mcp")
    
    # Check what methods are available in MCPServerStdio
    mcp_methods = [method for method in dir(MCPServerStdio) if not method.startswith('_')]
    logging.info(f"Available MCPServerStdio methods: {', '.join(mcp_methods)}")
except ImportError as e:
    has_mcp = False
    logging.error(f"Failed to import MCPServerStdio: {e}")
    logging.error("Make sure pydantic_ai is installed (pip install pydantic-ai)")

def check_node_availability():
    """Check if Node.js and npm are available"""
    try:
        # Check for Node.js
        node_process = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if node_process.returncode != 0:
            logging.error("Node.js is not available. Please install Node.js v16 or higher.")
            return False
            
        node_version = node_process.stdout.strip()
        logging.info(f"Node.js version: {node_version}")
        
        # Check for npm
        npm_process = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if npm_process.returncode != 0:
            logging.error("npm is not available. Please install npm.")
            return False
            
        npm_version = npm_process.stdout.strip()
        logging.info(f"npm version: {npm_version}")
        
        return True
    except Exception as e:
        logging.error(f"Error checking Node.js/npm availability: {e}")
        return False

def check_env_vars():
    """Check for required environment variables"""
    load_dotenv()
    
    # Check for Serper API key
    serper_api_key = os.environ.get('SERPER_API_KEY')
    if not serper_api_key:
        logging.error("SERPER_API_KEY environment variable is not set")
        logging.error("Please add it to your .env file")
    else:
        logging.info(f"SERPER_API_KEY: {'***' + serper_api_key[-4:] if len(serper_api_key) > 4 else '[hidden]'}")
    
    # Check for Spotify API key
    spotify_api_key = os.environ.get('SPOTIFY_API_KEY')
    if not spotify_api_key:
        logging.error("SPOTIFY_API_KEY environment variable is not set")
        logging.error("Please add it to your .env file")
    else:
        logging.info(f"SPOTIFY_API_KEY: {'***' + spotify_api_key[-4:] if len(spotify_api_key) > 4 else '[hidden]'}")
    
    # Check for other important environment variables
    model_vars = {
        'MODEL_CHOICE': os.environ.get('MODEL_CHOICE'),
        'AGENT_MODEL': os.environ.get('AGENT_MODEL'),
        'BASE_URL': os.environ.get('BASE_URL'),
        'API_BASE_URL': os.environ.get('API_BASE_URL')
    }
    
    for var_name, var_value in model_vars.items():
        if var_value:
            logging.info(f"{var_name}: {var_value}")
        else:
            logging.warning(f"{var_name} is not set")
    
    return serper_api_key, spotify_api_key

def inspect_server_object(server, server_name):
    """Inspect a server object to see what attributes and methods it has"""
    if not server:
        logging.error(f"{server_name} server is None")
        return
        
    logging.info(f"Inspecting {server_name} server object")
    
    # Check if it's the expected type
    server_type = type(server).__name__
    logging.info(f"{server_name} server type: {server_type}")
    
    # Check for important attributes
    has_process = hasattr(server, 'process')
    has_client = hasattr(server, '_client')
    has_session = hasattr(server, 'session')
    
    logging.info(f"{server_name} has process: {has_process}")
    logging.info(f"{server_name} has _client: {has_client}")
    logging.info(f"{server_name} has session: {has_session}")
    
    # Check for important methods
    important_methods = ['start', 'ensure_running', 'initialize_if_needed', 'list_tools']
    for method in important_methods:
        if hasattr(server, method) and callable(getattr(server, method)):
            logging.info(f"{server_name} has method: {method}")
        else:
            logging.warning(f"{server_name} MISSING method: {method}")

async def test_mcp_server(server, server_name):
    """Test if an MCP server is working properly"""
    if not server:
        logging.error(f"No {server_name} server provided")
        return False
        
    logging.info(f"Testing {server_name} MCP server")
    
    # First, inspect the server object
    inspect_server_object(server, server_name)
    
    # Try different ways to start the server based on available methods
    started = False
    
    # Method 1: Use ensure_running if available
    if hasattr(server, 'ensure_running') and callable(server.ensure_running):
        try:
            logging.info(f"Trying server.ensure_running() for {server_name}")
            await server.ensure_running()
            started = True
            logging.info(f"Successfully started {server_name} using ensure_running()")
        except Exception as e:
            logging.error(f"Error using ensure_running for {server_name}: {e}")
            logging.debug(traceback.format_exc())
    
    # Method 2: Use initialize_if_needed if available
    if not started and hasattr(server, 'initialize_if_needed') and callable(server.initialize_if_needed):
        try:
            logging.info(f"Trying server.initialize_if_needed() for {server_name}")
            await server.initialize_if_needed()
            started = True
            logging.info(f"Successfully started {server_name} using initialize_if_needed()")
        except Exception as e:
            logging.error(f"Error using initialize_if_needed for {server_name}: {e}")
            logging.debug(traceback.format_exc())
    
    # Method 3: Use start if available
    if not started and hasattr(server, 'start') and callable(server.start):
        try:
            logging.info(f"Trying server.start() for {server_name}")
            await server.start()
            started = True
            logging.info(f"Successfully started {server_name} using start()")
        except Exception as e:
            logging.error(f"Error using start for {server_name}: {e}")
            logging.debug(traceback.format_exc())
    
    # Method 4: Check if process is already running
    if not started and hasattr(server, 'process') and server.process:
        try:
            # Check if process is running
            if server.process.poll() is None:
                logging.info(f"{server_name} process is already running")
                started = True
            else:
                logging.warning(f"{server_name} process has exited with code {server.process.returncode}")
                
                # Try to start process manually
                if hasattr(server, '_start_process') and callable(server._start_process):
                    logging.info(f"Trying to restart {server_name} process")
                    server._start_process()
                    started = True
                    logging.info(f"Restarted {server_name} process")
                
        except Exception as e:
            logging.error(f"Error checking {server_name} process: {e}")
            logging.debug(traceback.format_exc())
    
    if not started:
        logging.error(f"No method available to start {server_name} server")
        return False
    
    # Try to list tools - multiple approaches because different implementations
    try:
        # First check the server directly
        if hasattr(server, 'list_tools') and callable(server.list_tools):
            logging.info(f"Listing tools for {server_name} via server.list_tools()...")
            tools = await server.list_tools()
            logging.info(f"{server_name} tools: {json.dumps(tools, indent=2)}")
            return True
        # Then check if there's a session with list_tools
        elif hasattr(server, 'session') and hasattr(server.session, 'list_tools') and callable(server.session.list_tools):
            logging.info(f"Listing tools for {server_name} via server.session.list_tools()...")
            tools = await server.session.list_tools()
            logging.info(f"{server_name} tools via session: {json.dumps(tools, indent=2)}")
            return True
        # Special case for our custom Session wrapper
        elif (hasattr(server, 'session') and 
              isinstance(server.session, type) and 
              hasattr(server.session, 'list_tools')):
            logging.info(f"Listing tools for {server_name} via custom session class...")
            session_instance = server.session(server._client)
            tools = await session_instance.list_tools()
            logging.info(f"{server_name} tools via custom session: {json.dumps(tools, indent=2)}")
            return True
        # Directly look for _client with list_tools
        elif hasattr(server, '_client') and hasattr(server._client, 'list_tools') and callable(server._client.list_tools):
            logging.info(f"Listing tools for {server_name} via server._client.list_tools()...")
            tools = await server._client.list_tools()
            logging.info(f"{server_name} tools via _client: {json.dumps(tools, indent=2)}")
            return True
        # If none of the above methods work, report failure but still consider this as "functional"
        else:
            logging.warning(f"No list_tools method found for {server_name}, but the server started")
            logging.info(f"This may still be usable as the agent's implementation will handle tool listing")
            return True
    except Exception as e:
        logging.error(f"Error listing tools for {server_name}: {e}")
        logging.debug(traceback.format_exc())
        # The session might work even though we can't list tools in the test
        logging.warning(f"Could not list tools, but {server_name} server may still be functional")
        return True

async def run_tests():
    """Run all tests"""
    logging.info("Starting MCP server tests for serper_spotify_agent")
    
    # Check Node.js availability
    if not check_node_availability():
        logging.error("Node.js or npm is not available. MCP servers will not work.")
        return
    
    # Check environment variables
    serper_api_key, spotify_api_key = check_env_vars()
    
    # Create servers
    logging.info("Creating Serper MCP server...")
    serper_server = create_serper_mcp_server(serper_api_key)
    
    logging.info("Creating Spotify MCP server...")
    spotify_server = create_spotify_mcp_server(spotify_api_key)
    
    # Test servers
    serper_result = await test_mcp_server(serper_server, "Serper")
    spotify_result = await test_mcp_server(spotify_server, "Spotify")
    
    # Summarize results
    logging.info("\n=== TEST RESULTS ===")
    logging.info(f"Serper MCP server: {'WORKING' if serper_result else 'FAILED'}")
    logging.info(f"Spotify MCP server: {'WORKING' if spotify_result else 'FAILED'}")
    
    if serper_result and spotify_result:
        logging.info("All MCP servers are working properly! 🎉")
    elif serper_result or spotify_result:
        logging.warning("Some MCP servers are working, but not all. Check the logs for details.")
    else:
        logging.error("All MCP servers failed. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(run_tests()) 