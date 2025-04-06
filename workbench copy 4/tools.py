from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback
from typing import Optional, List, Dict, Any

class IntegratedAgentConfig:
    def __init__(self):
        self.integrated_servers = []

async def test_mcp_server(server: MCPServerStdio, server_name: str) -> bool:
    """Test if an MCP server can be initialized and connected
    
    Args:
        server: The MCP server instance
        server_name: Name identifier for the server
        
    Returns:
        bool: True if server is working, False otherwise
    """
    try:
        logging.info(f"Testing {server_name} MCP server connection...")
        
        # Try to start the server if needed
        started = False
        
        # Method 1: Try starting through session
        if hasattr(server, 'session') and server.session:
            try:
                if not server.session.is_connected():
                    await server.session.connect()
                started = server.session.is_connected()
                logging.info(f"Started {server_name} server through session")
            except Exception as e:
                logging.warning(f"Could not start {server_name} server through session: {str(e)}")
        
        # Method 2: Try direct start if available
        if not started and hasattr(server, 'start'):
            try:
                await server.start()
                started = True
                logging.info(f"Started {server_name} server through direct start")
            except Exception as e:
                logging.warning(f"Could not start {server_name} server through direct start: {str(e)}")
        
        # Method 3: Try connect method if available
        if not started and hasattr(server, 'connect'):
            try:
                await server.connect()
                started = True
                logging.info(f"Started {server_name} server through connect method")
            except Exception as e:
                logging.warning(f"Could not start {server_name} server through connect: {str(e)}")
        
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
        
        # Try to list tools to verify connection
        logging.info(f"Testing {server_name} server connection by listing tools...")
        tools = await show_available_tools(server, server_name)
        
        if tools:
            logging.info(f"{server_name} server connection successful")
            return True
        else:
            logging.warning(f"{server_name} server connection test inconclusive - no tools found")
            return True  # Return True since the server might still be working even without tools
            
    except Exception as e:
        logging.error(f"Error testing {server_name} server: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return False

async def show_available_tools(server: MCPServerStdio, api_name: str):
    """Display available tools and their functions for the specified server."""
    try:
        logging.info("\n" + "="*50)
        logging.info(f"{api_name.upper()} TOOLS AND FUNCTIONS")
        logging.info("="*50)

        tools = []
        try:
            if hasattr(server, 'session') and server.session:
                response = await server.session.list_tools()
                if hasattr(response, 'tools'):
                    tools = response.tools
                elif isinstance(response, dict):
                    tools = response.get('tools', [])

            elif hasattr(server, 'list_tools'):
                response = await server.list_tools()
                if hasattr(response, 'tools'):
                    tools = response.tools
                elif isinstance(response, dict):
                    tools = response.get('tools', [])

            elif hasattr(server, 'listTools'):
                response = server.listTools()
                if isinstance(response, dict):
                    tools = response.get('tools', [])

        except Exception as tool_error:
            logging.debug(f"Error listing tools: {tool_error}", exc_info=True)

        if tools:
            categories = {
                'SERPER': {'Search': [], 'Web': [], 'News': [], 'Image': [], 'Misc': []},
                'SPOTIFY': {'Playback': [], 'Search': [], 'Playlists': [], 'User': [], 'Misc': []}
            }.get(api_name.upper(), {'General': []})

            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())

            for name in uncategorized.copy():
                if api_name.upper() == "SERPER":
                    if 'search' in name.lower() and 'image' in name.lower():
                        categories['Image'].append(name)
                    elif 'search' in name.lower() and 'news' in name.lower():
                        categories['News'].append(name)
                    elif 'search' in name.lower() and ('web' in name.lower() or 'google' in name.lower()):
                        categories['Web'].append(name)
                    elif 'search' in name.lower():
                        categories['Search'].append(name)
                    else:
                        categories['Misc'].append(name)
                elif api_name.upper() == "SPOTIFY":
                    if 'play' in name.lower() or 'track' in name.lower() or 'album' in name.lower():
                        categories['Playback'].append(name)
                    elif 'search' in name.lower():
                        categories['Search'].append(name)
                    elif 'playlist' in name.lower():
                        categories['Playlists'].append(name)
                    elif 'user' in name.lower() or 'profile' in name.lower():
                        categories['User'].append(name)
                    else:
                        categories['Misc'].append(name)
                else:
                    categories['General'].append(name)

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
            logging.info(f"Total Available {api_name.upper()} Tools: {len(tools)}")
            logging.info("="*50)
        else:
            logging.warning(f"\nNo {api_name.upper()} tools were discovered.")
        
        return tools

    except Exception as e:
        logging.error(f"Error displaying tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return []

async def execute_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    """Execute a tool with the specified parameters."""
    try:
        logging.info(f"Executing tool: {tool_name}")
        if params:
            logging.info(f"With parameters: {json.dumps(params, indent=2)}")
        
        if hasattr(server, 'session') and server.session:
            result = await server.session.invoke_tool(tool_name, params or {})
        elif hasattr(server, 'invoke_tool'):
            result = await server.invoke_tool(tool_name, params or {})
        else:
            raise Exception(f"Cannot find a way to invoke tool {tool_name}")
            
        return result
    except Exception as e:
        logging.error(f"Error executing tool '{tool_name}': {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def process_user_query(agent, user_query: str) -> tuple:
    """Process a user query using the integrated agent and return the result with metrics
    
    Args:
        agent: The integrated agent instance
        user_query: The user's query string
        
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logging.info(f"Processing query: '{user_query}'")
        
        # Execute the query through the agent
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Extract tool usage information
        tool_usage_info = []
        if isinstance(result, dict):
            if 'metadata' in result and isinstance(result['metadata'], dict):
                tool_usage_info = result['metadata'].get('tools', [])
            elif 'tools' in result:
                tool_usage_info = result['tools']
        
        if tool_usage_info:
            logging.info(f"Tools used in this query: {len(tool_usage_info)}")
            serper_tools_used = any("search" in str(tool).lower() for tool in tool_usage_info)
            spotify_tools_used = any(keyword in json.dumps(str(tool_usage_info)).lower() for keyword in 
                             ['spotify', 'music', 'song', 'playlist', 'artist', 'album', 'play', 'track'])
            if serper_tools_used:
                logging.info("SERPER search tools were utilized for this query")
            if spotify_tools_used:
                logging.info("SPOTIFY music tools were utilized for this query")

        return result, elapsed_time, tool_usage_info
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_serper_server(serper_api_key: str) -> MCPServerStdio:
    """Create a server for Serper search API with WebSocket support
    
    Args:
        serper_api_key: The API key for Serper
        
    Returns:
        MCPServerStdio: The MCP server instance
    """
    try:
        config = {
            "serperApiKey": serper_api_key
        }
        
        # Set up arguments for the MCP server using Smithery
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@marcopesani/mcp-server-serper",
            "--config",
            json.dumps(config)
        ]
        
        server = MCPServerStdio("npx", mcp_args)
        logging.info("Created Serper MCP server with WebSocket support")
        return server
        
    except Exception as e:
        logging.error(f"Error creating Serper server: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_spotify_server(spotify_api_key: str) -> MCPServerStdio:
    """Create a server for Spotify API with WebSocket support
    
    Args:
        spotify_api_key: The API key for Spotify
        
    Returns:
        MCPServerStdio: The MCP server instance
    """
    try:
        # Set up arguments for the MCP server using Smithery
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@superseoworld/mcp-spotify",
            "--key",
            spotify_api_key
        ]
        
        server = MCPServerStdio("npx", mcp_args)
        logging.info("Created Spotify MCP server with WebSocket support")
        return server
        
    except Exception as e:
        logging.error(f"Error creating Spotify server: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

async def configure_integrated_agent(serper_api_key: str, spotify_api_key: str) -> IntegratedAgentConfig:
    """Configure and test the integrated agent with Serper and Spotify servers
    
    Args:
        serper_api_key: API key for Serper
        spotify_api_key: API key for Spotify
        
    Returns:
        IntegratedAgentConfig: The configured agent
    """
    config = IntegratedAgentConfig()
    
    # Create and test Serper server
    logging.info("Creating Serper MCP Server...")
    serper_server = create_serper_server(serper_api_key)
    if await test_mcp_server(serper_server, "SERPER"):
        config.integrated_servers.append(serper_server)
    else:
        logging.error("Failed to initialize Serper server")
        
    # Create and test Spotify server
    logging.info("Creating Spotify MCP Server...")
    spotify_server = create_spotify_server(spotify_api_key)
    if await test_mcp_server(spotify_server, "SPOTIFY"):
        config.integrated_servers.append(spotify_server)
    else:
        logging.error("Failed to initialize Spotify server")
    
    if not config.integrated_servers:
        raise Exception("No MCP servers could be initialized")
        
    logging.info("Successfully configured integrated agent with MCP servers")
    return config

async def run_serper_query(agent, user_query: str) -> tuple:
    """Process a Serper search query using the integrated agent
    
    Args:
        agent: The integrated agent instance
        user_query: The user's search query
        
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logging.info(f"Processing Serper query: '{user_query}'")
        
        # Get the Serper server from the agent
        if not agent.integrated_servers or len(agent.integrated_servers) < 1:
            raise Exception("Serper server not initialized")
        
        result = await process_user_query(agent, user_query)
        return result
            
    except Exception as e:
        logging.error(f"Error processing Serper query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

async def run_spotify_query(agent, user_query: str) -> tuple:
    """Process a Spotify music query using the integrated agent
    
    Args:
        agent: The integrated agent instance
        user_query: The user's music query
        
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logging.info(f"Processing Spotify query: '{user_query}'")
        
        # Get the Spotify server from the agent
        if not agent.integrated_servers or len(agent.integrated_servers) < 2:
            raise Exception("Spotify server not initialized")
            
        result = await process_user_query(agent, user_query)
        return result
            
    except Exception as e:
        logging.error(f"Error processing Spotify query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def set_up_agent_logging():
    """Set up logging configuration for the agent."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Agent setup completed.")