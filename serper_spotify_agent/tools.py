from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback

async def display_mcp_tools(server: MCPServerStdio, server_name: str = "MCP"):
    """Display available MCP tools and their details for the specified server
    
    Args:
        server: The MCP server instance
        server_name: Name identifier for the server (SERPER or SPOTIFY)
    
    Returns:
        List of available tools
    """
    try:
        logging.info("\n" + "="*50)
        logging.info(f"{server_name} MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category - define different categories based on server type
            categories = {}
            
            if server_name == "SERPER":
                categories = {
                    'Search': [],
                    'Web Search': [],
                    'News Search': [],
                    'Image Search': [],
                    'Other': []
                }
            elif server_name == "SPOTIFY":
                categories = {
                    'Playback': [],
                    'Search': [],
                    'Playlists': [],
                    'User': [],
                    'Other': []
                }
            else:
                categories = {'All': []}
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns and server type
            for name in uncategorized.copy():
                if server_name == "SERPER":
                    if 'search' in name.lower() and 'image' in name.lower():
                        categories['Image Search'].append(name)
                    elif 'search' in name.lower() and 'news' in name.lower():
                        categories['News Search'].append(name)
                    elif 'search' in name.lower() and ('web' in name.lower() or 'google' in name.lower()):
                        categories['Web Search'].append(name)
                    elif 'search' in name.lower():
                        categories['Search'].append(name)
                    else:
                        categories['Other'].append(name)
                elif server_name == "SPOTIFY":
                    if 'play' in name.lower() or 'track' in name.lower() or 'album' in name.lower():
                        categories['Playback'].append(name)
                    elif 'search' in name.lower():
                        categories['Search'].append(name)
                    elif 'playlist' in name.lower():
                        categories['Playlists'].append(name)
                    elif 'user' in name.lower() or 'profile' in name.lower():
                        categories['User'].append(name)
                    else:
                        categories['Other'].append(name)
                else:
                    categories['All'].append(name)
                
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
            logging.info(f"Total Available {server_name} Tools: {len(tools)}")
            logging.info("="*50)
            
            # Display example usage depending on the server type
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            
            if server_name == "SERPER":
                logging.info("1. 'Search for recent news about artificial intelligence'")
                logging.info("2. 'Find information about climate change from scientific sources'")
                logging.info("3. 'Show me images of the Golden Gate Bridge'")
                logging.info("4. 'What are the latest updates on the Mars rover?'")
                logging.info("5. 'Search for recipes with chicken and rice'")
            elif server_name == "SPOTIFY":
                logging.info("1. 'Search for songs by Taylor Swift'")
                logging.info("2. 'Play the album Thriller by Michael Jackson'")
                logging.info("3. 'Create a playlist called Summer Hits'")
                logging.info("4. 'What are my top playlists?'")
                logging.info("5. 'Skip to the next track'")
            
        else:
            logging.warning(f"\nNo {server_name} MCP tools were discovered. This could mean either:")
            logging.warning("1. The MCP server doesn't expose any tools")
            logging.warning("2. The tools discovery mechanism is not supported")
            logging.warning("3. The server connection is not properly initialized")
            
        return tools
                
    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return []

async def execute_mcp_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    """Execute an MCP tool with the specified parameters
    
    Args:
        server: The MCP server instance
        tool_name: Name of the tool to execute
        params: Dictionary of parameters to pass to the tool
        
    Returns:
        The result of the tool execution
    """
    try:
        logging.info(f"Executing MCP tool: {tool_name}")
        if params:
            logging.info(f"With parameters: {json.dumps(params, indent=2)}")
        
        # Execute the tool through the server
        if hasattr(server, 'session') and server.session:
            result = await server.session.invoke_tool(tool_name, params or {})
        elif hasattr(server, 'invoke_tool'):
            result = await server.invoke_tool(tool_name, params or {})
        else:
            raise Exception(f"Cannot find a way to invoke tool {tool_name}")
            
        return result
    except Exception as e:
        logging.error(f"Error executing MCP tool '{tool_name}': {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def run_query(agent, user_query: str) -> tuple:
    """Process a user query using the MCP agent and return the result with metrics
    
    Args:
        agent: The MCP agent
        user_query: The user's query string
        
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Execute the query and track execution
        logging.info(f"Processing query: '{user_query}'")
        
        # Detect if this is a music-related query for Spotify
        is_spotify_query = any(keyword in user_query.lower() for keyword in 
                              ['spotify', 'music', 'song', 'playlist', 'artist', 'album', 'play', 'track', 'listen'])
        
        # Add market parameter for Spotify search queries
        if is_spotify_query and any(keyword in user_query.lower() for keyword in 
                                  ['song', 'track', 'artist', 'search', 'find', 'top']):
            # For Spotify, modify server handling to add market parameter
            if hasattr(agent, 'mcp_servers') and len(agent.mcp_servers) > 1:
                spotify_server = agent.mcp_servers[1]  # Second server should be Spotify
                
                if hasattr(spotify_server, 'session') and spotify_server.session:
                    # Store the original invoke_tool method
                    original_invoke = spotify_server.session.invoke_tool
                    
                    # Create a wrapper that adds market parameter when needed
                    async def invoke_tool_with_market(tool_name, params=None):
                        params = params or {}
                        
                        # Add market parameter for relevant tool calls
                        if tool_name in ['getTopTracks', 'searchTracks', 'getArtistTopTracks', 'searchArtists'] and 'market' not in params:
                            logging.info(f"Automatically adding 'market' parameter to {tool_name} call")
                            params['market'] = 'US'  # Default to US market
                        
                        return await original_invoke(tool_name, params)
                    
                    # Replace the invoke_tool method with our wrapped version
                    spotify_server.session.invoke_tool = invoke_tool_with_market
        
        # Execute the query
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Extract tool usage information if available
        tool_usage = []
        if hasattr(result, 'metadata') and result.metadata:
            try:
                # Try to extract tool usage information from metadata
                if isinstance(result.metadata, dict) and 'tools' in result.metadata:
                    tool_usage = result.metadata['tools']
                elif hasattr(result.metadata, 'tools'):
                    tool_usage = result.metadata.tools
                
                # Handle case where tools might be nested further
                if not tool_usage and isinstance(result.metadata, dict) and 'tool_calls' in result.metadata:
                    tool_usage = result.metadata['tool_calls']
            except Exception as tool_err:
                logging.debug(f"Could not extract tool usage: {tool_err}")
        
        # Log the tools used
        if tool_usage:
            logging.info(f"Tools used in this query: {len(tool_usage)}")
            
            # Identify which server was used
            serper_used = any("search" in tool.get('name', '').lower() for tool in tool_usage)
            spotify_used = any(keyword in json.dumps(tool_usage).lower() for keyword in 
                             ['spotify', 'music', 'song', 'playlist', 'artist', 'album', 'play', 'track'])
            
            if serper_used:
                logging.info("SERPER search tools were used for this query")
            if spotify_used:
                logging.info("SPOTIFY music tools were used for this query")
            
        return (result, elapsed_time, tool_usage)
            
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_serper_mcp_server(serper_api_key):
    """Create an MCP server for Serper search API
    
    Args:
        serper_api_key: The API key for Serper
        
    Returns:
        MCPServerStdio: The MCP server instance
    """
    try:
        # Create config with API key
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
        
        # Create and return the server
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating Serper MCP server: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_spotify_mcp_server(spotify_api_key):
    """Create an MCP server for Spotify API
    
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
        
        # Create and return the server
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating Spotify MCP server: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise