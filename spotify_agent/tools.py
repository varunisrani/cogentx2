from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback

async def display_mcp_tools(server: MCPServerStdio):
    """Display available Spotify MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("SPOTIFY MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category for Spotify
            categories = {
                'Playback': [],
                'Search': [],
                'Playlists': [],
                'User': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
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
            
            # Display example usage for Spotify
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'Search for songs by Taylor Swift'")
            logging.info("2. 'Play the album Thriller by Michael Jackson'")
            logging.info("3. 'Create a playlist called Summer Hits'")
            logging.info("4. 'What are my top playlists?'")
            logging.info("5. 'Skip to the next track'")
            
        else:
            logging.warning("\nNo Spotify MCP tools were discovered. This could mean either:")
            logging.warning("1. The Spotify MCP server doesn't expose any tools")
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

async def run_spotify_query(agent, user_query: str) -> tuple:
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
        
        # Check if the original agent has the tool schema information
        # This hack allows us to intercept and modify tool calls automatically
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            server = agent.mcp_servers[0]
            
            # Create a modified wrapper for the query
            async def process_with_market_parameter():
                # Set default market if we're dealing with track searches
                if any(keyword in user_query.lower() for keyword in ['song', 'track', 'artist', 'music', 'top']):
                    try:
                        # For search-related requests, we'll add the market parameter
                        if hasattr(server, 'session') and server.session:
                            # Store the original invoke_tool method
                            original_invoke = server.session.invoke_tool
                            
                            # Create a wrapper that adds market parameter when needed
                            async def invoke_tool_with_market(tool_name, params=None):
                                params = params or {}
                                
                                # Add market parameter for relevant tool calls
                                if tool_name in ['getTopTracks', 'searchTracks', 'getArtistTopTracks', 'searchArtists'] and 'market' not in params:
                                    logging.info(f"Automatically adding 'market' parameter to {tool_name} call")
                                    params['market'] = 'US'  # Default to US market
                                
                                return await original_invoke(tool_name, params)
                            
                            # Replace the method temporarily
                            server.session.invoke_tool = invoke_tool_with_market
                            
                            # Run the query
                            result = await agent.run(user_query)
                            
                            # Restore original method
                            server.session.invoke_tool = original_invoke
                            
                            return result
                        
                        # Alternative approach if no session
                        elif hasattr(server, 'invoke_tool'):
                            original_invoke = server.invoke_tool
                            
                            async def invoke_tool_with_market(tool_name, params=None):
                                params = params or {}
                                
                                # Add market parameter for relevant tool calls
                                if tool_name in ['getTopTracks', 'searchTracks', 'getArtistTopTracks', 'searchArtists'] and 'market' not in params:
                                    logging.info(f"Automatically adding 'market' parameter to {tool_name} call")
                                    params['market'] = 'US'  # Default to US market
                                
                                return await original_invoke(tool_name, params)
                            
                            server.invoke_tool = invoke_tool_with_market
                            
                            result = await agent.run(user_query)
                            
                            server.invoke_tool = original_invoke
                            
                            return result
                    except Exception as e:
                        logging.error(f"Error in market parameter wrapper: {str(e)}")
                        logging.debug("Error details:", exc_info=True)
                
                # Default case, run normally
                return await agent.run(user_query)
            
            # Use our wrapper
            result = await process_with_market_parameter()
        else:
            # Fallback to normal execution if we can't intercept
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
            logging.info(f"Tools used in this query: {json.dumps(tool_usage, indent=2)}")
            logging.info(f"Number of tools used: {len(tool_usage)}")
            
            # Log details about each tool used
            for i, tool in enumerate(tool_usage):
                tool_name = tool.get('name', 'Unknown Tool')
                logging.info(f"Tool {i+1}: {tool_name}")
        else:
            logging.info("No specific tools were recorded for this query")
            
        return result, elapsed_time, tool_usage
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_mcp_server(spotify_api_key):
    """Create MCP server instance for Spotify"""
    return MCPServerStdio(
        'npx',
        [
            '-y',
            '@smithery/cli@latest',
            'run',
            '@superseoworld/mcp-spotify',
            '--key',
            spotify_api_key
        ]
    ) 