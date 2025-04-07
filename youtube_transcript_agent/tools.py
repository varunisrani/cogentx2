from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback

async def display_mcp_tools(server: MCPServerStdio):
    """Display available YouTube Transcript MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("YOUTUBE TRANSCRIPT MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category for YouTube transcript
            categories = {
                'Transcripts': [],
                'Video Info': [],
                'Search': [],
                'Analysis': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
                if 'transcript' in name.lower() or 'captions' in name.lower() or 'text' in name.lower():
                    categories['Transcripts'].append(name)
                elif 'video' in name.lower() or 'info' in name.lower() or 'details' in name.lower():
                    categories['Video Info'].append(name)
                elif 'search' in name.lower() or 'find' in name.lower():
                    categories['Search'].append(name)
                elif 'analyze' in name.lower() or 'summary' in name.lower() or 'summarize' in name.lower():
                    categories['Analysis'].append(name)
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
            
            # Display example usage for YouTube transcript
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'Get transcript for YouTube video https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
            logging.info("2. 'Summarize the contents of this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
            logging.info("3. 'What are the key points discussed in this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
            logging.info("4. 'Find all mentions of AI in this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
            
        else:
            logging.warning("\nNo YouTube Transcript MCP tools were discovered. This could mean either:")
            logging.warning("1. The YouTube Transcript MCP server doesn't expose any tools")
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
            
        # Transform the result if necessary to match expected format
        if isinstance(result, dict) and "toolResult" in result:
            # Extract content from toolResult and add to root level if missing
            if "content" not in result and "content" in result.get("toolResult", {}):
                result["content"] = result["toolResult"]["content"]
                
        return result
    except Exception as e:
        logging.error(f"Error executing MCP tool '{tool_name}': {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def run_youtube_query(agent, user_query: str) -> tuple:
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
        
        # Extract YouTube URL if present to help the agent
        youtube_url = None
        import re
        url_pattern = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        match = re.search(url_pattern, user_query)
        if match:
            video_id = match.group(6)
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            logging.info(f"Extracted YouTube URL: {youtube_url}")
        
        # Add a wrapper for the tool call to fix validation issues
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            server = agent.mcp_servers[0]
            
            if hasattr(server, 'session') and server.session:
                # Store the original call_tool method
                original_call_tool = server.session.call_tool
                
                # Create a wrapped version
                async def wrapped_call_tool(tool_name, arguments):
                    result = await original_call_tool(tool_name, arguments)
                    
                    # Fix for MCP validation error by ensuring 'content' field exists
                    if isinstance(result, dict):
                        if 'toolResult' in result and 'content' in result['toolResult'] and 'content' not in result:
                            result['content'] = result['toolResult']['content']
                    
                    return result
                
                # Replace with wrapped version
                server.session.call_tool = wrapped_call_tool
        
        # Run the query
        result = await agent.run(user_query)
        
        # Restore original if we wrapped it
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            server = agent.mcp_servers[0]
            if hasattr(server, 'session') and server.session and 'original_call_tool' in locals():
                server.session.call_tool = original_call_tool
        
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
            for i, tool in enumerate(tool_usage):
                tool_name = tool.get('name', 'Unknown Tool')
                params = tool.get('parameters', {})
                logging.info(f"Tool {i+1}: {tool_name} with params: {json.dumps(params, indent=2)}")
        
        return (result, elapsed_time, tool_usage)
            
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_mcp_server(youtube_api_key):
    """Create an MCP server for YouTube transcript API
    
    Args:
        youtube_api_key: The API key for YouTube transcript service
        
    Returns:
        MCPServerStdio: The MCP server instance
    """
    try:
        # Set up arguments for the MCP server using Smithery
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@kimtaeyoon83/mcp-server-youtube-transcript",
            "--key",
            youtube_api_key
        ]
        
        # Create and return the server
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating YouTube Transcript MCP server: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise 