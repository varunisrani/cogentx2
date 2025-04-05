from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback

async def display_mcp_tools(server: MCPServerStdio):
    """Display available Firecrawl MCP tools and their details"""
    try:
        logging.info("\n" + "="*50)
        logging.info("FIRECRAWL MCP TOOLS AND FUNCTIONS")
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
            # Group tools by category (these categories will be modified once we know Firecrawl's tools)
            categories = {
                'Web Search': [],
                'Web Browsing': [],
                'Web Scraping': [],
                'Data Analysis': [],
                'Other': []
            }
            
            # Organize tools by category
            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())
            
            # Auto-categorize tools based on name patterns
            for name in uncategorized.copy():
                if 'search' in name.lower():
                    categories['Web Search'].append(name)
                elif 'browse' in name.lower() or 'navigate' in name.lower():
                    categories['Web Browsing'].append(name)
                elif 'scrape' in name.lower() or 'extract' in name.lower():
                    categories['Web Scraping'].append(name)
                elif 'analyze' in name.lower() or 'process' in name.lower():
                    categories['Data Analysis'].append(name)
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
            
            # Display example usage (will be updated with actual examples once we know Firecrawl's capabilities)
            logging.info("\nExample Queries:")
            logging.info("-"*50)
            logging.info("1. 'Search for recent news about artificial intelligence'")
            logging.info("2. 'Find information about climate change from scientific sources'")
            logging.info("3. 'Extract product details from amazon.com/dp/B07PXGQC1Q'")
            logging.info("4. 'Browse to wikipedia.org and find information about quantum computing'")
            logging.info("5. 'Compare prices of iPhone 15 across major retailers'")
            
        else:
            logging.warning("\nNo Firecrawl MCP tools were discovered. This could mean either:")
            logging.warning("1. The Firecrawl MCP server doesn't expose any tools")
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

async def run_firecrawl_query(agent, user_query: str) -> tuple:
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

def create_mcp_server(firecrawl_api_key, retry_max_attempts="3", retry_initial_delay="1000", 
                      retry_max_delay="5000", retry_backoff_factor="1.5", credit_warning_threshold="100",
                      credit_critical_threshold="50"):
    """Create MCP server instance for Firecrawl"""
    return MCPServerStdio(
        'npx',
        [
            '-y',
            'firecrawl-mcp'
        ],
        env={
            "FIRECRAWL_API_KEY": firecrawl_api_key,
            "FIRECRAWL_RETRY_MAX_ATTEMPTS": str(retry_max_attempts),
            "FIRECRAWL_RETRY_INITIAL_DELAY": str(retry_initial_delay),
            "FIRECRAWL_RETRY_MAX_DELAY": str(retry_max_delay),
            "FIRECRAWL_RETRY_BACKOFF_FACTOR": str(retry_backoff_factor),
            "FIRECRAWL_CREDIT_WARNING_THRESHOLD": str(credit_warning_threshold),
            "FIRECRAWL_CREDIT_CRITICAL_THRESHOLD": str(credit_critical_threshold)
        }
    ) 