from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback
from pydantic_ai import RunContext
from models import GitHubDeps

async def display_mcp_tools(server: MCPServerStdio, agent_name: str = "AGENT"):
    """Display available tools and functionalities for the specified agent

    Args:
        server: The MCP server instance
        agent_name: Name identifier for the agent (SPOTIFY_AGENT, GITHUB_AGENT)

    Returns:
        List of available tools
    """
    try:
        logging.info("\n" + "="*50)
        logging.info(f"{agent_name} TOOLS AND FUNCTIONS")
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
            categories = {}
            if agent_name == "SPOTIFY_AGENT":
                categories = {
                    'Song Recommendation': [],
                    'Playback': [],
                    'Search': [],
                    'Playlists': [],
                    'User': [],
                    'Other': []
                }
            elif agent_name == "GITHUB_AGENT":
                categories = {
                    'Repositories': [],
                    'Issue Management': [],
                    'Pull Requests': [],
                    'User Management': [],
                    'Other': []
                }
            else:
                categories = {'All': []}

            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())

            for name in uncategorized.copy():
                if agent_name == "SPOTIFY_AGENT":
                    if 'recommend' in name.lower():
                        categories['Song Recommendation'].append(name)
                    elif 'play' in name.lower() or 'track' in name.lower() or 'album' in name.lower():
                        categories['Playback'].append(name)
                    elif 'search' in name.lower():
                        categories['Search'].append(name)
                    elif 'playlist' in name.lower():
                        categories['Playlists'].append(name)
                    elif 'user' in name.lower() or 'profile' in name.lower():
                        categories['User'].append(name)
                    else:
                        categories['Other'].append(name)
                elif agent_name == "GITHUB_AGENT":
                    if 'repo' in name.lower():
                        categories['Repositories'].append(name)
                    elif 'issue' in name.lower():
                        categories['Issue Management'].append(name)
                    elif 'pull' in name.lower() or 'pr' in name.lower():
                        categories['Pull Requests'].append(name)
                    elif 'user' in name.lower():
                        categories['User Management'].append(name)
                    else:
                        categories['Other'].append(name)
                else:
                    categories['All'].append(name)

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
            logging.info(f"Total Available {agent_name} Tools: {len(tools)}")
            logging.info("="*50)

            return tools

    except Exception as e:
        logging.error(f"Error displaying agent tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return []

async def execute_agent_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    """Execute a tool with the specified parameters on the agent

    Args:
        server: The MCP server instance
        tool_name: Name of the tool to execute
        params: Dictionary of parameters to pass to the tool

    Returns:
        The result of the tool execution
    """
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

async def process_user_request(agent, user_request: str, agent_name: str = "AGENT") -> tuple:
    """Process a user request using the agent and return the result with metrics

    Args:
        agent: The agent
        user_request: The user's request string
        agent_name: Name identifier for the agent

    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    try:
        start_time = asyncio.get_event_loop().time()

        logging.info(f"Processing request: '{user_request}'")
        result = await agent.run(user_request)
        elapsed_time = asyncio.get_event_loop().time() - start_time

        tool_usage = []
        if hasattr(result, 'metadata') and result.metadata:
            try:
                if isinstance(result.metadata, dict) and 'tools' in result.metadata:
                    tool_usage = result.metadata['tools']
                elif hasattr(result.metadata, 'tools'):
                    tool_usage = result.metadata.tools

                if not tool_usage and isinstance(result.metadata, dict) and 'tool_calls' in result.metadata:
                    tool_usage = result.metadata['tool_calls']
            except Exception as tool_err:
                logging.debug(f"Could not extract tool usage: {tool_err}")

        if tool_usage:
            logging.info(f"Tools used in this request: {json.dumps(tool_usage, indent=2)}")
            logging.info(f"Number of tools used: {len(tool_usage)}")

            for i, tool in enumerate(tool_usage):
                tool_name = tool.get('name', 'Unknown Tool')
                logging.info(f"Tool {i+1}: {tool_name}")
        else:
            logging.info("No specific tools were recorded for this request")

        return result, elapsed_time, tool_usage
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def create_mcp_server(api_key):
    """Create MCP server instance based on the API key type"""
    # Determine if this is a Spotify or GitHub token based on format
    if len(api_key) == 36 and api_key.count('-') == 4:  # UUID format for Spotify
        return create_spotify_agent(api_key)
    else:  # Assume GitHub token
        return create_github_agent(api_key)

def create_spotify_agent(spotify_api_key):
    """Create agent instance for Spotify"""
    try:
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
    except Exception as e:
        logging.error(f"Error creating Spotify agent: {str(e)}")
        logging.error(f"Error details:", exc_info=True)
        raise

def create_github_agent(github_token):
    """Create agent instance for GitHub"""
    try:
        return MCPServerStdio(
            'npx',
            [
                '--yes',
                '--',
                'node',
                '--experimental-fetch',
                '--no-warnings',
                'node_modules/@modelcontextprotocol/server-github/dist/index.js'
            ],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
                "NODE_OPTIONS": "--no-deprecation"
            }
        )
    except Exception as e:
        logging.error(f"Error creating GitHub agent: {str(e)}")
        logging.error(f"Error details:", exc_info=True)
        raise

async def recommend_songs_from_spotify(agent, user_query: str) -> tuple:
    """Process a Spotify query using the agent and return the result with metrics

    Args:
        agent: The agent
        user_query: The user's query string

    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    return await process_user_request(agent, user_query, "SPOTIFY")

async def manage_github_repository(agent, user_query: str) -> tuple:
    """Process a GitHub query using the agent and return the result with metrics

    Args:
        agent: The agent
        user_query: The user's query string

    Returns:
        tuple: (result, elapsed_time, tool_usage)
    """
    return await process_user_request(agent, user_query, "GITHUB")