import asyncio
import json
import logging
import traceback
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import RunContext
from models import GitHubDeps

class Config:
    """Main configuration class for the agent."""

async def recommend_songs_from_spotify(agent, user_query: str):
    """Process a Spotify query using the agent and return the result with metrics."""
    try:
        return await run_query(agent, user_query, "SPOTIFY")
    except Exception as e:
        logging.error(f"Error recommending songs: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def manage_github_repository(agent, user_query: str):
    """Process a GitHub query using the agent and return the result with metrics."""
    try:
        return await run_query(agent, user_query, "GITHUB")
    except Exception as e:
        logging.error(f"Error managing GitHub repository: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

def create_mcp_server(api_key_or_token):
    """Create an MCP server based on the provided API key or token."""
    try:
        # Determine if this is a Spotify API key or GitHub token based on format/length
        if api_key_or_token and len(api_key_or_token) > 40:  # GitHub tokens are typically longer
            return create_github_mcp_server(api_key_or_token)
        else:
            return create_spotify_mcp_server(api_key_or_token)
    except Exception as e:
        logging.error(f"Error creating MCP server: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def display_tool_usage(server: MCPServerStdio, server_name: str = "MCP"):
    """Display available MCP tools and their details for the specified server."""
    try:
        logging.info("\n" + "="*50)
        logging.info(f"{server_name} MCP TOOLS AND FUNCTIONS")
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
            categories = {'All': []}
            if server_name == "SPOTIFY":
                categories = {
                    'Playback': [],
                    'Search': [],
                    'Playlists': [],
                    'User': [],
                    'Other': []
                }
            elif server_name == "GITHUB":
                categories = {
                    'Repositories': [],
                    'Issues': [],
                    'Pull Requests': [],
                    'Users': [],
                    'Other': []
                }

            tool_dict = {tool.get('name'): tool for tool in tools}
            uncategorized = set(tool_dict.keys())

            for name in uncategorized.copy():
                if server_name == "SPOTIFY":
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
                elif server_name == "GITHUB":
                    if 'repo' in name.lower():
                        categories['Repositories'].append(name)
                    elif 'issue' in name.lower():
                        categories['Issues'].append(name)
                    elif 'pull' in name.lower() or 'pr' in name.lower():
                        categories['Pull Requests'].append(name)
                    elif 'user' in name.lower():
                        categories['Users'].append(name)
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
            logging.info(f"Total Available {server_name} Tools: {len(tools)}")
            logging.info("="*50)
        else:
            logging.warning(f"\nNo {server_name} MCP tools were discovered.")

        return tools

    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return []

async def execute_mcp_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    """Execute an MCP tool with the specified parameters."""
    try:
        logging.info(f"Executing MCP tool: {tool_name}")
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
        logging.error(f"Error executing MCP tool '{tool_name}': {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

async def run_query(agent, user_query: str, server_name: str = "MCP") -> tuple:
    """Process a user query using the MCP agent and return the result with metrics."""
    try:
        start_time = asyncio.get_event_loop().time()

        logging.info(f"Processing query: '{user_query}'")
        result = await agent.run(user_query)
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
            logging.info(f"Tools used in this query: {json.dumps(tool_usage, indent=2)}")
            logging.info(f"Number of tools used: {len(tool_usage)}")

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

def create_spotify_mcp_server(spotify_api_key):
    """Create an MCP server for Spotify API."""
    try:
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@superseoworld/mcp-spotify",
            "--key",
            spotify_api_key
        ]
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating Spotify MCP server: {str(e)}")
        logging.error(f"Error details:", exc_info=True)
        raise

def create_github_mcp_server(github_token):
    """Create MCP server instance for GitHub."""
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
        logging.error(f"Error creating GitHub MCP server: {str(e)}")
        logging.error(f"Error details:", exc_info=True)
        raise

def initialize_agent():
    """Main configuration function for setup and initializations."""
    try:
        logging.info("Initialize the agent setup here...")
        # Initialization logic
    except Exception as e:
        logging.error(f"Error initializing the agent: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise

def setup_agent():
    """Agent setup configuration."""
    try:
        logging.info("Setting up the agent...")
        # Agent setup logic
    except Exception as e:
        logging.error(f"Error setting up the agent: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise