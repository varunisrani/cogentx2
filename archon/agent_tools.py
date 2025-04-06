from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
import sys
import os
import json
import logging
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

# Set up logging for MCP template operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_template_tools")

async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def retrieve_relevant_documentation_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str) -> str:
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 4,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}" 

async def list_documentation_pages_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

async def get_page_content_tool(supabase: Client, url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together but limit the characters in case the page is massive (there are a coule big ones)
        # This will be improved later so if the page is too big RAG will be performed on the page itself
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

def get_file_content_tool(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server

    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        print(f"Error retrieving file contents: {e}")
        return f"Error retrieving file contents: {str(e)}"           

def get_serper_spotify_example(file_type: str) -> str:
    """
    Returns example code for a serper-spotify agent that can be used as reference
    during template merging.
    
    Args:
        file_type: The type of file to get ("agent", "models", "tools", "main", "mcp_json")
        
    Returns:
        String containing the example code
    """
    examples = {
        "agent": """
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
import logging
import sys
import asyncio
import json
import traceback

from models import Config
from tools import create_serper_mcp_server, create_spotify_mcp_server, display_mcp_tools
from tools import execute_mcp_tool, run_query

# Combined system prompt for Serper-Spotify agent
system_prompt = \"\"\"
You are a powerful assistant with dual capabilities:

1. Web Search (Serper): You can search the web for information using the Serper API
   - Search for any information on the internet
   - Retrieve news articles, general knowledge, and up-to-date information
   - Find images and news about specific topics

2. Spotify Music: You can control and interact with Spotify
   - Search for songs, albums, and artists
   - Create and manage playlists
   - Control playback (play, pause, skip, etc.)
   - Get information about the user's library and recommendations

IMPORTANT USAGE NOTES:
- For Spotify search operations, a 'market' parameter (e.g., 'US') is required
- For web searches, try to be specific about what information you're looking for
- When the user asks a knowledge-based question, use web search
- When the user asks about music or wants to control Spotify, use the Spotify tools

When responding to the user, always be concise and helpful. If you don't know how to do something with 
the available tools, explain what you can do instead.
\"\"\"

def get_model(config: Config) -> OpenAIModel:
    \"\"\"Initialize the OpenAI model with the provided configuration.\"\"\"
    try:
        model = OpenAIModel(
            config.MODEL_CHOICE,
            provider=OpenAIProvider(
                base_url=config.BASE_URL,
                api_key=config.LLM_API_KEY
            )
        )
        logging.debug(f"Initialized model with choice: {config.MODEL_CHOICE}")
        return model
    except Exception as e:
        logging.error("Error initializing model: %s", e)
        sys.exit(1)

async def setup_agent(config: Config) -> Agent:
    \"\"\"Set up and initialize the combined Serper-Spotify agent with both MCP servers.\"\"\"
    try:
        # Create MCP server instances for both Serper and Spotify
        logging.info("Creating Serper MCP Server...")
        serper_server = create_serper_mcp_server(config.SERPER_API_KEY)
        
        logging.info("Creating Spotify MCP Server...")
        spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)
        
        # Create agent with both servers
        logging.info("Initializing agent with both Serper and Spotify MCP Servers...")
        agent = Agent(get_model(config), mcp_servers=[serper_server, spotify_server])
        
        # Set system prompt
        agent.system_prompt = system_prompt
        
        # Display and capture MCP tools for visibility from both servers
        try:
            serper_tools = await display_mcp_tools(serper_server, "SERPER")
            logging.info(f"Found {len(serper_tools) if serper_tools else 0} MCP tools available for Serper operations")
            
            spotify_tools = await display_mcp_tools(spotify_server, "SPOTIFY")
            logging.info(f"Found {len(spotify_tools) if spotify_tools else 0} MCP tools available for Spotify operations")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        logging.debug("Agent setup complete with both Serper and Spotify MCP servers.")
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)
""",
        "models": """
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import sys
import logging

class Config(BaseModel):
    \"\"\"Configuration for the Serper-Spotify Agent\"\"\"
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SERPER_API_KEY: str
    SPOTIFY_API_KEY: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        \"\"\"Load configuration from environment variables with better error handling\"\"\"
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("SERPER_API_KEY"):
            missing_vars.append("SERPER_API_KEY")
        if not os.getenv("SPOTIFY_API_KEY"):
            missing_vars.append("SPOTIFY_API_KEY")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error(\"\"\"
LLM_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
SPOTIFY_API_KEY=your_spotify_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            \"\"\")
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY"),
            SPOTIFY_API_KEY=os.getenv("SPOTIFY_API_KEY")
        )

def load_config() -> Config:
    \"\"\"Load the configuration from environment variables\"\"\"
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["SERPER_API_KEY"] = "***" if safe_config["SERPER_API_KEY"] else None
        safe_config["SPOTIFY_API_KEY"] = "***" if safe_config["SPOTIFY_API_KEY"] else None
        logging.debug(f"Config values: {safe_config}")
        return config
    except ValidationError as e:
        logging.error("Configuration validation error:")
        for error in e.errors():
            logging.error(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {str(e)}")
        sys.exit(1)
""",
        "tools": """
from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback

async def display_mcp_tools(server: MCPServerStdio, server_name: str = "MCP"):
    \"\"\"Display available MCP tools and their details for the specified server
    
    Args:
        server: The MCP server instance
        server_name: Name identifier for the server (SERPER or SPOTIFY)
    
    Returns:
        List of available tools
    \"\"\"
    try:
        logging.info("\\n" + "="*50)
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
                    logging.info(f"\\n{category}")
                    logging.info("="*50)
                    
                    for tool in category_tools:
                        logging.info(f"\\n📌 {tool.get('name')}")
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
            
            logging.info("\\n" + "="*50)
            logging.info(f"Total Available {server_name} Tools: {len(tools)}")
            logging.info("="*50)
            
            return tools
                
    except Exception as e:
        logging.error(f"Error displaying MCP tools: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        return []

async def execute_mcp_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    \"\"\"Execute an MCP tool with the specified parameters
    
    Args:
        server: The MCP server instance
        tool_name: Name of the tool to execute
        params: Dictionary of parameters to pass to the tool
        
    Returns:
        The result of the tool execution
    \"\"\"
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
    \"\"\"Process a user query using the MCP agent and return the result with metrics
    
    Args:
        agent: The MCP agent
        user_query: The user's query string
        
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    \"\"\"
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
    \"\"\"Create an MCP server for Serper search API
    
    Args:
        serper_api_key: The API key for Serper
        
    Returns:
        MCPServerStdio: The MCP server instance
    \"\"\"
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
    \"\"\"Create an MCP server for Spotify API
    
    Args:
        spotify_api_key: The API key for Spotify
        
    Returns:
        MCPServerStdio: The MCP server instance
    \"\"\"
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
"""
    }
    
    return examples.get(file_type, "Example not found for this file type")

# New MCP template-related functions

async def search_mcp_templates(supabase: Client, embedding_client: AsyncOpenAI, query: str, threshold: float = 0.65, count: int = 5) -> List[Dict[str, Any]]:
    """
    Search for MCP templates in Supabase based on semantic similarity to the query.
    
    Args:
        supabase: Supabase client
        embedding_client: OpenAI embedding client
        query: User query to search for similar templates
        threshold: Similarity threshold (0-1)
        count: Maximum number of templates to return
        
    Returns:
        List of matching template objects
    """
    try:
        logger.info(f"Searching for MCP templates with query: {query[:100]}...")
        
        # Get embedding for the query
        query_embedding = await get_embedding(query, embedding_client)
        
        # Search for templates in agent_embeddings
        result = supabase.rpc(
            'search_agent_embeddings',
            {
                'query_embedding': query_embedding,
                'similarity_threshold': threshold,
                'match_count': count
            }
        ).execute()
        
        templates = result.data
        logger.info(f"Found {len(templates)} matching templates")
        
        if not templates:
            # Try with a lower threshold if no results
            logger.info(f"Retrying with lower threshold: {threshold * 0.7}")
            result = supabase.rpc(
                'search_agent_embeddings',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': threshold * 0.7,
                    'match_count': count
                }
            ).execute()
            templates = result.data
            logger.info(f"Found {len(templates)} templates with lower threshold")
        
        return templates
    except Exception as e:
        logger.error(f"Error searching MCP templates: {str(e)}")
        return []

async def get_mcp_template_by_id(supabase: Client, template_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific MCP template by ID.
    
    Args:
        supabase: Supabase client
        template_id: The ID of the template to retrieve
        
    Returns:
        Template object or empty dict if not found
    """
    try:
        logger.info(f"Fetching MCP template with ID: {template_id}")
        result = supabase.from_('agent_embeddings') \
            .select('*') \
            .eq('id', template_id) \
            .execute()
        
        if result.data and len(result.data) > 0:
            logger.info(f"Found template: {result.data[0].get('folder_name', 'unknown')}")
            return {
                "found": True,
                "template": result.data[0]
            }
        else:
            logger.warning(f"Template ID {template_id} not found")
            return {
                "found": False,
                "message": f"Template with ID {template_id} not found"
            }
    except Exception as e:
        logger.error(f"Error fetching MCP template: {str(e)}")
        return {
            "found": False,
            "message": f"Error fetching template: {str(e)}"
        }

async def detect_mcp_needs(embedding_client: AsyncOpenAI, user_query: str) -> Dict[str, List[str]]:
    """
    Detect what MCP services might be needed based on the user query.
    
    Args:
        embedding_client: OpenAI embedding client for potential semantic matching
        user_query: The user's request
        
    Returns:
        Dict mapping service names to detected keywords
    """
    query_lower = user_query.lower()
    matched_services = {}
    
    # Define keywords for common MCP services
    service_keywords = {
        "github": ["github", "git", "repository", "repo", "pull request", "pr", "commit", "branch", "issue"],
        "spotify": ["spotify", "music", "playlist", "song", "track", "artist", "album"],
        "youtube": ["youtube", "video", "channel", "streaming", "youtube video"],
        "twitter": ["twitter", "tweet", "x.com", "tweets", "retweet"],
        "slack": ["slack", "message", "channel", "slack message", "workspace"],
        "gmail": ["gmail", "email", "mail", "inbox", "message", "send email"],
        "google_drive": ["google drive", "gdrive", "drive", "document", "spreadsheet", "slides"],
        "notion": ["notion", "page", "database", "notion page", "note"],
        "weather": ["weather", "forecast", "temperature", "climate"],
        "brave_search": ["search", "web search", "find information", "lookup", "research"]
    }
    
    # Check for keyword matches
    for service, keywords in service_keywords.items():
        matched_keywords = []
        for keyword in keywords:
            if keyword in query_lower:
                matched_keywords.append(keyword)
                
        if matched_keywords:
            matched_services[service] = matched_keywords
            logger.info(f"Detected potential need for {service} MCP via keywords: {', '.join(matched_keywords)}")
    
    return matched_services

async def extract_template_summary(template: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a summary of a template for AI analysis with enhanced MCP server detection."""
    folder_name = template.get("folder_name", "unknown")
    service_name = folder_name.replace("_agent", "").replace("-", "_").upper()
    
    # Extract basic information about the template
    summary = {
        "name": folder_name,
        "service": service_name,
        "has_files": {},
        "mcp_details": {}
    }
    
    # Check for the presence of different code files
    for file_type in ["agents_code", "models_code", "main_code", "tools_code"]:
        summary["has_files"][file_type.replace("_code", "")] = bool(template.get(file_type))
    
    # Extract MCP server information from tools code if available
    if template.get("tools_code"):
        code = template.get("tools_code")
        
        # Look for MCP server creation functions
        import re
        server_funcs = re.findall(r'def\s+create_([a-z_]+)_mcp_server', code)
        if server_funcs:
            summary["mcp_details"]["server_creation_functions"] = server_funcs
            
            # Specifically identify Serper and Spotify capabilities
            has_serper = any(func == "serper" for func in server_funcs)
            has_spotify = any(func == "spotify" for func in server_funcs)
            if has_serper:
                summary["has_serper"] = True
            if has_spotify:
                summary["has_spotify"] = True
        
        # Look for npx commands
        npx_patterns = [
            r'MCPServerStdio\s*\(\s*[\'"]npx[\'"]',
            r'MCPServerStdio\s*\(\s*[\'"]npx[\'"],\s*\[([^\]]+)\]'
        ]
        
        for pattern in npx_patterns:
            npx_matches = re.search(pattern, code, re.DOTALL)
            if npx_matches and len(npx_matches.groups()) > 0:
                # Extract the arguments to the npx command
                summary["mcp_details"]["has_npx_commands"] = True
                npx_args = npx_matches.group(1) if len(npx_matches.groups()) > 0 else "Found but couldn't extract args"
                summary["mcp_details"]["npx_args_sample"] = npx_args
        
        # Look for specific MCP server packages
        if "@marcopesani/mcp-server-serper" in code:
            summary["has_serper"] = True
            summary["mcp_details"]["has_serper_package"] = True
        
        if "@superseoworld/mcp-spotify" in code:
            summary["has_spotify"] = True
            summary["mcp_details"]["has_spotify_package"] = True
    
    # Check if there's a system prompt
    if template.get("agents_code"):
        code = template.get("agents_code")
        for prompt_pattern in [
            r'system_prompt\s*=\s*[\'"](.+?)[\'"]', 
            r'system_prompt\s*=\s*"""(.+?)"""',
            r'SYSTEM_PROMPT\s*=\s*[\'"](.+?)[\'"]',
            r'SYSTEM_PROMPT\s*=\s*"""(.+?)"""'
        ]:
            import re
            matches = re.search(prompt_pattern, code, re.DOTALL)
            if matches:
                # Just indicate that a system prompt was found
                summary["has_system_prompt"] = True
                prompt_content = matches.group(1).lower()
                
                # Check if the prompt mentions specific capabilities
                if any(word in prompt_content for word in ["serper", "web search", "google search", "internet search"]):
                    summary["has_serper_in_prompt"] = True
                
                if any(word in prompt_content for word in ["spotify", "music", "playlist", "song", "artist", "album"]):
                    summary["has_spotify_in_prompt"] = True
                
                break
    
    # Extract detailed MCP server information from MCP JSON if available
    if template.get("mcp_json"):
        try:
            mcp_data = json.loads(template.get("mcp_json", "{}"))
            if isinstance(mcp_data, dict) and "mcpServers" in mcp_data:
                # Store more detailed information about each server
                servers_info = []
                for server_name, server_data in mcp_data["mcpServers"].items():
                    server_info = {
                        "name": server_name,
                        "command": server_data.get("command", "unknown"),
                        "has_env_vars": "env" in server_data,
                    }
                    
                    # Add detailed environment variables if available
                    if "env" in server_data:
                        env_vars = []
                        for env_name in server_data["env"]:
                            env_vars.append(env_name)
                        server_info["env_vars"] = env_vars
                    
                    # Check for Serper and Spotify in server names or arguments
                    args_str = str(server_data.get("args", []))
                    if "serper" in server_name.lower() or "serper" in args_str.lower():
                        summary["has_serper"] = True
                        server_info["is_serper"] = True
                    
                    if "spotify" in server_name.lower() or "spotify" in args_str.lower():
                        summary["has_spotify"] = True
                        server_info["is_spotify"] = True
                    
                    servers_info.append(server_info)
                
                summary["mcp_details"]["servers"] = servers_info
                summary["mcp_servers"] = list(mcp_data["mcpServers"].keys())
        except json.JSONDecodeError:
            pass
    
    return summary

async def extract_system_prompt(code: str) -> Optional[str]:
    """Extract system prompt from agents code."""
    if not code:
        return None
    
    for prompt_pattern in [
        r'system_prompt\s*=\s*[\'"](.+?)[\'"]', 
        r'system_prompt\s*=\s*"""(.+?)"""',
        r'SYSTEM_PROMPT\s*=\s*[\'"](.+?)[\'"]',
        r'SYSTEM_PROMPT\s*=\s*"""(.+?)"""'
    ]:
        import re
        matches = re.search(prompt_pattern, code, re.DOTALL)
        if matches:
            return matches.group(1).strip()
    
    return None

async def create_unified_prompt_with_ai(system_prompts: List[Optional[str]], llm_client: AsyncOpenAI) -> str:
    """Use AI to create a unified system prompt from multiple templates."""
    # Filter out None values
    valid_prompts = [p for p in system_prompts if p]
    
    if not valid_prompts:
        return """You are a powerful multi-service assistant that combines multiple capabilities.
Please help the user with their requests using the appropriate tools and services."""
    
    prompt = """Create a unified system prompt for an AI assistant that combines multiple service capabilities.
Extract the core capabilities from each prompt and create a well-structured, non-repetitive system prompt.
The prompt should clearly explain all available capabilities while maintaining a consistent tone and style.

Here are the individual service prompts to merge:

"""
    
    for i, p in enumerate(valid_prompts, 1):
        prompt += f"\nPROMPT {i}:\n```\n{p}\n```\n"
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at creating clear, concise system prompts for AI assistants."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the content from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error creating unified prompt with AI: {e}")
        
        # Fallback: create a simple combined prompt
        combined = """You are a powerful multi-service assistant that combines the following capabilities:

"""
        for i, p in enumerate(valid_prompts, 1):
            # Just take the first few lines of each prompt to avoid repetition
            lines = p.split("\n")[:5]
            combined += f"{i}. Service {i} Capabilities:\n" + "\n".join(lines) + "\n\n"
            
        combined += """
IMPORTANT USAGE NOTES:
- When responding to the user, use the most appropriate service for their request
- If the user's request spans multiple services, use all relevant services in combination
- Always be concise, helpful, and accurate in your responses
"""
        return combined

async def merge_code_with_ai(file_type: str, code_blocks: List[str], llm_client: AsyncOpenAI, validation_prompt: str = "") -> str:
    """Enhanced code merging with validation."""
    if not code_blocks:
        return ""
    
    if len(code_blocks) == 1:
        return code_blocks[0]
    
    example_code = get_serper_spotify_example(file_type)
    has_example = example_code != "Example not found for this file type"
    
    merge_prompt = f"""Merge these Python code blocks for {file_type}.py into a single coherent file.
Follow these critical requirements:

{validation_prompt}

Example of well-structured code:
```python
{example_code if has_example else '# No example available'}
```

Code blocks to merge:
"""
    
    for i, block in enumerate(code_blocks, 1):
        merge_prompt += f"\nBLOCK {i}:\n```python\n{block}\n```\n"
    
    merge_prompt += "\nReturn ONLY the merged code without any markdown or explanations."
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Python developer specializing in code integration and validation."},
                {"role": "user", "content": merge_prompt}
            ]
        )
        
        merged_code = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if merged_code.startswith("```python"):
            merged_code = merged_code[10:]
        if merged_code.startswith("```"):
            merged_code = merged_code[3:]
        if merged_code.endswith("```"):
            merged_code = merged_code[:-3]
        
        return merged_code.strip()
        
    except Exception as e:
        logger.error(f"Error merging {file_type}.py: {e}")
        return "\n\n".join(code_blocks)  # Fallback to simple concatenation

async def ai_merge_mcp_templates(templates: List[Dict[str, Any]], user_query: str, llm_client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Merge multiple MCP templates using AI for intelligent integration.
    Enhanced with validation checks and consistency enforcement.
    
    Args:
        templates: List of template objects to merge
        user_query: Original user query to add context
        llm_client: AsyncOpenAI client for LLM calls
        
    Returns:
        Merged template with AI-integrated code
    """
    if not templates or len(templates) == 0:
        return {}
    
    logger.info(f"AI-merging {len(templates)} MCP templates")
    
    # Extract template names for better naming of the merged result
    template_names = []
    for template in templates:
        name = template.get("folder_name", "").replace("_agent", "").replace("_", "-")
        if name:
            template_names.append(name)
    
    # Create composite folder name
    folder_name = "_".join(template_names) + "_agent" if template_names else "combined_agent"
    
    try:
        # 1. Enhanced template analysis with specific focus on imports and function names
        template_summaries = [await extract_template_summary(t) for t in templates]
        
        # Get example MCP JSON for reference
        example_mcp_json = get_serper_spotify_example("mcp_json")
        
        # Enhanced analysis prompt with specific focus on common issues
        analysis_prompt = f"""Analyze these service templates and recommend a detailed integration strategy.
Focus on preventing common issues:

1. Import Consistency:
- Ensure all required imports are present and consistent across files
- Use absolute imports for project modules
- Maintain consistent import naming

2. Function Name Consistency:
- Use consistent function names across all files (e.g., initialize_agent vs load_config)
- Maintain clear naming patterns for similar functions
- Ensure exported function names match their imports

3. MCP Server Integration:
- Properly initialize all MCP servers with correct configurations
- Include all necessary environment variables
- Handle server startup and connection errors
- Implement proper server cleanup on exit

4. Agent Setup:
- Create a unified agent setup that handles all services
- Ensure consistent system prompt handling
- Properly configure all tools and APIs

Template details:
{json.dumps(template_summaries, indent=2)}

Example MCP JSON configuration for reference (Serper-Spotify agent):
{example_mcp_json}

Your analysis will be used to guide the AI-based merging of these templates.
"""
        
        analysis_response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a code integration expert specializing in merging multi-service agent templates."},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        analysis = analysis_response.choices[0].message.content
        logger.info(f"AI template analysis complete: {analysis[:100]}...")
        
        # 2. Enhanced code merging with validation checks
        merged_files = {}
        
        # Create a validation prompt that will be added to each file merge
        validation_prompt = """
CRITICAL VALIDATION REQUIREMENTS:

1. Import Validation:
- All imports must be at the top of the file
- Project imports must use consistent naming
- Required imports must not be missing
- No duplicate imports

2. Function Name Validation:
- Function names must match their usage across files
- Main configuration function must be named 'initialize_agent'
- Agent setup function must be named 'setup_agent'
- Tool functions must follow pattern 'create_*_mcp_server'

3. MCP Server Validation:
- Each service must have its MCP server creation function
- Server configurations must include all required parameters
- Error handling must be implemented for server startup
- Proper cleanup must be included

4. Code Structure:
- Main configuration class must be named 'AgentConfig'
- System prompts must be properly defined
- Tool configurations must be complete
- Error handling must be comprehensive

ENSURE ALL THESE REQUIREMENTS ARE MET IN THE MERGED CODE.
"""

        # Handle tools.py with enhanced validation
        tools_blocks = [t.get("tools_code", "") for t in templates if t.get("tools_code")]
        if tools_blocks:
            logger.info(f"AI-merging tools.py with enhanced validation...")
            tools_prompt = validation_prompt + """
SPECIFIC TOOLS.PY REQUIREMENTS:

1. MCP Server Functions:
- Must include create_mcp_server function for EACH service
- Must handle server initialization errors
- Must include proper environment variable handling
- Must implement proper cleanup

2. Tool Functions:
- Must include all service-specific tool functions
- Must maintain consistent error handling
- Must include proper type hints
- Must include proper documentation

3. Utility Functions:
- Must include display_tool_usage function
- Must include proper logging setup
- Must include proper type validation
"""
            merged_files["tools_code"] = await merge_code_with_ai("tools", tools_blocks, llm_client, tools_prompt)
            
        # Handle models.py with enhanced validation
        models_blocks = [t.get("models_code", "") for t in templates if t.get("models_code")]
        if models_blocks:
            logger.info(f"AI-merging models.py with enhanced validation...")
            models_prompt = validation_prompt + """
SPECIFIC MODELS.PY REQUIREMENTS:

1. Configuration Class:
- Must be named 'AgentConfig'
- Must include all required fields
- Must implement initialize_from_env method
- Must include proper validation

2. Environment Variables:
- Must handle all required API keys
- Must include proper error messages
- Must implement proper defaults
- Must include type hints
"""
            merged_files["models_code"] = await merge_code_with_ai("models", models_blocks, llm_client, models_prompt)
            
        # Handle agent.py with enhanced validation
        agents_blocks = [t.get("agents_code", "") for t in templates if t.get("agents_code")]
        if agents_blocks:
            logger.info(f"AI-merging agent.py with enhanced validation...")
            agents_prompt = validation_prompt + """
SPECIFIC AGENT.PY REQUIREMENTS:

1. Agent Setup:
- Must be named 'setup_agent'
- Must handle all service types
- Must include proper system prompts
- Must implement proper error handling

2. System Prompts:
- Must be clearly defined
- Must cover all service capabilities
- Must be properly formatted
- Must include usage instructions
"""
            merged_files["agents_code"] = await merge_code_with_ai("agents", agents_blocks, llm_client, agents_prompt)
            
        # Handle main.py with enhanced validation
        main_blocks = [t.get("main_code", "") for t in templates if t.get("main_code")]
        if main_blocks:
            logger.info(f"AI-merging main.py with enhanced validation...")
            main_prompt = validation_prompt + """
SPECIFIC MAIN.PY REQUIREMENTS:

1. Imports:
- Must use 'initialize_agent' from models
- Must use 'setup_agent' from agent
- Must include all required utilities

2. Main Function:
- Must implement proper argument parsing
- Must include interactive mode
- Must handle all service types
- Must implement proper cleanup

3. Error Handling:
- Must handle API failures
- Must handle server connection issues
- Must handle user interrupts
- Must implement proper logging
"""
            merged_files["main_code"] = await merge_code_with_ai("main", main_blocks, llm_client, main_prompt)
        
        # 3. Validate the merged code
        validation_errors = []
        
        # Check for import consistency
        for file_type, code in merged_files.items():
            if "initialize_agent" not in code and file_type == "models_code":
                validation_errors.append(f"Missing initialize_agent function in models.py")
            if "setup_agent" not in code and file_type == "agents_code":
                validation_errors.append(f"Missing setup_agent function in agent.py")
            if "AgentConfig" not in code and file_type == "models_code":
                validation_errors.append(f"Missing AgentConfig class in models.py")
                
        # If validation errors found, try to fix them
        if validation_errors:
            logger.warning(f"Found validation errors: {validation_errors}")
            logger.info("Attempting to fix validation errors...")
            
            # Try to fix models.py if it has issues
            if "models_code" in merged_files and any("models.py" in err for err in validation_errors):
                fix_prompt = f"""Fix the following issues in the models.py code:
{chr(10).join(err for err in validation_errors if 'models.py' in err)}

Current code:
{merged_files['models_code']}

Return ONLY the fixed code.
"""
                fix_response = await llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Python expert fixing code integration issues."},
                        {"role": "user", "content": fix_prompt}
                    ]
                )
                merged_files["models_code"] = fix_response.choices[0].message.content.strip()
        
        # 4. Create unified system prompt
        system_prompts = [await extract_system_prompt(t.get("agents_code", "")) for t in templates]
        unified_prompt = await create_unified_prompt_with_ai(system_prompts, llm_client)
        
        # 5. Enhanced MCP JSON merging
        merged_mcp = await merge_mcp_json(templates, user_query)
        
        # Return the merged template with enhanced validation
        return {
            **merged_files,
            "mcp_json": json.dumps(merged_mcp, indent=2),
            "folder_name": folder_name,
            "purpose": f"AI-merged template for: {user_query}",
            "metadata": {
                "merge_method": "ai-enhanced-validated",
                "created_at": datetime.now().isoformat(),
                "original_query": user_query,
                "source_templates": [t.get("folder_name") for t in templates],
                "template_count": len(templates),
                "ai_analysis": analysis,
                "merged_services": template_names,
                "validation_status": "passed" if not validation_errors else "fixed",
                "fixed_issues": validation_errors if validation_errors else []
            }
        }
        
    except Exception as e:
        logger.error(f"AI-based merging failed: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        logger.warning("Falling back to rule-based merging")
        return await rule_based_merge_mcp_templates(templates, user_query)

async def merge_mcp_json(templates: List[Dict[str, Any]], user_query: str) -> Dict[str, Any]:
    """Enhanced MCP JSON merging with validation."""
    try:
        merged_mcp = {
            "mcpServers": {},
            "metadata": {
                "generatedWith": "ai-enhanced",
                "mergeTimestamp": datetime.now().isoformat(),
                "userQuery": user_query,
                "sourceTemplates": [t.get("folder_name") for t in templates]
            }
        }
        
        for template in templates:
            if template.get("mcp_json"):
                try:
                    mcp_data = json.loads(template.get("mcp_json"))
                    if "mcpServers" in mcp_data:
                        for server_name, server_data in mcp_data["mcpServers"].items():
                            # Enhance server configuration
                            enhanced_server = {
                                **server_data,
                                "env": {
                                    **(server_data.get("env", {})),
                                    "RETRY_MAX_ATTEMPTS": "5",
                                    "RETRY_INITIAL_DELAY": "2000",
                                    "RETRY_MAX_DELAY": "30000",
                                    "RETRY_BACKOFF_FACTOR": "3"
                                }
                            }
                            merged_mcp["mcpServers"][server_name] = enhanced_server
                except json.JSONDecodeError:
                    logger.warning(f"Invalid MCP JSON in template {template.get('folder_name')}")
        
        return merged_mcp
    except Exception as e:
        logger.error(f"Error merging MCP JSON: {e}")
        return {
            "mcpServers": {},
            "metadata": {
                "error": str(e),
                "mergeTimestamp": datetime.now().isoformat()
            }
        }

async def merge_mcp_templates(templates: List[Dict[str, Any]], user_query: str) -> Dict[str, Any]:
    """
    Merge multiple MCP templates into a single coherent template.
    Attempts AI-based merging first, with fallback to rule-based approach.
    
    Args:
        templates: List of template objects to merge
        user_query: Original user query to add context
        
    Returns:
        Merged template with combined code for each file type
    """
    if not templates or len(templates) == 0:
        return {}
    
    # Check if we can use AI-based merging
    try:
        # Check if OPENAI_API_KEY is available
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if openai_key:
            # Initialize OpenAI client
            llm_client = AsyncOpenAI(api_key=openai_key)
            
            # Try AI-based merging
            logger.info("Attempting AI-based template merging with examples")
            
            # Check if templates involve Serper and/or Spotify
            serper_template = None
            spotify_template = None
            other_templates = []
            
            # Extract summaries to check for services
            for template in templates:
                summary = await extract_template_summary(template)
                
                if summary.get("has_serper"):
                    if not serper_template:
                        serper_template = template
                    else:
                        other_templates.append(template)
                elif summary.get("has_spotify"):
                    if not spotify_template:
                        spotify_template = template
                    else:
                        other_templates.append(template)
                else:
                    other_templates.append(template)
            
            # Log what we found
            logger.info(f"Found Serper template: {bool(serper_template)}")
            logger.info(f"Found Spotify template: {bool(spotify_template)}")
            logger.info(f"Other templates: {len(other_templates)}")
            
            # Use the Serper-Spotify example for the merging process
            logger.info("Using the Serper-Spotify example as reference for merging")
            
            # Prioritize templates based on services (for better merging)
            prioritized_templates = []
            if serper_template:
                prioritized_templates.append(serper_template)
            if spotify_template:
                prioritized_templates.append(spotify_template)
            prioritized_templates.extend(other_templates)
            
            # Perform AI-based merging with the prioritized templates
            return await ai_merge_mcp_templates(prioritized_templates, user_query, llm_client)
        else:
            logger.info("No OpenAI API key found, using rule-based merging")
            return await rule_based_merge_mcp_templates(templates, user_query)
    except Exception as e:
        logger.error(f"Error initializing AI-based merging: {e}")
        logger.info("Falling back to rule-based merging")
        return await rule_based_merge_mcp_templates(templates, user_query)

async def adapt_mcp_template(template: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Adapt an MCP template to the user's specific requirements.
    
    Args:
        template: The template to adapt
        user_query: The user's query describing what they want
        
    Returns:
        An adapted template with code tailored to the user's requirements
    """
    logger.info(f"Adapting MCP template to user query: {user_query[:100]}...")
    
    # Make a copy of the template to avoid modifying the original
    adapted_template = dict(template)
    
    # Initialize OpenAI client if possible
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    llm_client = None
    
    if openai_key:
        try:
            llm_client = AsyncOpenAI(api_key=openai_key)
            logger.info("Using AI to adapt template")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
    
    # Extract folder name for naming
    folder_name = template.get("folder_name", "unknown")
    project_name = folder_name.replace("_agent", "").replace("_", " ").title()
    
    # Try to adapt each code file using AI if available
    for file_type in ["agents_code", "models_code", "main_code", "tools_code"]:
        if template.get(file_type) and llm_client:
            code = template.get(file_type)
            file_name = file_type.replace("_code", ".py")
            
            try:
                logger.info(f"Adapting {file_name}...")
                
                # Create a prompt for the AI
                prompt = f"""Adapt this {file_name} code for the following user request:
                
User Request: {user_query}

Original Code:
```python
{code}
```

Modify the code to better suit the user's requirements. Make any necessary changes to:
1. Variable names, function names, and comments to match the user's intent
2. System prompts or agent capabilities to align with the requested functionality
3. Logic and functions to better match the user's goals

Return ONLY the adapted code with no explanations or markdown formatting.
"""
                
                # Get completion from OpenAI
                response = await llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert Python developer specializing in adapting code templates to specific requirements."},
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # Extract the code from the response
                adapted_code = response.choices[0].message.content.strip()
                
                # Remove any markdown code block formatting if present
                if adapted_code.startswith("```python"):
                    adapted_code = adapted_code[10:]
                if adapted_code.startswith("```"):
                    adapted_code = adapted_code[3:]
                if adapted_code.endswith("```"):
                    adapted_code = adapted_code[:-3]
                    
                # Update the template with the adapted code
                adapted_template[file_type] = adapted_code.strip()
                logger.info(f"Successfully adapted {file_name}")
                
            except Exception as e:
                logger.error(f"Error adapting {file_name}: {e}")
                # Keep the original code if there's an error
                logger.info(f"Keeping original {file_name}")
    
    # Adapt MCP JSON if present
    if template.get("mcp_json") and llm_client:
        try:
            logger.info("Adapting MCP JSON configuration...")
            
            # Try to parse the MCP JSON
            mcp_data = json.loads(template.get("mcp_json", "{}"))
            
            # Just update the metadata
            if "metadata" not in mcp_data:
                mcp_data["metadata"] = {}
                
            mcp_data["metadata"]["adaptedWith"] = "ai"
            mcp_data["metadata"]["adaptTimestamp"] = datetime.now().isoformat()
            mcp_data["metadata"]["userQuery"] = user_query
            mcp_data["metadata"]["originalTemplate"] = folder_name
            
            # Convert back to string
            adapted_template["mcp_json"] = json.dumps(mcp_data, indent=2)
            logger.info("Successfully adapted MCP JSON configuration")
            
        except json.JSONDecodeError:
            logger.warning("Couldn't adapt MCP JSON file - not valid JSON")
    
    # Add LLM_API_KEY to env example
    env_example_content = """# API Keys for External Services
"""
    
    # If we have MCP JSON, extract environment variables
    if template.get("mcp_json"):
        try:
            mcp_data = json.loads(template.get("mcp_json", "{}"))
            if "mcpServers" in mcp_data:
                for server_name, server_config in mcp_data["mcpServers"].items():
                    if "env" in server_config:
                        env_example_content += f"\n# {server_name.upper()} Configuration\n"
                        for env_var, value in server_config["env"].items():
                            env_example_content += f"{env_var}={value}\n"
        except json.JSONDecodeError:
            logger.warning("Couldn't adapt MCP JSON file - not valid JSON")
    
    # Add LLM_API_KEY to env example
    env_example_content += "\n# OpenAI API Key for Agent\nLLM_API_KEY=your_openai_api_key\n"
    env_example_content += "\n# Optional Configuration\nMODEL_CHOICE=gpt-4o-mini\n"
    
    # Add requirements.txt if not already present
    if "requirements_code" not in adapted_template or not adapted_template["requirements_code"]:
        adapted_template["requirements_code"] = """pydantic>=2.0.0
pydantic-ai>=0.7.0
python-dotenv>=1.0.0
colorlog>=6.0.0
typer>=0.9.0
rich>=13.0.0
httpx>=0.25.0
"""
    
    # Add .env.example if not already present
    if "env_example_code" not in adapted_template or not adapted_template["env_example_code"]:
        adapted_template["env_example_code"] = env_example_content
    
    # Add package.json for NPM dependencies if not already present
    if "package_json_code" not in adapted_template or not adapted_template["package_json_code"]:
        adapted_template["package_json_code"] = """{
  "name": "%s",
  "version": "1.0.0",
  "description": "Multi-service agent created with Archon",
  "dependencies": {
    "@smithery/cli": "latest"
  }
}""" % project_name.lower()
    
    # Add README if not already present
    if "readme_code" not in adapted_template or not adapted_template["readme_code"]:
        readme_content = f"""# {project_name}

This agent was adapted from the MCP template "{folder_name}".

## Description

An AI agent that {user_query}

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your API keys
3. Run the agent: `python main.py`

## Features

- Uses Pydantic AI for robust agent capabilities
- Includes MCP server integrations for external services
- Customized based on the user's request

## Available Files

"""
        # List the available files
        for file_type in ["agents_code", "models_code", "main_code", "tools_code", "mcp_json"]:
            if adapted_template.get(file_type):
                file_name = file_type.replace("_code", ".py")
                if file_type == "mcp_json":
                    file_name = "mcp.json"
                readme_content += f"- {file_name}\n"
                
        # Add MCP server information if available
        if adapted_template.get("mcp_json"):
            try:
                mcp_data = json.loads(adapted_template.get("mcp_json"))
                if "mcpServers" in mcp_data:
                    readme_content += "\n## MCP Servers\n\n"
                    for server_name in mcp_data["mcpServers"].keys():
                        readme_content += f"- {server_name}\n"
            except json.JSONDecodeError:
                pass
                
        adapted_template["readme_code"] = readme_content
    
    return adapted_template           