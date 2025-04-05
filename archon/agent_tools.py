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

async def merge_mcp_templates(templates: List[Dict[str, Any]], user_query: str) -> Dict[str, Any]:
    """
    Merge multiple MCP templates into a single coherent template.
    This enhanced version combines templates more intelligently,
    creating a proper multi-service agent like serper_spotify_agent.
    
    Args:
        templates: List of template objects to merge
        user_query: Original user query to add context
        
    Returns:
        Merged template with combined code for each file type
    """
    if not templates or len(templates) == 0:
        return {}
    
    logger.info(f"Merging {len(templates)} MCP templates")
    
    # Extract template names for better naming of the merged result
    template_names = []
    for template in templates:
        name = template.get("folder_name", "").replace("_agent", "").replace("_", "-")
        if name:
            template_names.append(name)
    
    # Create composite folder name
    folder_name = "_".join(template_names) + "_agent" if template_names else "combined_agent"
    
    # Initialize merged template with metadata
    merged = {
        "agents_code": "",
        "models_code": "",
        "main_code": "",
        "tools_code": "",
        "mcp_json": {},
        "purpose": f"Combined template for: {user_query}",
        "folder_name": folder_name,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "original_query": user_query,
            "source_templates": [t.get("folder_name") for t in templates],
            "template_count": len(templates)
        }
    }
    
    # Track imports, functions, and classes for deduplication
    all_imports = {
        "agents": set(),
        "models": set(),
        "main": set(),
        "tools": set()
    }
    
    # Track functions and classes to avoid duplication
    functions = {
        "agents": {},
        "models": {},
        "main": {},
        "tools": {}
    }
    
    # Keep track of detected service providers
    service_providers = []
    mcp_servers = []
    system_prompts = []
    
    # Temporary JSON data for MCP files
    mcp_data_list = []
    
    # Process each template
    for template in templates:
        folder_name = template.get("folder_name", "unknown")
        logger.info(f"Processing template: {folder_name}")
        
        # Extract service name from template folder name
        service_name = folder_name.replace("_agent", "").replace("-", "_").upper()
        service_providers.append(service_name)
        
        # MCP JSON (special handling as it's JSON)
        if template.get("mcp_json"):
            try:
                template_mcp = json.loads(template.get("mcp_json", "{}"))
                if isinstance(template_mcp, dict) and "mcpServers" in template_mcp:
                    # Store more information about the server
                    for server_key, server_data in template_mcp["mcpServers"].items():
                        mcp_servers.append({
                            "name": server_key,
                            "service": service_name,
                            "data": server_data
                        })
                    mcp_data_list.append(template_mcp)
            except json.JSONDecodeError:
                logger.warning(f"Couldn't parse MCP JSON from template {folder_name}")
        
        # Extract system prompt if present in agents_code
        if template.get("agents_code"):
            # Look for system_prompt pattern in agents code
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
                    prompt = matches.group(1).strip()
                    # Tag the prompt with service name
                    system_prompts.append({
                        "service": service_name,
                        "prompt": prompt
                    })
                    break
        
        # Process code files
        for file_type in ["agents_code", "models_code", "main_code", "tools_code"]:
            if not template.get(file_type):
                continue
                
            code = template.get(file_type)
            key_type = file_type.split("_")[0]
            
            # Extract imports
            import_lines = []
            code_lines = code.split("\n")
            for line in code_lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    if key_type in all_imports:
                        all_imports[key_type].add(line)
                        import_lines.append(line)
            
            # Extract functions and classes
            import re
            
            # Find all function definitions
            function_pattern = r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):\s*(?:(?:\'\'\'|\"\"\")[^\'\"]*(?:\'\'\'|\"\"\")\s*)?'
            for match in re.finditer(function_pattern, code):
                func_name = match.group(1)
                func_start = match.start()
                
                # Find the end of the function
                next_def = re.search(r'\ndef\s+', code[func_start + 1:])
                next_class = re.search(r'\nclass\s+', code[func_start + 1:])
                
                if next_def and next_class:
                    end_pos = func_start + 1 + min(next_def.start(), next_class.start())
                elif next_def:
                    end_pos = func_start + 1 + next_def.start()
                elif next_class:
                    end_pos = func_start + 1 + next_class.start()
                else:
                    end_pos = len(code)
                
                func_code = code[func_start:end_pos].strip()
                
                # Skip if it's a duplicate function, unless it's service-specific
                if func_name not in functions[key_type] or any(s.lower() in func_name.lower() for s in service_providers):
                    functions[key_type][func_name] = {
                        "code": func_code,
                        "service": service_name
                    }
            
            # Find all class definitions
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:'
            for match in re.finditer(class_pattern, code):
                class_name = match.group(1)
                class_start = match.start()
                
                # Find the end of the class
                next_def = re.search(r'\ndef\s+', code[class_start + 1:])
                next_class = re.search(r'\nclass\s+', code[class_start + 1:])
                
                if next_def and next_class:
                    end_pos = class_start + 1 + min(next_def.start(), next_class.start())
                elif next_def:
                    end_pos = class_start + 1 + next_def.start()
                elif next_class:
                    end_pos = class_start + 1 + next_class.start()
                else:
                    end_pos = len(code)
                
                class_code = code[class_start:end_pos].strip()
                
                # Skip if it's a duplicate class
                if class_name not in functions[key_type]:
                    functions[key_type][class_name] = {
                        "code": class_code,
                        "service": service_name
                    }
    
    # Create combined system prompt
    if system_prompts:
        combined_prompt = f"""You are a powerful multi-service assistant that combines the following capabilities:

"""
        for i, prompt_data in enumerate(system_prompts, 1):
            service = prompt_data["service"]
            prompt = prompt_data["prompt"]
            # Extract just the capabilities section, excluding boilerplate
            capabilities = "\n".join([line for line in prompt.split("\n") 
                                     if not line.lower().startswith(("you are", "i am", "important", "note:")) 
                                     and line.strip()])
            
            combined_prompt += f"{i}. {service} Capabilities:\n{capabilities}\n\n"
            
        combined_prompt += """
IMPORTANT USAGE NOTES:
- When responding to the user, use the most appropriate service for their request
- If the user's request spans multiple services, use all relevant services in combination
- Always be concise, helpful, and accurate in your responses
- If you don't know how to do something with the available tools, explain what you can do instead
"""
    
    # Generate merged code files
    for file_type in ["agents", "models", "main", "tools"]:
        # Start with imports
        code = "\n".join(sorted(all_imports[file_type])) + "\n\n" if all_imports[file_type] else ""
        
        # Add file header
        code += f"""# {folder_name} - {file_type}.py
# Generated by merging templates: {", ".join(template_names)}
# Based on request: {user_query}
# Using services: {", ".join(service_providers)}

"""
        
        # Special handling for agents.py - add system prompt first
        if file_type == "agents" and system_prompts:
            code += f'system_prompt = """{combined_prompt}"""\n\n'
            
            # Add standard imports that are always needed for agents.py
            standard_imports = """import logging
import traceback
import sys
from typing import List, Optional, Dict, Any
from pydantic_ai import Agent, get_model
from .models import Config
from .tools import *  # Import all tool functions
"""
            code = standard_imports + "\n" + code
        
        # Add all non-duplicate functions and classes
        for item_name, item_data in sorted(functions[file_type].items()):
            item_code = item_data["code"]
            service = item_data["service"]
            
            # Special handling for setup_agent function
            if item_name == "setup_agent" and file_type == "agents":
                # Create a new unified setup_agent function
                setup_code = """async def setup_agent(config: Config) -> Agent:
    \"\"\"Set up the multi-service agent with all required MCP servers.
    
    Args:
        config: Configuration object with API keys
        
    Returns:
        Agent: Configured agent instance with all MCP servers
    \"\"\"
    try:
        servers = []
        
"""
                # Add server creation for each service
                for service in service_providers:
                    setup_code += f"""        # Create {service} MCP server
        logging.info("Creating {service} MCP Server...")
        {service.lower()}_server = create_{service.lower()}_mcp_server(config.{service}_API_KEY)
        servers.append({service.lower()}_server)
        
"""
                
                setup_code += """        # Create agent with all servers
        agent = Agent(get_model(config), mcp_servers=servers)
        
        # Display and capture MCP tools for visibility
        try:
            for server in servers:
                tools = await display_mcp_tools(server)
                logging.info(f"Found {len(tools) if tools else 0} MCP tools available")
        except Exception as tool_err:
            logging.warning(f"Could not display MCP tools: {str(tool_err)}")
            logging.debug("Tool display error details:", exc_info=True)
        
        return agent
            
    except Exception as e:
        logging.error("Error setting up agent: %s", e)
        logging.error("Error details: %s", traceback.format_exc())
        sys.exit(1)
"""
                code += setup_code + "\n\n"
                continue
            
            # Special handling for run_query function in tools.py
            if file_type == "tools" and item_name.startswith("run_") and item_name.endswith("_query"):
                continue  # Skip individual query functions, we'll add a combined one later
            
            # Add the item with a comment indicating its source
            code += f"# From {service} template\n{item_code}\n\n"
        
        # Special handling for tools.py - add combined run_query function
        if file_type == "tools":
            # Add standard imports that are always needed for tools.py
            standard_imports = """import logging
import traceback
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, MCPServerStdio
"""
            code = standard_imports + "\n" + code

            # Add service-specific keywords
            keywords = """# Keywords for service detection"""
            
            for service in service_providers:
                service_upper = service.upper()
                if service_upper == "SERPER":
                    keywords += f"\n{service_upper}_KEYWORDS = ['search', 'google', 'find', 'lookup', 'web search']"
                elif service_upper == "FIRECRAWL":
                    keywords += f"\n{service_upper}_KEYWORDS = ['crawl', 'spider', 'scrape', 'extract', 'website', 'webpage']"
                elif service_upper == "SPOTIFY":
                    keywords += f"\n{service_upper}_KEYWORDS = ['spotify', 'music', 'playlist', 'song', 'track', 'artist', 'album']"
                elif service_upper == "GITHUB":
                    keywords += f"\n{service_upper}_KEYWORDS = ['github', 'git', 'repository', 'repo', 'pull request', 'pr', 'commit', 'branch']"
                else:
                    keywords += f"\n{service_upper}_KEYWORDS = ['{service.lower()}']"
            
            code = keywords + "\n\n" + code

            # Add standard tool functions
            standard_tools = """async def display_mcp_tools(server: MCPServerStdio) -> List[Dict[str, Any]]:
    \"\"\"Display available MCP tools from a server.
    
    Args:
        server: The MCP server to get tools from
        
    Returns:
        List of tool definitions
    \"\"\"
    try:
        tools = await server.list_tools()
        if tools:
            logging.info("Available MCP tools:")
            for tool in tools:
                logging.info(f"- {tool.get('name')}: {tool.get('description')}")
        return tools
    except Exception as e:
        logging.warning(f"Could not list MCP tools: {e}")
        return []

async def execute_mcp_tool(server: MCPServerStdio, tool_name: str, **kwargs) -> Any:
    \"\"\"Execute an MCP tool with the given arguments.
    
    Args:
        server: The MCP server to execute the tool on
        tool_name: Name of the tool to execute
        **kwargs: Tool arguments
        
    Returns:
        Tool execution result
    \"\"\"
    try:
        result = await server.execute_tool(tool_name, kwargs)
        return result
    except Exception as e:
        logging.error(f"Error executing MCP tool {tool_name}: {e}")
        raise
"""
            code += "\n" + standard_tools + "\n"

            # Add server creation functions
            for service in service_providers:
                service_lower = service.lower()
                server_code = f"""
def create_{service_lower}_mcp_server({service.upper()}_API_KEY: str) -> MCPServerStdio:
    \"\"\"Create an MCP server for {service}.
    
    Args:
        {service.upper()}_API_KEY: API key for {service} service
        
    Returns:
        MCPServerStdio: Configured MCP server
    \"\"\"
    try:
        # Set up arguments for the MCP server
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "mcp-server-{service_lower}",
            "--key",
            {service.upper()}_API_KEY
        ]
        
        # Create and return the server
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating {service} MCP server: {{str(e)}}")
        logging.error(f"Error details: {{traceback.format_exc()}}")
        raise
"""
                code += server_code + "\n"

            # Add service-specific query functions
            for service in service_providers:
                service_lower = service.lower()
                query_code = f"""
async def run_{service_lower}_query(query: str, agent: Agent) -> str:
    \"\"\"Execute a query using the {service} service.
    
    Args:
        query: The user's query
        agent: The agent instance
        
    Returns:
        str: Query result
    \"\"\"
    try:
        # Find the {service} server
        server = next((s for s in agent.mcp_servers if "mcp-server-{service_lower}" in str(s.args)), None)
        if not server:
            raise ValueError("No {service} server found")
            
        # Execute the appropriate tool based on the query
        result = await execute_mcp_tool(server, "query", input=query)
        return str(result)
        
    except Exception as e:
        logging.error(f"Error running {service} query: {{e}}")
        return f"Error: {{str(e)}}"
"""
                code += query_code + "\n"

            # Add the combined run_query function
            combined_query = """async def run_query(query: str, agent: Agent) -> str:
    \"\"\"Execute a query using the most appropriate service based on the query content.
    
    Args:
        query: The user's query
        agent: The agent instance with MCP servers
        
    Returns:
        str: The query result
    \"\"\"
    try:
        # Analyze query to determine which service to use
        query_lower = query.lower()
        
"""
            # Add service-specific handling
            for service in service_providers:
                service_lower = service.lower()
                combined_query += f"""        # Check if query is relevant for {service}
        if any(keyword in query_lower for keyword in {service.upper()}_KEYWORDS):
            logging.info("Using {service} for query")
            return await run_{service_lower}_query(query, agent)
            
"""
            
            combined_query += """        # If no specific service matches, try the most appropriate one
        logging.info("No specific service matched, using default")
        try:
            # Try each service in order until one succeeds
"""
            
            for service in service_providers:
                service_lower = service.lower()
                combined_query += f"""            try:
                return await run_{service_lower}_query(query, agent)
            except Exception as e:
                logging.debug(f"Failed to run query with {service}: {{e}}")
                
"""
            
            combined_query += """            # If all services fail, raise the last error
            raise Exception("All services failed to process the query")
            
        except Exception as e:
            logging.error(f"Error running query: {e}")
            return f"Error: {str(e)}"
            
    except Exception as e:
        logging.error(f"Error in run_query: {e}")
        return f"Error: {str(e)}"
"""
            
            code += combined_query + "\n"
        
        # Special handling for models.py
        if file_type == "models":
            # Add standard imports
            standard_imports = """from pydantic import BaseModel
import os
from typing import List, Optional, Dict, Any
"""
            code = standard_imports + "\n" + code

            # Add Config class if not present
            if "class Config" not in code:
                config_class = """class Config(BaseModel):
    \"\"\"Configuration for the agent.\"\"\"
"""
                # Add API key fields for each service
                for service in service_providers:
                    config_class += f"    {service.upper()}_API_KEY: str\n"
                
                # Add LLM API key
                config_class += """    LLM_API_KEY: str
    MODEL_CHOICE: str = "gpt-4-turbo-preview"
    
    @classmethod
    def load_from_env(cls) -> "Config":
        \"\"\"Load configuration from environment variables.\"\"\"
        missing_vars = []
"""
                # Add environment variable checks
                for service in service_providers:
                    config_class += f"""        if not os.getenv("{service.upper()}_API_KEY"):
            missing_vars.append("{service.upper()}_API_KEY")
"""
                
                config_class += """        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        return cls(
"""
                # Add return parameters
                for service in service_providers:
                    config_class += f"            {service.upper()}_API_KEY=os.getenv('{service.upper()}_API_KEY'),\n"
                
                config_class += """            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4-turbo-preview")
        )
"""
                code += config_class + "\n"
        
        # Special handling for main.py
        if file_type == "main":
            # Add standard imports
            standard_imports = """import asyncio
import logging
import os
from dotenv import load_dotenv
from rich.logging import RichHandler
from .agents import setup_agent
from .models import Config
from .tools import run_query

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
"""
            code = standard_imports + "\n" + code

            # Add main function if not present
            if "async def main" not in code:
                main_function = """async def main():
    \"\"\"Main entry point for the agent.\"\"\"
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config = Config.load_from_env()
        
        # Set up the agent
        agent = await setup_agent(config)
        logging.info("Agent setup complete")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                query = input("\\nEnter your query (or 'exit' to quit): ")
                if query.lower() in ["exit", "quit"]:
                    break
                    
                # Process the query
                result = await run_query(query, agent)
                print(f"\\nResult: {result}")
                
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                logging.debug("Error details:", exc_info=True)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        logging.error("Error details: %s", traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("\\nShutting down...")
        sys.exit(0)
"""
                code += main_function + "\n"
        
        # Store the merged code
        merged[f"{file_type}_code"] = code
    
    # Merge MCP JSON files
    if mcp_data_list:
        merged_mcp = {
            "mcpServers": {},
            "metadata": {
                "combinedServices": service_providers,
                "originalQuery": user_query,
                "generatedAt": datetime.now().isoformat(),
                "projectName": folder_name
            }
        }
        
        # Merge all server configurations
        for server in mcp_servers:
            server_name = server["name"]
            server_data = server["data"]
            service_name = server["service"]
            
            # Add server with properly structured API key reference and higher retry values
            server_data["env"] = server_data.get("env", {})
            server_data["env"].update({
                f"{service_name.upper()}_RETRY_MAX_ATTEMPTS": "5",
                f"{service_name.upper()}_RETRY_INITIAL_DELAY": "2000",
                f"{service_name.upper()}_RETRY_MAX_DELAY": "30000",
                f"{service_name.upper()}_RETRY_BACKOFF_FACTOR": "3"
            })
            merged_mcp["mcpServers"][server_name] = server_data
        
        merged["mcp_json"] = json.dumps(merged_mcp, indent=2)
    
    logger.info(f"Successfully merged {len(templates)} templates into {folder_name}")
    return merged

async def adapt_mcp_template(template: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Adapt an MCP template to fit the user's specific request.
    Enhanced version that handles both single and merged templates.
    
    Args:
        template: Template data to adapt
        user_query: User's original request
        
    Returns:
        Adapted template with customized code
    """
    adapted_template = template.copy()
    
    # Extract potential project name from user query
    project_name = "MyAgent"
    words = user_query.split()
    for i, word in enumerate(words):
        if word.lower() in ["for", "about", "using", "create"]:
            if i + 1 < len(words):
                project_name = words[i + 1].strip(",.:;")
                break
    
    # Clean the project name
    project_name = ''.join(c for c in project_name if c.isalnum() or c.isspace())
    project_name = project_name.title().replace(" ", "")
    if not project_name:
        project_name = "MyAgent"
    
    # Check if this is a merged template by looking at the folder_name
    is_merged = any(keyword in template.get("folder_name", "") for keyword in ["merged", "combined", "_"])
    
    # Extract service names from folder_name for merged templates
    service_names = []
    if is_merged:
        folder_name = template.get("folder_name", "")
        # Extract service names from folder name (e.g., spotify_github_agent -> Spotify, Github)
        parts = folder_name.replace("_agent", "").split("_")
        service_names = [part.capitalize() for part in parts if part and part not in ["merged", "combined"]]
    
    # Detect services from user query
    detected_services = await detect_mcp_needs(None, user_query)
    query_service_names = [service.capitalize() for service in detected_services.keys()]
    
    # Combine detected and folder name services without duplicates
    all_services = list(set(service_names + query_service_names))
    
    logger.info(f"Adapting template for project: {project_name} with services: {', '.join(all_services)}")
    
    # Create descriptive agent name based on services
    if all_services:
        agent_name = ''.join(all_services) + "Agent"
    else:
        agent_name = f"{project_name}Agent"
    
    # Update README or add one if it doesn't exist
    readme_content = f"""# {project_name}

A multi-service agent {"combining " + " and ".join(all_services) + " capabilities" if all_services else ""} created with Archon.

## Features

"""
    
    # Add features list to README separately (fixing the backslash in f-string issue)
    if all_services:
        for service in all_services:
            readme_content += f"- {service} integration\n"
    else:
        readme_content += "- Custom agent capabilities\n"
    
    # Continue with rest of README
    readme_content += """
## Setup

1. Clone this repository
2. Install dependencies with `pip install -r requirements.txt`
3. Install Node.js dependencies with `npm install`
4. Create a `.env` file with your API keys (see `.env.example`)
5. Run the agent with `python main.py`

## Environment Variables

"""
    
    # Basic adaptation - replace placeholder names and add comments
    for key in ["agents_code", "models_code", "main_code", "tools_code"]:
        if key not in adapted_template or not adapted_template[key]:
            continue
            
        code = adapted_template[key]
        
        # Add header comment
        header = f"""# {project_name} - {key.replace('_code', '.py')}
# Generated from template: {template.get('folder_name', 'unknown')}
# Based on request: {user_query}
# Using detected services: {', '.join(all_services) if all_services else 'None'}

"""
        
        # Replace generic agent names with project-specific ones
        code = code.replace("MyAgent", agent_name)
        code = code.replace("GenericAgent", agent_name)
        
        # Update system prompt name and description for merged templates
        if is_merged and key == "agents_code":
            # Check if there's a system prompt already defined
            if "system_prompt = " in code:
                # Update the system prompt with better name and description
                name_line = f"You are the {project_name} Assistant, a versatile AI capable of "
                if all_services:
                    name_line += "working with " + ", ".join(all_services[:-1])
                    if len(all_services) > 1:
                        name_line += f" and {all_services[-1]}"
                    else:
                        name_line += all_services[0]
                    name_line += " services."
                else:
                    name_line += "handling a wide range of tasks."
                
                import re
                # Find the system prompt and update the first line
                for pattern in [r'system_prompt\s*=\s*"""', r'system_prompt\s*=\s*\'\'\'', r'system_prompt\s*=\s*[\'"]']:
                    match = re.search(pattern, code)
                    if match:
                        start_pos = match.end()
                        next_line_end = code.find("\n", start_pos)
                        if next_line_end > start_pos:
                            # Replace the first line of the system prompt
                            code = code[:start_pos] + "\n" + name_line + code[next_line_end:]
                            break
        
        adapted_template[key] = header + code
    
    # Extract API keys from MCP JSON and update .env.example
    env_example_content = "# API Keys for MCP Servers\n"
    
    # Update MCP JSON if present
    if "mcp_json" in adapted_template and adapted_template["mcp_json"]:
        try:
            mcp_data = json.loads(adapted_template["mcp_json"])
            if "metadata" not in mcp_data:
                mcp_data["metadata"] = {}
            
            mcp_data["metadata"].update({
                "projectName": project_name,
                "userRequest": user_query,
                "generatedAt": "Generated for specific user request"
            })
            
            # Extract API key names for .env.example
            if "mcpServers" in mcp_data:
                for server_name, server_data in mcp_data["mcpServers"].items():
                    # Extract service name from server name
                    service_prefix = server_name.split("_")[0].upper()
                    env_example_content += f"{service_prefix}_API_KEY=your_{service_prefix.lower()}_api_key\n"
            
            adapted_template["mcp_json"] = json.dumps(mcp_data, indent=2)
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
        adapted_template["readme_code"] = readme_content
    
    return adapted_template           