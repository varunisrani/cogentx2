from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
import sys
import os
import json
import logging

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
    
    # Initialize merged template
    merged = {
        "agents_code": "",
        "models_code": "",
        "main_code": "",
        "tools_code": "",
        "mcp_json": {},
        "purpose": f"Combined template for: {user_query}",
        "folder_name": folder_name
    }
    
    # Track imports for deduplication and service providers
    all_imports = {
        "agents": set(),
        "models": set(),
        "main": set(),
        "tools": set()
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
            
            # Extract imports
            import_lines = []
            code_lines = code.split("\n")
            for line in code_lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Get the module being imported
                    key_type = file_type.split("_")[0]
                    if key_type in all_imports:
                        all_imports[key_type].add(line)
                        import_lines.append(line)
            
            # Store the processed content for merging
            if file_type == "agents_code":
                merged["agents_code"] += f"\n\n# From {service_name} template: {folder_name}\n\n"
                merged["agents_code"] += code
            elif file_type == "models_code":
                merged["models_code"] += f"\n\n# From {service_name} template: {folder_name}\n\n"
                merged["models_code"] += code
            elif file_type == "main_code":
                merged["main_code"] += f"\n\n# From {service_name} template: {folder_name}\n\n"
                merged["main_code"] += code
            elif file_type == "tools_code":
                merged["tools_code"] += f"\n\n# From {service_name} template: {folder_name}\n\n"
                merged["tools_code"] += code
    
    # Create combined system prompt from all extracted prompts
    if system_prompts:
        combined_prompt = f"""
You are a powerful assistant with multiple capabilities:

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
- If the user's request spans multiple services, use all relevant services
- Always be concise, helpful, and accurate in your responses
- If you don't know how to do something with the available tools, explain what you can do instead
"""
        
        # Update the system prompt in agents_code
        if merged["agents_code"]:
            # Create pattern to replace existing system prompts
            replace_pattern = r'(system_prompt|SYSTEM_PROMPT)\s*=\s*(\'\'\'|\"""|\"|\')'
            replacement = f'system_prompt = """{combined_prompt}'
            # Check if there's a pattern match
            import re
            if re.search(replace_pattern, merged["agents_code"]):
                merged["agents_code"] = re.sub(replace_pattern + r'.*?(\'\'\'|\"""|\"|\')', 
                                              replacement + '"""', 
                                              merged["agents_code"], 
                                              flags=re.DOTALL)
            else:
                # Add a new system prompt if one doesn't exist
                merged["agents_code"] = merged["agents_code"].replace("class ", 
                                                                     f"# Combined system prompt\nsystem_prompt = \"\"\"{combined_prompt}\"\"\"\n\nclass ", 
                                                                     1)
    
    # Merge MCP JSON files with enhanced server setup
    if mcp_data_list:
        merged_mcp = {
            "mcpServers": {},
            "metadata": {
                "combinedServices": service_providers,
                "originalQuery": user_query
            }
        }
        
        # Create proper variables for API keys in the setup
        added_env_vars = []
        
        for server in mcp_servers:
            server_name = server["name"]
            server_data = server["data"]
            service_name = server["service"]
            
            # Create standardized API key parameter
            api_key_var = f"{service_name}_API_KEY"
            added_env_vars.append(api_key_var)
            
            # Add server with properly structured API key reference
            merged_mcp["mcpServers"][server_name] = server_data
        
        merged["mcp_json"] = json.dumps(merged_mcp, indent=2)
        
        # Update models.py code to include all needed API keys
        if merged["models_code"]:
            # Find Config class definition
            import re
            config_class_match = re.search(r'class\s+Config\s*\([^)]*\):\s*[\'"]?[^\'"]?[\'"]?', merged["models_code"])
            if config_class_match:
                # Get class fields
                class_body_start = config_class_match.end()
                # Add missing API key fields
                api_key_fields = "\n".join([f"    {key}: str" for key in added_env_vars if key not in merged["models_code"]])
                if api_key_fields:
                    # Insert new fields after existing fields
                    config_class_end = merged["models_code"].find("@classmethod", class_body_start)
                    if config_class_end > 0:
                        merged["models_code"] = (merged["models_code"][:config_class_end] + 
                                               "\n" + api_key_fields + "\n" + 
                                               merged["models_code"][config_class_end:])
                
                # Update environment variable loading in Config.load_from_env
                load_env_match = re.search(r'def\s+load_from_env\s*\(cls\)', merged["models_code"])
                if load_env_match:
                    load_env_start = load_env_match.end()
                    missing_vars_check = "\n        ".join([f"if not os.getenv(\"{key}\"):\n            missing_vars.append(\"{key}\")" 
                                         for key in added_env_vars if f"os.getenv(\"{key}\")" not in merged["models_code"]])
                    
                    # Find missing_vars declaration and add after it
                    missing_vars_decl = re.search(r'missing_vars\s*=\s*\[\]', merged["models_code"][load_env_start:])
                    if missing_vars_decl and missing_vars_check:
                        insert_pos = load_env_start + missing_vars_decl.end()
                        merged["models_code"] = (merged["models_code"][:insert_pos] + 
                                               "\n        " + missing_vars_check + 
                                               merged["models_code"][insert_pos:])
                    
                    # Update return statement to include all API keys
                    return_match = re.search(r'return\s+cls\s*\(', merged["models_code"])
                    if return_match:
                        return_start = return_match.end()
                        return_end = merged["models_code"].find(")", return_start)
                        if return_end > 0:
                            missing_returns = "\n            ".join([f"{key}=os.getenv(\"{key}\")" 
                                             for key in added_env_vars if f"{key}=os.getenv" not in merged["models_code"]])
                            if missing_returns:
                                merged["models_code"] = (merged["models_code"][:return_end] + 
                                                       ",\n            " + missing_returns + 
                                                       merged["models_code"][return_end:])
    
    # Update tools.py to create all required MCP servers
    if merged["tools_code"] and service_providers:
        # Check if we need to add server creation functions
        for service in service_providers:
            create_func = f"create_{service.lower()}_mcp_server"
            if create_func not in merged["tools_code"]:
                server_code = f"""
def {create_func}({service}_API_KEY):
    \"\"\"Create an MCP server for {service}
    
    Args:
        {service}_API_KEY: The API key for {service}
        
    Returns:
        MCPServerStdio: The MCP server instance
    \"\"\"
    try:
        # Set up arguments for the MCP server
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "appropriate-mcp-server-for-{service.lower()}",
            "--key",
            {service}_API_KEY
        ]
        
        # Create and return the server
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating {service} MCP server: {{str(e)}}")
        logging.error(f"Error details: {{traceback.format_exc()}}")
        raise
"""
                merged["tools_code"] += "\n\n" + server_code
    
    # Update agent.py to set up all required MCP servers
    if merged["agents_code"] and service_providers:
        setup_func = "setup_agent"
        if setup_func in merged["agents_code"]:
            # Look for server creation pattern to add missing servers
            for service in service_providers:
                create_server_pattern = f"create_{service.lower()}_mcp_server"
                if create_server_pattern not in merged["agents_code"]:
                    # Find the setup_agent function
                    import re
                    setup_match = re.search(r'async\s+def\s+setup_agent\s*\([^)]*\):', merged["agents_code"])
                    if setup_match:
                        setup_start = setup_match.end()
                        # Find where to insert the new server
                        agent_creation = merged["agents_code"].find("agent = Agent(", setup_start)
                        if agent_creation > 0:
                            # Find the line with server creation
                            server_creation_line = merged["agents_code"][:agent_creation].rfind("\n") + 1
                            # Insert new server creation
                            server_code = f"""
        logging.info("Creating {service} MCP Server...")
        {service.lower()}_server = create_{service.lower()}_mcp_server(config.{service}_API_KEY)
        """
                            merged["agents_code"] = (merged["agents_code"][:server_creation_line] + 
                                                   server_code + 
                                                   merged["agents_code"][server_creation_line:])
                            
                            # Update agent creation to include new server
                            agent_line_end = merged["agents_code"].find("\n", agent_creation)
                            current_agent_line = merged["agents_code"][agent_creation:agent_line_end]
                            if "mcp_servers=" in current_agent_line:
                                # Add server to existing list
                                if f"{service.lower()}_server" not in current_agent_line:
                                    server_list_end = current_agent_line.find("]", current_agent_line.find("mcp_servers="))
                                    if server_list_end > 0:
                                        updated_line = (current_agent_line[:server_list_end] + 
                                                      f", {service.lower()}_server" + 
                                                      current_agent_line[server_list_end:])
                                        merged["agents_code"] = (merged["agents_code"][:agent_creation] + 
                                                               updated_line + 
                                                               merged["agents_code"][agent_line_end:])
                            else:
                                # Add mcp_servers parameter
                                new_line = current_agent_line.replace(")", f", mcp_servers=[{', '.join([f'{s.lower()}_server' for s in service_providers])}])")
                                merged["agents_code"] = (merged["agents_code"][:agent_creation] + 
                                                       new_line + 
                                                       merged["agents_code"][agent_line_end:])
        
    # Add imports for all required packages
    if merged["main_code"]:
        main_imports = "\n".join(sorted(all_imports["main"]))
        if main_imports:
            merged["main_code"] = main_imports + "\n\n" + merged["main_code"].lstrip()
    
    if merged["agents_code"]:
        agent_imports = "\n".join(sorted(all_imports["agents"]))
        if agent_imports:
            merged["agents_code"] = agent_imports + "\n\n" + merged["agents_code"].lstrip()
    
    if merged["models_code"]:
        model_imports = "\n".join(sorted(all_imports["models"]))
        if model_imports:
            merged["models_code"] = model_imports + "\n\n" + merged["models_code"].lstrip()
    
    if merged["tools_code"]:
        tool_imports = "\n".join(sorted(all_imports["tools"]))
        if tool_imports:
            merged["tools_code"] = tool_imports + "\n\n" + merged["tools_code"].lstrip()
    
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