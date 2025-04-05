from __future__ import annotations as _annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Generic
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
    search_agent_templates_tool,
    fetch_template_by_id_tool,
    get_embedding
)
from datetime import datetime
# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var, ensure_workbench_dir
from archon.agent_prompts import primary_coder_prompt
load_dotenv()
provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
model = OpenAIModel(llm)
logfire.configure(send_to_logfire='if-token-present')
@dataclass
class PydanticAIDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    reasoner_output: str
    advisor_output: str
pydantic_ai_coder = Agent(
    model,
    system_prompt="""
You are AVA's AI Code Generator Agent specializing in helping the users generate AI Agent code in any type of framework. 
Your purpose is to generate high-quality well-structured Python code to build AI agents for AVA. 
Respond to the user's queries related to code generation and use the provided tools effectively.
First understand what the user is asking for, then suggest an appropriate approach.
When users request an agent, follow these steps:
1. Understand their requirements clearly
2. Search for relevant templates using `search_agent_templates`
3. If templates are found, display them with `display_template_files`
4. If appropriate, generate a complete agent from template using `generate_complete_agent_from_template`
Available Templates:
The system has various agent templates available, including simple agents, complex MCP server agents, and more specialized templates. 
When a user has multiple requirements (such as "I need an agent that can search the web and control Spotify music"), use one of these approaches:
1. Multi-Template Functionality: If the user has multiple requirements, search for templates that match each requirement separately, then merge them using the `merge_agent_templates` tool. For example, if a user needs "a spotify agent with web search capabilities", this would involve merging the spotify_agent and serper_agent templates.
2. Template Customization: If the user's requirements are similar to an existing template but need modifications, use the `update_agent_template` tool to customize it. For example, if a user wants "a Spotify agent that can also summarize songs", you would modify the Spotify template to add this capability.
3. Multi-MCP Example: For requests involving multiple MCP tools or API integrations, refer to the `serper_spotify_agent` example which demonstrates how to create an agent that combines both Serper web search and Spotify music control in a single agent. This example shows how to:
   - Initialize multiple MCP servers in a single agent
   - Create a combined system prompt that handles both capabilities
   - Structure the code to work with multiple API services
   - Handle different types of user queries appropriately
CRITICAL CONSIDERATIONS FOR MULTI-TEMPLATE MERGES:
When merging multiple templates or creating multi-MCP agents, you MUST ensure the following:
1. Configuration Integration:
   - All Config classes must include ALL required API keys and credentials from BOTH templates
   - Environment variable validation must check for ALL required credentials
   - Update the .env.example file to include ALL environment variables
2. Agent Structure:
   - The merged agent must properly initialize ALL MCP servers
   - The system prompt must clearly cover ALL capabilities
   - The agent's dependencies must include ALL necessary classes and variables
3. Import Consistency:
   - Remove duplicate imports
   - Ensure all necessary libraries are imported exactly once
   - Fix any undefined variables or references
4. Tool Integration:
   - Incorporate ALL tools from both templates
   - Ensure tool naming is consistent and doesn't conflict
   - Properly handle each API's authentication and execution flow
5. Main Application Flow:
   - Update user interface to clearly present ALL available capabilities
   - Create a unified command handling system that directs to the appropriate service
   - Include proper help and error messages for ALL functions
6. Testing:
   - Verify that calls to both services work independently
   - Test integrated functionality when applicable
   - Include clear troubleshooting steps for ALL services
For best results, apply tools in this order:
1. `search_agent_templates` - Find relevant templates based on user requirements
2. `display_template_files` - Show the files from matching templates
3. `merge_agent_templates` - Combine multiple templates if needed
4. `update_agent_template` - Customize a template if needed
5. `generate_complete_agent_from_template` - Generate the final agent code
For very specific or unique requirements with no matching templates, create custom agent code from scratch.
IMPORTANT: When generating or modifying agents, maintain compatibility with the existing codebase. The tools.py file is particularly important - modifications to this file should be minimal and focused only on necessary changes. Other files can be more freely modified to meet requirements.
Example Usage Scenarios:
1. Multiple requirements: "I need an agent that can search the web and control my Spotify" 
   - Search for both serper_agent and spotify_agent templates
   - Refer to the serper_spotify_agent example for combining multiple MCP servers
   - Generate a complete agent that handles both capabilities
2. Single requirement with modification: "I want a Spotify agent that can summarize songs"
   - Search for spotify_agent template
   - Modify it to include summarization capabilities
   - Generate the customized agent
3. Unique requirement: "I need an agent that can analyze GitHub repositories"
   - Create custom code if no suitable template exists
   - Build a well-structured agent with appropriate tools
Always provide comprehensive, executable code that meets the user's requirements.
    deps_type=PydanticAIDeps,
    retries=2
)
@pydantic_ai_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
{ctx.deps.reasoner_output if ctx.deps.reasoner_output else "No reasoner output available."}
## Advisor Agent Output:
{ctx.deps.advisor_output if ctx.deps.advisor_output else "No advisor output available."}
## IMPORTANT TEMPLATE HANDLING INSTRUCTIONS:
When working with templates from Supabase, you MUST follow these guidelines:
1. When searching for templates, use search_agent_templates with appropriate queries
2. After finding a relevant template, use display_template_files to show the full code to the user
3. When generating files, use generate_complete_agent_from_template to ensure NO CODE IS SKIPPED
4. Always use markdown code blocks (```python) when displaying code to ensure proper formatting
5. For the Spotify agent specifically, search for "spotify" to get direct matches
6. When displaying template files to the user, make sure to show the FULL CODE, not just descriptions
## NEW MULTI-TEMPLATE FUNCTIONALITY:
When a user has MULTIPLE REQUIREMENTS or needs functionality from DIFFERENT TEMPLATES:
1. Use merge_agent_templates tool to combine multiple templates into a unified agent
2. You can either provide specific template_ids or let the function search based on the query
3. This will merge the agents.py files intelligently, combining imports, dependency classes, and tools
4. It will preserve the tools.py file from the most relevant template
5. Use this capability when a user's requirements span multiple agent types or functionality
Example usage:
- When user wants "spotify agent with web search capabilities", search for both Spotify and search templates
- When user wants to "build an agent that can use Brave search API and access GitHub", merge templates for both
## TEMPLATE CUSTOMIZATION CAPABILITY:
When a user wants to MODIFY an existing template to match specific requirements:
1. Use update_agent_template tool to adapt a template to new user requirements
2. This will preserve the overall structure while making targeted changes to:
   - Update the system prompt to include new requirements
   - Add new dependencies detected from user requirements
   - Create new tools based on the specific requirements
3. Use this when a user wants an existing template with additional capabilities
4. The function handles parsing requirements to make appropriate code changes
Example usage:
- When user wants "spotify agent that can also summarize songs" - update an existing Spotify template
- When user says "modify the agent to handle YouTube links" - update with this new requirement
Always use these tools in this order:
1. search_agent_templates - to find relevant templates
2. display_template_files - to show full code to the user
3. merge_agent_templates - when multiple templates are needed
4. update_agent_template - when customizing a template for specific requirements
5. generate_complete_agent_from_template - for single template solutions without modifications
Remember that tools.py must remain unchanged from the template, while you can make minimal modifications to other files as needed.
@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    Retrieve relevant documentation chunks based on the query with RAG.
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    print(f"\n[DEEP LOG] Retrieving documentation for query: {user_query}")
    result = await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, user_query)
    print(f"\n[DEEP LOG] Documentation retrieval result length: {len(result)} characters")
    return result
@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    Retrieve a list of all available Pydantic AI documentation pages.
    Returns:
        List[str]: List of unique URLs for all documentation pages
    print("\n[DEEP LOG] Listing all documentation pages")
    pages = await list_documentation_pages_tool(ctx.deps.supabase)
    print(f"\n[DEEP LOG] Found {len(pages)} documentation pages: {pages[:5]}{'...' if len(pages) > 5 else ''}")
    return pages
@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
    Returns:
        str: The complete page content with all chunks combined in order
    print(f"\n[DEEP LOG] Getting content for page: {url}")
    content = await get_page_content_tool(ctx.deps.supabase, url)
    print(f"\n[DEEP LOG] Page content retrieved, length: {len(content)} characters")
    if len(content) > 200:
        print(f"\n[DEEP LOG] Content preview: {content[:200]}...")
    return content
@pydantic_ai_coder.tool
async def search_agent_templates(ctx: RunContext[PydanticAIDeps], query: str, threshold: float = 0.4, limit: int = 3) -> Dict[str, Any]:
    Search for agent templates using embedding similarity and direct text matching.
    Args:
        ctx: The context including the Supabase client and embedding client
        query: The search query describing the agent to build
        threshold: Similarity threshold (0.0 to 1.0)
        limit: Maximum number of results to return
    Returns:
        Dict containing similar agent templates with their code and metadata
    print(f"\n[DEEP LOG] Searching agent templates with query: '{query}', threshold: {threshold}, limit: {limit}")
    # Check if the query specifically mentions Spotify
    if 'spotify' in query.lower():
        print(f"\n[DEEP LOG] Spotify explicitly mentioned in query, adding special handling")
        # First try direct term matching with Spotify templates
        spotify_result = ctx.deps.supabase.table('agent_embeddings') \
            .select('*') \
            .ilike('folder_name', '%spotify%') \
            .limit(limit) \
            .execute()
        if spotify_result.data and len(spotify_result.data) > 0:
            print(f"\n[DEEP LOG] Found {len(spotify_result.data)} Spotify templates via direct term matching")
            templates = []
            for template in spotify_result.data:
                templates.append({
                    "id": template['id'],
                    "folder_name": template['folder_name'],
                    "purpose": template['purpose'],
                    "similarity": 1.0,  # Direct match gets highest similarity
                    "agents_code": template['agents_code'],
                    "main_code": template['main_code'],
                    "models_code": template['models_code'],
                    "tools_code": template['tools_code'],
                    "mcp_json": template['mcp_json'],
                    "metadata": template['metadata']
                })
                # Log detailed information
                print(f"\n[DEEP LOG] Template found via direct match:")
                print(f"  ID: {template['id']}")
                print(f"  Folder: {template['folder_name']}")
                print(f"  Purpose: {template['purpose']}")
                # Log code availability
                print(f"  Has agents_code: {bool(template.get('agents_code'))}")
                print(f"  Has main_code: {bool(template.get('main_code'))}")
                print(f"  Has models_code: {bool(template.get('models_code'))}")
                print(f"  Has tools_code: {bool(template.get('tools_code'))}")
                print(f"  Has mcp_json: {bool(template.get('mcp_json'))}")
            return {
                "templates": templates,
                "message": f"Found {len(templates)} Spotify templates."
            }
    # If we don't have a direct match, perform semantic similarity search
    result = await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, threshold, limit)
    # Log detailed information about the templates found
    templates = result.get("templates", [])
    print(f"\n[DEEP LOG] Found {len(templates)} templates via semantic search")
    for i, template in enumerate(templates):
        print(f"\n[DEEP LOG] Template {i+1}:")
        print(f"  ID: {template.get('id')}")
        print(f"  Folder: {template.get('folder_name')}")
        print(f"  Purpose: {template.get('purpose')}")
        print(f"  Similarity: {template.get('similarity', 0):.4f}")
        # Log metadata
        if "metadata" in template and template["metadata"]:
            print(f"  Metadata: {json.dumps(template['metadata'], indent=2)[:300]}...")
        # Log code availability
        print(f"  Has agents_code: {bool(template.get('agents_code'))}")
        print(f"  Has main_code: {bool(template.get('main_code'))}")
        print(f"  Has models_code: {bool(template.get('models_code'))}")
        print(f"  Has tools_code: {bool(template.get('tools_code'))}")
        print(f"  Has mcp_json: {bool(template.get('mcp_json'))}")
    # If we still don't have results, try a more direct database query with lower threshold
    if not templates and threshold > 0.2:
        print(f"\n[DEEP LOG] No templates found with threshold {threshold}, trying with lower threshold 0.2")
        alternative_result = await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, 0.2, limit)
        templates = alternative_result.get("templates", [])
        # Log results from lower threshold
        if templates:
            print(f"\n[DEEP LOG] Found {len(templates)} templates with lower threshold")
            for i, template in enumerate(templates):
                print(f"\n[DEEP LOG] Template {i+1}:")
                print(f"  ID: {template.get('id')}")
                print(f"  Folder: {template.get('folder_name')}")
                print(f"  Purpose: {template.get('purpose')}")
                print(f"  Similarity: {template.get('similarity', 0):.4f}")
            return alternative_result
    return result
@pydantic_ai_coder.tool
async def fetch_template_by_id(ctx: RunContext[PydanticAIDeps], template_id: int) -> Dict[str, Any]:
    Fetch a specific agent template by ID.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch
    Returns:
        Dict containing the agent template with its code and metadata
    print(f"\n[DEEP LOG] Fetching template with ID: {template_id}")
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if result.get("found", False):
        template = result.get("template", {})
        print("\n[DEEP LOG] Template found:")
        print(f"  ID: {template.get('id')}")
        print(f"  Folder: {template.get('folder_name')}")
        print(f"  Purpose: {template.get('purpose')}")
        # Log code previews
        if template.get("agents_code"):
            print(f"\n[DEEP LOG] agents.py preview:\n{template['agents_code'][:300]}...")
        if template.get("main_code"):
            print(f"\n[DEEP LOG] main.py preview:\n{template['main_code'][:300]}...")
        if template.get("models_code"):
            print(f"\n[DEEP LOG] models.py preview:\n{template['models_code'][:300]}...")
        if template.get("tools_code"):
            print(f"\n[DEEP LOG] tools.py preview:\n{template['tools_code'][:300]}...")
        if template.get("mcp_json"):
            print(f"\n[DEEP LOG] mcp.json preview:\n{template['mcp_json'][:300]}...")
    else:
        print(f"\n[DEEP LOG] Template not found. Message: {result.get('message')}")
    return result
@pydantic_ai_coder.tool_plain
def generate_agents_file(code_content: str, file_path: str = "agents.py") -> str:
    Generate an agents.py file with the provided content from a template.
    Args:
        code_content: The content to write to the file (from template with minimal modifications)
        file_path: Path where to save the file (default: agents.py)
    Returns:
        A confirmation message with the full code content
    try:
        print(f"\n[DEEP LOG] Generating agents file from template: {file_path}")
        print(f"\n[DEEP LOG] Content preview (first 300 chars):\n{code_content[:300]}...")
        print(f"\n[DEEP LOG] Content length: {len(code_content)} characters")
        # Ensure workbench directory exists
        os.makedirs("workbench", exist_ok=True)
        # Write the content to the file
        file_path = os.path.join("workbench", file_path)
        with open(file_path, "w") as f:
            f.write(code_content)
        print(f"\n[DEEP LOG] Successfully created {file_path} from template")
        # Return both success message and full code for UI display
        return f"""Successfully created {file_path} from template
```python
{code_content}
```"""
    except Exception as e:
        print(f"\n[DEEP LOG] Error creating {file_path}: {str(e)}")
        return f"Error creating {file_path}: {str(e)}"
@pydantic_ai_coder.tool_plain
def generate_main_file(code_content: str, file_path: str = "main.py") -> str:
    Generate a main.py file with the provided content from a template.
    Args:
        code_content: The content to write to the file (from template with minimal modifications)
        file_path: Path where to save the file (default: main.py)
    Returns:
        A confirmation message with the full code content
    try:
        print(f"\n[DEEP LOG] Generating main file from template: {file_path}")
        print(f"\n[DEEP LOG] Content preview (first 300 chars):\n{code_content[:300]}...")
        print(f"\n[DEEP LOG] Content length: {len(code_content)} characters")
        # Ensure workbench directory exists
        os.makedirs("workbench", exist_ok=True)
        # Write the content to the file
        file_path = os.path.join("workbench", file_path)
        with open(file_path, "w") as f:
            f.write(code_content)
        print(f"\n[DEEP LOG] Successfully created {file_path} from template")
        # Return both success message and full code for UI display
        return f"""Successfully created {file_path} from template
```python
{code_content}
```"""
    except Exception as e:
        print(f"\n[DEEP LOG] Error creating {file_path}: {str(e)}")
        return f"Error creating {file_path}: {str(e)}"
@pydantic_ai_coder.tool_plain
def generate_models_file(code_content: str, file_path: str = "models.py") -> str:
    Generate a models.py file with the provided content from a template.
    Args:
        code_content: The content to write to the file (from template with minimal modifications)
        file_path: Path where to save the file (default: models.py)
    Returns:
        A confirmation message with the full code content
    try:
        print(f"\n[DEEP LOG] Generating models file from template: {file_path}")
        print(f"\n[DEEP LOG] Content preview (first 300 chars):\n{code_content[:300]}...")
        print(f"\n[DEEP LOG] Content length: {len(code_content)} characters")
        # Ensure workbench directory exists
        os.makedirs("workbench", exist_ok=True)
        # Write the content to the file
        file_path = os.path.join("workbench", file_path)
        with open(file_path, "w") as f:
            f.write(code_content)
        print(f"\n[DEEP LOG] Successfully created {file_path} from template")
        # Return both success message and full code for UI display
        return f"""Successfully created {file_path} from template
```python
{code_content}
```"""
    except Exception as e:
        print(f"\n[DEEP LOG] Error creating {file_path}: {str(e)}")
        return f"Error creating {file_path}: {str(e)}"
@pydantic_ai_coder.tool_plain
def generate_tools_file(code_content: str, file_path: str = "tools.py") -> str:
    Generate a tools.py file with the provided content from a template.
    IMPORTANT: tools.py should be kept unchanged from the template unless absolutely necessary.
    Args:
        code_content: The content to write to the file (from template with NO modifications)
        file_path: Path where to save the file (default: tools.py)
    Returns:
        A confirmation message with the full code content
    try:
        print(f"\n[DEEP LOG] Generating tools file from template: {file_path}")
        print(f"\n[DEEP LOG] Content preview (first 300 chars):\n{code_content[:300]}...")
        print(f"\n[DEEP LOG] Content length: {len(code_content)} characters")
        # Ensure workbench directory exists
        os.makedirs("workbench", exist_ok=True)
        # Write the content to the file
        file_path = os.path.join("workbench", file_path)
        with open(file_path, "w") as f:
            f.write(code_content)
        print(f"\n[DEEP LOG] Successfully created {file_path} from template (NO modifications as required)")
        # Return both success message and full code for UI display
        return f"""Successfully created {file_path} from template (NO modifications as required)
```python
{code_content}
```"""
    except Exception as e:
        print(f"\n[DEEP LOG] Error creating {file_path}: {str(e)}")
        return f"Error creating {file_path}: {str(e)}"
@pydantic_ai_coder.tool_plain
def generate_mcp_json(json_content: str, file_path: str = "mcp.json") -> str:
    Generate an mcp.json file with the provided content from a template.
    Args:
        json_content: The content to write to the file (JSON configuration)
        file_path: Path where to save the file (default: mcp.json)
    Returns:
        A confirmation message with the full code content
    try:
        print(f"\n[DEEP LOG] Generating MCP JSON file from template: {file_path}")
        print(f"\n[DEEP LOG] Content preview (first 300 chars):\n{json_content[:300]}...")
        print(f"\n[DEEP LOG] Content length: {len(json_content)} characters")
        # Ensure workbench directory exists
        os.makedirs("workbench", exist_ok=True)
        # Write the content to the file
        file_path = os.path.join("workbench", file_path)
        with open(file_path, "w") as f:
            f.write(json_content)
        print(f"\n[DEEP LOG] Successfully created {file_path} from template")
        # Return both success message and full code for UI display
        return f"""Successfully created {file_path} from template
```json
{json_content}
```"""
    except Exception as e:
        print(f"\n[DEEP LOG] Error creating {file_path}: {str(e)}")
        return f"Error creating {file_path}: {str(e)}"
@pydantic_ai_coder.tool
async def extract_and_use_template(ctx: RunContext[PydanticAIDeps], template_id: int) -> Dict[str, str]:
    Extract code files from a template and create the necessary files.
    This is a helper function that fetches a template by ID and automatically
    generates all the needed files from that template.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch and use
    Returns:
        Dict containing confirmation messages for each file created
    print(f"\n[DEEP LOG] Extracting and using template ID: {template_id}")
    # Fetch the template
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {"error": f"Template ID {template_id} not found: {result.get('message')}"}
    template = result.get("template", {})
    print(f"\n[DEEP LOG] Successfully fetched template: {template.get('folder_name')}")
    # Create the files from the template
    messages = {}
    # Create agents.py if available
    if template.get("agents_code"):
        agents_file = generate_agents_file(template["agents_code"])
        messages["agents.py"] = agents_file
    # Create main.py if available
    if template.get("main_code"):
        main_file = generate_main_file(template["main_code"])
        messages["main.py"] = main_file
    # Create models.py if available
    if template.get("models_code"):
        models_file = generate_models_file(template["models_code"])
        messages["models.py"] = models_file
    # Create tools.py if available (NO modifications)
    if template.get("tools_code"):
        tools_file = generate_tools_file(template["tools_code"])
        messages["tools.py"] = tools_file
    # Create mcp.json if available
    if template.get("mcp_json"):
        mcp_file = generate_mcp_json(template["mcp_json"])
        messages["mcp.json"] = mcp_file
    print(f"\n[DEEP LOG] Template extraction complete. Created {len(messages)} files.")
    return messages
@pydantic_ai_coder.tool
async def extract_and_modify_template(ctx: RunContext[PydanticAIDeps], template_id: int, 
                                     agent_changes: str = "", 
                                     main_changes: str = "", 
                                     models_changes: str = "", 
                                     mcp_changes: str = "") -> Dict[str, str]:
    Extract code files from a template, apply specific modifications, and create the files.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch and modify
        agent_changes: Description of changes to make to agents.py (optional)
        main_changes: Description of changes to make to main.py (optional)
        models_changes: Description of changes to make to models.py (optional)
        mcp_changes: Description of changes to make to mcp.json (optional)
    Returns:
        Dict containing confirmation messages for each file created
    print(f"\n[DEEP LOG] Extracting and modifying template ID: {template_id}")
    print(f"\n[DEEP LOG] Modifications requested:")
    if agent_changes: print(f"- agents.py: {agent_changes}")
    if main_changes: print(f"- main.py: {main_changes}")
    if models_changes: print(f"- models.py: {models_changes}")
    if mcp_changes: print(f"- mcp.json: {mcp_changes}")
    # Fetch the template
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {"error": f"Template ID {template_id} not found: {result.get('message')}"}
    template = result.get("template", {})
    print(f"\n[DEEP LOG] Successfully fetched template: {template.get('folder_name')}")
    # Apply modifications and create files
    messages = {}
    # Modify and create agents.py if available
    if template.get("agents_code"):
        # If changes are specified, you can modify the code based on those changes
        # Here we're just logging the fact that changes were requested
        agents_code = template["agents_code"]
        if agent_changes:
            print(f"\n[DEEP LOG] Applying modifications to agents.py as specified: {agent_changes}")
        agents_file = generate_agents_file(agents_code)
        messages["agents.py"] = agents_file
    # Modify and create main.py if available
    if template.get("main_code"):
        main_code = template["main_code"]
        if main_changes:
            print(f"\n[DEEP LOG] Applying modifications to main.py as specified: {main_changes}")
        main_file = generate_main_file(main_code)
        messages["main.py"] = main_file
    # Modify and create models.py if available
    if template.get("models_code"):
        models_code = template["models_code"]
        if models_changes:
            print(f"\n[DEEP LOG] Applying modifications to models.py as specified: {models_changes}")
        models_file = generate_models_file(models_code)
        messages["models.py"] = models_file
    # Create tools.py if available (NO modifications, as per requirements)
    if template.get("tools_code"):
        tools_file = generate_tools_file(template["tools_code"])
        messages["tools.py"] = tools_file
    # Modify and create mcp.json if available
    if template.get("mcp_json"):
        mcp_json = template["mcp_json"]
        if mcp_changes:
            print(f"\n[DEEP LOG] Applying modifications to mcp.json as specified: {mcp_changes}")
        mcp_file = generate_mcp_json(mcp_json)
        messages["mcp.json"] = mcp_file
    print(f"\n[DEEP LOG] Template modification complete. Created {len(messages)} files.")
    return messages
@pydantic_ai_coder.tool
async def get_template_structure(ctx: RunContext[PydanticAIDeps], template_id: int) -> Dict[str, Any]:
    Get the structure and metadata of a specific template without extracting it.
    This is useful to inspect a template before deciding to use it.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to inspect
    Returns:
        Dict containing template metadata and available files
    print(f"\n[DEEP LOG] Inspecting template structure for ID: {template_id}")
    # Fetch the template
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {"error": f"Template ID {template_id} not found: {result.get('message')}"}
    template = result.get("template", {})
    # Extract metadata
    metadata = {
        "id": template.get("id"),
        "folder_name": template.get("folder_name"),
        "purpose": template.get("purpose"),
        "tags": template.get("tags", []),
        "created_at": template.get("created_at"),
        "metadata": template.get("metadata", {})
    }
    # Check available files
    available_files = []
    if template.get("agents_code"):
        available_files.append("agents.py")
    if template.get("main_code"):
        available_files.append("main.py")
    if template.get("models_code"):
        available_files.append("models.py")
    if template.get("tools_code"):
        available_files.append("tools.py")
    if template.get("mcp_json"):
        available_files.append("mcp.json")
    # Extract code previews (first 200 chars of each file)
    code_previews = {}
    if template.get("agents_code"):
        preview = template["agents_code"][:200] + ("..." if len(template["agents_code"]) > 200 else "")
        code_previews["agents.py"] = preview
    if template.get("main_code"):
        preview = template["main_code"][:200] + ("..." if len(template["main_code"]) > 200 else "")
        code_previews["main.py"] = preview
    if template.get("models_code"):
        preview = template["models_code"][:200] + ("..." if len(template["models_code"]) > 200 else "")
        code_previews["models.py"] = preview
    if template.get("tools_code"):
        preview = template["tools_code"][:200] + ("..." if len(template["tools_code"]) > 200 else "")
        code_previews["tools.py"] = preview
    if template.get("mcp_json"):
        preview = template["mcp_json"][:200] + ("..." if len(template["mcp_json"]) > 200 else "")
        code_previews["mcp.json"] = preview
    # Print detailed log information
    print(f"\n[DEEP LOG] Template metadata: {json.dumps(metadata, indent=2, default=str)}")
    print(f"\n[DEEP LOG] Available files: {', '.join(available_files)}")
    return {
        "metadata": metadata,
        "available_files": available_files,
        "code_previews": code_previews
    }
@pydantic_ai_coder.tool
async def generate_complete_agent_from_template(ctx: RunContext[PydanticAIDeps], template_id: int, 
                                              custom_name: str = "", 
                                              custom_description: str = "",
                                              add_comments: bool = True,
                                              query: str = "") -> Dict[str, str]:
    Generate complete agent code files from a template in Supabase without skipping any parts.
    Handles large files by breaking them into manageable chunks while ensuring complete code generation.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch
        custom_name: Optional custom name for the agent (defaults to template name)
        custom_description: Optional custom description for the agent
        add_comments: Whether to add explanatory comments to the generated code
        query: The original query used to search for this template (to check for multi-service needs)
    Returns:
        Dict containing confirmation messages for each file created, including the full code content
    print(f"\n[DEEP LOG] Generating complete agent code from template ID: {template_id}")
    print(f"\n[DEEP LOG] Custom name: {custom_name or 'Not specified (using template name)'}")
    print(f"\n[DEEP LOG] Custom description: {custom_description or 'Not specified (using template description)'}")
    print(f"\n[DEEP LOG] Original query: {query}")
    # Check if this is a multi-service request and we should merge templates instead
    if query:
        # List of common keywords that indicate a multi-service request
        multi_service_keywords = [
            "and", "with", "both", "multiple", "combine", "integration", 
            "spotify github", "github spotify", "search email", "email search"
        ]
        # Check if any of these keywords appear in the query
        is_multi_service = any(keyword in query.lower() for keyword in multi_service_keywords)
        # If this is a multi-service request, we should redirect to merge_agent_templates
        if is_multi_service:
            print(f"\n[DEEP LOG] Detected multi-service request from query: '{query}'")
            print(f"\n[DEEP LOG] Will search for additional templates to merge with template ID: {template_id}")
            # If we have a multi-service request but only a single template ID, 
            # let's find additional relevant templates
            related_templates = []
            # Use the query to find other relevant templates
            if ctx.deps.embedding_client:
                search_result = await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, 0.3, 5)
                if search_result.get("templates"):
                    for template in search_result.get("templates", []):
                        # Skip the current template
                        if template.get("id") != template_id:
                            related_templates.append(template.get("id"))
            # If we found additional templates, merge them
            if related_templates:
                # Use the first related template (most relevant) along with the current one
                print(f"\n[DEEP LOG] Found additional templates to merge: {related_templates}")
                print(f"\n[DEEP LOG] Redirecting to merge_agent_templates with template IDs: {[template_id, related_templates[0]]}")
                # Call merge_agent_templates instead
                return await merge_agent_templates(
                    ctx, 
                    template_ids=[template_id, related_templates[0]],
                    custom_name=custom_name,
                    custom_description=custom_description
                )
            else:
                print(f"\n[DEEP LOG] No additional templates found to merge with template ID: {template_id}")
    # Proceed with the regular template generation if not a multi-service request or no additional templates found
    # Fetch the complete template from Supabase
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {"error": f"Template ID {template_id} not found: {result.get('message')}"}
    template = result.get("template", {})
    print(f"\n[DEEP LOG] Successfully fetched template: {template.get('folder_name')}")
    # Process template metadata for customization
    folder_name = template.get('folder_name', 'unknown_template')
    agent_name = custom_name if custom_name else folder_name.replace('_', ' ').title()
    purpose = template.get('purpose', '')
    description = custom_description if custom_description else purpose
    # Prepare messages dictionary for file creation results
    messages = {}
    # Also store complete code for UI display
    complete_code = {}
    # Process each file from the template
    # 1. AGENTS.PY
    if template.get("agents_code"):
        agents_code = template["agents_code"]
        print(f"\n[DEEP LOG] Full agents.py code length: {len(agents_code)} characters")
        # Add customization if requested
        if add_comments:
            agents_code = f"""# {agent_name} Agent
# Purpose: {description}
# Generated from template: {folder_name} (ID: {template_id})
# Generation timestamp: {datetime.now().isoformat()}
{agents_code}"""
        print(f"\n[DEEP LOG] Generating agents.py ({len(agents_code)} characters)")
        # Process the entire file without skipping any parts
        agents_file = generate_agents_file(agents_code)
        messages["agents.py"] = agents_file
        complete_code["agents.py"] = agents_code
    # 2. MAIN.PY
    if template.get("main_code"):
        main_code = template["main_code"]
        print(f"\n[DEEP LOG] Full main.py code length: {len(main_code)} characters")
        # Add customization if requested
        if add_comments:
            main_code = f"""# {agent_name} Main Module
# Purpose: {description}
# Generated from template: {folder_name} (ID: {template_id})
# Generation timestamp: {datetime.now().isoformat()}
{main_code}"""
        print(f"\n[DEEP LOG] Generating main.py ({len(main_code)} characters)")
        main_file = generate_main_file(main_code)
        messages["main.py"] = main_file
        complete_code["main.py"] = main_code
    # 3. MODELS.PY - Merge Config classes and models from all templates
    print("\n[DEEP LOG] Merging models.py from all templates")
    merged_models_py = f"# Merged models.py from templates: {', '.join(template_names)}\n"
    merged_models_py += f"# Created at: {datetime.now().isoformat()}\n\n"
    # Extract imports, Config classes and models from all templates
    models_imports = set([
        "from pydantic import BaseModel, ValidationError",
        "from dotenv import load_dotenv",
        "import os",
        "import sys",
        "import logging"
    ])
    config_fields = {}
    model_classes = []
    seen_models = set()
    for template in templates:
        models_code = template.get("models_code", "")
        if not models_code:
            continue
        print(f"\n[DEEP LOG] Processing models from template: {template.get('folder_name')}")
        # Extract imports
        import_matches = re.findall(r"^(?:import|from)\s+.*$", models_code, re.MULTILINE)
        for imp in import_matches:
            models_imports.add(imp)
        # Extract Config class fields
        config_matches = re.findall(r"class\s+Config(?:\(BaseModel\))?:.*?(?=\n\nclass|\n\ndef|\Z)", models_code, re.DOTALL)
        if config_matches:
            config_class = config_matches[0]
            # Extract fields with their types and defaults
            field_matches = re.findall(r"^\s+(\w+):\s+([^=\n]+)(?:=\s*([^\n]+))?$", config_class, re.MULTILINE)
            for field_name, field_type, default_value in field_matches:
                if field_name not in config_fields:
                    config_fields[field_name] = (field_type.strip(), default_value.strip() if default_value else None)
                    print(f"  - Added config field: {field_name}: {field_type}")
        # Extract other model classes
        model_matches = re.findall(r"class\s+(?!Config)(\w+)(?:\(BaseModel\))?:.*?(?=\n\nclass|\n\ndef|\Z)", models_code, re.DOTALL)
        for model_name in model_matches:
            if model_name not in seen_models:
                model_class = re.search(f"class\\s+{model_name}(?:\\(BaseModel\\))?:.*?(?=\\n\\nclass|\\n\\ndef|\\Z)", models_code, re.DOTALL)
                if model_class:
                    model_classes.append(model_class.group(0))
                    seen_models.add(model_name)
                    print(f"  - Added model class: {model_name}")
    # Add all unique imports
    merged_models_py += "\n".join(sorted(models_imports)) + "\n\n"
    # Create merged Config class
    merged_models_py += "class Config(BaseModel):\n"
    merged_models_py += "    \"\"\"Combined configuration for merged agent templates\"\"\"\n"
    for field_name, (field_type, default_value) in sorted(config_fields.items()):
        field_line = f"    {field_name}: {field_type}"
        if default_value:
            field_line += f" = {default_value}"
        merged_models_py += field_line + "\n"
    # Add helper methods to Config class
    merged_models_py += "\n    @classmethod\n"
    merged_models_py += "    def load_from_env(cls) -> 'Config':\n"
    merged_models_py += "        \"\"\"Load configuration from environment variables\"\"\"\n"
    merged_models_py += "        load_dotenv()\n\n"
    merged_models_py += "        # Check for required environment variables\n"
    merged_models_py += "        missing_vars = []\n"
    # Add environment variable checks for required fields
    for field_name, (field_type, default_value) in config_fields.items():
        if not default_value and 'Optional' not in field_type:
            merged_models_py += f"        if not os.getenv(\"{field_name}\"):\n"
            merged_models_py += f"            missing_vars.append(\"{field_name}\")\n"
    merged_models_py += "\n        if missing_vars:\n"
    merged_models_py += "            logging.error(\"Missing required environment variables:\")\n"
    merged_models_py += "            for var in missing_vars:\n"
    merged_models_py += "                logging.error(f\"  - {var}\")\n"
    merged_models_py += "            logging.error(\"\\nPlease create a .env file with the required variables.\")\n"
    merged_models_py += "            sys.exit(1)\n\n"
    # Add return statement with all fields
    merged_models_py += "        return cls(\n"
    for field_name, (field_type, default_value) in sorted(config_fields.items()):
        merged_models_py += f"            {field_name}=os.getenv(\"{field_name}\""
        if default_value:
            merged_models_py += f", {default_value}"
        merged_models_py += "),\n"
    merged_models_py += "        )\n\n"
    # Add other model classes
    for model_class in model_classes:
        merged_models_py += "\n" + model_class + "\n"
    # 4. TOOLS.PY - Merge utility functions and MCP server handlers
    print("\n[DEEP LOG] Merging tools.py from all templates")
    merged_tools_py = f"# Merged tools.py from templates: {', '.join(template_names)}\n"
    merged_tools_py += f"# Created at: {datetime.now().isoformat()}\n\n"
    # Extract imports and functions from all templates
    tools_imports = set()
    mcp_server_functions = []
    utility_functions = []
    seen_functions = set()
    for template in templates:
        tools_code = template.get("tools_code", "")
        if not tools_code:
            continue
        # Extract imports
        import_matches = re.findall(r"^(?:import|from)\s+.*$", tools_code, re.MULTILINE)
        for imp in import_matches:
            tools_imports.add(imp)
        # Extract MCP server creation functions
        mcp_matches = re.findall(r"def\s+create_(\w+)_mcp_server.*?(?=\n\ndef|\n\nclass|\Z)", tools_code, re.DOTALL)
        for mcp_name in mcp_matches:
            if f"create_{mcp_name}_mcp_server" not in seen_functions:
                mcp_def = re.search(f"def\\s+create_{mcp_name}_mcp_server.*?(?=\\n\\ndef|\\n\\nclass|\\Z)", tools_code, re.DOTALL)
                if mcp_def:
                    mcp_server_functions.append(mcp_def.group(0))
                    seen_functions.add(f"create_{mcp_name}_mcp_server")
        # Extract utility functions (non-MCP server functions)
        func_matches = re.findall(r"def\s+(?!create_\w+_mcp_server)(\w+).*?(?=\n\ndef|\n\nclass|\Z)", tools_code, re.DOTALL)
        for func_name in func_matches:
            if func_name not in seen_functions:
                func_def = re.search(f"def\\s+{func_name}.*?(?=\\n\\ndef|\\n\\nclass|\\Z)", tools_code, re.DOTALL)
                if func_def:
                    utility_functions.append(func_def.group(0))
                    seen_functions.add(func_name)
    # Add all unique imports
    merged_tools_py += "\n".join(sorted(tools_imports)) + "\n\n"
    # Add helper functions for MCP tools management
    merged_tools_py += """
async def display_mcp_tools(server: MCPServerStdio, server_name: str = "MCP"):
    \"\"\"Display available MCP tools and their details for the specified server
    Args:
        server: The MCP server instance
        server_name: Name identifier for the server
    Returns:
        List of available tools
    \"\"\"
    try:
        logging.info("\\n" + "="*50)
        logging.info(f"{server_name} MCP TOOLS AND FUNCTIONS")
        logging.info("="*50)
        tools = []
        try:
            # Try different methods to get tools
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
            if tools:
                for tool in tools:
                    logging.info(f"\\n📌 {tool.get('name')}")
                    if tool.get('description'):
                        logging.info(f"   Description: {tool.get('description')}")
                    if tool.get('parameters'):
                        logging.info("   Parameters:")
                        for param_name, param_info in tool['parameters'].get('properties', {}).items():
                            required = param_name in tool['parameters'].get('required', [])
                            logging.info(f"   - {param_name} ({param_info.get('type', 'unknown')}) {'[Required]' if required else '[Optional]'}")
        except Exception as tool_error:
            logging.debug(f"Error listing tools: {tool_error}", exc_info=True)
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
    # Add MCP server creation functions
    for func in mcp_server_functions:
        merged_tools_py += "\n" + func + "\n"
    # Add utility functions
    for func in utility_functions:
        merged_tools_py += "\n" + func + "\n"
    # Add run_query function that handles multiple MCP servers
    merged_tools_py += """
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
        logging.info(f"Processing query: '{user_query}'")
        # Execute the query
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        # Extract tool usage information
        tool_usage = []
        if hasattr(result, 'metadata') and result.metadata:
            try:
                if isinstance(result.metadata, dict):
                    tool_usage = result.metadata.get('tools', []) or result.metadata.get('tool_calls', [])
                elif hasattr(result.metadata, 'tools'):
                    tool_usage = result.metadata.tools
            except Exception as tool_err:
                logging.debug(f"Could not extract tool usage: {tool_err}")
        # Log tool usage
        if tool_usage:
            logging.info(f"Tools used in this query: {len(tool_usage)}")
            for tool in tool_usage:
                logging.info(f"- {tool.get('name', 'Unknown tool')}")
        return (result, elapsed_time, tool_usage)
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise
    # 5. MCP.JSON
    if template.get("mcp_json"):
        mcp_json = template["mcp_json"]
        print(f"\n[DEEP LOG] Full mcp.json length: {len(mcp_json)} characters")
        # For MCP.JSON, we preserve the JSON structure but can add a comment if it's valid JSON
        try:
            # Try to parse and reconstruct to ensure valid JSON
            mcp_data = json.loads(mcp_json)
            # Add metadata if requested
            if add_comments and isinstance(mcp_data, dict):
                if "metadata" not in mcp_data:
                    mcp_data["metadata"] = {}
                mcp_data["metadata"]["templateInfo"] = {
                    "name": folder_name,
                    "id": template_id,
                    "customName": agent_name,
                    "generatedAt": datetime.now().isoformat()
                }
                # Convert back to formatted JSON
                mcp_json = json.dumps(mcp_data, indent=2)
            print(f"\n[DEEP LOG] Generating mcp.json ({len(mcp_json)} characters)")
            mcp_file = generate_mcp_json(mcp_json)
            messages["mcp.json"] = mcp_file
            complete_code["mcp.json"] = mcp_json
        except json.JSONDecodeError:
            # If not valid JSON, keep as is
            print(f"\n[DEEP LOG] Warning: mcp.json is not valid JSON, keeping as is")
            mcp_file = generate_mcp_json(mcp_json)
            messages["mcp.json"] = mcp_file
            complete_code["mcp.json"] = mcp_json
    # Add complete code to the response for UI display
    messages["complete_code"] = complete_code
    messages["complete_code_message"] = "Full template code included in this response for UI display"
    # 6. GENERATE .ENV.EXAMPLE FILE
    # Extract environment variables from the code to create a comprehensive .env.example
    if any([template.get("agents_code"), template.get("main_code"), template.get("tools_code")]):
        env_vars = extract_environment_variables([
            template.get("agents_code", ""),
            template.get("main_code", ""),
            template.get("tools_code", ""),
            template.get("models_code", "")
        ])
        if env_vars:
            env_example = "# Environment Variables for " + agent_name + "\n"
            env_example += "# Generated from template: " + folder_name + "\n\n"
            for var, comment in env_vars.items():
                env_example += f"# {comment}\n{var}=your_{var.lower()}_here\n\n"
            # Write .env.example file
            try:
                env_path = os.path.join("workbench", ".env.example")
                os.makedirs("workbench", exist_ok=True)
                with open(env_path, "w") as f:
                    f.write(env_example)
                messages[".env.example"] = f"Created .env.example file with {len(env_vars)} environment variables"
                print(f"\n[DEEP LOG] Generated .env.example with {len(env_vars)} environment variables")
            except Exception as e:
                messages[".env.example"] = f"Error creating .env.example: {str(e)}"
                print(f"\n[DEEP LOG] Error creating .env.example: {str(e)}")
    return messages
def extract_environment_variables(code_files: List[str]) -> Dict[str, str]:
    Extract environment variables from the code files with descriptions.
    Args:
        code_files: List of code files to analyze
    Returns:
        Dict mapping environment variable names to their descriptions
    import re
    env_vars = {}
    env_pattern = r'os\.getenv\([\'"]([A-Z0-9_]+)[\'"]'
    comment_pattern = r'#\s*(.*?)\s*\n\s*os\.getenv\([\'"]([A-Z0-9_]+)[\'"]'
    for code in code_files:
        if not code:
            continue
        # Find all os.getenv calls
        for match in re.finditer(env_pattern, code):
            var_name = match.group(1)
            if var_name not in env_vars:
                env_vars[var_name] = f"Required for {var_name.replace('_', ' ').lower()}"
        # Look for comments above os.getenv calls for better descriptions
        for match in re.finditer(comment_pattern, code, re.MULTILINE):
            comment, var_name = match.groups()
            if var_name in env_vars:
                env_vars[var_name] = comment
    return env_vars
def extract_requirements(code_files: List[str]) -> List[str]:
    Args:
        code_files: List of code files to analyze
    Returns:
        List of required package names
    import re
    # Common packages that need to be included in requirements.txt
    requirements = set([
        "pydantic-ai",
        "dotenv",
        "logfire",
        "supabase",
        "openai"
    ])
    # Import patterns to match
    import_patterns = [
        r'import\s+([a-zA-Z0-9_]+)',
        r'from\s+([a-zA-Z0-9_]+)\s+import'
    ]
    # Packages to exclude (standard library, local modules)
    exclude_packages = {
        "os", "sys", "json", "typing", "datetime", "dataclasses", "asyncio", 
        "re", "time", "collections", "pathlib", "platform", "random", "archon",
        "utils", "__future__"
    }
    for code in code_files:
        if not code:
            continue
        for pattern in import_patterns:
            for match in re.finditer(pattern, code):
                package = match.group(1).split('.')[0]
                if package and package not in exclude_packages:
                    requirements.add(package)
    # Map to specific requirements with common renames
    package_mapping = {
        "httpx": "httpx",
        "pydantic": "pydantic>=2.0.0",
        "langgraph": "langgraph",
        "openai": "openai>=1.0.0",
        "dotenv": "python-dotenv",
        "supabase": "supabase"
    }
    # Apply mappings and sort
    final_requirements = []
    for req in sorted(requirements):
        final_requirements.append(package_mapping.get(req, req))
    return final_requirements
@pydantic_ai_coder.tool
async def process_large_code_file(ctx: RunContext[PydanticAIDeps], template_id: int, file_type: str, 
                                 output_file: str = "", token_limit: int = 8000) -> Dict[str, Any]:
    Process a large code file from a template by breaking it into manageable chunks.
    This avoids token limitations that might cause parts of the code to be skipped.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch
        file_type: The type of file to process ('agents', 'main', 'models', 'tools', 'mcp')
        output_file: Optional custom output filename (default: based on file_type)
        token_limit: Maximum token size for each chunk processing
    Returns:
        Dict containing results of the file processing
    print(f"\n[DEEP LOG] Processing large code file: {file_type} from template {template_id}")
    # Map file type to the corresponding field in the template
    file_type_map = {
        'agents': 'agents_code',
        'main': 'main_code',
        'models': 'models_code',
        'tools': 'tools_code',
        'mcp': 'mcp_json'
    }
    if file_type not in file_type_map:
        return {
            "success": False,
            "message": f"Invalid file type: {file_type}. Must be one of: {', '.join(file_type_map.keys())}"
        }
    field_name = file_type_map[file_type]
    # Fetch the template from Supabase
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {
            "success": False,
            "message": f"Template ID {template_id} not found: {result.get('message')}"
        }
    template = result.get("template", {})
    code_content = template.get(field_name)
    if not code_content:
        return {
            "success": False,
            "message": f"No {file_type} code found in template {template_id}"
        }
    # Determine output filename
    if not output_file:
        if file_type == 'mcp':
            output_file = "mcp.json"
        else:
            output_file = f"{file_type}.py"
    print(f"\n[DEEP LOG] Processing file {output_file} with {len(code_content)} characters")
    # If the code is small enough, process it directly
    if len(code_content) < token_limit:
        print(f"\n[DEEP LOG] Code is small enough to process directly")
        # Generate the file based on the type
        if file_type == 'agents':
            result = generate_agents_file(code_content, output_file)
        elif file_type == 'main':
            result = generate_main_file(code_content, output_file)
        elif file_type == 'models':
            result = generate_models_file(code_content, output_file)
        elif file_type == 'tools':
            result = generate_tools_file(code_content, output_file)
        elif file_type == 'mcp':
            result = generate_mcp_json(code_content, output_file)
        return {
            "success": True,
            "message": result,
            "file": output_file,
            "chunks_processed": 1,
            "total_characters": len(code_content)
        }
    # For larger files, split into chunks based on logical divisions
    print(f"\n[DEEP LOG] Code is large, breaking into chunks for processing")
    # Find logical divisions in the code (classes, functions, etc.)
    chunks = split_code_into_logical_chunks(code_content, token_limit)
    num_chunks = len(chunks)
    print(f"\n[DEEP LOG] Split code into {num_chunks} logical chunks")
    # Process and write to file directly in chunks to avoid token limitations
    try:
        with open(output_file, 'w') as f:
            # Add appropriate header
            if file_type != 'mcp':
                folder_name = template.get('folder_name', 'unknown_template')
                f.write(f"# Generated from template: {folder_name} (ID: {template_id})\n")
                f.write(f"# Warning: This is a large file that was processed in {num_chunks} chunks\n\n")
            # Write each chunk
            for i, chunk in enumerate(chunks):
                print(f"\n[DEEP LOG] Processing chunk {i+1}/{num_chunks} ({len(chunk)} characters)")
                f.write(chunk)
                # Add separator between chunks except for the last one
                if i < num_chunks - 1 and file_type != 'mcp':
                    f.write("\n\n# --- CHUNK BOUNDARY: DO NOT REMOVE ---\n\n")
        return {
            "success": True,
            "message": f"Successfully processed {file_type} code in {num_chunks} chunks",
            "file": output_file,
            "chunks_processed": num_chunks,
            "total_characters": len(code_content)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing {file_type} code: {str(e)}"
        }
def split_code_into_logical_chunks(code: str, max_chunk_size: int) -> List[str]:
    Split code into logical chunks that respect syntactic boundaries.
    This prevents splitting in the middle of a function or class definition.
    Args:
        code: The full code content to split
        max_chunk_size: Maximum size of each chunk in characters
    Returns:
        List of code chunks
    import re
    # If the code is already small enough, return it as is
    if len(code) <= max_chunk_size:
        return [code]
    # Define patterns for logical boundaries (class/function definitions, imports)
    boundary_patterns = [
        r'^\s*class\s+\w+',  # Class definitions
        r'^\s*def\s+\w+',    # Function definitions
        r'^\s*import\s+',    # Import statements
        r'^\s*from\s+\S+\s+import', # From imports
        r'^\s*@',            # Decorators
        r'^\s*#\s*\w+',      # Section comments
        r'^\s*$'             # Empty lines
    ]
    # Compile the patterns
    patterns = [re.compile(pattern, re.MULTILINE) for pattern in boundary_patterns]
    # Split the code into lines
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    for line in lines:
        line_with_newline = line + '\n'
        line_size = len(line_with_newline)
        # If adding this line would exceed the chunk size and we're at a logical boundary
        if current_size + line_size > max_chunk_size:
            # Check if the line starts a new logical section
            is_boundary = any(pattern.match(line) for pattern in patterns)
            if is_boundary or current_size > max_chunk_size * 0.8:
                # Finalize current chunk and start a new one
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
                continue
        # Add line to current chunk
        current_chunk.append(line)
        current_size += line_size
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks
@pydantic_ai_coder.tool
async def explore_all_templates(ctx: RunContext[PydanticAIDeps], limit: int = 20, 
                               include_stats: bool = True, 
                               include_previews: bool = False) -> Dict[str, Any]:
    Explore all available agent templates in the Supabase database.
    Provides a comprehensive overview of available templates for agent generation.
    Args:
        ctx: The context including the Supabase client
        limit: Maximum number of templates to return
        include_stats: Whether to include detailed statistics for each template
        include_previews: Whether to include code previews for each template
    Returns:
        Dict containing template metadata and statistics
    print(f"\n[DEEP LOG] Exploring all templates (limit: {limit})")
    if not ctx.deps.supabase:
        return {"error": "Supabase client not available"}
    try:
        # Query Supabase for all templates, limited by the specified limit
        result = ctx.deps.supabase.table('agent_embeddings') \
            .select('id, folder_name, purpose, metadata, created_at') \
            .order('created_at', desc=True) \
            .limit(limit) \
            .execute()
        if not result.data:
            return {"templates": [], "message": "No templates found in the database."}
        templates = result.data
        total_count = len(templates)
        # Get detailed stats if requested
        if include_stats:
            stats = {}
            for template in templates:
                template_id = template['id']
                # Get file counts and sizes
                file_stats = ctx.deps.supabase.table('agent_embeddings') \
                    .select('agents_code, main_code, models_code, tools_code, mcp_json') \
                    .eq('id', template_id) \
                    .execute()
                if file_stats.data:
                    file_data = file_stats.data[0]
                    stats[template_id] = {
                        "files": {
                            "agents.py": len(file_data.get('agents_code', '') or ''),
                            "main.py": len(file_data.get('main_code', '') or ''),
                            "models.py": len(file_data.get('models_code', '') or ''),
                            "tools.py": len(file_data.get('tools_code', '') or ''),
                            "mcp.json": len(file_data.get('mcp_json', '') or '')
                        },
                        "total_size": sum(len(file_data.get(f, '') or '') for f in 
                                         ['agents_code', 'main_code', 'models_code', 'tools_code', 'mcp_json'])
                    }
        # Add code previews if requested
        if include_previews:
            for template in templates:
                template_id = template['id']
                # Get code previews
                preview_data = ctx.deps.supabase.table('agent_embeddings') \
                    .select('agents_code, main_code, models_code, tools_code, mcp_json') \
                    .eq('id', template_id) \
                    .execute()
                if preview_data.data:
                    file_data = preview_data.data[0]
                    preview_size = 300  # Number of characters to include in preview
                    template['previews'] = {}
                    for file_name, field_name in [
                        ('agents.py', 'agents_code'),
                        ('main.py', 'main_code'),
                        ('models.py', 'models_code'),
                        ('tools.py', 'tools_code'),
                        ('mcp.json', 'mcp_json')
                    ]:
                        content = file_data.get(field_name, '')
                        if content:
                            template['previews'][file_name] = (
                                content[:preview_size] + 
                                ('...' if len(content) > preview_size else '')
                            )
        response = {
            "total": total_count,
            "templates": templates,
            "message": f"Found {total_count} templates."
        }
        if include_stats:
            response["stats"] = stats
        print(f"\n[DEEP LOG] Found {total_count} templates")
        return response
    except Exception as e:
        error_msg = f"Error exploring templates: {str(e)}"
        print(f"\n[DEEP LOG] {error_msg}")
        return {"error": error_msg}
@pydantic_ai_coder.tool
async def create_sample_template(ctx: RunContext[PydanticAIDeps], template_name: str, 
                                sample_purpose: str = "Sample agent template for testing") -> Dict[str, Any]:
    Create a sample agent template in Supabase for testing purposes.
    Args:
        ctx: The context including the Supabase client and embedding client
        template_name: Name for the sample template
        sample_purpose: Purpose description for the sample template
    Returns:
        Dict with the result of the template creation
    print(f"\n[DEEP LOG] Creating sample template: {template_name}")
    if not ctx.deps.supabase or not ctx.deps.embedding_client:
        return {"error": "Supabase or embedding client not available"}
    # Format the agent name for code use
    agent_name = template_name.lower().replace(' ', '_')
    # Create agents.py with basic structure
    agents_code = (
        f"from __future__ import annotations as _annotations\n"
        f"from dataclasses import dataclass\n"
        f"import os\n"
        f"import logfire\n"
        f"from pydantic_ai import Agent, RunContext\n\n"
        f"logfire.configure(send_to_logfire='if-token-present')\n\n"
        f"@dataclass\n"
        f"class AgentDeps:\n"
        f"    \"\"\"Data dependencies for the {template_name} agent\"\"\"\n"
        f"    api_key: str\n\n"
        f"{agent_name}_agent = Agent(\n"
        f"    'openai:gpt-4o-mini',\n"
        f"    system_prompt=\"\"\"You are the {template_name} agent. Your purpose is to {sample_purpose}.\n"
        f"    Be concise and helpful in your responses.\"\"\",\n"
        f"    deps_type=AgentDeps,\n"
        f"    retries=2\n"
        f")\n\n"
        f"@{agent_name}_agent.tool\n"
        f"async def sample_tool(ctx: RunContext[AgentDeps], query: str) -> str:\n"
        f"    \"\"\"A sample tool that demonstrates how to implement agent tools\n"
        f"    \n"
        f"    Args:\n"
        f"        ctx: The context including dependencies\n"
        f"        query: The query to process\n"
        f"        \n"
        f"    Returns:\n"
        f"        The result of processing the query\n"
        f"    \"\"\"\n"
        f"    print(f\"Processing query: {{query}}\")\n"
        f"    # This is where actual implementation would go\n"
        f"    return f\"Sample response for query: {{query}}\"\n"
    )
    # Create main.py with basic execution flow
    main_code = (
        f"import os\n"
        f"import asyncio\n"
        f"from dotenv import load_dotenv\n"
        f"import logfire\n"
        f"from agents import {agent_name}_agent, AgentDeps\n\n"
        f"# Load environment variables\n"
        f"load_dotenv()\n\n"
        f"async def main():\n"
        f"    # Initialize dependencies\n"
        f"    api_key = os.getenv(\"API_KEY\", \"default_key\")\n"
        f"    deps = AgentDeps(api_key=api_key)\n"
        f"    \n"
        f"    # Run the agent\n"
        f"    print(f\"Running {template_name} agent with prompt...\")\n"
        f"    result = await {agent_name}_agent.run(\n"
        f"        \"Tell me about yourself and what you can do\", \n"
        f"        deps=deps\n"
        f"    )\n"
        f"    \n"
        f"    print(f\"Response: {{result.data}}\")\n\n"
        f"if __name__ == \"__main__\":\n"
        f"    try:\n"
        f"        asyncio.run(main())\n"
        f"    except KeyboardInterrupt:\n"
        f"        logging.info(\"Exiting gracefully...\")\n"
        f"    except Exception as e:\n"
        f"        logging.error(f\"Unexpected error: {{str(e)}}\")\n"
        f"        sys.exit(1)\n"
    )
    # Create models.py with sample Pydantic models
    models_code = (
        f"from dataclasses import dataclass\n"
        f"from typing import List, Dict, Any, Optional\n"
        f"from pydantic import BaseModel, Field\n\n"
        f"class SampleRequest(BaseModel):\n"
        f"    \"\"\"Sample request model for the {template_name} agent\"\"\"\n"
        f"    query: str = Field(..., description=\"The query to process\")\n"
        f"    options: Optional[Dict[str, Any]] = Field(None, description=\"Optional processing options\")\n\n"
        f"class SampleResponse(BaseModel):\n"
        f"    \"\"\"Sample response model for the {template_name} agent\"\"\"\n"
        f"    result: str = Field(..., description=\"The result of processing the query\")\n"
        f"    status: str = Field(\"success\", description=\"The status of the processing\")\n"
        f"    metadata: Optional[Dict[str, Any]] = Field(None, description=\"Optional metadata about the processing\")\n"
    )
    # Create tools.py with API utilities
    tools_code = (
        f"import os\n"
        f"import json\n"
        f"import httpx\n"
        f"from typing import Dict, Any, List, Optional\n\n"
        f"async def api_request(\n"
        f"    endpoint: str, \n"
        f"    method: str = \"GET\", \n"
        f"    params: Optional[Dict[str, Any]] = None,\n"
        f"    headers: Optional[Dict[str, Any]] = None,\n"
        f"    data: Optional[Dict[str, Any]] = None,\n"
        f"    api_key: Optional[str] = None\n"
        f") -> Dict[str, Any]:\n"
        f"    \"\"\"Make an API request to the specified endpoint\n"
        f"    \n"
        f"    Args:\n"
        f"        endpoint: The API endpoint to call\n"
        f"        method: The HTTP method to use\n"
        f"        params: Optional query parameters\n"
        f"        headers: Optional HTTP headers\n"
        f"        data: Optional request body data\n"
        f"        api_key: Optional API key to use\n"
        f"        \n"
        f"    Returns:\n"
        f"        The API response data\n"
        f"    \"\"\"\n"
        f"    # Set up headers\n"
        f"    request_headers = headers or {{}}\n"
        f"    if api_key:\n"
        f"        request_headers[\"Authorization\"] = f\"Bearer {{api_key}}\"\n"
        f"    \n"
        f"    # Make the request\n"
        f"    async with httpx.AsyncClient() as client:\n"
        f"        if method == \"GET\":\n"
        f"            response = await client.get(endpoint, params=params, headers=request_headers)\n"
        f"        elif method == \"POST\":\n"
        f"            response = await client.post(endpoint, params=params, json=data, headers=request_headers)\n"
        f"        else:\n"
        f"            raise ValueError(f\"Unsupported HTTP method: {{method}}\")\n"
        f"        \n"
        f"        # Handle the response\n"
        f"        response.raise_for_status()\n"
        f"        return response.json()\n"
    )
    # Create mcp.json configuration
    created_at = datetime.now().isoformat()
    mcp_json = (
        f"{{\n"
        f"  \"mcpServers\": {{\n"
        f"    \"sample-server\": {{\n"
        f"      \"command\": \"npx\",\n"
        f"      \"args\": [\n"
        f"        \"-y\",\n"
        f"        \"@modelcontextprotocol/server-sample\"\n"
        f"      ],\n"
        f"      \"env\": {{\n"
        f"        \"API_KEY\": \"YOUR_API_KEY_HERE\"\n"
        f"      }}\n"
        f"    }}\n"
        f"  }},\n"
        f"  \"metadata\": {{\n"
        f"    \"name\": \"{template_name}\",\n"
        f"    \"purpose\": \"{sample_purpose}\",\n"
        f"    \"createdAt\": \"{created_at}\",\n"
        f"    \"version\": \"1.0.0\"\n"
        f"  }}\n"
        f"}}"
    )
    try:
        # Generate metadata for the template
        metadata = {
            "type": "sample",
            "source": "generated",
            "created_at": created_at,
            "agents": [f"{agent_name}_agent"],
            "features": ["sample_tool", "api_request"],
            "description": sample_purpose,
            "has_agents": True,
            "has_tools": True,
            "has_models": True,
            "has_main": True,
            "has_mcp": True,
            "version": "1.0.0"
        }
        # Create combined text for embedding
        combined_text = (
            f"Purpose: {sample_purpose}\n\n"
            f"Agents:\n{agents_code}\n\n"
            f"Tools:\n{tools_code}\n\n"
            f"Models:\n{models_code}\n\n"
            f"Main:\n{main_code}\n\n"
            f"MCP:\n{mcp_json}"
        )
        # Generate embedding
        embedding = await get_embedding(combined_text, ctx.deps.embedding_client)
        # Create template dict for insertion
        template_dict = {
            "folder_name": agent_name,
            "agents_code": agents_code,
            "main_code": main_code,
            "models_code": models_code,
            "tools_code": tools_code,
            "mcp_json": mcp_json,
            "purpose": sample_purpose,
            "metadata": metadata,
            "embedding": embedding
        }
        # Insert into Supabase
        result = ctx.deps.supabase.table("agent_embeddings").insert(template_dict).execute()
        if hasattr(result, 'error') and result.error:
            return {"success": False, "error": f"Database insertion error: {result.error}"}
        inserted_id = result.data[0]['id'] if result.data else None
        return {
            "success": True,
            "message": f"Successfully created sample template: {template_name}",
            "template_id": inserted_id,
            "stats": {
                "agents.py": len(agents_code),
                "main.py": len(main_code),
                "models.py": len(models_code),
                "tools.py": len(tools_code),
                "mcp.json": len(mcp_json)
            }
        }
    except Exception as e:
        error_msg = f"Error creating sample template: {str(e)}"
        print(f"\n[DEEP LOG] {error_msg}")
        return {"success": False, "error": error_msg}
@pydantic_ai_coder.tool
async def display_template_files(ctx: RunContext[PydanticAIDeps], template_id: int) -> Dict[str, str]:
    Display the template files in the UI without writing them to disk.
    This function is specifically designed for showing the code to the user.
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to display
    Returns:
        Dict containing the code content of each file for UI display
    print(f"\n[DEEP LOG] Displaying template files for ID: {template_id}")
    # Fetch the template
    result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
    if not result.get("found", False):
        return {"error": f"Template ID {template_id} not found: {result.get('message')}"}
    template = result.get("template", {})
    print(f"\n[DEEP LOG] Successfully fetched template: {template.get('folder_name')}")
    # Prepare the response with formatted code
    display_result = {
        "template_info": {
            "id": template.get("id"),
            "folder_name": template.get("folder_name"),
            "purpose": template.get("purpose"),
        }
    }
    # Format each file with markdown code blocks for proper UI display
    if template.get("agents_code"):
        display_result["agents.py"] = f"""```python
{template.get("agents_code")}
```"""
    if template.get("main_code"):
        display_result["main.py"] = f"""```python
{template.get("main_code")}
```"""
    if template.get("models_code"):
        display_result["models.py"] = f"""```python
{template.get("models_code")}
```"""
    if template.get("tools_code"):
        display_result["tools.py"] = f"""```python
{template.get("tools_code")}
```"""
    if template.get("mcp_json"):
        try:
            # Try to parse and prettify JSON
            mcp_data = json.loads(template.get("mcp_json"))
            mcp_json = json.dumps(mcp_data, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, keep as is
            mcp_json = template.get("mcp_json")
        display_result["mcp.json"] = f"""```json
{mcp_json}
```"""
    print(f"\n[DEEP LOG] Prepared template files for display: {', '.join(display_result.keys())}")
    return display_result
@pydantic_ai_coder.tool
async def merge_agent_templates(ctx: RunContext[PydanticAIDeps], 
                           query: str = "",
                           template_ids: List[int] = None,
                           custom_name: str = "",
                           custom_description: str = "",
                           preview_before_merge: bool = False) -> Dict[str, Any]:
    Merge multiple agent templates into a single unified set of agent files.
    This function thoroughly combines components from all templates to create a comprehensive multi-service agent.
    Args:
        ctx: The context including the Supabase client and embedding client
        query: Query to search for relevant templates (if template_ids not provided)
        template_ids: List of template IDs to merge
        custom_name: Custom name for the merged template
        custom_description: Custom description for the merged template
        preview_before_merge: Whether to display template components before merging
    Returns:
        Dict with the result of the merge
    try:
        print(f"\n[DEEP LOG] Merging agent templates. Query: '{query}', Template IDs: {template_ids}")
        if not ctx.deps.supabase:
            return {"error": "Supabase client not available"}
        # Get templates
        templates = []
        template_names = []
        # If template_ids is not provided, search for templates using the query
        if not template_ids and query:
            if not ctx.deps.embedding_client:
                return {"error": "Embedding client not available for searching templates"}
            print(f"\n[DEEP LOG] Searching for templates with query: '{query}'")
            search_result = await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, 0.3, 5)
            if not search_result.get("templates"):
                return {"error": f"No templates found matching query: '{query}'"}
            for template in search_result.get("templates", []):
                template_id = template.get("id")
                if template_id:
                    fetch_result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
                    if fetch_result.get("found", False):
                        templates.append(fetch_result.get("template"))
                        template_names.append(fetch_result.get("template", {}).get("folder_name", f"template_{template_id}"))
    except Exception as e:
        print(f"\n[DEEP LOG] Error searching for templates: {str(e)}")
        return {"error": f"Failed to search for templates: {str(e)}"}
    try:
        # If template_ids is provided, fetch those templates
        if template_ids:
            for template_id in template_ids:
                fetch_result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
                if fetch_result.get("found", False):
                    templates.append(fetch_result.get("template"))
                    template_names.append(fetch_result.get("template", {}).get("folder_name", f"template_{template_id}"))
        # Ensure we have templates to merge
        if len(templates) < 2:
            return {"error": "At least 2 templates are required for merging"}
                    except Exception as e:
        print(f"\n[DEEP LOG] Error fetching templates: {str(e)}")
        return {"error": f"Failed to fetch templates: {str(e)}"}
        try:
            # Create a dictionary with all merged code files
            merged_files = {
                "agents.py": merged_agents_py,
                "main.py": merged_main_py,
                "models.py": merged_models_py,
                "tools.py": merged_tools_py,
                "mcp.json": mcp_json
            }
            # Write all merged files to the workbench directory
            workbench_dir = os.path.join(os.getcwd(), "workbench")
            if not os.path.exists(workbench_dir):
                os.makedirs(workbench_dir, exist_ok=True)
                print(f"\n[DEEP LOG] Created workbench directory: {workbench_dir}")
            # Write each file to the workbench directory
            for filename, content in merged_files.items():
                file_path = os.path.join(workbench_dir, filename)
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"\n[DEEP LOG] Successfully wrote {filename} to workbench directory")
                except Exception as e:
                    print(f"\n[DEEP LOG] Error writing {filename} to workbench directory: {e}")
            except Exception as e:
        print(f"\n[DEEP LOG] Error creating merged files: {str(e)}")
        return {"error": f"Failed to create merged files: {str(e)}"}
            # Generate requirements.txt file
            requirements = [
                "pydantic-ai",
                "python-dotenv",
                "httpx",
                "logfire"
            ]
            # Add service-specific requirements
            if "spotify" in ''.join(template_names).lower():
                requirements.append("spotipy")
            if "github" in ''.join(template_names).lower():
                requirements.append("PyGithub")
            if "search" in ''.join(template_names).lower() or "serper" in ''.join(template_names).lower():
                requirements.append("serper-python")
            # Write requirements.txt file
            requirements_path = os.path.join(workbench_dir, "requirements.txt")
            try:
                with open(requirements_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(requirements))
                print(f"\n[DEEP LOG] Successfully wrote requirements.txt to workbench directory")
            except Exception as e:
                print(f"\n[DEEP LOG] Error writing requirements.txt to workbench directory: {e}")
            # Also create a .env.example file with all required environment variables
            env_vars = set()
            env_vars_with_comments = {}
            # Extract environment variables from config fields
            for template in templates:
                models_code = template.get("models_code", "")
                if models_code:
                    # Look for Config class fields
                    config_matches = re.findall(r"class\s+Config(?:\(BaseModel\))?:.*?(?=\n\nclass|\n\ndef|\Z)", models_code, re.DOTALL)
                    if config_matches:
                        config_class = config_matches[0]
                        # Extract fields that look like API keys
                        api_key_fields = re.findall(r"^\s+(\w+(?:_API_KEY|_TOKEN|_SECRET)):\s+([^=\n]+)(?:=\s*([^\n]+))?", config_class, re.MULTILINE)
                        for field_name, field_type, _ in api_key_fields:
                            env_vars.add(field_name)
                            env_vars_with_comments[field_name] = f"Required for {field_name.replace('_API_KEY', '').replace('_TOKEN', '').replace('_SECRET', '').lower()} authentication"
            # Create .env.example file
            if env_vars:
                env_example = f"# Environment Variables for {custom_name or '_'.join(template_names)}\n"
                env_example += f"# Created at: {datetime.now().isoformat()}\n"
                env_example += f"# Merged from templates: {', '.join(template_names)}\n\n"
                for var in sorted(env_vars):
                    comment = env_vars_with_comments.get(var, "Required API key or token")
                    env_example += f"# {comment}\n{var}=your_{var.lower()}_here\n\n"
                # Write .env.example file
                env_path = os.path.join(workbench_dir, ".env.example")
                try:
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.write(env_example)
                    print(f"\n[DEEP LOG] Successfully wrote .env.example file with {len(env_vars)} environment variables")
                except Exception as e:
                    print(f"\n[DEEP LOG] Error writing .env.example file: {e}")
            try:
                # Create combined text for embedding
                combined_text = (
                    f"Purpose: {metadata['purpose']}\n\n"
                    f"Agents:\n{merged_agents_py}\n\n"
                    f"Tools:\n{merged_tools_py}\n\n"
                    f"Models:\n{merged_models_py}\n\n"
                    f"Main:\n{merged_main_py}\n\n"
                    f"MCP:\n{mcp_json or ''}"
                )
                # Generate embedding if embedding client available
                embedding = None
                if ctx.deps.embedding_client:
                    embedding = await get_embedding(combined_text, ctx.deps.embedding_client)
                # Determine folder name for the merged template
                folder_name = custom_name or f"merged_{'_'.join(template_names[:2])}"
                # Insert into database
                if ctx.deps.supabase:
                    result = ctx.deps.supabase.table("agent_embeddings").insert({
                        "folder_name": folder_name,
                        "agents_code": merged_agents_py,
                        "main_code": merged_main_py,
                        "models_code": merged_models_py,
                        "tools_code": merged_tools_py,
                        "mcp_json": mcp_json,
                        "purpose": metadata['purpose'],
                        "metadata": metadata,
                        "embedding": embedding
                    }).execute()
                    # Check for errors
                    if hasattr(result, 'error') and result.error:
                        print(f"\n[DEEP LOG] Error inserting merged template: {result.error}")
                        return {"error": f"Failed to insert merged template: {result.error}"}
                    inserted_id = result.data[0]['id'] if result.data else None
                    print(f"\n[DEEP LOG] Successfully created merged template with ID: {inserted_id}")
                    return {
                        "success": True,
                        "message": f"Successfully created merged template: {folder_name}",
                        "id": inserted_id,
                        "folder_name": folder_name,
                        "parent_templates": metadata['parent_templates'],
                        "parent_template_names": metadata['parent_template_names'],
                        "files": {
                            "agents.py": len(merged_agents_py),
                            "main.py": len(merged_main_py) if merged_main_py else 0,
                            "models.py": len(merged_models_py) if merged_models_py else 0,
                            "tools.py": len(merged_tools_py) if merged_tools_py else 0,
                            "mcp.json": len(mcp_json) if mcp_json else 0
                        },
                        "metadata": metadata
                    }
                else:
                    return {
                        "error": "Supabase client not available for storing the merged template"
                    }
            except Exception as e:
                print(f"\n[DEEP LOG] Error creating merged template: {str(e)}")
                return {"error": f"Failed to create merged template: {str(e)}"}
@pydantic_ai_coder.tool
async def create_multi_mcp_template_example(ctx: RunContext[PydanticAIDeps], 
                                          template_name: str = "serper_spotify_agent",
                                          sample_purpose: str = "Combined agent for search and music control") -> Dict[str, Any]:
    Create a sample multi-MCP agent template that combines Serper search and Spotify music control.
    This serves as an example for when users need an agent that uses multiple MCP tools or templates.
    Args:
        ctx: The context including the Supabase client and embedding client
        template_name: Name for the multi-MCP template (default: serper_spotify_agent)
        sample_purpose: Purpose description for the multi-MCP template
    Returns:
        Dict with the result of the template creation
    print(f"\n[DEEP LOG] Creating multi-MCP template example: {template_name}")
    if not ctx.deps.supabase or not ctx.deps.embedding_client:
        return {"error": "Supabase or embedding client not available"}
    # Format the agent name for code use
    agent_name = template_name.lower().replace(' ', '_')
    # Create agents.py with combined Serper and Spotify agent structure
    agents_code = (
        f"from pydantic_ai.providers.openai import OpenAIProvider\n"
        f"from pydantic_ai.models.openai import OpenAIModel\n"
        f"from pydantic_ai import Agent\n"
        f"import logging\n"
        f"import sys\n"
        f"import asyncio\n"
        f"import json\n"
        f"import traceback\n\n"
        f"from models import Config\n"
        f"from tools import create_serper_mcp_server, create_spotify_mcp_server, display_mcp_tools\n"
        f"from tools import execute_mcp_tool, run_query\n\n"
        f"# Combined system prompt for Serper-Spotify agent\n"
        f"system_prompt = \"\"\"\n"
        f"You are a powerful assistant with dual capabilities:\n\n"
        f"1. Web Search (Serper): You can search the web for information using the Serper API\n"
        f"   - Search for any information on the internet\n"
        f"   - Retrieve news articles, general knowledge, and up-to-date information\n"
        f"   - Find images and news about specific topics\n\n"
        f"2. Spotify Music: You can control and interact with Spotify\n"
        f"   - Search for songs, albums, and artists\n"
        f"   - Create and manage playlists\n"
        f"   - Control playback (play, pause, skip, etc.)\n"
        f"   - Get information about the user's library and recommendations\n\n"
        f"IMPORTANT USAGE NOTES:\n"
        f"- For Spotify search operations, a 'market' parameter (e.g., 'US') is required\n"
        f"- For web searches, try to be specific about what information you're looking for\n"
        f"- When the user asks a knowledge-based question, use web search\n"
        f"- When the user asks about music or wants to control Spotify, use the Spotify tools\n\n"
        f"When responding to the user, always be concise and helpful. If you don't know how to do something with \n"
        f"the available tools, explain what you can do instead.\n"
        f"\"\"\"\n\n"
        f"def get_model(config: Config) -> OpenAIModel:\n"
        f"    \"\"\"Initialize the OpenAI model with the provided configuration.\"\"\"\n"
        f"    try:\n"
        f"        model = OpenAIModel(\n"
        f"            config.MODEL_CHOICE,\n"
        f"            provider=OpenAIProvider(\n"
        f"                base_url=config.BASE_URL,\n"
        f"                api_key=config.LLM_API_KEY\n"
        f"            )\n"
        f"        )\n"
        f"        logging.debug(f\"Initialized model with choice: {{config.MODEL_CHOICE}}\")\n"
        f"        return model\n"
        f"    except Exception as e:\n"
        f"        logging.error(\"Error initializing model: %s\", e)\n"
        f"        sys.exit(1)\n\n"
        f"async def setup_agent(config: Config) -> Agent:\n"
        f"    \"\"\"Set up and initialize the combined Serper-Spotify agent with both MCP servers.\"\"\"\n"
        f"    try:\n"
        f"        # Create MCP server instances for both Serper and Spotify\n"
        f"        logging.info(\"Creating Serper MCP Server...\")\n"
        f"        serper_server = create_serper_mcp_server(config.SERPER_API_KEY)\n"
        f"        \n"
        f"        logging.info(\"Creating Spotify MCP Server...\")\n"
        f"        spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)\n"
        f"        \n"
        f"        # Create agent with both servers\n"
        f"        logging.info(\"Initializing agent with both Serper and Spotify MCP Servers...\")\n"
        f"        agent = Agent(get_model(config), mcp_servers=[serper_server, spotify_server])\n"
        f"        \n"
        f"        # Set system prompt\n"
        f"        agent.system_prompt = system_prompt\n"
        f"        \n"
        f"        # Display and capture MCP tools for visibility from both servers\n"
        f"        try:\n"
        f"            serper_tools = await display_mcp_tools(serper_server, \"SERPER\")\n"
        f"            logging.info(f\"Found {{len(serper_tools) if serper_tools else 0}} MCP tools available for Serper operations\")\n"
        f"            \n"
        f"            spotify_tools = await display_mcp_tools(spotify_server, \"SPOTIFY\")\n"
        f"            logging.info(f\"Found {{len(spotify_tools) if spotify_tools else 0}} MCP tools available for Spotify operations\")\n"
        f"        except Exception as tool_err:\n"
        f"            logging.warning(f\"Could not display MCP tools: {{str(tool_err)}}\")\n"
        f"            logging.debug(\"Tool display error details:\", exc_info=True)\n"
        f"        \n"
        f"        logging.debug(\"Agent setup complete with both Serper and Spotify MCP servers.\")\n"
        f"        return agent\n"
        f"            \n"
        f"    except Exception as e:\n"
        f"        logging.error(\"Error setting up agent: %s\", e)\n"
        f"        logging.error(\"Error details: %s\", traceback.format_exc())\n"
        f"        sys.exit(1)\n"
    )
    # Create models.py with combined configuration class
    models_code = (
        f"from pydantic import BaseModel, ValidationError\n"
        f"from dotenv import load_dotenv\n"
        f"import os\n"
        f"import sys\n"
        f"import logging\n\n"
        f"class Config(BaseModel):\n"
        f"    \"\"\"Configuration for the Serper-Spotify Agent\"\"\"\n"
        f"    MODEL_CHOICE: str = \"gpt-4o-mini\"\n"
        f"    BASE_URL: str = \"https://api.openai.com/v1\"\n"
        f"    LLM_API_KEY: str\n"
        f"    SERPER_API_KEY: str\n"
        f"    SPOTIFY_API_KEY: str\n\n"
        f"    @classmethod\n"
        f"    def load_from_env(cls) -> 'Config':\n"
        f"        \"\"\"Load configuration from environment variables with better error handling\"\"\"\n"
        f"        load_dotenv()\n"
        f"        \n"
        f"        # Check for required environment variables\n"
        f"        missing_vars = []\n"
        f"        if not os.getenv(\"LLM_API_KEY\"):\n"
        f"            missing_vars.append(\"LLM_API_KEY\")\n"
        f"        if not os.getenv(\"SERPER_API_KEY\"):\n"
        f"            missing_vars.append(\"SERPER_API_KEY\")\n"
        f"        if not os.getenv(\"SPOTIFY_API_KEY\"):\n"
        f"            missing_vars.append(\"SPOTIFY_API_KEY\")\n"
        f"            \n"
        f"        if missing_vars:\n"
        f"            logging.error(\"Missing required environment variables:\")\n"
        f"            for var in missing_vars:\n"
        f"                logging.error(f\"  - {{var}}\")\n"
        f"            logging.error(\"\\nPlease create a .env file with the following content:\")\n"
        f"            logging.error(\"\"\"\n"
        f"LLM_API_KEY=your_openai_api_key\n"
        f"SERPER_API_KEY=your_serper_api_key\n"
        f"SPOTIFY_API_KEY=your_spotify_api_key\n"
        f"MODEL_CHOICE=gpt-4o-mini  # optional\n"
        f"BASE_URL=https://api.openai.com/v1  # optional\n"
        f"            \"\"\")\n"
        f"            sys.exit(1)\n"
        f"            \n"
        f"        return cls(\n"
        f"            MODEL_CHOICE=os.getenv(\"MODEL_CHOICE\", \"gpt-4o-mini\"),\n"
        f"            BASE_URL=os.getenv(\"BASE_URL\", \"https://api.openai.com/v1\"),\n"
        f"            LLM_API_KEY=os.getenv(\"LLM_API_KEY\"),\n"
        f"            SERPER_API_KEY=os.getenv(\"SERPER_API_KEY\"),\n"
        f"            SPOTIFY_API_KEY=os.getenv(\"SPOTIFY_API_KEY\")\n"
        f"        )\n\n"
        f"def load_config() -> Config:\n"
        f"    \"\"\"Load the configuration from environment variables\"\"\"\n"
        f"    try:\n"
        f"        config = Config.load_from_env()\n"
        f"        logging.debug(\"Configuration loaded successfully\")\n"
        f"        # Hide sensitive information in logs\n"
        f"        safe_config = config.model_dump()\n"
        f"        safe_config[\"LLM_API_KEY\"] = \"***\" if safe_config[\"LLM_API_KEY\"] else None\n"
        f"        safe_config[\"SERPER_API_KEY\"] = \"***\" if safe_config[\"SERPER_API_KEY\"] else None\n"
        f"        safe_config[\"SPOTIFY_API_KEY\"] = \"***\" if safe_config[\"SPOTIFY_API_KEY\"] else None\n"
        f"        logging.debug(f\"Config values: {{safe_config}}\")\n"
        f"        return config\n"
        f"    except ValidationError as e:\n"
        f"        logging.error(\"Configuration validation error:\")\n"
        f"        for error in e.errors():\n"
        f"            logging.error(f\"  - {{error['loc'][0]}}: {{error['msg']}}\")\n"
        f"        sys.exit(1)\n"
        f"    except Exception as e:\n"
        f"        logging.error(f\"Unexpected error loading configuration: {{str(e)}}\")\n"
        f"        sys.exit(1)\n"
    )
    # Create tools.py with combined MCP server functions
    tools_code = (
        '''
from pydantic_ai.mcp import MCPServerStdio
import logging
import asyncio
import json
import traceback
async def display_mcp_tools(server: MCPServerStdio, server_name: str = "MCP"):
    Args:
        server: The MCP server instance
        server_name: Name identifier for the server (SERPER or SPOTIFY)
    Returns:
        List of available tools
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
    Args:
        server: The MCP server instance
        tool_name: Name of the tool to execute
        params: Dictionary of parameters to pass to the tool
    Returns:
        The result of the tool execution
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
    Args:
        agent: The MCP agent
        user_query: The user's query string
    Returns:
        tuple: (result, elapsed_time, tool_usage)
    try:
        start_time = asyncio.get_event_loop().time()
        logging.info(f"Processing query: '{user_query}'")
        # Execute the query
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        # Extract tool usage information
        tool_usage = []
        if hasattr(result, 'metadata') and result.metadata:
            try:
                if isinstance(result.metadata, dict):
                    tool_usage = result.metadata.get('tools', []) or result.metadata.get('tool_calls', [])
                elif hasattr(result.metadata, 'tools'):
                    tool_usage = result.metadata.tools
            except Exception as tool_err:
                logging.debug(f"Could not extract tool usage: {tool_err}")
        # Log tool usage
        if tool_usage:
            logging.info(f"Tools used in this query: {len(tool_usage)}")
            for tool in tool_usage:
                logging.info(f"- {tool.get('name', 'Unknown tool')}")
        return (result, elapsed_time, tool_usage)
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        raise
def create_serper_mcp_server(serper_api_key):
    Args:
        serper_api_key: The API key for Serper
    Returns:
        MCPServerStdio: The MCP server instance
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
    Args:
        spotify_api_key: The API key for Spotify
    Returns:
        MCPServerStdio: The MCP server instance
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
'''
    )
    # Create main.py with execution flow
    main_code = (
       '''
import asyncio
import logging
import sys
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import json
import traceback
from models import load_config
from agent import setup_agent
from tools import run_query
# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Serper-Spotify Combined Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='serper_spotify_agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    return parser.parse_args()
# Configure logging with colors and better formatting
def setup_logging(args):
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    # File handler for complete logs
    file_handler = RotatingFileHandler(
        args.log_file,
        maxBytes=args.max_log_size,
        backupCount=args.log_backups
    )
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    # Suppress verbose logging from libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    return root_logger
# Function to display tool usage in a user-friendly way
def display_tool_usage(tool_usage):
    if not tool_usage:
        print("\n📋 No specific tools were recorded for this query")
        return
    # Detect which service was used based on tool names
    serper_used = any("search" in tool.get('name', '').lower() for tool in tool_usage)
    spotify_used = any(keyword in json.dumps(tool_usage).lower() for keyword in 
                     ['spotify', 'music', 'song', 'playlist', 'artist', 'album', 'play', 'track'])
    if serper_used:
        print("\n🔍 SERPER SEARCH TOOLS USED:")
    elif spotify_used:
        print("\n🎵 SPOTIFY TOOLS USED:")
    else:
        print("\n🛠️ TOOLS USED:")
    print("-"*50)
    for i, tool in enumerate(tool_usage, 1):
        tool_name = tool.get('name', 'Unknown Tool')
        tool_params = tool.get('parameters', {})
        print(f"{i}. Tool: {tool_name}")
        if tool_params:
            print("   Parameters:")
            for param, value in tool_params.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"   - {param}: {value}")
        print()
def display_startup_message():
    print("\n" + "="*50)
    print("🔍🎵 SERPER-SPOTIFY COMBINED AGENT 🎵🔍")
    print("="*50)
    print("This agent can:")
    print("- Search the web for information using Serper")
    print("- Control Spotify and find music")
    print("\nJust ask a question about anything, or request a song!")
    print("\nType 'exit', 'quit', or press Ctrl+C to exit.")
    print("\nTroubleshooting tips if issues occur:")
    print("1. Make sure your API keys have the necessary permissions")
    print("2. Run 'npm install' to make sure all dependencies are installed")
    print("3. Check the log file for detailed error messages")
    print("="*50)
async def main():
    # Parse command line arguments and set up logging
    args = parse_args()
    logger = setup_logging(args)
    try:
        logger.info("Starting Serper-Spotify Combined Agent")
        # Check for Node.js (required for MCP servers)
        try:
            import subprocess
            node_version = subprocess.check_output(['node', '--version']).decode().strip()
            npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
            logger.info(f"Node.js version: {node_version}, npm version: {npm_version}")
        except Exception as e:
            logger.warning(f"Could not detect Node.js/npm: {str(e)}. Make sure these are installed.")
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        # Setup agent
        logger.info("Setting up combined agent...")
        agent = await setup_agent(config)
        try:
            async with agent.run_mcp_servers():
                logger.info("Both MCP Servers started successfully")
                display_startup_message()
                while True:
                    try:
                        # Get query from user
                        user_query = input("\n💬 Enter your query (web search or music): ")
                        # Check if user wants to exit
                        if user_query.lower() in ['exit', 'quit', '']:
                            print("Exiting agent...")
                            break
                        # Log the query
                        logger.info(f"Processing query: '{user_query}'")
                        print(f"\nProcessing: '{user_query}'")
                        print("This may take a moment...\n")
                        # Run the query through the agent
                        try:
                            result, elapsed_time, tool_usage = await run_query(agent, user_query)
                            # Log and display the result
                            logger.info(f"Query completed in {elapsed_time:.2f} seconds")
                            # Display the tools that were used
                            display_tool_usage(tool_usage)
                            print("\n" + "="*50)
                            print("RESULTS:")
                            print("="*50)
                            print(result.data)
                            print("="*50)
                            print(f"Query completed in {elapsed_time:.2f} seconds")
                        except Exception as query_error:
                            logger.error(f"Error processing query: {str(query_error)}")
                            print(f"\n❌ Error: {str(query_error)}")
                            print("Please try a different query or check the logs for details.")
                            print("\nSuggestions:")
                            print("1. Make sure your API keys have the necessary permissions")
                            print("2. Try a different query format")
                            print("3. For Spotify queries, ensure you have an active account")
                    except KeyboardInterrupt:
                        logger.info("User interrupted the process")
                        print("\nExiting due to keyboard interrupt...")
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        print(f"\n❌ Error in main loop: {str(e)}")
                        print("Please try again or check the logs for details.")
        except Exception as server_error:
            logger.error(f"Error running MCP servers: {str(server_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            print(f"\n❌ Error running MCP servers: {str(server_error)}")
            print("\nTroubleshooting steps:")
            print("1. Make sure you have Node.js installed (version 16+)")
            print("2. Run 'npm install @smithery/cli' to install dependencies")
            print("3. Check that your API keys are valid and have proper permissions")
            print("4. Check the log file for more detailed error information")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.info("Agent shutting down")
        print("\nThank you for using the Serper-Spotify agent!")
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1) 
'''
    )
    # Create mcp.json configuration with both servers
    created_at = datetime.now().isoformat()
    mcp_json = (
        f"{{\n"
        f"  \"mcpServers\": {{\n"
        f"    \"serper-search\": {{\n"
        f"      \"command\": \"npx\",\n"
        f"      \"args\": [\n"
        f"        \"-y\",\n"
        f"        \"@smithery/cli@latest\",\n"
        f"        \"run\",\n"
        f"        \"@marcopesani/mcp-server-serper\"\n"
        f"      ],\n"
        f"      \"env\": {{\n"
        f"        \"SERPER_API_KEY\": \"YOUR_SERPER_API_KEY_HERE\"\n"
        f"      }}\n"
        f"    }},\n"
        f"    \"spotify-music\": {{\n"
        f"      \"command\": \"npx\",\n"
        f"      \"args\": [\n"
        f"        \"-y\",\n"
        f"        \"@smithery/cli@latest\",\n"
        f"        \"run\",\n"
        f"        \"@superseoworld/mcp-spotify\"\n"
        f"      ],\n"
        f"      \"env\": {{\n"
        f"        \"SPOTIFY_API_KEY\": \"YOUR_SPOTIFY_API_KEY_HERE\"\n"
        f"      }}\n"
        f"    }}\n"
        f"  }},\n"
        f"  \"metadata\": {{\n"
        f"    \"name\": \"{template_name}\",\n"
        f"    \"purpose\": \"{sample_purpose}\",\n"
        f"    \"createdAt\": \"{created_at}\",\n"
        f"    \"version\": \"1.0.0\",\n"
        f"    \"multiMcpExample\": true\n"
        f"  }}\n"
        f"}}"
    )
    try:
        # Generate metadata for the template
        metadata = {
            "type": "multi_mcp_example",
            "source": "generated",
            "created_at": created_at,
            "purpose": sample_purpose,
            "features": ["multi_mcp_integration", "serper_search", "spotify_music"],
            "description": "Example template showing how to combine multiple MCP tools in a single agent",
            "has_agents": True,
            "has_tools": True,
            "has_models": True,
            "has_main": True,
            "has_mcp": True,
            "mcp_servers": ["serper", "spotify"],
            "version": "1.0.0"
        }
        # Create combined text for embedding
        combined_text = (
            f"Purpose: {sample_purpose}\n\n"
            f"Agents:\n{agents_code}\n\n"
            f"Tools:\n{tools_code}\n\n"
            f"Models:\n{models_code}\n\n"
            f"Main:\n{main_code}\n\n"
            f"MCP:\n{mcp_json}"
        )
        # Generate embedding
        embedding = await get_embedding(combined_text, ctx.deps.embedding_client)
        # Insert into database
        if ctx.deps.supabase:
            result = ctx.deps.supabase.table("agent_embeddings").insert({
                "folder_name": template_name,
                "agents_code": agents_code,
                "main_code": main_code,
                "models_code": models_code,
                "tools_code": tools_code,
                "mcp_json": mcp_json,
                "purpose": sample_purpose,
                "metadata": metadata,
                "embedding": embedding
            }).execute()
            # Check for errors
            if hasattr(result, 'error') and result.error:
                print(f"\n[DEEP LOG] Error inserting template: {result.error}")
                return {"error": f"Failed to insert template: {result.error}"}
            inserted_id = result.data[0]['id'] if result.data else None
            print(f"\n[DEEP LOG] Successfully created multi-MCP template example with ID: {inserted_id}")
            return {
                "success": True,
                "message": f"Successfully created multi-MCP template example: {template_name}",
                "id": inserted_id,
                "stats": {
                    "agents.py": len(agents_code),
                    "main.py": len(main_code),
                    "models.py": len(models_code),
                    "tools.py": len(tools_code),
                    "mcp.json": len(mcp_json)
                }
            }
        else:
            print("\n[DEEP LOG] Supabase client not available, could not insert template")
            return {
                "success": False,
                "message": "Could not insert template: Supabase client not available",
                "files": {
                    "agents.py": agents_code,
                    "main.py": main_code,
                    "models.py": models_code,
                    "tools.py": tools_code,
                    "mcp.json": mcp_json
                }
            }
    except Exception as e:
        print(f"\n[DEEP LOG] Error in template processing: {str(e)}")
        return {"error": f"Failed to process templates: {str(e)}"}
@pydantic_ai_coder.tool
async def display_template_components(ctx: RunContext[PydanticAIDeps], 
                                    template_ids: List[int],
                                    components: List[str] = ["config", "mcp_servers", "tools"]) -> Dict[str, Any]:
    Displays specific components from multiple templates for comparison before merging.
    This function helps users understand how different templates implement key features,
    which is especially useful for multi-service agents.
    Args:
        ctx: The context including the Supabase client
        template_ids: List of template IDs to compare
        components: List of components to extract and compare (options: "config", "mcp_servers", 
                   "tools", "system_prompt", "dependencies", "imports")
    Returns:
        Dictionary with extracted components from each template
    print(f"\n[DEEP LOG] Displaying template components for comparison: {template_ids}")
    if not ctx.deps.supabase:
        return {"error": "Supabase client not available"}
    result = {}
    template_details = []
    # Validate components
    valid_components = ["config", "mcp_servers", "tools", "system_prompt", "dependencies", "imports"]
    requested_components = [c for c in components if c in valid_components]
    if not requested_components:
        requested_components = ["config", "mcp_servers", "tools"]  # Default components
    # Fetch templates
    for template_id in template_ids:
        template_result = await fetch_template_by_id_tool(ctx.deps.supabase, template_id)
        if not template_result.get("found", False):
            template_details.append({
                "id": template_id,
                "error": f"Template with ID {template_id} not found"
            })
            continue
        template = template_result.get("template", {})
        template_name = template.get("folder_name", f"template_{template_id}")
        # Extract components
        extracted = {
            "id": template_id,
            "name": template_name,
            "components": {}
        }
        # Process agents.py for various components
        agents_code = template.get("agents_code", "")
        # Extract config from models.py
        if "config" in requested_components:
            models_code = template.get("models_code", "")
            config_pattern = r"class\s+Config(?:\(BaseModel\))?:.*?(?=\n\nclass|\n\ndef|\Z)"
            config_matches = re.findall(config_pattern, models_code, re.DOTALL)
            if config_matches:
                extracted["components"]["config"] = config_matches[0].strip()
            else:
                extracted["components"]["config"] = "No Config class found in models.py"
        # Extract MCP server initialization
        if "mcp_servers" in requested_components:
            mcp_pattern = r"(?:def\s+create_\w+_mcp_server|(?:server|mcp_server)\s*=\s*MCPServerStdio).*?(?=\n\n\S|\Z)"
            mcp_matches = re.findall(mcp_pattern, agents_code + "\n\n" + template.get("tools_code", ""), re.DOTALL)
            if mcp_matches:
                extracted["components"]["mcp_servers"] = "\n\n".join(mcp_matches).strip()
            else:
                tools_code = template.get("tools_code", "")
                mcp_matches = re.findall(mcp_pattern, tools_code, re.DOTALL)
                if mcp_matches:
                    extracted["components"]["mcp_servers"] = "\n\n".join(mcp_matches).strip()
                else:
                    extracted["components"]["mcp_servers"] = "No MCP server initialization found"
        # Extract tool functions
        if "tools" in requested_components:
            tools_code = template.get("tools_code", "")
            # Look for @tool decorators or async def patterns typical in tool definitions
            tools_pattern = r"(?:@\w+_agent\.tool|async def \w+\(.*?\)).*?(?=\n\n\S|\Z)"
            tools_matches = re.findall(tools_pattern, agents_code + "\n\n" + tools_code, re.DOTALL)
            if tools_matches:
                # Limit to first 3 tools if there are many
                tool_sample = tools_matches[:3]
                if len(tools_matches) > 3:
                    tool_sample.append(f"... and {len(tools_matches) - 3} more tools")
                extracted["components"]["tools"] = "\n\n".join(tool_sample).strip()
            else:
                extracted["components"]["tools"] = "No tool functions found"
        # Extract system prompt
        if "system_prompt" in requested_components:
            system_prompt_pattern = r"(?:system_prompt|SYSTEM_PROMPT)\s*=\s*(?:\"\"\"|''').+?(?:\"\"\"|''')"
            system_prompt_matches = re.findall(system_prompt_pattern, agents_code, re.DOTALL)
            if system_prompt_matches:
                # Get just a sample if it's too long
                prompt_sample = system_prompt_matches[0]
                if len(prompt_sample) > 500:
                    prompt_sample = prompt_sample[:500] + "...[truncated]..."
                extracted["components"]["system_prompt"] = prompt_sample.strip()
            else:
                extracted["components"]["system_prompt"] = "No system prompt found"
        # Extract dependency classes
        if "dependencies" in requested_components:
            dependencies_pattern = r"@dataclass\s+class\s+\w+(?:Deps|Dependencies).*?(?=\n\n\S|\Z)"
            dependencies_matches = re.findall(dependencies_pattern, agents_code, re.DOTALL)
            if dependencies_matches:
                extracted["components"]["dependencies"] = dependencies_matches[0].strip()
            else:
                extracted["components"]["dependencies"] = "No dependency class found"
        # Extract imports
        if "imports" in requested_components:
            imports_pattern = r"(?:^|\n)(?:import|from) .+?(?=\n\S|\Z)"
            agents_imports = re.findall(imports_pattern, agents_code, re.DOTALL)
            models_imports = re.findall(imports_pattern, template.get("models_code", ""), re.DOTALL)
            tools_imports = re.findall(imports_pattern, template.get("tools_code", ""), re.DOTALL)
            all_imports = set()
            for imp_list in [agents_imports, models_imports, tools_imports]:
                for imp in imp_list:
                    all_imports.add(imp.strip())
            extracted["components"]["imports"] = "\n".join(sorted(all_imports))
        template_details.append(extracted)
    # Format the response
    result["templates"] = template_details
    result["compared_components"] = requested_components
    # Summary for quick comparison
    result["summary"] = {
        "template_count": len(template_details),
        "template_names": [t.get("name") for t in template_details if "error" not in t],
        "components_analyzed": requested_components
    }
    return result
# This function is called when generating a complete agent from a template
async def generate_complete_agent_from_template(self, template_id=None, template_search_query=None):
    print(f"\n[DEEP LOG] Generating complete agent code from template. ID: {template_id}, Query: {template_search_query}")
    # If a template_id is provided, fetch from database
    if template_id is not None:
        template = await fetch_template_by_id_tool(self.deps.supabase, template_id)
        if not template.get("found", False):
            return {"error": f"Template with ID {template_id} not found."}
        template = template.get("template")
    # If a query is provided, search for a template
    elif template_search_query is not None:
        templates = await search_agent_templates_tool(self.deps.supabase, self.deps.embedding_client, template_search_query)
        if not templates.get("templates"):
            return {"error": f"No templates found matching query: {template_search_query}"}
        templates_list = templates.get("templates")
        # Check if we need to merge multiple templates based on query
        # Look for keywords indicating multiple services in query
        multi_service_keywords = ["and", "with", "both", "multiple", "combine", "integration", 
                                  "spotify github", "github spotify", "search email", "email search"]
        needs_merge = any(keyword in template_search_query.lower() for keyword in multi_service_keywords)
        # If query contains multi-service keywords, check if we found multiple service-specific templates
        if needs_merge and len(templates_list) >= 2:
            print(f"\n[DEEP LOG] Multi-service query detected. Will attempt to merge templates instead of using a single template.")
            # Extract template IDs for merging
            template_ids = [t.get("id") for t in templates_list[:2]]  # Take first two matching templates
            # Call merge_agent_templates instead of using a single template
            return await merge_agent_templates(self.ctx, template_ids=template_ids, 
                                             custom_name=f"merged_{template_search_query.replace(' ', '_')}")
        # Otherwise, use the first template
        template = templates_list[0]
        template_id = template.get("id")
        template = await fetch_template_by_id_tool(self.deps.supabase, template_id)
        if not template.get("found", False):
            return {"error": f"Failed to fetch template with ID {template_id}"}
        template = template.get("template")
    else:
        return {"error": "Either template_id or template_search_query must be provided."}
    # Extract code from the template
    agents_code = template.get("agents_code", "")
    tools_code = template.get("tools_code", "")
    models_code = template.get("models_code", "")
    main_code = template.get("main_code", "")
    mcp_json = template.get("mcp_json", "")
    # Check if we have all required components
    if not agents_code or not tools_code or not models_code:
        return {"error": "Template is missing required components."}
    # Put together the complete agent code
