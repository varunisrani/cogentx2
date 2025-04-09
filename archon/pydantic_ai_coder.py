from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var
from archon.agent_prompts import primary_coder_prompt
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
    search_mcp_templates,
    get_mcp_template_by_id,
    detect_mcp_needs,
    merge_mcp_templates,
    adapt_mcp_template
)

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

# Initialize the model based on provider
if provider == "Anthropic":
    model = AnthropicModel(llm, api_key=api_key)
else:
    # OpenAI model initialization - using default configuration with env vars
    # The OpenAIModel reads configuration from environment variables
    model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    reasoner_output: str
    advisor_output: str
    mcp_templates: Optional[List[Dict[str, Any]]] = None
    selected_template: Optional[Dict[str, Any]] = None
    merged_templates: Optional[Dict[str, Any]] = None

# Helper function to safely write files to the workbench directory
def write_to_workbench(filename: str, content: str) -> Tuple[bool, str]:
    """
    Safely write content to a file in the workbench directory.

    Args:
        filename: The name of the file to write
        content: The content to write to the file

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Determine the absolute path to the workbench directory
        workbench_dir = os.path.join(os.getcwd(), "workbench")

        # Create the full path to the file
        file_path = os.path.join(workbench_dir, filename)

        # Write the content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully wrote {filename} to workbench directory: {file_path}")
        return True, f"Successfully created {filename}"
    except Exception as e:
        error_message = f"Error writing {filename} to workbench directory: {str(e)}"
        print(error_message)
        return False, error_message

pydantic_ai_coder = Agent(
    model,
    system_prompt=primary_coder_prompt,
    deps_type=PydanticAIDeps,
    retries=5
)

@pydantic_ai_coder.system_prompt
def add_reasoner_output(ctx: RunContext[str]) -> str:
    base_prompt = f"""
    Additional thoughts/instructions from the reasoner LLM.
    This scope includes documentation pages for you to search as well:
    {ctx.deps.reasoner_output}

    Recommended starting point from the advisor agent:
    {ctx.deps.advisor_output}
    """

    # Add MCP template information if available
    if ctx.deps.mcp_templates:
        template_info = "\n\nMCP Templates are available for this request. Use the search_mcp_templates, get_mcp_template_by_id, " \
                        "and analyze_mcp_templates tools to explore and utilize these templates.\n"

        # If we have template details, show a summary
        if len(ctx.deps.mcp_templates) > 0:
            template_info += "Available Templates:\n"
            for i, template in enumerate(ctx.deps.mcp_templates[:5]):  # Show first 5 templates
                template_info += f"- {template.get('folder_name', 'Unknown')} (ID: {template.get('id', 'Unknown')}): " \
                                f"{template.get('purpose', 'No description')[:100]}...\n"

        base_prompt += template_info

    # Add selected or merged template information if available
    if ctx.deps.selected_template:
        # Create file list with a safer approach
        available_files = []
        for f in ['agents.py', 'models.py', 'mains.py', 'tools.py', 'mcp.json', '__init__.py']:
            field_name = f.split('.')[0] + '_code'
            if f == 'mcp.json':
                field_name = 'mcp_json'
            elif f == '__init__.py':
                field_name = 'init_code'
            if ctx.deps.selected_template.get(field_name):
                available_files.append(f)

        template_details = f"\n\nSelected MCP Template: {ctx.deps.selected_template.get('folder_name', 'Unknown')}" \
                          f"\nPurpose: {ctx.deps.selected_template.get('purpose', 'No description')[:200]}...\n" \
                          f"Contains files: {', '.join(available_files)}\n" \
                          f"Use the generate_files_from_template tool to create files from this template."

        base_prompt += template_details

    if ctx.deps.merged_templates:
        # Create file list with a safer approach for merged templates
        available_merged_files = []
        for f in ['agents.py', 'models.py', 'mains.py', 'tools.py', 'mcp.json', '__init__.py']:
            field_name = f.split('.')[0] + '_code'
            if f == 'mcp.json':
                field_name = 'mcp_json'
            elif f == '__init__.py':
                field_name = 'init_code'
            if ctx.deps.merged_templates.get(field_name):
                available_merged_files.append(f)

        merged_details = f"\n\nMerged MCP Templates are available. This is a combination of multiple templates." \
                        f"\nContains files: {', '.join(available_merged_files)}\n" \
                        f"Use the generate_files_from_merged_templates tool to create files from these merged templates."

        base_prompt += merged_details

    return base_prompt

@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, user_query)

@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)

@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)

@pydantic_ai_coder.tool
async def search_mcp_templates_for_query(ctx: RunContext[PydanticAIDeps], query: str) -> List[Dict[str, Any]]:
    """
    Search for MCP templates in the database based on their semantic similarity to the query.

    Args:
        ctx: The run context with dependencies
        query: The search query for finding relevant MCP templates

    Returns:
        List of template metadata objects with similarity scores
    """
    templates = await search_mcp_templates(ctx.deps.supabase, ctx.deps.embedding_client, query)

    # Store the templates in the dependencies for future use
    ctx.deps.mcp_templates = templates

    # Format the response to be more readable
    result = []
    for template in templates:
        # Get available files with a safer approach
        available_files = []
        for f in ["agents.py", "models.py", "mains.py", "tools.py", "mcp.json"]:
            field_name = f.split('.')[0] + '_code'
            if f == 'mcp.json':
                field_name = 'mcp_json'
            if template.get(field_name):
                available_files.append(f)

        result.append({
            "id": template.get("id"),
            "folder_name": template.get("folder_name"),
            "purpose": template.get("purpose", "No description available")[:250] + "...",
            "similarity": template.get("similarity", 0),
            "available_files": available_files
        })

    return result

@pydantic_ai_coder.tool
async def get_mcp_template_details(ctx: RunContext[PydanticAIDeps], template_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific MCP template.

    Args:
        ctx: The run context with dependencies
        template_id: The ID of the template to fetch

    Returns:
        Detailed information about the template
    """
    result = await get_mcp_template_by_id(ctx.deps.supabase, template_id)

    if result.get("found", False):
        template = result.get("template", {})

        # Store the selected template in the context
        ctx.deps.selected_template = template

        # Format the response to be more readable
        formatted_result = {
            "id": template.get("id"),
            "folder_name": template.get("folder_name"),
            "purpose": template.get("purpose", "No description"),
            "available_files": [],
            "file_previews": {}
        }

        # Get available files with a safer approach
        for f in ["agents.py", "models.py", "mains.py", "tools.py", "mcp.json"]:
            field_name = f.split('.')[0] + '_code'
            if f == 'mcp.json':
                field_name = 'mcp_json'
            if template.get(field_name):
                formatted_result["available_files"].append(f)

        # Add previews of the files (first 200 chars)
        for file_type in ["agents_code", "models_code", "main_code", "tools_code", "mcp_json"]:
            if template.get(file_type):
                file_name = file_type.replace("_code", ".py")
                if file_type == "mcp_json":
                    file_name = "mcp.json"

                content = template.get(file_type)
                preview = content[:200] + "..." if len(content) > 200 else content
                formatted_result["file_previews"][file_name] = preview

        return formatted_result
    else:
        return {
            "error": result.get("message", "Template not found")
        }

@pydantic_ai_coder.tool
async def get_template_file_content(ctx: RunContext[PydanticAIDeps], template_id: int, file_name: str) -> str:
    """
    Get the content of a specific file from an MCP template.

    Args:
        ctx: The run context with dependencies
        template_id: The ID of the template
        file_name: The name of the file to retrieve (e.g., "agents.py", "models.py", "mcp.json")

    Returns:
        The content of the requested file
    """
    result = await get_mcp_template_by_id(ctx.deps.supabase, template_id)

    if not result.get("found", False):
        return f"Error: Template with ID {template_id} not found."

    template = result.get("template", {})

    # Map file name to the corresponding field in the template
    field_mapping = {
        "agents.py": "agents_code",
        "models.py": "models_code",
        "main.py": "main_code",
        "mains.py": "main_code",
        "tools.py": "tools_code",
        "mcp.json": "mcp_json",
        "__init__.py": "init_code"
    }

    field_name = field_mapping.get(file_name)
    if not field_name:
        return f"Error: Unknown file name '{file_name}'. Available files: {', '.join(field_mapping.keys())}"

    content = template.get(field_name)
    if not content:
        return f"Error: File '{file_name}' not found in template {template.get('folder_name', 'unknown')}."

    return f"Content of {file_name} from template {template.get('folder_name', 'unknown')}:\n\n{content}"

@pydantic_ai_coder.tool
async def detect_mcp_services(ctx: RunContext[PydanticAIDeps], query: str) -> Dict[str, List[str]]:
    """
    Detect which MCP services might be needed based on keywords in the query.

    Args:
        ctx: The run context with dependencies
        query: The user query to analyze

    Returns:
        Dictionary mapping service names to the keywords that triggered them
    """
    return await detect_mcp_needs(ctx.deps.embedding_client, query)

@pydantic_ai_coder.tool
async def merge_multiple_templates(ctx: RunContext[PydanticAIDeps], template_ids: List[int], user_query: str) -> Dict[str, Any]:
    """
    Merge multiple MCP templates into a single coherent template.

    Args:
        ctx: The run context with dependencies
        template_ids: List of template IDs to merge
        user_query: The original user query for context

    Returns:
        Summary of the merged template
    """
    # Fetch all templates
    templates = []
    for template_id in template_ids:
        result = await get_mcp_template_by_id(ctx.deps.supabase, template_id)
        if result.get("found", False):
            templates.append(result.get("template"))

    if len(templates) < 1:
        return {"error": "No valid templates found to merge."}

    # Merge the templates
    merged = await merge_mcp_templates(templates, user_query)

    # Store the merged templates in the context
    ctx.deps.merged_templates = merged

    # Format the response
    return {
        "success": True,
        "message": f"Successfully merged {len(templates)} templates",
        "templates_merged": [t.get("folder_name", f"ID: {t.get('id')}") for t in templates],
        "available_files": get_available_files(merged),
        "file_previews": {
            "agents.py": merged.get("agents_code", "")[:200] + "..." if merged.get("agents_code") and len(merged.get("agents_code")) > 200 else merged.get("agents_code", ""),
            "models.py": merged.get("models_code", "")[:200] + "..." if merged.get("models_code") and len(merged.get("models_code")) > 200 else merged.get("models_code", ""),
            "main.py": merged.get("main_code", "")[:200] + "..." if merged.get("main_code") and len(merged.get("main_code")) > 200 else merged.get("main_code", ""),
            "tools.py": merged.get("tools_code", "")[:200] + "..." if merged.get("tools_code") and len(merged.get("tools_code")) > 200 else merged.get("tools_code", ""),
            "mcp.json": merged.get("mcp_json", "")[:200] + "..." if merged.get("mcp_json") and len(merged.get("mcp_json")) > 200 else merged.get("mcp_json", "")
        }
    }

# Helper function to get available files from a template
def get_available_files(template_data):
    available_files = []
    for f in ["agents.py", "models.py", "mains.py", "tools.py", "mcp.json"]:
        field_name = f.split('.')[0] + '_code'
        if f == 'mcp.json':
            field_name = 'mcp_json'

        # Special handling for mcp.json in merged templates
        if f == 'mcp.json' and template_data.get('mcp_json'):
            available_files.append(f)
        elif template_data.get(field_name):
            available_files.append(f)

    return available_files

@pydantic_ai_coder.tool
async def generate_files_from_template(ctx: RunContext[PydanticAIDeps], template_id: int, user_query: str) -> Dict[str, str]:
    """
    Generate code files from an MCP template and adapt them to the user's query.

    Args:
        ctx: The run context with dependencies
        template_id: The ID of the template to use
        user_query: The user's query for adapting the template

    Returns:
        Dictionary mapping file names to their generated content
    """
    # Get the template
    result = await get_mcp_template_by_id(ctx.deps.supabase, template_id)

    if not result.get("found", False):
        return {"error": f"Template with ID {template_id} not found."}

    template = result.get("template")

    # Adapt the template to the user's query
    adapted_template = await adapt_mcp_template(template, user_query)

    # Set the template as the selected template
    ctx.deps.selected_template = adapted_template

    # Extract the files
    files = {}
    file_creation_results = {}
    formatted_code_for_ui = ["# Generated Agent Code\n\nHere are all the generated files for your agent:\n"]

    if adapted_template.get("agents_code"):
        files["agents.py"] = adapted_template.get("agents_code")
        success, message = write_to_workbench("agents.py", adapted_template.get("agents_code"))
        file_creation_results["agents.py"] = message
        formatted_code_for_ui.append(f"\n## agents.py\n```python\n{adapted_template.get('agents_code')}\n```\n")

    if adapted_template.get("models_code"):
        files["models.py"] = adapted_template.get("models_code")
        success, message = write_to_workbench("models.py", adapted_template.get("models_code"))
        file_creation_results["models.py"] = message
        formatted_code_for_ui.append(f"\n## models.py\n```python\n{adapted_template.get('models_code')}\n```\n")

    if adapted_template.get("main_code"):
        files["main.py"] = adapted_template.get("main_code")
        success, message = write_to_workbench("main.py", adapted_template.get("main_code"))
        file_creation_results["main.py"] = message
        formatted_code_for_ui.append(f"\n## main.py\n```python\n{adapted_template.get('main_code')}\n```\n")

    if adapted_template.get("tools_code"):
        files["tools.py"] = adapted_template.get("tools_code")
        success, message = write_to_workbench("tools.py", adapted_template.get("tools_code"))
        file_creation_results["tools.py"] = message
        formatted_code_for_ui.append(f"\n## tools.py\n```python\n{adapted_template.get('tools_code')}\n```\n")

    if adapted_template.get("mcp_json"):
        files["mcp.json"] = adapted_template.get("mcp_json")
        success, message = write_to_workbench("mcp.json", adapted_template.get("mcp_json"))
        file_creation_results["mcp.json"] = message
        formatted_code_for_ui.append(f"\n## mcp.json\n```json\n{adapted_template.get('mcp_json')}\n```\n")

    if adapted_template.get("init_code"):
        files["__init__.py"] = adapted_template.get("init_code")
        success, message = write_to_workbench("__init__.py", adapted_template.get("init_code"))
        file_creation_results["__init__.py"] = message
        formatted_code_for_ui.append(f"\n## __init__.py\n```python\n{adapted_template.get('init_code')}\n```\n")

    # Create a .env.example file with MCP-specific environment variables
    if adapted_template.get("mcp_json"):
        try:
            mcp_data = json.loads(adapted_template.get("mcp_json"))
            env_vars = []

            if "mcpServers" in mcp_data:
                for server_name, server_config in mcp_data["mcpServers"].items():
                    if "env" in server_config:
                        for env_var, value in server_config["env"].items():
                            env_vars.append(f"# API key for {server_name} MCP server\n{env_var}={value}")

            if env_vars:
                env_example = "\n\n".join(env_vars)
                files[".env.example"] = env_example
                success, message = write_to_workbench(".env.example", env_example)
                file_creation_results[".env.example"] = message
                formatted_code_for_ui.append(f"\n## .env.example\n```\n{env_example}\n```\n")
        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple .env.example
            env_example = "# Add your MCP server environment variables here"
            files[".env.example"] = env_example
            success, message = write_to_workbench(".env.example", env_example)
            file_creation_results[".env.example"] = message
            formatted_code_for_ui.append(f"\n## .env.example\n```\n{env_example}\n```\n")

    # Create a requirements.txt file with the necessary dependencies
    requirements = [
        "pydantic-ai",
        "httpx",
        "python-dotenv",
        "logfire"
    ]

    # Add MCP-specific requirements if MCP JSON is present
    if adapted_template.get("mcp_json"):
        requirements.append("pydantic-ai-mcp")

    requirements_content = "\n".join(requirements)
    files["requirements.txt"] = requirements_content
    success, message = write_to_workbench("requirements.txt", requirements_content)
    file_creation_results["requirements.txt"] = message
    formatted_code_for_ui.append(f"\n## requirements.txt\n```\n{requirements_content}\n```\n")

    # Generate a README.md file
    project_name = adapted_template.get("folder_name", "MCP Agent").title().replace("_", " ")
    readme_content = f"""# {project_name}

This agent was generated from the MCP template "{adapted_template.get('folder_name')}".

## Description

{adapted_template.get('purpose', 'A Pydantic AI agent with MCP integrations.')}

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run the agent: `python main.py`

## Files

{', '.join(list(files.keys()))}

## MCP Servers

This agent uses the following MCP servers:
"""

    # Add MCP server information to the README if available
    if adapted_template.get("mcp_json"):
        try:
            mcp_data = json.loads(adapted_template.get("mcp_json"))
            if "mcpServers" in mcp_data:
                for server_name, server_config in mcp_data["mcpServers"].items():
                    readme_content += f"\n- {server_name}"
        except json.JSONDecodeError:
            pass

    files["README.md"] = readme_content
    success, message = write_to_workbench("README.md", readme_content)
    file_creation_results["README.md"] = message
    formatted_code_for_ui.append(f"\n## README.md\n```markdown\n{readme_content}\n```\n")

    # Combine all formatted code blocks for UI display
    combined_code_for_ui = "\n".join(formatted_code_for_ui)

    # Return both the file contents, creation results, and formatted code for UI
    return {
        "files": files,
        "results": file_creation_results,
        "code_for_ui": combined_code_for_ui,
        "message": "Agent files have been successfully generated and written to the workbench directory."
    }

@pydantic_ai_coder.tool
async def generate_files_from_merged_templates(ctx: RunContext[PydanticAIDeps], user_query: str) -> Dict[str, str]:
    """
    Generate code files from merged MCP templates and adapt them to the user's query.

    Args:
        ctx: The run context with dependencies
        user_query: The user's query for adapting the merged templates

    Returns:
        Dictionary mapping file names to their generated content
    """
    if not ctx.deps.merged_templates:
        return {"error": "No merged templates available. Use the merge_multiple_templates tool first."}

    merged_templates = ctx.deps.merged_templates

    # Adapt the merged templates to the user's query
    adapted_templates = await adapt_mcp_template(merged_templates, user_query)

    # Extract the files
    files = {}
    file_creation_results = {}
    formatted_code_for_ui = ["# Generated Agent Code (Merged Templates)\n\nHere are all the generated files for your agent:\n"]

    if adapted_templates.get("agents_code"):
        files["agents.py"] = adapted_templates.get("agents_code")
        success, message = write_to_workbench("agents.py", adapted_templates.get("agents_code"))
        file_creation_results["agents.py"] = message
        formatted_code_for_ui.append(f"\n## agents.py\n```python\n{adapted_templates.get('agents_code')}\n```\n")

    if adapted_templates.get("models_code"):
        files["models.py"] = adapted_templates.get("models_code")
        success, message = write_to_workbench("models.py", adapted_templates.get("models_code"))
        file_creation_results["models.py"] = message
        formatted_code_for_ui.append(f"\n## models.py\n```python\n{adapted_templates.get('models_code')}\n```\n")

    if adapted_templates.get("main_code"):
        files["main.py"] = adapted_templates.get("main_code")
        success, message = write_to_workbench("main.py", adapted_templates.get("main_code"))
        file_creation_results["main.py"] = message
        formatted_code_for_ui.append(f"\n## main.py\n```python\n{adapted_templates.get('main_code')}\n```\n")

    if adapted_templates.get("tools_code"):
        files["tools.py"] = adapted_templates.get("tools_code")
        success, message = write_to_workbench("tools.py", adapted_templates.get("tools_code"))
        file_creation_results["tools.py"] = message
        formatted_code_for_ui.append(f"\n## tools.py\n```python\n{adapted_templates.get('tools_code')}\n```\n")

    if adapted_templates.get("mcp_json"):
        files["mcp.json"] = adapted_templates.get("mcp_json")
        success, message = write_to_workbench("mcp.json", adapted_templates.get("mcp_json"))
        file_creation_results["mcp.json"] = message
        formatted_code_for_ui.append(f"\n## mcp.json\n```json\n{adapted_templates.get('mcp_json')}\n```\n")

    if adapted_templates.get("init_code"):
        files["__init__.py"] = adapted_templates.get("init_code")
        success, message = write_to_workbench("__init__.py", adapted_templates.get("init_code"))
        file_creation_results["__init__.py"] = message
        formatted_code_for_ui.append(f"\n## __init__.py\n```python\n{adapted_templates.get('init_code')}\n```\n")

    # Add any other files from adapted_templates
    for key, value in adapted_templates.items():
        if key.endswith("_code") and key not in ["agents_code", "models_code", "main_code", "tools_code", "init_code", "mcp_json"] and value:
            file_name = key.replace("_code", ".py")
            files[file_name] = value
            success, message = write_to_workbench(file_name, value)
            file_creation_results[file_name] = message
            formatted_code_for_ui.append(f"\n## {file_name}\n```python\n{value}\n```\n")

    # Create a .env.example file with MCP-specific environment variables
    if adapted_templates.get("env_example_code"):
        env_example = adapted_templates.get("env_example_code")
        files[".env.example"] = env_example
        success, message = write_to_workbench(".env.example", env_example)
        file_creation_results[".env.example"] = message
        formatted_code_for_ui.append(f"\n## .env.example\n```\n{env_example}\n```\n")
    elif adapted_templates.get("mcp_json"):
        try:
            mcp_data = json.loads(adapted_templates.get("mcp_json"))
            env_vars = []

            if "mcpServers" in mcp_data:
                for server_name, server_config in mcp_data["mcpServers"].items():
                    if "env" in server_config:
                        for env_var, value in server_config["env"].items():
                            env_vars.append(f"# API key for {server_name} MCP server\n{env_var}={value}")

            if env_vars:
                env_example = "\n\n".join(env_vars)
                files[".env.example"] = env_example
                success, message = write_to_workbench(".env.example", env_example)
                file_creation_results[".env.example"] = message
                formatted_code_for_ui.append(f"\n## .env.example\n```\n{env_example}\n```\n")
        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple .env.example
            env_example = "# Add your MCP server environment variables here"
            files[".env.example"] = env_example
            success, message = write_to_workbench(".env.example", env_example)
            file_creation_results[".env.example"] = message
            formatted_code_for_ui.append(f"\n## .env.example\n```\n{env_example}\n```\n")

    # Create a requirements.txt file with the necessary dependencies
    if adapted_templates.get("requirements_code"):
        requirements_content = adapted_templates.get("requirements_code")
        files["requirements.txt"] = requirements_content
        success, message = write_to_workbench("requirements.txt", requirements_content)
        file_creation_results["requirements.txt"] = message
        formatted_code_for_ui.append(f"\n## requirements.txt\n```\n{requirements_content}\n```\n")
    else:
        requirements = [
            "pydantic-ai",
            "httpx",
            "python-dotenv",
            "logfire"
        ]

        # Add MCP-specific requirements if MCP JSON is present
        if adapted_templates.get("mcp_json"):
            requirements.append("pydantic-ai-mcp")

        requirements_content = "\n".join(requirements)
        files["requirements.txt"] = requirements_content
        success, message = write_to_workbench("requirements.txt", requirements_content)
        file_creation_results["requirements.txt"] = message
        formatted_code_for_ui.append(f"\n## requirements.txt\n```\n{requirements_content}\n```\n")

    # Generate a README.md file
    if adapted_templates.get("readme_code"):
        readme_content = adapted_templates.get("readme_code")
        files["README.md"] = readme_content
        success, message = write_to_workbench("README.md", readme_content)
        file_creation_results["README.md"] = message
        formatted_code_for_ui.append(f"\n## README.md\n```markdown\n{readme_content}\n```\n")
    else:
        project_name = "Merged MCP Agent"
        readme_content = f"""# {project_name}

This agent was generated by merging multiple MCP templates.

## Description

An agent created from multiple MCP templates based on: "{user_query}"

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run the agent: `python main.py`

## Files

{', '.join(list(files.keys()))}

## MCP Servers

This agent uses the following MCP servers:
"""

        # Add MCP server information to the README if available
        if adapted_templates.get("mcp_json"):
            try:
                mcp_data = json.loads(adapted_templates.get("mcp_json"))
                if "mcpServers" in mcp_data:
                    for server_name, server_config in mcp_data["mcpServers"].items():
                        readme_content += f"\n- {server_name}"
            except json.JSONDecodeError:
                pass

        files["README.md"] = readme_content
        success, message = write_to_workbench("README.md", readme_content)
        file_creation_results["README.md"] = message
        formatted_code_for_ui.append(f"\n## README.md\n```markdown\n{readme_content}\n```\n")

    # Check if package.json is provided
    if adapted_templates.get("package_json_code"):
        package_json = adapted_templates.get("package_json_code")
        files["package.json"] = package_json
        success, message = write_to_workbench("package.json", package_json)
        file_creation_results["package.json"] = message
        formatted_code_for_ui.append(f"\n## package.json\n```json\n{package_json}\n```\n")

    # Combine all formatted code blocks for UI display
    combined_code_for_ui = "\n".join(formatted_code_for_ui)

    # Return both the file contents, creation results, and formatted code for UI
    return {
        "files": files,
        "results": file_creation_results,
        "code_for_ui": combined_code_for_ui,
        "message": "Agent files have been successfully generated and written to the workbench directory."
    }