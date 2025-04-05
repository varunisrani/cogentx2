from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import get_env_var
from archon.agent_prompts import tools_refiner_prompt
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
    get_file_content_tool,
    search_agent_templates_tool,
    fetch_template_by_id_tool
)

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

model = OpenAIModel(llm)
embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class ToolsRefinerDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    file_list: List[str]

tools_refiner_agent = Agent(
    model,
    system_prompt=tools_refiner_prompt + """

ENHANCED GUIDANCE FOR MULTI-SERVICE INTEGRATION:

When working with tools for multi-service agents (e.g., combined Spotify and GitHub functionality):

1. DO NOT DISCARD tool functions from either service
   - Ensure BOTH services' tool functions are properly included
   - Check for name collisions and resolve them with clear naming
   - Validate that ALL tool functions have correct dependencies

2. VERIFY proper server initialization
   - Confirm that EACH service has its server initialization function
   - Check that ALL required environment variables are properly accessed
   - Ensure ALL servers are properly created in the main file

3. COMPARE the tool implementations against BOTH templates
   - Verify that no functionality is lost from either template
   - Ensure authentication is properly handled for ALL services
   - Check for consistent error handling across ALL tools

4. MERGING CHECKLIST - Verify the merged code includes:
   ✓ ALL tool functions from BOTH templates
   ✓ ALL MCP server initializations from BOTH templates
   ✓ Proper authentication for ALL services
   ✓ Clear error handling for ALL functions
   ✓ No duplicate or conflicting function names

The success of a multi-service agent depends on proper integration of tools from ALL templates!
""",
    deps_type=ToolsRefinerDeps,
    retries=2
)


@tools_refiner_agent.system_prompt  
def add_file_list(ctx: RunContext[str]) -> str:
    return f"""
    
Here is the list of all the files that you can pull the contents of with the
'get_file_content' tool if the example/tool/MCP server is relevant to the
agent the user is trying to build:
 
""" + "\n".join(ctx.deps.file_list) + """
    """

@tools_refiner_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[ToolsRefinerDeps], query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Make sure your searches always focus on implementing tools.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        query: Your query to retrieve relevant documentation for implementing tools
        
    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, query)

@tools_refiner_agent.tool
async def list_documentation_pages(ctx: RunContext[ToolsRefinerDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    This will give you all pages available, but focus on the ones related to tools.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)

@tools_refiner_agent.tool
async def get_page_content(ctx: RunContext[ToolsRefinerDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Only use this tool to get pages related to using tools with Pydantic AI.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)

@tools_refiner_agent.tool_plain
def get_file_content(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)    

@tools_refiner_agent.tool
async def search_agent_templates(ctx: RunContext[ToolsRefinerDeps], query: str, threshold: float = 0.4, limit: int = 3) -> Dict[str, Any]:
    """
    Search for agent templates using embedding similarity.
    Use this to find existing agent templates similar to the tools needed for the agent.
    
    Args:
        ctx: The context including the Supabase client and embedding client
        query: The search query describing the tools needed
        threshold: Similarity threshold (0.0 to 1.0)
        limit: Maximum number of results to return
        
    Returns:
        Dict containing similar agent templates with their code and metadata
    """
    return await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, threshold, limit)

@tools_refiner_agent.tool
async def fetch_template_by_id(ctx: RunContext[ToolsRefinerDeps], template_id: int) -> Dict[str, Any]:
    """
    Fetch a specific agent template by ID.
    Use this to get the full details of a template after finding it with search_agent_templates.
    
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch
        
    Returns:
        Dict containing the agent template with its code and metadata
    """
    return await fetch_template_by_id_tool(ctx.deps.supabase, template_id)    