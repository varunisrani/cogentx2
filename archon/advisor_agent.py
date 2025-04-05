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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var
from archon.agent_prompts import advisor_prompt
from archon.agent_tools import get_file_content_tool, search_agent_templates_tool, fetch_template_by_id_tool

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AdvisorDeps:
    file_list: List[str]
    supabase: Client = None
    embedding_client: AsyncOpenAI = None

advisor_agent = Agent(
    model,
    system_prompt=advisor_prompt + """

ADDITIONAL GUIDANCE FOR MULTI-SERVICE AGENTS:

When the user requests an agent that combines multiple services (e.g., "Spotify and GitHub" or "search and email"), 
you must recommend a proper merging approach that combines code from ALL relevant templates.

For multi-service agents:
1. Identify templates for EACH required service
2. Emphasize that proper merging requires combining:
   - Config class fields from BOTH templates
   - Dependency class fields from BOTH templates
   - Tool functions from BOTH templates
   - System prompts from BOTH templates
   - MCP server initialization from BOTH templates

3. Explain that simply using one template's code is NOT sufficient
4. IMPORTANT: EXPLICITLY RECOMMEND USING THE merge_agent_templates TOOL WITH TEMPLATE IDs FROM SEPARATE SERVICES
5. AVOID recommending pre-built combined templates (like SpotifyGitHubAgent - ID 5) even if they appear to match - 
   the proper approach is to merge separate templates to ensure complete integration
6. Use language like: "I recommend merging the spotify_agent (ID 4) and github_agent (ID 2) templates using the merge_agent_templates tool"

REMEMBER: The goal is comprehensive integration of ALL services by combining separate templates, 
not using a pre-existing template that might be incomplete.

TEMPLATE MERGING OVERRIDE:

Even if you see a pre-built combined template like "SpotifyGitHubAgent" (ID 5) that matches the user's request,
DO NOT recommend it. Instead, ALWAYS recommend merging separate templates using merge_agent_templates.

SPECIFIC TEMPLATE EXAMPLES:
- For Spotify functionality: Use spotify_agent (ID 4)
- For GitHub functionality: Use github_agent (ID 2)
- For Search functionality: Use search_agent (ID 1)
- For Gmail functionality: Use gmail_agent (ID 3)

When a user asks for multiple services (e.g., "Spotify and GitHub"), your recommendation should be:
"I recommend merging separate templates - spotify_agent (ID 4) and github_agent (ID 2) - using the merge_agent_templates tool."
""",
    deps_type=AdvisorDeps,
    retries=2
)

@advisor_agent.system_prompt  
def add_file_list(ctx: RunContext[str]) -> str:
    base_prompt = f"""
    
Here is the list of all the files that you can pull the contents of with the
'get_file_content' tool if the example/tool/MCP server is relevant to the
agent the user is trying to build:
 
""" + "\n".join(ctx.deps.file_list) + """
    """
    
    if ctx.deps.supabase and ctx.deps.embedding_client:
        base_prompt += """
    
You can also search for similar agent templates with the 'search_agent_templates' tool.
These templates contain pre-built agent.py, main.py, models.py, tools.py, and mcp.json files
that can serve as excellent starting points for generating new agents.

When you find relevant templates, you should:
1. Analyze their structure and purpose
2. Adapt them to the user's specific requirements
3. Recommend the most suitable one(s) for the user's needs
        """
    
    return base_prompt

@advisor_agent.tool_plain
def get_file_content(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)

@advisor_agent.tool
async def search_agent_templates(ctx: RunContext[AdvisorDeps], query: str, threshold: float = 0.4, limit: int = 3) -> Dict[str, Any]:
    """
    Search for agent templates using embedding similarity.
    Use this to find existing agent templates similar to what the user wants to build.
    
    Args:
        ctx: The context including the Supabase client and embedding client
        query: The search query describing the agent to build
        threshold: Similarity threshold (0.0 to 1.0)
        limit: Maximum number of results to return
        
    Returns:
        Dict containing similar agent templates with their code and metadata
    """
    if not ctx.deps.supabase or not ctx.deps.embedding_client:
        return {"templates": [], "message": "Agent embedding search is not available."}
    
    return await search_agent_templates_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, threshold, limit)

@advisor_agent.tool
async def fetch_template_by_id(ctx: RunContext[AdvisorDeps], template_id: int) -> Dict[str, Any]:
    """
    Fetch a specific agent template by ID.
    Use this to get the full details of a template after finding it with search_agent_templates.
    
    Args:
        ctx: The context including the Supabase client
        template_id: The ID of the template to fetch
        
    Returns:
        Dict containing the agent template with its code and metadata
    """
    if not ctx.deps.supabase:
        return {"found": False, "message": "Agent embedding search is not available."}
    
    return await fetch_template_by_id_tool(ctx.deps.supabase, template_id)