from __future__ import annotations as _annotations
from .pydantic_ai_coder import detect_mcp_tool_keywords
from .mcp_tools.mcp_tool_selector import (
    extract_structured_requirements,
    UserRequirements,
    ToolRequirement,
    get_required_tools,
    rank_tools_by_requirement_match,
    get_crewai_tool_requirements,
    detect_keywords_in_query
)
from .mcp_tools.mcp_tool_coder import (
    mcp_tool_agent,
    MCPToolDeps,
    find_relevant_mcp_tools,
    integrate_mcp_tool_with_code,
    create_connection_script,
    generate_mcp_integration_readme,
    setup_mcp_tool_structure,
    analyze_tool_code,
    customize_tool_implementation,
    verify_tool_integration,
    create_mcp_context,
    generate_complete_crewai_project,
    extract_class_names_from_tools,
    generate_requirements,
    generate_crewai_readme,
    generate_tools_py,
)
import importlib

import os
import asyncio
import json
import logging
import time
import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import RunContext

# Add the parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main Archon files - delay import to avoid circular dependencies

# Import MCP tool specific functions directly

# Import the tool selector functions

# Import from pydantic_ai_coder directly

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'mcp_tool_graph.log')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mcp_tool_graph')

# Load environment variables
load_dotenv()


def get_tool_name_from_purpose(purpose: str) -> str:
    """
    Extract a reasonable tool name from the purpose description.

    Args:
        purpose: The purpose description string

    Returns:
        A cleaned tool name suitable for code generation
    """
    try:
        # First try to find explicit tool mentions
        tool_patterns = [
            r'(?:the |a |an )?([A-Za-z0-9]+)(?:Tool|Agent|API|SDK)',
            r'([A-Za-z0-9]+) (?:tool|agent|api|integration)',
            r'integrat(?:e|ion with) ([A-Za-z0-9]+)',
            r'([A-Za-z0-9]+) service',
            r'using ([A-Za-z0-9]+) to'
        ]

        for pattern in tool_patterns:
            match = re.search(pattern, purpose, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # If no specific pattern matched, use the first word that might be a tool name
        words = purpose.split()
        for word in words:
            # Skip common words and look for capitalized words that might be service names
            if len(word) > 3 and word[0].isupper() and word.lower() not in [
                'this', 'that', 'tool', 'with', 'from', 'using', 'agent', 'generate', 'create',
                'implement', 'design', 'build', 'develop', 'the', 'for', 'and', 'integration'
            ]:
                return word.lower()

        # Fall back to "mcp" + a word that might indicate the tool's purpose
        purpose_words = ['search', 'music', 'mail', 'chat',
                         'news', 'file', 'data', 'text', 'image']
        for word in purpose_words:
            if word in purpose.lower():
                return f"{word}"

        # Last resort: return "tool"
        return "tool"
    except Exception as e:
        logger.error(f"Error extracting tool name from purpose: {e}")
        return "tool"


async def mcp_tool_flow(
    user_request: str,
    openai_client: AsyncOpenAI,
    supabase_client: Any,
    base_dir: str = "output"
) -> Dict[str, Any]:
    """
    Main MCP tool flow for integrating CrewAI with MCP tools.

    This is the primary flow for handling MCP tool operations. 
    It orchestrates the entire process from tool selection to code generation.

    Args:
        user_request: User's natural language request
        openai_client: AsyncOpenAI client for embeddings and completions
        supabase_client: Supabase client for database access
        base_dir: Base directory for output files

    Returns:
        Dictionary with results of the flow
    """
    try:
        logger.info(
            f"Starting MCP tool flow for request: {user_request[:100]}...")

        # Set up dependencies
        deps = MCPToolDeps(
            supabase=supabase_client,
            openai_client=openai_client
        )

        # Create context with the full user request
        context = create_mcp_context(
            deps=deps,
            prompt=user_request
        )

        # Extract CrewAI tool requirements from query
        crewai_requirements = await get_crewai_tool_requirements(
            user_request,
            openai_client
        )

        logger.info(
            f"Extracted CrewAI requirements for {len(crewai_requirements.get('tools', []))} tools")

        # Always use the template-based approach
        logger.info("Always using template-based approach")
        return await combined_adaptive_flow(
            user_request=user_request,
            openai_client=openai_client,
            supabase_client=supabase_client,
            base_dir=base_dir
        )
    except Exception as e:
        logger.error(f"Error in MCP tool flow: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error in MCP tool flow: {str(e)}"
        }


def detect_multi_tool_request(query: str) -> bool:
    """
    Detect if a user request is likely asking for multiple integrated tools.

    Args:
        query: The user's query string

    Returns:
        Boolean indicating if this is a multi-tool request
    """
    query_lower = query.lower()

    # Check for keywords suggesting multiple tools
    multi_tool_indicators = [
        "multiple tools",
        "several tools",
        "tools that work together",
        "integrate tools",
        "combination of tools",
        "several services",
        "multiple services",
        "integrated services"
    ]

    for indicator in multi_tool_indicators:
        if indicator in query_lower:
            logger.info(
                f"MCP MULTI-TOOL DETECTION: Found indicator '{indicator}'")
            return True

    # Check for "and" connecting two service names
    # This pattern matches requests like "create a tool for spotify and github"
    service_names = [
        "github", "spotify", "youtube", "twitter", "slack", "gmail",
        "google drive", "discord", "notion", "trello", "asana", "jira",
        "instagram", "linkedin", "facebook", "calendar", "weather",
        "maps", "news", "dropbox", "shopify", "stripe", "aws"
    ]

    # Check for explicit "and" between service names
    for i, service1 in enumerate(service_names):
        if service1 in query_lower:
            # Look for phrases like "spotify and github" or "tools for spotify and github"
            for service2 in service_names[i+1:]:
                patterns = [
                    f"{service1} and {service2}",
                    f"{service2} and {service1}",
                    f"tools for {service1} and {service2}",
                    f"tool for {service1} and {service2}",
                    f"using {service1} and {service2}",
                    f"with {service1} and {service2}"
                ]

                for pattern in patterns:
                    if pattern in query_lower:
                        logger.info(
                            f"MCP MULTI-TOOL DETECTION: Found services connected by 'and': '{pattern}'")
                        return True

    # Count service names mentioned in the query
    service_count = 0
    mentioned_services = []

    for service in service_names:
        if service in query_lower:
            service_count += 1
            mentioned_services.append(service)
            if service_count >= 2:
                logger.info(
                    f"MCP MULTI-TOOL DETECTION: Found multiple services: {', '.join(mentioned_services)}")
                return True

    # If we're still here, check for comma-separated service names
    for service1 in service_names:
        if service1 in query_lower:
            for service2 in service_names:
                if service1 != service2 and service2 in query_lower:
                    # Check for patterns like "spotify, github" or "github, spotify"
                    if f"{service1}, {service2}" in query_lower or f"{service2}, {service1}" in query_lower:
                        logger.info(
                            f"MCP MULTI-TOOL DETECTION: Found comma-separated services: '{service1}, {service2}'")
                        return True

    return False


async def find_complementary_tools(
    query: str,
    existing_tools: List[str],
    deps: MCPToolDeps,
    max_additional: int = 2
) -> List[str]:
    """
    Find complementary tools that would work well with the tools already mentioned.

    Args:
        query: User's query
        existing_tools: Tools already identified
        deps: MCPToolDeps object with clients
        max_additional: Maximum number of additional tools to find

    Returns:
        List of additional tool names
    """
    try:
        prompt = f"""
        Based on this user request and the already identified tools, suggest {max_additional} additional tools that would integrate well with them:
        
        USER REQUEST: "{query}"
        
        ALREADY IDENTIFIED TOOLS: {', '.join(existing_tools)}
        
        Return ONLY a comma-separated list of additional tool names from this list:
        github, spotify, youtube, twitter, slack, gmail, google_drive, discord, notion, trello, asana, jira, instagram, linkedin, facebook
        
        Choose tools that would complement the existing ones and fulfill the user's request.
        """

        response = await deps.openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )

        tools_text = response.choices[0].message.content.strip().lower()

        # Clean up the response
        tools_text = tools_text.replace(".", "").replace("and", ",")

        # Parse the comma-separated list
        additional_tools = [tool.strip()
                            for tool in tools_text.split(",") if tool.strip()]

        # Remove any duplicates with existing tools
        additional_tools = [
            tool for tool in additional_tools if tool not in existing_tools]

        # Limit to the maximum requested
        additional_tools = additional_tools[:max_additional]

        return additional_tools

    except Exception as e:
        logger.error(f"Error finding complementary tools: {str(e)}")
        return []


async def generate_multi_tool_example(
    ctx: RunContext,
    tools: List[Dict[str, Any]],
    mentioned_tools: List[str],
    user_request: str
) -> Dict[str, str]:
    """
    Generate example files showing how to use multiple tools together.
    Creates agents.py, tasks.py, and crew.py files for a complete CrewAI setup.

    Args:
        ctx: RunContext with dependencies
        tools: List of tools being integrated
        mentioned_tools: List of tool names mentioned by the user
        user_request: Original user request

    Returns:
        Dictionary containing code for each file
    """
    try:
        # Extract tool purposes for the prompt
        tool_purposes = [tool.get('purpose', 'Unknown Tool') for tool in tools]

        # Extract tool class names if available
        tool_names = []
        for tool in tools:
            tool_purpose = tool.get('purpose', '')
            name = get_tool_name_from_purpose(tool_purpose)
            if name:
                tool_names.append(name.capitalize() + "Tool")

        # If we couldn't extract names, use simple placeholders
        if not tool_names or len(tool_names) < len(tools):
            tool_names = [f"Tool{i+1}" for i in range(len(tools))]

        # Create agents.py first
        agents_prompt = f"""
        Create a comprehensive agents.py file for a CrewAI project that uses these tools:
        
        USER REQUEST: "{user_request}"
        
        TOOLS:
        {chr(10).join([f"- {purpose}" for purpose in tool_purposes])}
        
        MENTIONED SERVICES: {', '.join(mentioned_tools)}
        
        TOOL CLASSES (from tools.py): {', '.join(tool_names)}
        
        IMPORTANT: Create a complete, runnable agents.py file that:
        1. Imports all tools from tools.py
        2. Creates 2-4 specialized CrewAI agents that use ALL the tools (make sure each tool is used by at least one agent)
        3. Give each agent a clear, distinct role, goal, and backstory
        4. Distribute the tools appropriately among the agents based on their roles
        5. Follow CrewAI best practices
        
        The code should be complete, well-commented, and ready to use.
        Do not include explanations outside the code - just create the agents.py file.
        """

        # Create tasks.py next
        tasks_prompt = f"""
        Create a comprehensive tasks.py file for a CrewAI project that uses these tools:
        
        USER REQUEST: "{user_request}"
        
        TOOLS:
        {chr(10).join([f"- {purpose}" for purpose in tool_purposes])}
        
        MENTIONED SERVICES: {', '.join(mentioned_tools)}
        
        TOOL CLASSES (from tools.py): {', '.join(tool_names)}
        
        IMPORTANT: Create a complete, runnable tasks.py file that:
        1. Imports the agents from agents.py
        2. Creates tasks for each agent
        3. Ensures each task description clearly explains what the agent should do
        4. Describes how the agent should use their tools to accomplish the task
        5. Sets appropriate expected_output for each task
        6. Follows a logical sequence for task execution
        
        The code should be complete, well-commented, and ready to use.
        Do not include explanations outside the code - just create the tasks.py file.
        """

        # Create crew.py
        crew_prompt = f"""
        Create a comprehensive crew.py file for a CrewAI project that uses these tools:
        
        USER REQUEST: "{user_request}"
        
        TOOLS:
        {chr(10).join([f"- {purpose}" for purpose in tool_purposes])}
        
        MENTIONED SERVICES: {', '.join(mentioned_tools)}
        
        TOOL CLASSES (from tools.py): {', '.join(tool_names)}
        
        IMPORTANT: Create a complete, runnable crew.py file that:
        1. Imports the agents from agents.py
        2. Imports the tasks from tasks.py
        3. Creates a Crew with all the agents
        4. Assigns tasks to the Crew
        5. Sets up the proper process (sequential or hierarchical)
        6. Includes code to run the Crew and handle its output
        7. Includes a main() function with proper execution
        
        The code should be complete, well-commented, and ready to use.
        Do not include explanations outside the code - just create the crew.py file.
        """

        # Generate crew.py
        crew_response = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": crew_prompt}],
            max_tokens=3000,
            temperature=1.0  # Changed from 0.3 to 1.0
        )
        crew_code = crew_response.choices[0].message.content
        crew_code = crew_code.replace(
            "```python", "").replace("```", "").strip()

        # Generate main.py
        main_prompt = f"""
        Create a complete main.py file for a CrewAI project that:
        
        1. Uses the agents, tasks, and crew defined in the project
        2. Provides a simple command-line interface
        3. Demonstrates how to run the project
        4. Handles errors appropriately
        5. Includes logging setup
        
        The code should be fully functional and ready to run on its own.
        
        The user request is: "{user_request}"
        
        The available tools are:
        {', '.join(mentioned_tools)}
        
        Make sure main.py integrates with the following components:
        - agents.py which defines specialized agents
        - tasks.py which defines the tasks
        - crew.py which orchestrates the whole process
        
        Return ONLY the complete Python code for the main.py file.
        """

        main_response = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": main_prompt}],
            max_tokens=3000,
            temperature=1.0
        )
        main_code = main_response.choices[0].message.content
        main_code = main_code.replace(
            "```python", "").replace("```", "").strip()

        # Add headers to each file
        header = f"""#!/usr/bin/env python3
# Generated for: {user_request}
# This file is part of a CrewAI project using multiple MCP tools
        """

        results = {}

        # Generate agents.py
        agents_response = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": agents_prompt}],
            max_tokens=3000,
            temperature=1.0  # Changed from 0.3 to 1.0
        )
        agents_code = agents_response.choices[0].message.content
        agents_code = agents_code.replace(
            "```python", "").replace("```", "").strip()

        # Generate tasks.py
        tasks_response = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": tasks_prompt}],
            max_tokens=3000,
            temperature=1.0  # Changed from 0.3 to 1.0
        )
        tasks_code = tasks_response.choices[0].message.content
        tasks_code = tasks_code.replace(
            "```python", "").replace("```", "").strip()

        results['agents.py'] = f"{header}\\n{agents_code}"
        results['tasks.py'] = f"{header}\\n{tasks_code}"
        results['crew.py'] = f"{header}\\n{crew_code}"
        results['main.py'] = f"{header}\\n{main_code}"

        logger.info(
            "Successfully generated agents.py, tasks.py, crew.py, and main.py")
        return results

    except Exception as e:
        logger.error(f"Error generating CrewAI files: {str(e)}")

        # Create fallback files if generation fails
        results = {}

        # Get tool names for the fallback example
        tool_names = []
        for tool in tools:
            purpose = tool.get('purpose', '')
            tool_name = get_tool_name_from_purpose(purpose)
            if tool_name:
                # Make a reasonable class name guess
                class_name = tool_name.capitalize() + "Tool"
                if "MCP" not in class_name:
                    class_name = tool_name.capitalize() + "MCPTool"
                tool_names.append(class_name)

        # Default to generic names if we couldn't extract them
        if not tool_names or len(tool_names) < len(tools):
            tool_names = [f"Tool{i+1}" for i in range(len(tools))]

        # Create a better fallback with the actual tool names
        tool_instantiations = "\n".join(
            [f"{tool_name.lower()} = {tool_name}()" for tool_name in tool_names])
        first_half = len(tool_names) // 2 or 1
        agent_tools = ", ".join(
            [f"{tool_name.lower()}" for tool_name in tool_names[:first_half]])
        agent2_tools = ", ".join(
            [f"{tool_name.lower()}" for tool_name in tool_names[first_half:]])

        # Fallback agents.py
        results['agents.py'] = f"""#!/usr/bin/env python3
# Generated for: {user_request}
# This file is part of a CrewAI project using multiple MCP tools
# NOTE: This is a fallback implementation

from tools import *
from crewai import Agent

# Initialize all the tools
{tool_instantiations}

# Create agents that use the tools
research_agent = Agent(
    name="research_agent",
    role="Research and Data Collection Agent",
    goal="Gather and process information using external services",
    backstory="You are an expert at gathering and analyzing data from various sources. You know how to extract valuable insights and prepare them for action.",
    verbose=True,
    tools=[{agent_tools}]
)

action_agent = Agent(
    name="action_agent",
    role="Action and Integration Agent",
    goal="Execute actions and integrate data across services",
    backstory="You specialize in taking actionable steps based on information. You're skilled at integrating multiple services to achieve complex goals.",
    verbose=True,
    tools=[{agent2_tools}]
)
"""

        # Fallback tasks.py
        results['tasks.py'] = f"""#!/usr/bin/env python3
# Generated for: {user_request}
# This file is part of a CrewAI project using multiple MCP tools
# NOTE: This is a fallback implementation

from crewai import Task
from agents import research_agent, action_agent

# Create tasks for the agents - make sure to use ALL tools
research_task = Task(
    description="Research and collect all necessary information related to: {user_request}. Make sure to utilize all available tools to gather comprehensive data. Document your findings in a structured format.",
    agent=research_agent,
    expected_output="Comprehensive research results and collected data"
)

action_task = Task(
    description="Based on the research, take all necessary actions to fulfill: {user_request}. Integrate the data across services and execute the required operations. Document all actions taken and their results.",
    agent=action_agent,
    expected_output="Completed actions and integration results"
)
"""

        # Fallback crew.py
        results['crew.py'] = f"""#!/usr/bin/env python3
# Generated for: {user_request}
# This file is part of a CrewAI project using multiple MCP tools
# NOTE: This is a fallback implementation

from crewai import Crew, Process
from agents import research_agent, action_agent
from tasks import research_task, action_task
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a crew with the agents and tasks
crew = Crew(
    agents=[research_agent, action_agent],
    tasks=[research_task, action_task],
    process=Process.sequential,
    verbose=True
)

def main():
    print(f"Starting crew to handle request: {user_request}")
    result = crew.kickoff()
    print("\\nResults:")
    print(result)
    return result

if __name__ == "__main__":
    main()
"""

        logger.info("Created fallback CrewAI files due to generation error")
        return results


async def adaptive_code_synthesis_flow(
    user_request: str,
    openai_client: AsyncOpenAI,
    supabase_client: Any,
    base_dir: str = "output"
) -> Dict[str, Any]:
    """
    Implements the Adaptive Code Synthesis approach for MCP tool generation.
    This advanced flow analyzes user requirements deeply, understands reference code,
    and adapts it to meet specific needs.

    Args:
        user_request: User's natural language request
        openai_client: AsyncOpenAI client for embeddings and completions
        supabase_client: Supabase client for database access
        base_dir: Directory to use for output files (full path, not a base directory)

    Returns:
        Dictionary with results of the MCP tool integration
    """
    try:
        logger.info(
            f"Starting Adaptive Code Synthesis for request: {user_request[:100]}...")

        # Set up dependencies
        mcp_deps = MCPToolDeps(
            supabase=supabase_client,
            openai_client=openai_client
        )

        # Step 1: Perform deep user intent analysis
        user_requirements = await extract_structured_requirements(
            query=user_request,
            openai_client=openai_client
        )

        logger.info(f"Extracted user requirements: {len(user_requirements.primary_tools)} primary tools, " +
                    f"customization level: {user_requirements.customization_level}")

        # If no primary tools were identified, use fallback
        if not user_requirements.primary_tools:
            logger.info(
                "No specific tools identified, falling back to standard approach")
            return await mcp_tool_flow(
                user_request=user_request,
                openai_client=openai_client,
                supabase_client=supabase_client,
                base_dir=base_dir
            )

        # Step 2: Create a RunContext for the tools using the helper function
        context = create_mcp_context(
            deps=mcp_deps,
            prompt=user_request
        )

        # Step 3: Find relevant reference tools from Supabase with better error handling
        try:
            tool_search_result = await find_relevant_mcp_tools(
                ctx=context,
                user_query=user_request
            )
        except Exception as search_error:
            logger.error(
                f"Error finding relevant MCP tools: {str(search_error)}")
            # Fall back to standard approach if tool search fails
            logger.info(
                "Falling back to standard MCP tool flow due to tool search error")
            return await mcp_tool_flow(
                user_request=user_request,
                openai_client=openai_client,
                supabase_client=supabase_client,
                base_dir=base_dir
            )

        if not tool_search_result.get("found", False):
            logger.info("No relevant MCP tools found for this request")
            return {
                "success": False,
                "message": "No relevant MCP tools found. Using standard code generation.",
                "files": [],
                "output_dir": base_dir
            }

        # Extract the found tools
        tools = tool_search_result.get("tools", [])

        # Step 4: Rank tools based on requirement match with better error handling
        try:
            ranked_tools = await rank_tools_by_requirement_match(
                tools=tools,
                requirements=user_requirements
            )
        except Exception as rank_error:
            logger.error(f"Error ranking tools: {str(rank_error)}")
            ranked_tools = tools  # Use unranked tools as fallback

        if not ranked_tools:
            logger.info("No tools matched the requirements after ranking")
            return {
                "success": False,
                "message": "No tools matched the requirements after ranking. Using standard generation.",
                "files": [],
                "output_dir": base_dir
            }

        # Step 5: Set up output directory structure
        # Use the provided base_dir directly as the output directory
        output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"Using output directory for all generated files: {output_dir}")

        # Generate a README.md file with overview
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(
                f"# MCP Tool Integration for {user_requirements.primary_tools[0].tool_name if user_requirements.primary_tools else 'Custom Request'}\n\n")
            f.write(
                f"This project was generated by the MCP Adaptive Code Synthesis workflow.\n\n")
            f.write(f"## Overview\n\n")
            f.write(f"**Original Request:** {user_request}\n\n")
            f.write(
                f"**Customization Level:** {user_requirements.customization_level}\n\n")
            f.write(f"**Primary Tools:**\n\n")
            for tool in user_requirements.primary_tools:
                f.write(f"- {tool.tool_name}\n")

        files_written = [readme_path]

        # Create mcp_tools directory inside output dir
        mcp_tools_dir = os.path.join(output_dir, "mcp_tools")
        os.makedirs(mcp_tools_dir, exist_ok=True)

        # Create __init__.py in mcp_tools directory
        init_file = os.path.join(mcp_tools_dir, "__init__.py")
        with open(init_file, "w") as f:
            f.write("# MCP Tools Package\n")
            f.write("# Generated with Adaptive Code Synthesis\n\n")

            for tool in user_requirements.primary_tools:
                f.write(f"from .{tool.tool_name.lower()} import *\n")

        files_written.append(init_file)

        # Step 6: Process each tool requirement
        tool_directories = []
        processed_tools = []
        for tool_req in user_requirements.primary_tools:
            tool_name = tool_req.tool_name.lower().replace(" ", "_")

            logger.info(
                f"Processing tool: {tool_name}, importance: {tool_req.importance}")

            # Find the best matching reference tool for this requirement
            reference_tool = None
            for ranked_tool in ranked_tools:
                if tool_name in ranked_tool.get('purpose', '').lower():
                    reference_tool = ranked_tool
                    break

            if not reference_tool:
                logger.warning(
                    f"No reference tool found for {tool_name}, skipping")
                continue

            # Create tool directory
            tool_dir = os.path.join(mcp_tools_dir, tool_name)
            os.makedirs(tool_dir, exist_ok=True)
            tool_directories.append(tool_dir)

            # Create __init__.py in tool directory
            init_file = os.path.join(tool_dir, "__init__.py")
            with open(init_file, "w") as f:
                f.write(f"# {tool_name.capitalize()} MCP Tool\n")
                f.write(
                    f"# Purpose: {reference_tool.get('purpose', 'Integration tool')}\n")
                f.write(
                    f"# Customization level: {user_requirements.customization_level}\n")

            files_written.append(init_file)

            # Step 7: Analyze the reference tool code
            reference_code = reference_tool.get("tool_code", "")
            code_analysis = await analyze_tool_code(
                ctx=context,
                tool_code=reference_code,
                purpose=reference_tool.get("purpose", "")
            )

            # Step 8: Customize the tool implementation based on requirements
            # Extract custom features from the requirements
            custom_features = tool_req.custom_features

            # Convert the tool requirement to a dictionary for the API
            tool_requirements = {
                "tool_name": tool_req.tool_name,
                "importance": tool_req.importance,
                "custom_features": tool_req.custom_features,
                "integration_points": tool_req.integration_points,
                "authentication_requirements": tool_req.authentication_requirements,
                "specific_endpoints": tool_req.specific_endpoints,
                "customization_level": user_requirements.customization_level,
                "special_instructions": user_requirements.special_instructions
            }

            customized_code = await customize_tool_implementation(
                ctx=context,
                reference_code=reference_code,
                code_analysis=code_analysis,
                requirements=tool_requirements,
                custom_features=custom_features
            )

            # Step 9: Verify the customized tool
            verification = await verify_tool_integration(
                ctx=context,
                customized_code=customized_code,
                requirements=tool_requirements
            )

            # Step 10: Write the customized tool code
            tool_code_path = os.path.join(tool_dir, "tool.py")
            with open(tool_code_path, "w") as f:
                f.write(customized_code)
            files_written.append(tool_code_path)

            # Step 11: Generate a verification report
            verification_path = os.path.join(tool_dir, "verification.json")
            with open(verification_path, "w") as f:
                json.dump(verification, f, indent=2)
            files_written.append(verification_path)

            # Step 12: Write an example file
            example_code_path = os.path.join(tool_dir, "example.py")
            with open(example_code_path, "w") as f:
                f.write(f"# Example usage of the {tool_name} tool\n")
                f.write(f"# This example shows how to use the tool directly\n\n")
                f.write(f"from mcp_tools.{tool_name}.tool import *\n\n")
                f.write("def main():\n")
                f.write(f"    # Initialize the tool\n")
                f.write(
                    f"    # Example based on: {reference_tool.get('purpose', 'MCP Tool')}\n")
                f.write("    \n")
                f.write("    # TODO: Add initialization and usage code\n")
                custom_features_str = ", ".join(
                    custom_features) if custom_features else "None"
                f.write(
                    f"    # See customized features: {custom_features_str}\n")
                f.write("    \n")
                f.write(f'    print(f"Example usage of {tool_name} tool")\n')
                f.write("    \n")
                f.write('if __name__ == "__main__":\n')
                f.write("    main()\n")

            files_written.append(example_code_path)

            # Add to processed tools list
            processed_tools.append({
                "tool_name": tool_name,
                "importance": tool_req.importance,
                "reference_tool": reference_tool.get("purpose", ""),
                "customization_level": user_requirements.customization_level
            })

        # Step 13: Generate a top-level example file showing how to use all tools together
        if len(processed_tools) > 1:
            multi_example_path = os.path.join(
                output_dir, "multi_tool_example.py")
            with open(multi_example_path, "w") as f:
                f.write("# Example usage of multiple MCP tools together\n\n")
                for tool in processed_tools:
                    f.write(
                        f"from mcp_tools.{tool['tool_name']}.tool import *\n")
                f.write("\ndef main():\n")
                f.write("    # Initialize all tools\n")
                for tool in processed_tools:
                    f.write(f"    # Initialize {tool['tool_name']} tool\n")
                    f.write("    # TODO: Add initialization code\n")
                f.write("\n    # Example workflow\n")
                f.write("    # TODO: Add example workflow code\n\n")
                f.write('if __name__ == "__main__":\n')
                f.write("    main()\n")

            files_written.append(multi_example_path)

        # Step 14: Generate requirements.txt
        requirements_path = os.path.join(output_dir, "requirements.txt")
        with open(requirements_path, "w") as f:
            f.write("# Requirements for MCP tool integration\n")
            f.write("crewai>=0.28.0\n")
            f.write("langchain-core>=0.1.0\n")
            f.write("pydantic>=2.0.0\n")
            f.write("python-dotenv>=1.0.0\n")

            # Add tool-specific requirements
            for tool in processed_tools:
                if "github" in tool["tool_name"]:
                    f.write("PyGithub>=2.0.0\n")
                elif "spotify" in tool["tool_name"]:
                    f.write("spotipy>=2.0.0\n")
                elif "youtube" in tool["tool_name"]:
                    f.write("google-api-python-client>=2.0.0\n")

        files_written.append(requirements_path)

        # Create .env file with placeholders for necessary credentials
        env_file_path = os.path.join(output_dir, ".env.example")
        with open(env_file_path, "w") as f:
            f.write("# Environment variables for MCP tool integration\n")
            f.write("# Rename this file to .env and add your actual credentials\n\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")

            # Add tool-specific environment variables
            for tool in processed_tools:
                if "github" in tool["tool_name"]:
                    f.write("# GitHub credentials\n")
                    f.write("GITHUB_TOKEN=your_github_token_here\n\n")
                elif "spotify" in tool["tool_name"]:
                    f.write("# Spotify credentials\n")
                    f.write("SPOTIFY_CLIENT_ID=your_spotify_client_id_here\n")
                    f.write(
                        "SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here\n\n")
                elif "youtube" in tool["tool_name"]:
                    f.write("# YouTube credentials\n")
                    f.write("YOUTUBE_API_KEY=your_youtube_api_key_here\n\n")

        files_written.append(env_file_path)

        # Generate a comprehensive README explaining the tool architecture
        readme_context = create_mcp_context(
            deps=mcp_deps,
            prompt=f"Generate README for {user_request}"
        )

        # Get a summary of what each tool does
        tools_summary = []
        for tool in processed_tools:
            tools_summary.append({
                "name": tool["tool_name"],
                "purpose": tool["reference_tool"],
                "importance": tool["importance"]
            })

        # Generate a comprehensive README
        readme_content = await generate_mcp_integration_readme(
            ctx=readme_context,
            mcp_tool_data={
                "purposes": [tool.get("tool_name", "") for tool in processed_tools],
                "tools": tools,
                "processed_tools": tools_summary,
                "user_requirements": user_requirements.dict(),
                "approach": "Adaptive Code Synthesis"
            },
            generated_files=[os.path.basename(file) for file in files_written]
        )

        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        files_written.append(readme_file)

        # Return results
        return {
            "success": True,
            "message": f"Successfully generated {len(processed_tools)} customized MCP tool(s) using Adaptive Code Synthesis",
            "files": files_written,
            "output_dir": output_dir,
            "tool_info": {
                "tool_count": len(processed_tools),
                "tool_names": [tool.get("tool_name", "") for tool in processed_tools],
                "directory": output_dir,
                "synthesis_approach": "Adaptive Code Synthesis",
                "customization_level": user_requirements.customization_level
            }
        }

    except Exception as e:
        logger.error(
            f"Error in Adaptive Code Synthesis flow: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error in Adaptive Code Synthesis: {str(e)}",
            "files": [],
            "output_dir": base_dir
        }


def get_main_tool_type(structured_requirements: UserRequirements) -> str:
    """
    Extract the main tool type from structured requirements for naming purposes.

    Args:
        structured_requirements: Parsed requirements with primary tools

    Returns:
        String representing the main tool type (github, spotify, youtube, etc.)
    """
    main_tool_type = "generic"
    if not structured_requirements.primary_tools:
        return "generic"

    # Log the primary tools to help with debugging
    primary_tool_names = [
        pt.tool_name for pt in structured_requirements.primary_tools]
    logger.info(
        f"Primary tool names in get_main_tool_type: {primary_tool_names}")

    main_tool = structured_requirements.primary_tools[0].tool_name.lower()
    if "github" in main_tool:
        main_tool_type = "github"
    elif "spotify" in main_tool:
        main_tool_type = "spotify"
    elif "youtube" in main_tool:
        main_tool_type = "youtube"
    elif "serper" in main_tool or "search" in main_tool:
        main_tool_type = "search"
    elif "stock" in main_tool or "finance" in main_tool:
        main_tool_type = "stock"
    else:
        # Use first word of tool name
        main_tool_type = main_tool.split()[0]

    logger.info(f"Identified main tool type: {main_tool_type}")
    return main_tool_type


async def combined_adaptive_flow(
    user_request: str,
    openai_client: AsyncOpenAI,
    supabase_client: Any,
    base_dir: str = "output",
    use_adaptive_synthesis: bool = True
) -> Dict[str, Any]:
    """
    Combined adaptive flow for MCP tool integration.

    This flow combines:
    1. MCP tool selection
    2. MCP tool integration
    3. Template-based code generation

    Args:
        user_request: User's request or query
        openai_client: OpenAI client
        supabase_client: Supabase client
        base_dir: Base directory for output
        use_adaptive_synthesis: Whether to use adaptive synthesis

    Returns:
        Dictionary containing the results
    """
    try:
        logger.info(
            f"Starting combined adaptive flow for request: {user_request[:100]}...")

        # Create a context for MCP selector
        deps = MCPToolDeps(
            supabase=supabase_client,
            openai_client=openai_client
        )
        ctx = create_mcp_context(deps)

        # Step 1: Perform deep requirement analysis
        logger.info("Performing deep requirement analysis for query")
        structured_requirements = await get_crewai_tool_requirements(
            user_request,
            openai_client
        )

        # Extract tools information and log them
        tools = structured_requirements.get('tools', [])
        logger.info(
            f"Successfully extracted requirements with {len(tools)} tools")

        # Step 2: Determine the main tool type
        main_tool_type = "generic"
        if tools:
            # Get the name of the first tool as the main tool type
            main_tool = tools[0].get('name', '').lower()
            logger.info(f"Primary tool name: {main_tool}")

            if "github" in main_tool:
                main_tool_type = "github"
            elif "spotify" in main_tool:
                main_tool_type = "spotify"
            elif "youtube" in main_tool:
                main_tool_type = "youtube"
            elif "serper" in main_tool or "search" in main_tool:
                main_tool_type = "search"
            elif "stock" in main_tool or "finance" in main_tool:
                main_tool_type = "stock"
            else:
                # Use first word of tool name
                main_tool_type = main_tool.split()[0]

        logger.info(f"Identified main tool type: {main_tool_type}")

        # Step 3: Find the most relevant MCP tools
        tools_result = await find_relevant_mcp_tools(ctx, user_request)

        if not tools_result.get("tools", []):
            logger.warning("No MCP tools found for the request")
            return {"error": "No suitable MCP tools found for your request"}

        tools_data = {
            "query": user_request,
            "tools": tools_result.get("tools", []),
            "purpose": f"MCP Tools for {user_request[:50]}..."
        }

        # Log the structured requirements
        if tools:
            logger.info(
                f"Successfully extracted structured requirements with {len(tools)} tools")
            logger.info(f"Main tool type identified: {main_tool_type}")

        # Ensure output directory exists
        os.makedirs(base_dir, exist_ok=True)
        logger.info(
            f"Using output directory for all generated files: {base_dir}")

        # Create a dated directory for the output
        project_type = main_tool_type if main_tool_type else "multi_tool"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = os.path.join(
            base_dir, f"{project_type}_tool_{timestamp}")
        os.makedirs(project_dir, exist_ok=True)
        logger.info(f"Created project directory: {project_dir}")
        
        # Create an error_list.md file that will document known common errors and their fixes
        error_list_path = os.path.join(project_dir, "error_list.md")
        with open(error_list_path, "w") as f:
            f.write("# Common Errors in tools.py and Their Fixes\n\n")
            f.write("This file documents common errors that can occur in tools.py files and their automatic fixes.\n\n")
            
            f.write("## Class naming inconsistency\n\n")
            f.write("**Error Type:** class_naming_inconsistency\n\n")
            f.write("**Solution:** Detected class naming inconsistency. Adding alias class for backward compatibility.\n\n")
            
            f.write("## Missing 'Tool' suffix in class names\n\n")
            f.write("**Error Type:** missing_tool_suffix\n\n")
            f.write("**Solution:** All tool classes should have the 'Tool' suffix for consistency.\n\n")
            
            f.write("## Inconsistent method names across files\n\n")
            f.write("**Error Type:** inconsistent_method_names\n\n")
            f.write("**Solution:** Method names should be consistent across all files (e.g., use 'get_transcript' instead of 'extract_transcript').\n\n")
            
            f.write("## Missing alias class for backward compatibility\n\n")
            f.write("**Error Type:** missing_alias_class\n\n")
            f.write("**Solution:** Adding alias class with MCPTool suffix for backward compatibility.\n\n")
            
            f.write("## Applied Fixes in this Project\n\n")
            f.write("This section will be updated during the generation process if any fixes are applied.\n")

        # Generate the tools.py file
        tools_py = await generate_tools_py(ctx, tools_data, project_dir)

        # Extract tool class names from the generated tools.py
        if os.path.exists(os.path.join(project_dir, "tools.py")):
            with open(os.path.join(project_dir, "tools.py"), "r") as f:
                tools_py_content = f.read()
                tool_class_names = extract_class_names_from_tools(
                    tools_py_content)
                logger.info(f"Extracted tool class names: {tool_class_names}")
        else:
            tool_class_names = []
            logger.warning(
                "tools.py not found, cannot extract tool class names")

        # Generate requirements.txt
        requirements_txt = generate_requirements(tools_data, tool_class_names)
        with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
            f.write(requirements_txt)
        logger.info(
            f"Generated requirements.txt at {os.path.join(project_dir, 'requirements.txt')}")

        # Use the template integration module to generate files
        from .mcp_tools.mcp_template_integration import generate_from_template

        # Use a more capable model for template generation
        template_model = os.getenv("TEMPLATE_MODEL", "gpt-4o-mini")
        logger.info(f"Using model {template_model} for template generation")

        # Add tool class names and detected tool types to tools_data
        tools_data["tool_class_names"] = tool_class_names
        if main_tool_type:
            tools_data["detected_tool_types"] = [main_tool_type]

        # Generate from template directly into the project directory
        template_results = await generate_from_template(
            user_request,
            tools_data,
            tool_class_names,
            project_dir,
            supabase_client,
            openai_client,
            template_model
        )

        # Update the error_list.md file with info about applied fixes
        try:
            from .mcp_tools.mcp_tool_coder import TOOLS_PY_COMMON_ERRORS
            error_list_path = os.path.join(project_dir, "error_list.md")
            if os.path.exists(error_list_path):
                with open(error_list_path, "a") as f:
                    f.write("\n\n## Class Naming Validation\n\n")
                    f.write("The tools.py file was validated against common errors and the following validations were performed:\n\n")
                    for error in TOOLS_PY_COMMON_ERRORS:
                        f.write(f"- {error['description']}: {error['error_message']}\n")
                    
                    # Add special note for YouTube tools
                    if "youtube" in main_tool_type.lower():
                        f.write("\n### Special YouTube Tool Validation\n\n")
                        f.write("For YouTube tools, we specifically check for both `YouTubeTranscriptTool` and `YouTubeTranscriptMCPTool` " +
                                "classes and ensure both exist with one inheriting from the other for complete backward compatibility.\n")
        except Exception as e:
            logger.warning(f"Could not update error_list.md with validation details: {e}")

        # If template generation was successful
        if template_results and len(template_results) > 0:
            logger.info(
                "Template generation successful! Created CrewAI project files.")

            # Generate a README.md file
            readme_content = await generate_crewai_readme(
                ctx,
                tools_data,
                list(template_results.keys())
            )

            with open(os.path.join(project_dir, "README.md"), "w") as f:
                f.write(readme_content)
            logger.info(
                f"Generated README.md at {os.path.join(project_dir, 'README.md')}")

            logger.info(
                f"Successfully generated complete CrewAI project in {project_dir}")
            return {
                "message": f"Successfully generated complete CrewAI project in {project_dir}",
                "project_dir": project_dir,
                "template_used": True
            }
        else:
            # Template generation failed - return error
            logger.error(
                "Template generation failed and no fallback is available")
            return {
                "error": "Template generation failed and no fallback is available",
                "project_dir": project_dir
            }

    except Exception as e:
        logger.error(f"Error in combined adaptive flow: {e}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}


async def generate_multi_tool_reasoning(
    user_request: str,
    structured_requirements: UserRequirements,
    openai_client: AsyncOpenAI
) -> str:
    """
    Generate enhanced reasoning for multi-tool scenarios to guide the code generation process.

    Args:
        user_request: Original user request
        structured_requirements: Structured requirements extracted from the request
        openai_client: AsyncOpenAI client

    Returns:
        String containing reasoning about the multi-tool implementation
    """
    try:
        prompt = f"""
        Analyze this user request for a multi-tool CrewAI project and provide detailed reasoning 
        about how the tools should be integrated and work together.
        
        USER REQUEST: "{user_request}"
        
        PRIMARY TOOLS: {', '.join([t.tool_name for t in structured_requirements.primary_tools])}
        
        WORKFLOW DESCRIPTION: "{structured_requirements.workflow_description}"
        
        Please provide a detailed analysis covering:
        1. How the tools should integrate and share data
        2. What each tool's primary responsibility should be
        3. How the workflow between tools should be structured
        4. What agent roles are needed to effectively use these tools
        5. Potential challenges in the integration and how to address them
        
        Focus on technical requirements and architecture decisions, not implementation details.
        """

        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2
        )

        reasoning = response.choices[0].message.content
        logger.info(f"Generated multi-tool reasoning ({len(reasoning)} chars)")
        return reasoning

    except Exception as e:
        logger.error(f"Error generating multi-tool reasoning: {str(e)}")
        return "Error generating reasoning for multi-tool scenario."


async def generate_architecture_plan(
    user_request: str,
    structured_requirements: UserRequirements,
    openai_client: AsyncOpenAI
) -> str:
    """
    Generate a high-level architecture plan for the multi-tool CrewAI project.

    Args:
        user_request: Original user request
        structured_requirements: Structured requirements extracted from the request
        openai_client: AsyncOpenAI client

    Returns:
        String containing a high-level architecture plan
    """
    try:
        prompt = f"""
        Create a high-level architecture plan for a CrewAI project that integrates 
        multiple tools based on this user request.
        
        USER REQUEST: "{user_request}"
        
        PRIMARY TOOLS: {', '.join([t.tool_name for t in structured_requirements.primary_tools])}
        INTEGRATION PATTERN: {structured_requirements.integration_pattern}
        
        Your architecture plan should include:
        1. Overall system architecture (components and their relationships)
        2. Data flow diagram (how data moves between components)
        3. Agent-tool assignment strategy
        4. Process flow for the CrewAI workflow
        5. Key interfaces between components
        
        Format as a concise but comprehensive architecture plan.
        """

        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2
        )

        plan = response.choices[0].message.content
        logger.info(f"Generated architecture plan ({len(plan)} chars)")
        return plan

    except Exception as e:
        logger.error(f"Error generating architecture plan: {str(e)}")
        return "Error generating architecture plan for multi-tool scenario."

__all__ = [
    'mcp_tool_flow',
    'adaptive_code_synthesis_flow',
    'combined_adaptive_flow'
]
