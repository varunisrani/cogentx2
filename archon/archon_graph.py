from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client
import logfire
import os
import sys

# Import the message classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps
from archon.advisor_agent import advisor_agent, AdvisorDeps
from archon.refiner_agents.prompt_refiner_agent import prompt_refiner_agent
from archon.refiner_agents.tools_refiner_agent import tools_refiner_agent, ToolsRefinerDeps
from archon.refiner_agents.agent_refiner_agent import agent_refiner_agent, AgentRefinerDeps
from archon.agent_tools import list_documentation_pages_tool
from utils.utils import get_env_var, get_clients, ensure_workbench_dir

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

# Always use OpenAI models
is_openai = True
is_anthropic = False

reasoner_llm_model_name = get_env_var('REASONER_MODEL') or 'o3-mini'
reasoner_llm_model = OpenAIModel(reasoner_llm_model_name)

reasoner = Agent(  
    reasoner_llm_model,
    system_prompt='You are an expert at coding AI agents with Pydantic AI and defining the scope for doing so.',  
)

primary_llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
primary_llm_model = OpenAIModel(primary_llm_model_name)

router_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding the AI agent.',  
)

end_conversation_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to end a conversation for creating an AI agent by giving instructions for how to execute the agent and they saying a nice goodbye to the user.',  
)

# Initialize clients
embedding_client, supabase = get_clients()

# Define state schema
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]

    scope: str
    advisor_output: str
    file_list: List[str]

    refined_prompt: str
    refined_tools: str
    refined_agent: str

# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # First, get the documentation pages so the reasoner can decide which ones are necessary
    documentation_pages = await list_documentation_pages_tool(supabase)
    documentation_pages_str = "\n".join(documentation_pages)

    # Then, use the reasoner to define the scope
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

    {documentation_pages_str}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    result = await reasoner.run(prompt)
    scope = result.data
    
    # Ensure workbench directory exists
    workbench_dir = ensure_workbench_dir()
    
    # Create scope.md file path
    scope_path = os.path.join(workbench_dir, "scope.md")
    
    try:
        with open(scope_path, "w", encoding="utf-8") as f:
            f.write(scope)
        print(f"Successfully wrote scope document to: {scope_path}")
    except Exception as e:
        print(f"Error writing scope document: {e}")
        # Don't fail, continue with the scope in memory
    
    return {"scope": scope}

# Advisor agent - create a starting point based on examples and prebuilt tools/MCP servers
async def advisor_with_examples(state: AgentState):
    # Get the directory one level up from the current file (archon_graph.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # The agent-resources folder is adjacent to the parent folder of archon_graph.py
    agent_resources_dir = os.path.join(parent_dir, "agent-resources")
    
    # Get a list of all files in the agent-resources directory and its subdirectories
    file_list = []
    
    for root, dirs, files in os.walk(agent_resources_dir):
        for file in files:
            # Get the full path to the file
            file_path = os.path.join(root, file)
            # Use the full path instead of relative path
            file_list.append(file_path)
    
    # Analyze the user message to check if they need an agent with multiple MCP tools
    user_message = state['latest_user_message'].lower()
    multi_mcp_example_note = ""
    
    # Check if the request might involve multiple MCP tools or template merging
    multi_mcp_keywords = [
        "multiple", "combine", "both", "together", "and", 
        "with", "plus", "as well as", "along with",
        "multi-tool", "multi tool", "multiple tools",
        "multiple api", "multiple apis", "two tools",
        "several tools", "integrate", "integration"
    ]
    
    # Important MCP service keywords to detect
    mcp_services = {
        "search": ["search", "web", "serper", "google", "brave", "find information"],
        "spotify": ["spotify", "music", "song", "playlist", "track", "album", "artist"],
        "github": ["github", "repository", "repo", "git", "pull request", "issue", "code"],
        "gmail": ["gmail", "email", "message", "inbox", "send mail"],
        "firecrawl": ["firecrawl", "crawl", "website", "scrape", "page content"],
        "maps": ["maps", "location", "directions", "places", "navigate"]
    }
    
    # Count how many different MCP services are mentioned
    mentioned_services = []
    for service, keywords in mcp_services.items():
        if any(keyword in user_message for keyword in keywords):
            mentioned_services.append(service)
    
    multi_tool_detected = False
    
    # Check if multiple services are mentioned
    if len(mentioned_services) >= 2:
        multi_tool_detected = True
        print(f"[INFO] Multiple MCP services detected: {', '.join(mentioned_services)}")
        
        # Add information about the detected services to the state
        multiple_services_info = {
            "detected": True,
            "services": mentioned_services,
            "force_template_merge": True
        }
        
        # Store directly in the state dictionary
        state["multiple_services"] = multiple_services_info
        print(f"[INFO] Added multi-service information to state: {mentioned_services}")
    
    # Also check for multi-tool keywords
    if any(keyword in user_message for keyword in multi_mcp_keywords):
        print(f"[INFO] Multi-tool keywords detected in message: {user_message}")
        multi_tool_detected = True
        
        # If we haven't already added the multi-service info
        if "multiple_services" not in state or len(mentioned_services) < 2:
            state["multiple_services"] = {
                "detected": True,
                "services": ["general_multi_service"],
                "force_template_merge": True
            }
            print(f"[INFO] Added general multi-service information to state")
    
    print(f"[INFO] Multi-tool detected: {multi_tool_detected}")
    
    # If multiple tools are needed, explain the merging approach
    if multi_tool_detected:
        multi_mcp_example_note = """
        Note: The user is requesting an agent that combines multiple capabilities.
        This requires combining multiple templates or using a multi-MCP example.
        
        A thorough implementation MUST address these critical integration points:
        
        1. Configuration Integration:
           - Ensure Config class includes ALL required API keys for each service
           - Validate ALL required environment variables
           - Update .env.example with ALL needed variables
        
        2. Agent Structure:
           - Initialize ALL MCP servers properly
           - Create a combined system prompt covering ALL capabilities
           - Ensure proper dependency classes and imports
        
        3. Tool Integration:
           - Implement functions for ALL services
           - Maintain clear naming conventions to avoid conflicts
           - Properly handle authentication for each service
        
        4. Main Application Flow:
           - Create a unified interface for ALL capabilities
           - Add clear command handling between services
           - Implement proper error handling for each service
        
        5. DETAILED TEMPLATE MERGING INSTRUCTIONS:
           - File-by-file Integration:
             • agents.py: Combine ALL imports, dependency classes, tools from both templates
             • models.py: Merge ALL Config class fields and model classes
             • tools.py: Include ALL utility functions from both templates
             • main.py: Initialize ALL MCP servers and create a unified interface
             • mcp.json: Include ALL server configurations
           
           - System Prompt Integration:
             • Explicitly describe capabilities from ALL services
             • Provide guidance on when to use each service
             • Include examples for ALL capabilities
        """
        
        # Add service-specific implementation suggestions
        if "search" in mentioned_services and "spotify" in mentioned_services:
            multi_mcp_example_note += """
           - For Search + Spotify: Use the serper_spotify_agent example which shows how to combine 
             web search and music control in a single agent. This template has been specifically 
             created to demonstrate multi-MCP integration.
        """
        
        if "github" in mentioned_services:
            multi_mcp_example_note += """
           - For GitHub integration: Use the github_agent template as a base and ensure proper
             GitHub authentication and repository management functions
        """
        
        if "gmail" in mentioned_services:
            multi_mcp_example_note += """
           - For Gmail integration: Use the gmail_agent template and ensure proper
             email handling capabilities
        """
        
        # Add general implementation approach
        multi_mcp_example_note += """
        
        IMPROVED TEMPLATE MERGING WORKFLOW:
        
        1. Search for relevant templates:
           - Use search_agent_templates to find templates for each required capability
           - Look for templates that specifically match your requirements
        
        2. View BOTH templates BEFORE merging:
           - Use display_template_files to examine EACH template individually
           - Pay special attention to:
             * Config class fields needed for authentication
             * Dependency class fields required by each service  
             * MCP server initialization code
             * Tool functions for each capability
        
        3. Merging approach:
           a. DO NOT just take code from one template and ignore the other
           b. PROPERLY COMBINE code from BOTH templates:
              - Keep ALL imports from BOTH templates (remove duplicates)
              - Merge dependency classes with ALL fields from BOTH templates
              - Include ALL tool functions from BOTH templates
              - Create a system prompt that covers ALL capabilities
              - Initialize ALL MCP servers in main.py
              - Include ALL Config fields in models.py
           c. After merging, carefully review the combined code
        
        4. Post-merge validation:
           - Verify ALL imports are included (no undefined references)
           - Check that ALL fields are in the Config class
           - Confirm ALL MCP servers are properly initialized
           - Ensure the system prompt covers ALL capabilities
           - Test that ALL tools/functions work correctly
        """
    
    # Then, prompt the advisor with the list of files it can use for examples and tools
    deps = AdvisorDeps(
        file_list=file_list,
        supabase=supabase,
        embedding_client=embedding_client
    )
    
    # Add the multi-MCP note to the user message if applicable
    if multi_mcp_example_note:
        modified_message = f"{state['latest_user_message']}\n\n{multi_mcp_example_note}"
        result = await advisor_agent.run(modified_message, deps=deps)
    else:
        result = await advisor_agent.run(state['latest_user_message'], deps=deps)
        
    advisor_output = result.data
    
    return {"file_list": file_list, "advisor_output": advisor_output}

# Coding Node with Feedback Handling
async def coder_agent(state: AgentState, writer):    
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        reasoner_output=state.get('scope', ''),
        advisor_output=state.get('advisor_output', '')
    )
    
    print(f"\n[DEEP LOG] Initialized PydanticAIDeps with supabase client: {supabase is not None}")
    print(f"\n[DEEP LOG] Initialized PydanticAIDeps with embedding client: {embedding_client is not None}")

    # Check if multiple services were detected and force template merging
    if 'multiple_services' in state and state['multiple_services'].get('detected', False):
        print(f"\n[DEEP LOG] Multiple services detected: {state['multiple_services']['services']}")
        print(f"\n[DEEP LOG] FORCING TEMPLATE MERGING instead of using pre-built templates")
        
        # Add explicit instruction to the prompt to merge templates
        services_str = ", ".join(state['multiple_services']['services'])
        merging_instruction = f"""
        IMPORTANT INSTRUCTION: This is a multi-service request combining {services_str}.
        
        DO NOT use a pre-built combined template (like SpotifyGitHubAgent - ID 5).
        
        INSTEAD, YOU MUST:
        1. Find separate templates for each service (e.g., spotify_agent - ID 4, github_agent - ID 2)
        2. Use the merge_agent_templates function to combine them
        3. Ensure ALL functionality from BOTH templates is included
        
        This is CRITICAL to ensure proper integration of all services.
        """
        
        if 'latest_user_message' in state:
            state['latest_user_message'] = state['latest_user_message'] + "\n\n" + merging_instruction

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # The prompt either needs to be the user message (initial agent request or feedback)
    # or the refined prompt/tools/agent if we are in that stage of the agent creation process
    if 'refined_prompt' in state and state['refined_prompt']:
        prompt = f"""
        I need you to refine the agent you created. 
        
        Here is the refined prompt:\n
        {state['refined_prompt']}\n\n

        Here are the refined tools:\n
        {state['refined_tools']}\n

        And finally, here are the changes to the agent definition to make if any:\n
        {state['refined_agent']}\n\n

        Output any changes necessary to the agent code based on these refinements.
        """
    else:
        prompt = state['latest_user_message']

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await pydantic_ai_coder.run(prompt, deps=deps, message_history=message_history)
        writer(result.data)
    else:
        async with pydantic_ai_coder.run_stream(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    # print(ModelMessagesTypeAdapter.validate_json(result.new_messages_json()))

    # Add the new conversation history (including tool calls)
    # Reset the refined properties in case they were just used to refine the agent
    return {
        "messages": [result.new_messages_json()],
        "refined_prompt": "",
        "refined_tools": "",
        "refined_agent": ""
    }

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# Determine if the user is finished creating their AI agent or not
async def route_user_message(state: AgentState):
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent and gave feedback, respond with just the text "coder_agent".
    If the user asks specifically to "refine" the agent, respond with just the text "refine".
    """

    result = await router_agent.run(prompt)
    
    if result.data == "finish_conversation": return "finish_conversation"
    if result.data == "refine": return ["refine_prompt", "refine_tools", "refine_agent"]
    return "coder_agent"

# Refines the prompt for the AI agent
async def refine_prompt(state: AgentState):
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the prompt for the agent."

    # Run the agent to refine the prompt for the agent being created
    result = await prompt_refiner_agent.run(prompt, message_history=message_history)

    return {"refined_prompt": result.data}

# Refines the tools for the AI agent
async def refine_tools(state: AgentState):
    # Prepare dependencies
    deps = ToolsRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        file_list=state['file_list']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the tools for the agent."

    # Run the agent to refine the tools for the agent being created
    result = await tools_refiner_agent.run(prompt, deps=deps, message_history=message_history)

    return {"refined_tools": result.data}

# Refines the defintion for the AI agent
async def refine_agent(state: AgentState):
    # Prepare dependencies
    deps = AgentRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the agent definition."

    # Run the agent to refine the definition for the agent being created
    result = await agent_refiner_agent.run(prompt, deps=deps, message_history=message_history)

    return {"refined_agent": result.data}

# End of conversation agent to give instructions for executing the agent
async def finish_conversation(state: AgentState, writer):    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await end_conversation_agent.run(state['latest_user_message'], message_history= message_history)
        writer(result.data)   
    else: 
        async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history= message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return {"messages": [result.new_messages_json()]}        

# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("advisor_with_examples", advisor_with_examples)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("refine_prompt", refine_prompt)
builder.add_node("refine_tools", refine_tools)
builder.add_node("refine_agent", refine_agent)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge(START, "advisor_with_examples")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("advisor_with_examples", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    ["coder_agent", "finish_conversation", "refine_prompt", "refine_tools", "refine_agent"]
)
builder.add_edge("refine_prompt", "coder_agent")
builder.add_edge("refine_tools", "coder_agent")
builder.add_edge("refine_agent", "coder_agent")
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)