import streamlit as st
import asyncio
import uuid
import os
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from openai import AsyncOpenAI
from supabase import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chat')

# Lazy imports to avoid circular dependencies
_agentic_flow = None
_mcp_tool_flow = None
_openai_client = None
_supabase_client = None

def _get_agentic_flow():
    """Lazy import of agentic_flow to avoid circular imports."""
    global _agentic_flow
    if _agentic_flow is None:
        try:
            from archon.archon_graph import agentic_flow
            _agentic_flow = agentic_flow
        except ImportError:
            logger.warning("Failed to import agentic_flow")
            return None
    return _agentic_flow

def _get_mcp_tool_flow():
    """Lazy import of MCP tool flow to avoid circular imports."""
    global _mcp_tool_flow
    if _mcp_tool_flow is None:
        try:
            from archon.mcp_tools.mcp_tool_graph import combined_adaptive_flow
            _mcp_tool_flow = combined_adaptive_flow
        except ImportError:
            logger.warning("Failed to import MCP tool flow")
            return None
    return _mcp_tool_flow

def initialize_clients(openai_client: AsyncOpenAI = None, supabase_client: Client = None):
    """Initialize OpenAI and Supabase clients."""
    global _openai_client, _supabase_client
    _openai_client = openai_client
    _supabase_client = supabase_client

@st.cache_resource
def get_thread_id() -> str:
    """Generate and cache a unique thread ID for the session."""
    return str(uuid.uuid4())

async def process_mcp_request(user_request: str, tool_mode: str = "single", output_dir: str = "output") -> str:
    """Process a user request through the MCP tool flow."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For multiple tools mode, enhance the request
        if tool_mode == "multiple":
            # Create a more specific directory
            timestamp = asyncio.get_event_loop().time()
            output_dir = os.path.join(output_dir, f"multi_tools_{int(timestamp)}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Enhance the user request for multiple tools
            enhanced_request = f"""Create multiple integrated MCP tools that work together: {user_request}
            
This should generate MULTIPLE tools that can be used together in a CrewAI setup, ensuring they're properly integrated.
"""
            user_request = enhanced_request

        # Get MCP tool flow
        mcp_flow = _get_mcp_tool_flow()
        if not mcp_flow:
            return "❌ MCP tools are not available. Please check your installation."

        # Check if clients are available
        if not _openai_client or not _supabase_client:
            return "❌ OpenAI or Supabase clients are not initialized. Please check your configuration."

        # Process the request through MCP tool flow
        result = await mcp_flow(
            user_request=user_request,
            openai_client=_openai_client,
            supabase_client=_supabase_client,
            base_dir=output_dir,
            use_adaptive_synthesis=True
        )

        if result.get("success", False):
            # Format the response with file information
            files = result.get("files", [])
            tool_info = result.get("tool_info", {})
            
            response = f"""✅ **Successfully processed MCP request**\n\n"""
            
            if tool_info:
                response += f"**Tool Info:**\n"
                if "purpose" in tool_info:
                    response += f"- Purpose: {tool_info['purpose']}\n"
                if "tool_count" in tool_info:
                    response += f"- Tools Generated: {tool_info['tool_count']}\n"
            
            if files:
                response += f"\n**Generated Files:**\n"
                for file in files:
                    response += f"- `{os.path.basename(file)}`\n"
            
            response += f"\nAll files are saved in: `{output_dir}`"
            return response
        else:
            return f"❌ **Error processing request**\n\n{result.get('message', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}", exc_info=True)
        return f"❌ **Error processing your request**\n\n{str(e)}"

async def run_agent_with_streaming(user_input: str, thread_id: str, agent_mode: str = "standard") -> AsyncGenerator[str, None]:
    """Run the agent with streaming text for the user_input prompt."""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        # Check which mode to use (standard or MCP)
        if agent_mode == "mcp":
            try:
                mcp_response = await process_mcp_request(user_input, "single")
                yield mcp_response
                return
            except Exception as e:
                error_message = f"Error in MCP mode: {str(e)}"
                yield error_message
                return

        # Standard mode using agentic_flow
        agentic_flow = _get_agentic_flow()
        if not agentic_flow:
            yield "Standard mode is not available. Please check if archon components are installed."
            return

        # First message from user
        if len(st.session_state.messages) == 1:
            async for msg in agentic_flow.astream(
                    {"latest_user_message": user_input}, config, stream_mode="custom"
                ):
                    yield msg
        # Continue the conversation
        else:
            async for msg in agentic_flow.astream(
                {"latest_user_message": user_input, "resume": user_input}, config, stream_mode="custom"
            ):
                yield msg
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        yield error_message

async def display_chat_interface():
    """Display the chat interface for talking to the Agent Builder."""
    st.markdown("""
    <h1>Cogentx Agent Chat</h1>
    <p style="font-size: 1.2rem; margin-bottom: 2rem;">Create intelligent agents that help with various tasks using natural language instructions.</p>
    """, unsafe_allow_html=True)

    # Create a container at the top for mode selection
    with st.container(border=True):
        st.markdown("### Agent Creation Mode")
        st.markdown("Select how you want to create your agent:")
        
        # Initialize mode selection in session state if not present
        if "agent_mode" not in st.session_state:
            st.session_state.agent_mode = "standard"
        
        # Create two columns for mode selection cards
        mode_col1, mode_col2 = st.columns(2)
        
        with mode_col1:
            standard_card_style = "background-color: " + ("#E3F2FD" if st.session_state.agent_mode == "standard" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.agent_mode == "standard" else "#e0e0e0")
            
            st.markdown(f"""
            <div style="{standard_card_style}" onclick="document.getElementById('standard_mode_btn').click()">
                <div style="font-size: 32px; margin-bottom: 10px;">🤖</div>
                <h3 style="margin: 0; margin-bottom: 5px;">Standard Mode</h3>
                <p style="margin: 0; color: #666;">Create agents using templates and standard approach</p>
            </div>
            """, unsafe_allow_html=True)
            
            standard_mode = st.button("🤖 Standard Mode", key="standard_mode_btn", use_container_width=True)
            if standard_mode:
                st.session_state.agent_mode = "standard"
                st.rerun()
        
        with mode_col2:
            mcp_card_style = "background-color: " + ("#E3F2FD" if st.session_state.agent_mode == "mcp" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.agent_mode == "mcp" else "#e0e0e0")
            
            st.markdown(f"""
            <div style="{mcp_card_style}" onclick="document.getElementById('mcp_mode_btn').click()">
                <div style="font-size: 32px; margin-bottom: 10px;">🔌</div>
                <h3 style="margin: 0; margin-bottom: 5px;">MCP Mode</h3>
                <p style="margin: 0; color: #666;">Create agents with advanced MCP tool integrations</p>
            </div>
            """, unsafe_allow_html=True)
            
            mcp_mode = st.button("🔌 MCP Mode", key="mcp_mode_btn", use_container_width=True)
            if mcp_mode:
                st.session_state.agent_mode = "mcp"
                st.rerun()
    
    # Display current mode info
    st.markdown(f"""
    <div style="background-color: #E3F2FD; border-left: 4px solid #1E88E5; padding: 12px; border-radius: 4px; margin: 16px 0;">
        <p style="margin: 0;"><strong>Currently in {st.session_state.agent_mode.title()} Mode</strong>: {
            "Using traditional template-based approach for agent creation." if st.session_state.agent_mode == "standard" 
            else "Using MCP tools for advanced API integrations and agent creation."
        }</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"type": "system", "content": "Welcome to Cogentx Agent Builder! Describe what kind of agent you'd like to build, and I'll help you create it."}]

    # Create a container to hold the chat messages with fixed height
    with st.container(border=False, height=400):
        # Create a scrollable area for messages
        st.markdown("""
        <style>
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 20px;
        }
        </style>
        <div class="chat-container">
        """, unsafe_allow_html=True)
        
        # Display chat messages from history
        for message in st.session_state.messages:
            message_type = message["type"]
            if message_type == "system":
                with st.chat_message("assistant", avatar="⚡"):
                    st.markdown(message["content"])
            elif message_type in ["human", "ai"]:
                with st.chat_message("user" if message_type == "human" else "assistant", 
                                    avatar="👤" if message_type == "human" else "⚡"):
                    st.markdown(message["content"])
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Create a container at the bottom for the chat input
    with st.container(border=True):
        # Chat input with mode-specific placeholder
        placeholder_text = "What kind of agent would you like to build?" if st.session_state.agent_mode == "standard" else "Describe the agent with integrations you want to build using MCP..."
        user_input = st.chat_input(placeholder_text)

    if user_input:
        # Append user message to conversation
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Display assistant response
        response_content = ""
        with st.chat_message("assistant", avatar="⚡"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            
            try:
                thread_id = get_thread_id()
                async for chunk in run_agent_with_streaming(user_input, thread_id, st.session_state.agent_mode):
                    if chunk.startswith("An error occurred") or chunk.startswith("Connection interrupted"):
                        message_placeholder.error(chunk)
                        return
                    response_content += chunk
                    message_placeholder.markdown(response_content)
                
                # Append successful response to history
                st.session_state.messages.append({"type": "ai", "content": response_content})
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                message_placeholder.error(error_message) 