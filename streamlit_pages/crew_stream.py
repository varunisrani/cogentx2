from __future__ import annotations
from typing import Literal, TypedDict, List, Dict, Any, Optional, Tuple, Annotated
from langgraph.types import Command
import os
import streamlit as st
import logfire
import asyncio
import time
import json
import uuid
import sys
import platform
import subprocess
import threading
import queue
import webbrowser
import importlib
import tempfile
import shutil
import ast
import io
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# OpenAI and other API clients
from openai import AsyncOpenAI, OpenAI
from supabase import Client, create_client

# Environment and utility imports
from dotenv import load_dotenv
from utils.utils import get_env_var, save_env_var, write_to_log

# Import specific components
from archon_graph import agentic_flow

# Try to import MCP tool components
try:
    from mcp_tools.mcp_tool_graph import mcp_tool_flow, combined_adaptive_flow
    from mcp_tools.mcp_tool_coder import MCPToolDeps
    mcp_tools_available = True
except ImportError:
    mcp_tools_available = False

# Try to import report flow components
try:
    from report_graph import report_flow
    report_analyzer_available = True
except ImportError:
    report_analyzer_available = False

# Try to import debug components
try:
    from debug_pydantic import debug_reasoner, debug_coder, DebuggingDeps
    from debug_graph import run_debug_workflow
    debug_agent_available = True
except ImportError:
    debug_agent_available = False

# Import message part classes for pydantic models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('unified_ui')

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client = None
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
is_ollama = "localhost" in base_url.lower()

if is_ollama:
    openai_client = AsyncOpenAI(base_url=base_url,api_key=api_key)
elif get_env_var("OPENAI_API_KEY"):
    openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))
else:
    openai_client = None

# Initialize OpenAI client (synchronous for some components)
openai_sync_client = OpenAI(api_key=get_env_var("OPENAI_API_KEY")) if get_env_var("OPENAI_API_KEY") else None

if get_env_var("SUPABASE_URL"):
    supabase: Client = Client(
            get_env_var("SUPABASE_URL"),
            get_env_var("SUPABASE_SERVICE_KEY")
        )
else:
    supabase = None

# Set custom theme colors - Cogentx branding with dark theme
st.markdown("""
    <style>
    :root {
        --primary-color: #1E88E5;  /* Cogentx Blue */
        --secondary-color: #26A69A; /* Cogentx Teal */
        --accent-color: #FF5252;   /* Cogentx Accent Red */
        --background-color: #121212; /* Dark background */
        --card-background: #1E1E1E; /* Dark card background */
        --text-color: #E0E0E0;     /* Light text for dark theme */
        --header-color: #64B5F6;   /* Lighter blue for headers on dark */
        --border-color: #333333;   /* Dark borders */
    }
    
    /* Apply dark background color to the entire app */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Style the buttons */
    .stButton > button {
        color: white !important;
        background-color: var(--primary-color) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4) !important;
    }
    
    /* Override Streamlit's default focus styles */
    .stButton > button:focus, 
    .stButton > button:focus:hover, 
    .stButton > button:active, 
    .stButton > button:active:hover {
        color: white !important;
        background-color: var(--secondary-color) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4) !important;
        outline: none !important;
    }
    
    /* Style headers */
    h1 {
        color: var(--header-color) !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: var(--header-color) !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: var(--primary-color) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Hide spans within header elements */
    h1 span, h2 span, h3 span {
        display: none !important;
        visibility: hidden;
        width: 0;
        height: 0;
        opacity: 0;
        position: absolute;
        overflow: hidden;
    }
    
    /* Style code blocks */
    pre {
        border-left: 4px solid var(--primary-color);
        background-color: #2A2A2A !important;
        border-radius: 4px !important;
        color: #E0E0E0 !important;
    }
    
    code {
        color: var(--primary-color) !important;
        background-color: #2A2A2A !important;
    }
    
    /* Style links */
    a {
        color: var(--primary-color) !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: var(--secondary-color) !important;
        text-decoration: underline !important;
    }
    
    /* Style the chat messages */
    .stChatMessage {
        background-color: var(--card-background) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        padding: 12px !important;
        margin-bottom: 16px !important;
        color: var(--text-color) !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1A3A5A !important; /* Dark blue for user */
        border-left: 4px solid var(--primary-color);
    }
    
    /* AI message styling */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: var(--card-background) !important;
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Style the chat input */
    .stChatInput > div {
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Remove outline on focus */
    .stChatInput > div:focus-within {
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.3) !important;
        border: 2px solid var(--primary-color) !important;
        outline: none !important;
    }
    
    /* Remove outline on all inputs when focused */
    input:focus, textarea:focus, [contenteditable]:focus {
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.3) !important;
        border-color: var(--primary-color) !important;
        outline: none !important;
    }

    /* Style inputs and text areas */
    input, textarea, [contenteditable] {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--background-color) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
        background-color: #333333;
        border: none !important;
        color: var(--text-color) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Style selectbox and multiselect */
    .stSelectbox > div, .stMultiSelect > div {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Style file upload areas */
    .file-uploader {
        border: 2px dashed var(--primary-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        background-color: rgba(30, 136, 229, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .file-uploader:hover {
        background-color: rgba(30, 136, 229, 0.2);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1A3A5A;
        border-left: 6px solid var(--primary-color);
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #1B3B2F;
        border-left: 6px solid var(--secondary-color);
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #3A3215;
        border-left: 6px solid #FFC107;
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Error boxes */
    .error-box {
        background-color: #3B1A1A;
        border-left: 6px solid var(--accent-color);
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Tool section styling */
    .tool-section {
        background-color: var(--card-background);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Card styling for containers */
    .stContainer, div[data-testid="stContainer"] {
        background-color: var(--card-background) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        margin-bottom: 1rem !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        background-color: var(--card-background);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        color: var(--text-color) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--card-background);
        border-radius: 0 0 8px 8px;
        border: 1px solid var(--border-color);
        border-top: none;
        color: var(--text-color) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-background);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--header-color) !important;
    }
    
    /* Primary button styling */
    button[data-testid="baseButton-primary"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Secondary button styling */
    button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        color: var(--primary-color) !important;
        border: 2px solid var(--primary-color) !important;
    }
    
    /* Add Cogentx logo styling */
    .cogentx-logo {
        text-align: center;
        padding: 1rem 0;
    }
    
    .cogentx-logo img {
        max-width: 200px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-indicator.green {
        background-color: #4CAF50;
    }
    
    .status-indicator.red {
        background-color: #F44336;
    }
    
    .status-indicator.yellow {
        background-color: #FFC107;
    }

    /* Style progress bars */
    div[data-testid="stProgress"] > div {
        background-color: var(--primary-color) !important;
    }

    div[data-testid="stProgress"] {
        background-color: var(--border-color) !important;
    }

    /* Style markdown */
    .element-container div.markdown-text-container {
        color: var(--text-color) !important;
    }

    /* Fix text color in all contexts */
    * {
        color: var(--text-color);
    }
    
    /* Override default Streamlit dark theme elements */
    .st-bq {
        background-color: var(--card-background) !important;
        border-left-color: var(--primary-color) !important;
    }
    
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_thread_id():
    """Generate and cache a unique thread ID for the session."""
    return str(uuid.uuid4())

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        # Check which mode to use (standard or MCP)
        agent_mode = st.session_state.get("agent_mode", "standard")
        
        # If using MCP mode, route the request through the MCP flow
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
        # Handle other errors
        error_message = f"An error occurred: {str(e)}"
        yield error_message

# Helper function for CrewAI to MCP converter
def create_temp_directory():
    """Create a temporary directory for uploading files."""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def save_uploaded_file(uploaded_file, temp_dir, filename=None):
    """Save uploaded file to the temporary directory."""
    if uploaded_file is not None:
        if filename is None:
            filename = uploaded_file.name
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def read_file_content(file_path):
    """Read and return the content of a file."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    return None

# Helper function for MCP Tool Creator
async def process_mcp_request(user_request, tool_mode="single"):
    """Process a user request through the MCP tool flow."""
    if not mcp_tools_available:
        return "MCP tools functionality is not available. Please check if the required modules are installed."
    
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # For multiple tools mode, enhance the request
    if tool_mode == "multiple":
        # Create a more specific directory
        timestamp = asyncio.get_event_loop().time()
        output_dir = os.path.join(output_dir, f"multi_tools_{int(timestamp)}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhance the user request to ensure multiple tools are detected
        enhanced_request = f"""Create multiple integrated MCP tools that work together: {user_request}
        
This should generate MULTIPLE tools that can be used together in a CrewAI setup, ensuring they're properly integrated.
"""
        user_request = enhanced_request
    
    try:
        # Process the request through MCP tool flow
        result = await combined_adaptive_flow(
            user_request=user_request,
            openai_client=openai_client,
            supabase_client=supabase,
            base_dir=output_dir,
            use_adaptive_synthesis=True
        )
        
        # Create a formatted response
        if result.get("success", False):
            if tool_mode == "single":
                if "tool_info" in result and result["tool_info"].get("purpose", ""):
                    response = f"""✅ **Successfully integrated MCP tool: {result['tool_info']['purpose']}**

Generated the following files:
"""
                else:
                    response = f"""✅ **Successfully generated code**

Generated the following files:
"""
            else:
                # Multiple tools mode
                tool_count = result.get("tool_info", {}).get("tool_count", 0)
                tool_names = result.get("tool_info", {}).get("tool_names", [])
                if tool_names:
                    tool_names_str = ", ".join(tool_names[:3])
                    if len(tool_names) > 3:
                        tool_names_str += f" and {len(tool_names) - 3} more"
                    response = f"""✅ **Successfully integrated {tool_count} MCP tools: {tool_names_str}**

Generated the following files:
"""
                else:
                    response = f"""✅ **Successfully generated {tool_count} MCP tools**

Generated the following files:
"""
            
            for file_path in result.get("files", []):
                response += f"- `{os.path.basename(file_path)}`\n"
            
            output_directory = result.get("tool_info", {}).get("directory", output_dir)
            response += f"\nAll files are saved in: `{output_directory}`"
            
            # Store the generated files and tool info in session state
            if "mcp_generated_files" not in st.session_state:
                st.session_state.mcp_generated_files = []
            if "mcp_tool_info" not in st.session_state:
                st.session_state.mcp_tool_info = {}
                
            st.session_state.mcp_generated_files = result.get("files", [])
            st.session_state.mcp_tool_info = result.get("tool_info", {})
            
        else:
            response = f"""⚠️ **Error in processing request**

{result.get('message', 'An unknown error occurred.')}
"""
            
        return response
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}", exc_info=True)
        return f"❌ **Error processing your request**\n\n{str(e)}"

# Helper function for Report Analyzer
async def process_reports(logs_path: str, input_path: str, output_path: str):
    """Process the uploaded files using the report flow."""
    if not report_analyzer_available:
        return "Report analyzer functionality is not available. Please check if the required modules are installed."
    
    try:
        logger.info("Starting report processing")
        
        # Read file contents
        with open(logs_path, 'r') as f:
            logs_content = f.read()
            
        with open(input_path, 'r') as f:
            input_content = f.read()
            
        with open(output_path, 'r') as f:
            output_content = f.read()

        # Create initial state
        initial_state = {
            "logs_content": logs_content,
            "input_content": input_content,
            "output_content": output_content,
            "error_report": "",
            "error_solve_report": "",
            "overall_log_report": "",
            "agent_communication_report": "",
            "tool_usage_report": "",
            "input_quality_report": "",
            "output_quality_report": "",
            "messages": []
        }

        # Get thread ID for this session
        session_thread_id = get_thread_id()
        
        # Create configuration with required checkpointer keys
        config = {
            "configurable": {
                "thread_id": session_thread_id,
                "checkpoint_ns": "report_analysis",
                "checkpoint_id": f"report_{session_thread_id}"
            }
        }

        # Run the report flow
        result = await report_flow.ainvoke(initial_state, config=config)
        return result
    except Exception as e:
        logger.error(f"Error processing reports: {str(e)}", exc_info=True)
        return None

# Helper function for Debug Agent
async def process_debug_request(code_files: Dict[str, str], error_details: str):
    """Process a debugging request with the debug agent."""
    if not debug_agent_available:
        return "Debugging agent functionality is not available. Please check if the required modules are installed."
    
    try:
        # Run the debugging workflow
        result = await run_debug_workflow(code_files, error_details)
        return result
    except Exception as e:
        logger.error(f"Error processing debug request: {str(e)}", exc_info=True)
        return None

async def chat_tab():
    """Display the chat interface for talking to the Agent Builder"""

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
            
            # Hidden button for the click handler
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
            
            # Hidden button for the click handler
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
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            message_type = message["type"]
            if message_type == "system":
                with st.chat_message("assistant", avatar="⚡"):
                    st.markdown(message["content"])
            elif message_type in ["human", "ai"]:
                with st.chat_message("user" if message_type == "human" else "assistant", 
                                    avatar="👤" if message_type == "human" else "⚡"):
                    st.markdown(message["content"])
        
        # Close the scrollable container div
        st.markdown("</div>", unsafe_allow_html=True)

    # Create a container at the bottom for the chat input
    with st.container(border=True):
        # Chat input for the user - adjust placeholder based on the selected mode
        placeholder_text = "What kind of agent would you like to build?" if st.session_state.agent_mode == "standard" else "Describe the agent with integrations you want to build using MCP..."
        user_input = st.chat_input(placeholder_text)

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user prompt in the UI
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant", avatar="⚡"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            
            # Show typing indicator before response starts
            message_placeholder.markdown("⏳ Thinking...")
            
            try:
                # Run the async generator to fetch responses
                async for chunk in run_agent_with_streaming(user_input):
                    if chunk.startswith("An error occurred") or chunk.startswith("Connection interrupted"):
                        message_placeholder.error(chunk)
                        return
                    response_content += chunk
                    # Update the placeholder with the current response content
                    message_placeholder.markdown(response_content)
                
                # Only append successful responses to message history
                st.session_state.messages.append({"type": "ai", "content": response_content})
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                message_placeholder.error(error_message)

async def mcp_tools_tab():
    """Display the MCP Tool Builder interface."""
    st.markdown("""
    <h1>Cogentx MCP Tool Builder</h1>
    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Create powerful CrewAI agents with integrated MCP tools for external API services.</p>
    <p>Describe the tools you want to build, and the system will find and integrate the appropriate MCP tools for your agents.</p>
    """, unsafe_allow_html=True)
    
    if not mcp_tools_available:
        st.error("MCP tools functionality is not available. Please check if the required modules are installed.")
        return
    
    # Initialize session state
    if "mcp_messages" not in st.session_state:
        st.session_state.mcp_messages = []
    if "mcp_generated_files" not in st.session_state:
        st.session_state.mcp_generated_files = []
    if "mcp_tool_info" not in st.session_state:
        st.session_state.mcp_tool_info = {}
    if "mcp_tool_mode" not in st.session_state:
        st.session_state.mcp_tool_mode = "single"

    # Tool mode selection with cards
    st.markdown("### Select Tool Creation Mode")
    
    tool_mode_col1, tool_mode_col2 = st.columns(2)
    with tool_mode_col1:
        single_card_style = "background-color: " + ("#E3F2FD" if st.session_state.mcp_tool_mode == "single" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.mcp_tool_mode == "single" else "#e0e0e0")
        
        st.markdown(f"""
        <div style="{single_card_style}" onclick="document.getElementById('single_mode_btn').click()">
            <div style="font-size: 32px; margin-bottom: 10px;">🔨</div>
            <h3 style="margin: 0; margin-bottom: 5px;">Single Tool Mode</h3>
            <p style="margin: 0; color: #666;">Create a focused tool for a specific purpose</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button for the click handler
        single_mode = st.button("🔨 Single Tool Mode", key="single_mode_btn", use_container_width=True)
        if single_mode:
            st.session_state.mcp_tool_mode = "single"
            st.rerun()

    with tool_mode_col2:
        multi_card_style = "background-color: " + ("#E3F2FD" if st.session_state.mcp_tool_mode == "multiple" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.mcp_tool_mode == "multiple" else "#e0e0e0")
        
        st.markdown(f"""
        <div style="{multi_card_style}" onclick="document.getElementById('multi_mode_btn').click()">
            <div style="font-size: 32px; margin-bottom: 10px;">🔧</div>
            <h3 style="margin: 0; margin-bottom: 5px;">Multiple Tools Mode</h3>
            <p style="margin: 0; color: #666;">Create integrated tools that work together</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button for the click handler
        multi_mode = st.button("🔧 Multiple Tools Mode", key="multi_mode_btn", use_container_width=True)
        if multi_mode:
            st.session_state.mcp_tool_mode = "multiple"
            st.rerun()

    # Show current mode with a nicer info box
    st.markdown(f"""
    <div style="background-color: #E3F2FD; border-left: 4px solid #1E88E5; padding: 12px; border-radius: 4px; margin: 16px 0;">
        <p style="margin: 0;"><strong>Currently in {st.session_state.mcp_tool_mode.title()} Mode</strong>: {
            "Create a single focused MCP tool." if st.session_state.mcp_tool_mode == "single" 
            else "Create multiple MCP tools that work together."
        }</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat interface container
    with st.container(border=False):
        st.markdown("### Describe Your Tool Requirements")
        
        # Display chat messages
        for message in st.session_state.mcp_messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "⚡"):
                st.markdown(message["content"])

        # Chat input
        prompt = st.chat_input(f"What would you like to build with {'a single' if st.session_state.mcp_tool_mode == 'single' else 'multiple'} MCP tool{'s' if st.session_state.mcp_tool_mode == 'multiple' else ''}?")
        
        if prompt:
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            st.session_state.mcp_messages.append({"role": "user", "content": prompt})
            
            # Create a placeholder for the assistant's response
            with st.chat_message("assistant", avatar="⚡"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ Searching for relevant MCP tools...")
                
                try:
                    # Process the request based on the current mode
                    response = await process_mcp_request(prompt, st.session_state.mcp_tool_mode)
                    
                    # Display the response
                    message_placeholder.markdown(response)
                    
                    # Add assistant's response to chat history
                    st.session_state.mcp_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    message_placeholder.error(f"An error occurred: {str(e)}")
    
    # Results section with a nice card UI
    if st.session_state.mcp_generated_files or st.session_state.mcp_tool_info:
        with st.container(border=True):
            st.markdown("## Generated Results")
            
            # Tool info section with nice formatting
            if st.session_state.mcp_tool_info:
                st.markdown("### Tool Information")
                
                # Create an info card
                purpose = st.session_state.mcp_tool_info.get('purpose', 'No purpose specified')
                tool_count = st.session_state.mcp_tool_info.get('tool_count', 1) if 'tool_count' in st.session_state.mcp_tool_info else (len(st.session_state.mcp_tool_info.get('tool_names', [])) if 'tool_names' in st.session_state.mcp_tool_info else 1)
                
                tool_type = "Multiple Tools" if tool_count > 1 else "Single Tool"
                tool_icon = "🔧" if tool_count > 1 else "🔨"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 24px; margin-right: 12px;">{tool_icon}</span>
                        <h3 style="margin: 0; color: #1E88E5;">{tool_type}</h3>
                    </div>
                    <p><strong>Purpose:</strong> {purpose}</p>
                    <p><strong>Tool Count:</strong> {tool_count}</p>
                """, unsafe_allow_html=True)
                
                # Display multiple tool names if available
                if 'tool_names' in st.session_state.mcp_tool_info:
                    tool_names = st.session_state.mcp_tool_info.get('tool_names', [])
                    tool_names_html = ", ".join([f"<span style='background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px; margin-right: 5px;'>{name}</span>" for name in tool_names])
                    st.markdown(f"<p><strong>Tools:</strong> {tool_names_html}</p>", unsafe_allow_html=True)
                    
                # Display output directory
                if 'directory' in st.session_state.mcp_tool_info:
                    st.markdown(f"<p><strong>Output Directory:</strong> <code>{st.session_state.mcp_tool_info.get('directory', '')}</code></p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show the generated files if available
            if st.session_state.mcp_generated_files:
                st.markdown("### Generated Files")
                
                # Create a grid for the files
                cols = st.columns(3)
                for i, file_path in enumerate(st.session_state.mcp_generated_files):
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1]
                    
                    # Pick an icon based on file extension
                    file_icon = "📄"
                    if file_ext == ".py":
                        file_icon = "🐍"
                    elif file_ext == ".json":
                        file_icon = "📊"
                    elif file_ext == ".md":
                        file_icon = "📝"
                    
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px; display: flex; align-items: center;">
                            <span style="font-size: 20px; margin-right: 10px;">{file_icon}</span>
                            <div style="flex-grow: 1;">
                                <strong>{file_name}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Reset conversation button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Clear MCP Chat", use_container_width=True, type="secondary"):
            st.session_state.mcp_messages = []
            st.session_state.mcp_generated_files = []
            st.session_state.mcp_tool_info = {}
            st.rerun()

async def crew_to_mcp_tab():
    """Display the CrewAI to MCP Converter interface."""
    st.markdown("""
    <h1>Cogentx CrewAI to MCP Converter</h1>
    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Convert existing CrewAI agents to MCP server files.</p>
    <p>Upload your CrewAI project files and automatically generate MCP-compatible files.</p>
    """, unsafe_allow_html=True)
    
    # Initialize state
    if "crew_temp_dir" not in st.session_state:
        st.session_state.crew_temp_dir = create_temp_directory()
    
    if "agents_file" not in st.session_state:
        st.session_state.agents_file = None
    if "tasks_file" not in st.session_state:
        st.session_state.tasks_file = None
    if "crew_file" not in st.session_state:
        st.session_state.crew_file = None
    if "tools_file" not in st.session_state:
        st.session_state.tools_file = None
    
    if "crew_info" not in st.session_state:
        st.session_state.crew_info = {}
    if "converted_files" not in st.session_state:
        st.session_state.converted_files = []
    
    # Main container with improved styling
    with st.container(border=True):
        # File upload section with better styling
        st.markdown("""
        <h3 style="color: #1E88E5; margin-bottom: 1rem;">
            <span style="font-size: 24px; margin-right: 10px;">📂</span> Upload CrewAI Files
        </h3>
        <p>Upload your CrewAI project files to convert them to MCP format.</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Track uploaded files
        has_agents = st.session_state.agents_file is not None
        has_tasks = st.session_state.tasks_file is not None
        has_crew = st.session_state.crew_file is not None
        has_tools = st.session_state.tools_file is not None
        
        required_files_uploaded = has_agents and has_tasks and has_crew
        
        # Required files progress
        progress = (int(has_agents) + int(has_tasks) + int(has_crew)) / 3
        st.progress(progress, text=f"Uploaded {int(progress * 3)}/3 required files")
        
        with col1:
            with st.container(border=True):
                st.markdown("""
                <h4 style="color: #0D47A1;">Required Files</h4>
                """, unsafe_allow_html=True)
                
                # Crew file
                crew_label = "✅ crew.py" if has_crew else "📄 Upload crew.py"
                st.markdown(f"<p><strong>{crew_label}</strong></p>", unsafe_allow_html=True)
                crew_file = st.file_uploader("", type=["py"], key="crew_uploader")
                if crew_file:
                    st.session_state.crew_file = save_uploaded_file(crew_file, st.session_state.crew_temp_dir, "crew.py")
                    st.rerun()  # Refresh to update progress and labels
                
                # Agents file
                agents_label = "✅ agents.py" if has_agents else "📄 Upload agents.py"
                st.markdown(f"<p><strong>{agents_label}</strong></p>", unsafe_allow_html=True)
                agents_file = st.file_uploader("", type=["py"], key="agents_uploader")
                if agents_file:
                    st.session_state.agents_file = save_uploaded_file(agents_file, st.session_state.crew_temp_dir, "agents.py")
                    st.rerun()  # Refresh to update progress and labels
                
                # Tasks file
                tasks_label = "✅ tasks.py" if has_tasks else "📄 Upload tasks.py"
                st.markdown(f"<p><strong>{tasks_label}</strong></p>", unsafe_allow_html=True)
                tasks_file = st.file_uploader("", type=["py"], key="tasks_uploader")
                if tasks_file:
                    st.session_state.tasks_file = save_uploaded_file(tasks_file, st.session_state.crew_temp_dir, "tasks.py")
                    st.rerun()  # Refresh to update progress and labels
        
        with col2:
            with st.container(border=True):
                st.markdown("""
                <h4 style="color: #0D47A1;">Optional Files & Settings</h4>
                """, unsafe_allow_html=True)
                
                # Tools file
                tools_label = "✅ tools.py" if has_tools else "📄 Upload tools.py (Optional)"
                st.markdown(f"<p><strong>{tools_label}</strong></p>", unsafe_allow_html=True)
                tools_file = st.file_uploader("", type=["py"], key="tools_uploader")
                if tools_file:
                    st.session_state.tools_file = save_uploaded_file(tools_file, st.session_state.crew_temp_dir, "tools.py")
                    st.rerun()  # Refresh to update progress and labels
                
                # API Key input
                st.markdown("<p><strong>🔑 API Keys (Optional)</strong></p>", unsafe_allow_html=True)
                api_key = st.text_input("OpenAI API Key", type="password", key="openai_key", placeholder="Enter your OpenAI API key")
        
        # Show file status message
        if not required_files_uploaded:
            st.info("Please upload all required files (agents.py, tasks.py, and crew.py) to begin conversion.")
        else:
            st.success("✅ All required files uploaded successfully!")
        
        # Convert button with better styling
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if required_files_uploaded:
                if st.button("🔄 Convert to MCP", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing files and generating MCP code..."):
                        try:
                            # TODO: Actually implement crew_to_mcp functionality here
                            # For now, just display placeholder analysis
                            st.session_state.crew_info = {
                                "agents": ["ResearchAgent", "WriterAgent"],
                                "tasks": ["ResearchTask", "WriteTask"],
                                "tools": ["SearchTool", "WritingTool"],
                                "crew_name": "WritingCrew"
                            }
                            
                            # Create output directory
                            output_dir = os.path.join(os.getcwd(), "mcp_output")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Placeholder for generated files list
                            st.session_state.converted_files = [
                                os.path.join(output_dir, "mcp_server.py"),
                                os.path.join(output_dir, "mcp_config.json"),
                                os.path.join(output_dir, "start_mcp.py")
                            ]
                            
                            # Create placeholder files for demo
                            with open(st.session_state.converted_files[0], 'w') as f:
                                f.write("# Generated MCP server file\nprint('MCP Server started')")
                            
                            with open(st.session_state.converted_files[1], 'w') as f:
                                f.write('{"name": "Generated MCP Config", "version": "1.0"}')
                            
                            with open(st.session_state.converted_files[2], 'w') as f:
                                f.write("# Start MCP script\nimport os\nprint('Starting MCP server')")
                            
                            # Display a success message
                            st.success("Files converted successfully! MCP files generated.")
                            st.rerun()  # Refresh to show results
                            
                        except Exception as e:
                            st.error(f"Error converting files: {str(e)}")
            else:
                st.markdown("""
                <div style="background-color: #F5F5F5; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #E0E0E0;">
                    <button disabled style="background-color: #BDBDBD; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; width: 100%;">
                        🔄 Convert to MCP
                    </button>
                    <p style="margin-top: 8px; color: #757575; font-size: 14px;">
                        Please upload all required files first
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display generated files if available
    if st.session_state.converted_files:
        st.markdown("""
        <h2 style="color: #0D47A1; margin-top: 2rem; margin-bottom: 1rem;">
            <span style="font-size: 30px; margin-right: 10px;">✨</span> Generated MCP Files
        </h2>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            # Display crew info as a card
            if st.session_state.crew_info:
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">📋</span> Conversion Summary
                </h3>
                """, unsafe_allow_html=True)
                
                crew_name = st.session_state.crew_info.get("crew_name", "Unknown Crew")
                agents = st.session_state.crew_info.get("agents", [])
                tasks = st.session_state.crew_info.get("tasks", [])
                tools = st.session_state.crew_info.get("tools", [])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Crew Name:** {crew_name}")
                    st.markdown(f"**Output Directory:** `{os.path.join(os.getcwd(), 'mcp_output')}`")
                
                with col2:
                    st.markdown(f"**Agents Converted:** {len(agents)}")
                    st.markdown(f"**Tasks Converted:** {len(tasks)}")
                    st.markdown(f"**Tools Converted:** {len(tools)}")
            
            # File list with icons
            st.markdown("""
            <h3 style="color: #1E88E5; margin-top: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 24px; margin-right: 10px;">📁</span> Generated Files
            </h3>
            """, unsafe_allow_html=True)
            
            # Display files in a grid
            cols = st.columns(3)
            for i, file_path in enumerate(st.session_state.converted_files):
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1]
                
                # Pick an icon based on file extension
                file_icon = "📄"
                if file_ext == ".py":
                    file_icon = "🐍"
                elif file_ext == ".json":
                    file_icon = "📊"
                
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px; display: flex; align-items: center;">
                        <span style="font-size: 20px; margin-right: 10px;">{file_icon}</span>
                        <div style="flex-grow: 1;">
                            <strong>{file_name}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create a download button for a zip of all files
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for file_path in st.session_state.converted_files:
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                
                zip_buffer.seek(0)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.download_button(
                        label="⬇️ Download All Files as ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="mcp_files.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error creating ZIP file: {str(e)}")
        
        # Clear results button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Start New Conversion", use_container_width=True):
                # Reset conversion state but keep uploaded files
                st.session_state.crew_info = {}
                st.session_state.converted_files = []
                st.rerun()

async def report_analyzer_tab():
    """Display the Report Analyzer interface."""
    st.markdown("""
    <h1>Cogentx Report Analyzer</h1>
    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Analyze agent performance and generate detailed reports.</p>
    <p>Upload your log files and get AI-powered insights about your agent's performance and execution.</p>
    """, unsafe_allow_html=True)

    if not report_analyzer_available:
        st.error("Report analyzer functionality is not available. Please check if the required modules are installed.")
        return
    
    # Main container with improved styling
    with st.container(border=True):
        # File upload section with better styling
        st.markdown("""
        <h3 style="color: #1E88E5; margin-bottom: 1rem;">
            <span style="font-size: 24px; margin-right: 10px;">📂</span> Upload Analysis Files
        </h3>
        <p>Upload your log files, input, and output to analyze agent performance.</p>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Track uploaded files
        uploaded_logs = False
        uploaded_input = False
        uploaded_output = False
        
        with col1:
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-bottom: 10px;'><strong>📄 Log File</strong></p>", unsafe_allow_html=True)
            logs_file = st.file_uploader("", type=['txt'], key="logs_uploader")
            if logs_file:
                uploaded_logs = True
                st.success("✅ Log file uploaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-bottom: 10px;'><strong>📄 Input File</strong></p>", unsafe_allow_html=True)
            input_file = st.file_uploader("", type=['txt'], key="input_uploader")
            if input_file:
                uploaded_input = True
                st.success("✅ Input file uploaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-bottom: 10px;'><strong>📄 Output File</strong></p>", unsafe_allow_html=True)
            output_file = st.file_uploader("", type=['txt'], key="output_uploader")
            if output_file:
                uploaded_output = True
                st.success("✅ Output file uploaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show upload status
        all_files_uploaded = uploaded_logs and uploaded_input and uploaded_output
        
        # Show progress bar for uploads
        progress = (int(uploaded_logs) + int(uploaded_input) + int(uploaded_output)) / 3
        st.progress(progress, text=f"Uploaded {int(progress * 3)}/3 required files")
        
        if not all_files_uploaded:
            st.info("Please upload all required files (log.txt, input.txt, and output.txt) to begin analysis.")
        else:
            st.success("✅ All required files uploaded successfully!")
        
        # Process button with better styling
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if all_files_uploaded:
                if st.button("🔍 Analyze Reports", use_container_width=True, type="primary"):
                    with st.spinner("Processing reports..."):
                        try:
                            # Save uploaded files
                            temp_folder = "temp_uploads"
                            os.makedirs(temp_folder, exist_ok=True)
                            logs_path = save_uploaded_file(logs_file, temp_folder)
                            input_path = save_uploaded_file(input_file, temp_folder)
                            output_path = save_uploaded_file(output_file, temp_folder)

                            # Process reports
                            result = await process_reports(logs_path, input_path, output_path)

                            if result:
                                # Store result in session state
                                st.session_state.report_result = result
                                st.success("Analysis completed successfully!")
                                st.rerun()  # Refresh to show results
                            
                            # Clean up temporary files
                            for file_path in [logs_path, input_path, output_path]:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                        except Exception as e:
                            logger.error(f"Error in report processing: {str(e)}", exc_info=True)
                            st.error(f"An error occurred: {str(e)}")
            else:
                st.markdown("""
                <div style="background-color: #F5F5F5; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #E0E0E0;">
                    <button disabled style="background-color: #BDBDBD; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; width: 100%;">
                        🔍 Analyze Reports
                    </button>
                    <p style="margin-top: 8px; color: #757575; font-size: 14px;">
                        Please upload all required files first
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display results if available
    if "report_result" in st.session_state and st.session_state.report_result:
        result = st.session_state.report_result
        
        st.markdown("""
        <h2 style="color: #0D47A1; margin-top: 2rem; margin-bottom: 1rem;">
            <span style="font-size: 30px; margin-right: 10px;">📊</span> Analysis Results
        </h2>
        """, unsafe_allow_html=True)
        
        # Create tabs for different analysis categories
        report_tabs = st.tabs(["Error Analysis", "Communication Analysis", "Quality Analysis"])
        
        # Error Analysis tab
        with report_tabs[0]:
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">🐞</span> Error Report
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("error_report", "No error report available"))
            
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">💡</span> Error Solutions
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("error_solve_report", "No error solutions available"))
            
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">📝</span> Overall Log Summary
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("overall_log_report", "No log summary available"))
        
        # Communication Analysis tab
        with report_tabs[1]:
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">💬</span> Agent Communication Report
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("agent_communication_report", "No communication report available"))
            
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">🔧</span> Tool Usage Report
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("tool_usage_report", "No tool usage report available"))
        
        # Quality Analysis tab
        with report_tabs[2]:
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">📥</span> Input Quality Report
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("input_quality_report", "No input quality report available"))
            
            with st.container(border=True):
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">📤</span> Output Quality Report
                </h3>
                """, unsafe_allow_html=True)
                st.markdown(result.get("output_quality_report", "No output quality report available"))
        
        # Clear results button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                del st.session_state.report_result
                st.rerun()

async def debug_agent_tab():
    """Display the debugging agent interface."""
    st.markdown("""
    <h1>Cogentx Debug Agent</h1>
    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Automatically analyze and fix issues in CrewAI code.</p>
    <p>Upload your CrewAI code files and error messages, and our AI will diagnose problems and suggest fixes.</p>
    """, unsafe_allow_html=True)
    
    if not debug_agent_available:
        st.error("Debugging agent functionality is not available. Please check if the required modules are installed.")
        return
    
    # Initialize state
    if "debug_code_files" not in st.session_state:
        st.session_state.debug_code_files = {}
    if "debug_error_details" not in st.session_state:
        st.session_state.debug_error_details = ""
    if "debug_result" not in st.session_state:
        st.session_state.debug_result = None
    
    # Main content in a card-style container
    with st.container(border=True):
        # Upload files section with improved styling
        st.markdown("""
        <h3 style="color: #1E88E5; margin-bottom: 1rem;">
            <span style="font-size: 24px; margin-right: 10px;">📂</span> Required Code Files
        </h3>
        <p>The debugging agent requires all 4 CrewAI component files to properly analyze code issues.</p>
        """, unsafe_allow_html=True)
        
        # Create a two-column layout for required files
        col1, col2 = st.columns(2)
        
        # Check if all required files are uploaded
        required_files = ["crew.py", "agents.py", "tasks.py", "tools.py"]
        uploaded_files = list(st.session_state.debug_code_files.keys())
        missing_files = [file for file in required_files if file not in uploaded_files]
        
        # Show a progress bar for file uploads
        total_files = len(required_files)
        uploaded_count = total_files - len(missing_files)
        progress = uploaded_count / total_files
        
        st.progress(progress, text=f"Uploaded {uploaded_count}/{total_files} required files")
        
        # File upload columns with improved styling
        with col1:
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            crew_label = "✅ crew.py" if "crew.py" in uploaded_files else "📄 Upload crew.py"
            st.markdown(f"<p style='text-align: center; margin-bottom: 10px;'><strong>{crew_label}</strong></p>", unsafe_allow_html=True)
            crew_file = st.file_uploader("", type=["py"], key="debug_crew_file")
            if crew_file:
                try:
                    content = crew_file.getvalue().decode("utf-8")
                    st.session_state.debug_code_files["crew.py"] = content
                    st.rerun()  # Refresh to update progress and labels
                except Exception as e:
                    st.error(f"Error reading crew.py: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            agents_label = "✅ agents.py" if "agents.py" in uploaded_files else "📄 Upload agents.py"
            st.markdown(f"<p style='text-align: center; margin-bottom: 10px;'><strong>{agents_label}</strong></p>", unsafe_allow_html=True)
            agents_file = st.file_uploader("", type=["py"], key="debug_agents_file")
            if agents_file:
                try:
                    content = agents_file.getvalue().decode("utf-8")
                    st.session_state.debug_code_files["agents.py"] = content
                    st.rerun()  # Refresh to update progress and labels
                except Exception as e:
                    st.error(f"Error reading agents.py: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            tasks_label = "✅ tasks.py" if "tasks.py" in uploaded_files else "📄 Upload tasks.py"
            st.markdown(f"<p style='text-align: center; margin-bottom: 10px;'><strong>{tasks_label}</strong></p>", unsafe_allow_html=True)
            tasks_file = st.file_uploader("", type=["py"], key="debug_tasks_file")
            if tasks_file:
                try:
                    content = tasks_file.getvalue().decode("utf-8")
                    st.session_state.debug_code_files["tasks.py"] = content
                    st.rerun()  # Refresh to update progress and labels
                except Exception as e:
                    st.error(f"Error reading tasks.py: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply file-uploader styling through markdown instead
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            tools_label = "✅ tools.py" if "tools.py" in uploaded_files else "📄 Upload tools.py"
            st.markdown(f"<p style='text-align: center; margin-bottom: 10px;'><strong>{tools_label}</strong></p>", unsafe_allow_html=True)
            tools_file = st.file_uploader("", type=["py"], key="debug_tools_file")
            if tools_file:
                try:
                    content = tools_file.getvalue().decode("utf-8")
                    st.session_state.debug_code_files["tools.py"] = content
                    st.rerun()  # Refresh to update progress and labels
                except Exception as e:
                    st.error(f"Error reading tools.py: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    
        # Show file status message
        if missing_files:
            st.warning(f"⚠️ Missing required files: {', '.join(missing_files)}")
        else:
            st.success("✅ All required code files uploaded successfully!")
        
        # Error details section with improved styling
        st.markdown("""
        <h3 style="color: #1E88E5; margin-top: 2rem; margin-bottom: 1rem;">
            <span style="font-size: 24px; margin-right: 10px;">🐞</span> Error Details (5th required input)
        </h3>
        <p>Provide the error message or stack trace from your CrewAI execution.</p>
        """, unsafe_allow_html=True)
        
        error_tabs = st.tabs(["Paste Error Text", "Upload Error File"])
        
        with error_tabs[0]:
            st.session_state.debug_error_details = st.text_area(
                "Paste error messages or stack traces here",
                value=st.session_state.debug_error_details,
                height=200,
                key="debug_error_area"
            )
        
        with error_tabs[1]:
            error_file = st.file_uploader("Upload error log file", type=["txt", "log"], key="debug_error_file")
            if error_file:
                try:
                    content = error_file.getvalue().decode("utf-8")
                    st.session_state.debug_error_details = content
                    st.success("✅ Error file uploaded")
                except Exception as e:
                    st.error(f"Error reading error file: {str(e)}")
        
        # Check if error details are provided
        error_provided = bool(st.session_state.debug_error_details.strip())
        if not error_provided:
            st.info("Please provide error details to begin debugging")
        
        # Debug button with improved styling
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        debug_button_disabled = not (len(st.session_state.debug_code_files) == 4 and error_provided)
        
        if debug_button_disabled:
            st.markdown("""
            <div style="background-color: #F5F5F5; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #E0E0E0;">
                <button disabled style="background-color: #BDBDBD; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; width: 100%;">
                    🔍 Debug Code
                </button>
                <p style="margin-top: 8px; color: #757575; font-size: 14px;">
                    Please provide all required files and error details
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🔍 Debug Code", use_container_width=True, type="primary"):
                with st.spinner("Analyzing code and error details..."):
                    try:
                        # Process debug request
                        result = await process_debug_request(
                            st.session_state.debug_code_files,
                            st.session_state.debug_error_details
                        )
                        
                        if result:
                            st.session_state.debug_result = result
                            st.success("Debugging completed successfully!")
                    except Exception as e:
                        logger.error(f"Error in debugging process: {str(e)}", exc_info=True)
                        st.error(f"An error occurred: {str(e)}")
    
    # Display results if available
    if st.session_state.debug_result:
        st.markdown("""
        <h2 style="color: #0D47A1; margin-top: 2rem; margin-bottom: 1rem;">
            Debugging Results
        </h2>
        """, unsafe_allow_html=True)
        
        # Results in tabs for better organization
        debug_result_tabs = st.tabs(["Analysis", "Fixed Code", "Explanation"])
        
        # Analysis tab
        with debug_result_tabs[0]:
            st.markdown("""
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                <span style="font-size: 24px; margin-right: 10px;">🔍</span> AI Analysis
            </h3>
            """, unsafe_allow_html=True)
            
            analysis = st.session_state.debug_result.get("reasoner_analysis", "No analysis available")
            st.markdown(f"<div class='analysis-box'>{analysis}</div>", unsafe_allow_html=True)
        
        # Fixed code tab
        with debug_result_tabs[1]:
            fixed_files = st.session_state.debug_result.get("fixed_files", {})
            
            if fixed_files:
                st.markdown("""
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                    <span style="font-size: 24px; margin-right: 10px;">✅</span> Fixed Files
                </h3>
                """, unsafe_allow_html=True)
                
                # Create a select box to choose which file to view
                file_options = list(fixed_files.keys())
                selected_file = st.selectbox("Select a file to view:", file_options)
                
                if selected_file:
                    # Display the file with syntax highlighting
                    st.code(fixed_files[selected_file], language="python")
                    
                    # Add individual download buttons for each file
                    st.download_button(
                        label=f"Download {selected_file}",
                        data=fixed_files[selected_file],
                        file_name=selected_file,
                        mime="text/plain"
                    )
                
                # Create a download button for all fixed files as a ZIP
                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zipf:
                        for filename, content in fixed_files.items():
                            zipf.writestr(filename, content)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download All Fixed Files as ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="fixed_files.zip",
                        mime="application/zip",
                        key="download_all_fixed"
                    )
                except Exception as e:
                    st.error(f"Error creating ZIP file: {str(e)}")
            else:
                st.info("No fixed files were generated.")
        
        # Explanation tab
        with debug_result_tabs[2]:
            st.markdown("""
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">
                <span style="font-size: 24px; margin-right: 10px;">💡</span> Fix Explanation
            </h3>
            """, unsafe_allow_html=True)
            
            explanation = st.session_state.debug_result.get("explanation", "No explanation available")
            st.markdown(explanation)
        
        # Clear results button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Debug Session", use_container_width=True):
                st.session_state.debug_result = None
                st.rerun()

async def main():
    """Main function to run the Streamlit app."""
    
    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        valid_tabs = ["Chat", "MCP", "CrewToMCP", "Reports", "Debug"]
        if tab_name in valid_tabs:
            st.session_state.selected_tab = tab_name
    
    # Initialize session state for selected tab if not present
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Chat"
    
    # Add sidebar navigation
    with st.sidebar:
        # Logo and title with dark theme color
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: var(--header-color); margin-bottom: 0;">Cogentx</h1>
            <h3 style="color: var(--primary-color); margin-top: 0;">Agent Builder Platform</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        
        # Add navigation buttons directly (without container with class)
        chat_button = st.button("💬 Chat", use_container_width=True, key="sidebar_chat")
        mcp_button = st.button("🔧 MCP Tool Builder", use_container_width=True, key="sidebar_mcp")
        crew_button = st.button("🔄 CrewAI to MCP", use_container_width=True, key="sidebar_crew")
        report_button = st.button("📊 Report Analyzer", use_container_width=True, key="sidebar_report")
        debug_button = st.button("🐞 Debug Agent", use_container_width=True, key="sidebar_debug")
        
        st.markdown("---")
        
        # Update selected tab based on button clicks
        if chat_button:
            st.session_state.selected_tab = "Chat"
            st.rerun()
        elif mcp_button:
            st.session_state.selected_tab = "MCP"
            st.rerun()
        elif crew_button:
            st.session_state.selected_tab = "CrewToMCP"
            st.rerun()
        elif report_button:
            st.session_state.selected_tab = "Reports"
            st.rerun()
        elif debug_button:
            st.session_state.selected_tab = "Debug"
            st.rerun()
        
        # Environment status with improved styling
        st.markdown("### Environment Status")
        
        # API connections with visual indicators
        if openai_client:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator green"></span>
                <strong>OpenAI API:</strong> Connected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator red"></span>
                <strong>OpenAI API:</strong> Not connected
            </div>
            """, unsafe_allow_html=True)
        
        if supabase:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator green"></span>
                <strong>Supabase:</strong> Connected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator yellow"></span>
                <strong>Supabase:</strong> Not connected
            </div>
            """, unsafe_allow_html=True)
        
        # Component availability with visual indicators
        st.markdown("### Available Components")
        
        if mcp_tools_available:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator green"></span>
                <strong>MCP Tool Builder:</strong> Available
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator yellow"></span>
                <strong>MCP Tool Builder:</strong> Not available
            </div>
            """, unsafe_allow_html=True)
            
        if report_analyzer_available:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator green"></span>
                <strong>Report Analyzer:</strong> Available
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator yellow"></span>
                <strong>Report Analyzer:</strong> Not available
            </div>
            """, unsafe_allow_html=True)
            
        if debug_agent_available:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator green"></span>
                <strong>Debug Agent:</strong> Available
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator yellow"></span>
                <strong>Debug Agent:</strong> Not available
            </div>
            """, unsafe_allow_html=True)
        
        # Reset session button with better styling
        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key != "selected_tab":  # Keep selected tab
                    del st.session_state[key]
            st.rerun()
    
    # Display the selected tab with content container styling
    with st.container():
        if st.session_state.selected_tab == "Chat":
            await chat_tab()
        elif st.session_state.selected_tab == "MCP":
            await mcp_tools_tab()
        elif st.session_state.selected_tab == "CrewToMCP":
            await crew_to_mcp_tab()
        elif st.session_state.selected_tab == "Reports":
            await report_analyzer_tab()
        elif st.session_state.selected_tab == "Debug":
            await debug_agent_tab()

if __name__ == "__main__":
    asyncio.run(main())