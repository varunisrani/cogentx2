from __future__ import annotations
from dotenv import load_dotenv
import streamlit as st
import logfire
import asyncio
from concurrent.futures import TimeoutError
import time
import os
import zipfile
import io
import shutil
import subprocess
import threading
import queue
import sys
from datetime import datetime
import re
from code_editor import code_editor
from PIL import Image
import logging
import traceback
import uuid

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Cogentx",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import crew_stream components
from crew_stream import (
    agentic_flow, mcp_tools_available, report_analyzer_available,
    debug_agent_available, process_mcp_request, process_reports,
    process_debug_request, create_temp_directory, save_uploaded_file,
    read_file_content
)

# Log setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the streamlit_terminal package
try:
    from streamlit_terminal import st_terminal
    HAS_STREAMLIT_TERMINAL = True
except Exception:
    HAS_STREAMLIT_TERMINAL = False

# Import our custom terminal implementation as a fallback
from custom_terminal import st_custom_terminal

# Import page modules
try:
    from streamlit_pages.home import home_tab
except ImportError:
    logger.error("Failed to import home_tab")
    def home_tab():
        st.error("Home tab module not found")

try:
    from streamlit_pages.agent_runner import agent_runner_tab
except ImportError:
    logger.error("Failed to import agent_runner_tab")
    def agent_runner_tab():
        st.error("Agent Runner tab module not found")

try:
    from streamlit_pages.template_browser import template_browser_tab
except ImportError:
    logger.error("Failed to import template_browser_tab")
    def template_browser_tab():
        st.error("Template Browser tab module not found")

try:
    from streamlit_pages.workbench import workbench_tab
except ImportError:
    logger.error("Failed to import workbench_tab")
    def workbench_tab():
        st.error("Workbench tab module not found")

try:
    from streamlit_pages.nocode_editor import nocode_editor_tab
except ImportError:
    logger.error("Failed to import nocode_editor_tab")
    def nocode_editor_tab():
        st.error("Code Editor tab module not found")

try:
    from streamlit_pages.nocode_builder import nocode_builder_tab
except ImportError:
    logger.error("Failed to import nocode_builder_tab")
    def nocode_builder_tab():
        st.error("No-Code Builder tab module not found")

try:
    from streamlit_pages.chat import chat_tab
except ImportError:
    logger.error("Failed to import chat_tab")
    async def chat_tab():
        st.error("Chat tab module not found")
        return None

try:
    from streamlit_pages.crew_tab import crew_tab
except ImportError:
    logger.error("Failed to import crew_tab")
    async def crew_tab():
        st.error("Crew tab module not found")
        return None

try:
    from streamlit_pages.terminal import terminal_tab
except ImportError:
    logger.error("Failed to import terminal_tab")
    def terminal_tab():
        st.error("Terminal tab module not found")

# Display icon and title in the sidebar
try:
    logo_paths = [
        'public/CogentxLightGrey.png',
        'assets/cogentx-logo.png',
        'public/Cogentx.png'
    ]

    logo_loaded = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.image(logo_path, width=1000)
            logo_loaded = True
            break

    if not logo_loaded:
        # Fallback to text header if no logo found
        st.title("⚡ Cogentx")
except Exception as e:
    # Fallback to text header on any error
    st.title("⚡ Cogentx")
    logger.debug(f"Logo loading error (using text fallback): {e}")

# Utilities and styles
from utils.utils import get_clients
from streamlit_pages.styles import load_css

# Streamlit pages
from streamlit_pages.intro import intro_tab
from streamlit_pages.environment import environment_tab
from streamlit_pages.database import database_tab
from streamlit_pages.documentation import documentation_tab
from streamlit_pages.agent_service import agent_service_tab
from streamlit_pages.mcp import mcp_tab
from streamlit_pages.logs import logs_tab, capture_terminal_output

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client, supabase = get_clients()

# Load custom CSS styles
load_css()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Helper functions from crew_stream.py
async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text for the user_input prompt."""
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
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

# Add crew_stream tab function
async def crew_stream_tab():
    """Display the Crew Stream interface."""
    st.markdown("""
    <h1>Cogentx Crew Stream</h1>
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
        st.session_state.messages = []

    # Display chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            async for chunk in run_agent_with_streaming(prompt):
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Rest of your existing code...

async def main():
    """Main function to run the Streamlit app."""

    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        if tab_name in ["Home", "Chat", "Template Browser", "Code Editor", "No-Code Builder", "Generated Code", "Crew"]:
            st.session_state.selected_tab = tab_name

    # Add sidebar navigation
    with st.sidebar:
        try:
            # Try multiple logo paths
            logo_paths = [
                'public/CogentxLightGrey.png',
                'assets/cogentx-logo.png',
                'public/Cogentx.png'
            ]

            logo_loaded = False
            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    st.image(logo_path, width=1000)
                    logo_loaded = True
                    break

            if not logo_loaded:
                # Fallback to text header if no logo found
                st.title("⚡ Cogentx")

        except Exception as e:
            # Fallback to text header on any error
            st.title("⚡ Cogentx")
            logger.debug(f"Logo loading error (using text fallback): {e}")

        # Navigation options with vertical buttons
        st.write("### Navigation")

        # Initialize session state for selected tab if not present
        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Home"

        # Vertical navigation buttons
        intro_button = st.button("Home", use_container_width=True, key="home_button")
        chat_button = st.button("Chat", use_container_width=True, key="chat_button")
        crew_button = st.button("Crew", use_container_width=True, key="crew_button")
        template_browser_button = st.button("Template Browser", use_container_width=True, key="template_browser_button")
        code_editor_button = st.button("Code Editor", use_container_width=True, key="code_editor_button")
        nocode_builder_button = st.button("No-Code Builder", use_container_width=True, key="nocode_builder_button")
        generated_code_button = st.button("Generated Code", use_container_width=True, key="generated_code_button")

        # Update selected tab based on button clicks
        if intro_button:
            st.session_state.selected_tab = "Home"
        elif chat_button:
            st.session_state.selected_tab = "Chat"
        elif crew_button:
            st.session_state.selected_tab = "Crew"
        elif template_browser_button:
            st.session_state.selected_tab = "Template Browser"
        elif code_editor_button:
            st.session_state.selected_tab = "Code Editor"
        elif nocode_builder_button:
            st.session_state.selected_tab = "No-Code Builder"
        elif generated_code_button:
            st.session_state.selected_tab = "Generated Code"

    # Display the selected tab
    if st.session_state.selected_tab == "Home":
        st.title("Cogentx - Home")
        home_tab()
    elif st.session_state.selected_tab == "Template Browser":
        st.title("Cogentx - Template Browser")
        template_browser_tab()
    elif st.session_state.selected_tab == "Code Editor":
        st.title("Cogentx - Code Editor")
        nocode_editor_tab()
    elif st.session_state.selected_tab == "No-Code Builder":
        st.title("Cogentx - No-Code Builder")
        nocode_builder_tab()
    elif st.session_state.selected_tab == "Chat":
        st.title("Cogentx - Agent Builder")
        try:
            await chat_tab()
        except Exception as e:
            st.error(f"Error in chat tab: {str(e)}")
            logger.error(f"Chat tab error: {str(e)}")
    elif st.session_state.selected_tab == "Crew":
        st.title("Cogentx - Crew")
        try:
            await crew_tab()
        except Exception as e:
            st.error(f"Error in crew tab: {str(e)}")
            logger.error(f"Crew tab error: {str(e)}")
    elif st.session_state.selected_tab == "Generated Code":
        st.title("Cogentx - Generated Code")

        # Create tabs for Code Editor and Terminal
        code_tab, terminal_tab_content = st.tabs(["Code Editor", "Terminal"])

        with code_tab:
            nocode_editor_tab()  # Use the existing nocode_editor_tab function

        with terminal_tab_content:
            terminal_tab()  # Use the existing terminal_tab function

if __name__ == "__main__":
    asyncio.run(main())
