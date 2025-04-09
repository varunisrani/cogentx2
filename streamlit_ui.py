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

# Import Crew AI Chat Tab
try:
    logger.info("Attempting to import crew_ai_chat_tab from streamlit_pages")
    from streamlit_pages.crew_ai_chat_tab import crew_ai_chat_tab
    logger.info("Successfully imported crew_ai_chat_tab")
except ImportError as e:
    logger.error(f"Failed to import crew_ai_chat_tab: {str(e)}")
    # Try the original import path as fallback
    try:
        logger.info("Attempting to import crew_ai_chat_tab from crewaii")
        from streamlit_pages.crew_ai_chat_tab import crew_ai_chat_tab
        logger.info("Successfully imported crew_ai_chat_tab from crewaii")
    except ImportError as e2:
        logger.error(f"Failed to import crew_ai_chat_tab from crewaii: {str(e2)}")
        async def crew_ai_chat_tab():
            st.error("Crew AI Chat tab module not found")
            return None
except Exception as e:
    logger.error(f"Unexpected error importing crew_ai_chat_tab: {str(e)}")
    async def crew_ai_chat_tab():
        st.error(f"Crew AI Chat tab error: {str(e)}")
        return None


try:
    from streamlit_pages.terminal import terminal_tab
except ImportError:
    logger.error("Failed to import terminal_tab")
    async def terminal_tab():
        st.error("Terminal tab module not found")
        return None

# Import the streamlit_terminal package
try:
    from streamlit_terminal import st_terminal
    HAS_STREAMLIT_TERMINAL = True
except Exception:
    HAS_STREAMLIT_TERMINAL = False

# Import our custom terminal implementation as a fallback
from custom_terminal import st_custom_terminal

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

async def main():
    """Main function to run the Streamlit app."""

    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        if tab_name in ["Home", "Chat", "Crew AI Chat", "Template Browser", "Code Editor", "No-Code Builder", "Generated Code"]:
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
        crew_ai_chat_button = st.button("Crew AI Chat", use_container_width=True, key="crew_ai_chat_button")  # New Crew AI Chat Button
        template_browser_button = st.button("Template Browser", use_container_width=True, key="template_browser_button")
        code_editor_button = st.button("Code Editor", use_container_width=True, key="code_editor_button")
        nocode_builder_button = st.button("No-Code Builder", use_container_width=True, key="nocode_builder_button")
        generated_code_button = st.button("Generated Code", use_container_width=True, key="generated_code_button")

        # Update selected tab based on button clicks
        if intro_button:
            st.session_state.selected_tab = "Home"
        elif chat_button:
            st.session_state.selected_tab = "Chat"
        elif crew_ai_chat_button:  # Handle Crew AI Chat button click
            st.session_state.selected_tab = "Crew AI Chat"
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
    elif st.session_state.selected_tab == "Crew AI Chat":  # Handle Crew AI Chat tab
        st.title("Cogentx - Crew AI Chat")
        try:
            # Import the initialize_clients function from crew_ai_chat_tab
            from streamlit_pages.crew_ai_chat_tab import initialize_clients
            # Initialize the clients in the crew_ai_chat_tab module
            initialize_clients(openai_client, supabase)
            # Call the crew_ai_chat_tab function
            await crew_ai_chat_tab()
        except Exception as e:
            st.error(f"Error in Crew AI Chat tab: {str(e)}")
            logger.error(f"Crew AI Chat tab error: {str(e)}")
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