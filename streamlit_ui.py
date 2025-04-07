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
    def chat_tab():
        st.error("Chat tab module not found")

try:
    from streamlit_pages.terminal import terminal_tab
except ImportError:
    logger.error("Failed to import terminal_tab")
    def terminal_tab():
        st.error("Terminal tab module not found")

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

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Archon",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Display icon and title in the sidebar
try:
    logo_path = 'assets/archon-logo.png'
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        st.sidebar.image(image, width=250)
    else:
        # Use emoji as fallback
        st.sidebar.title("🧙‍♂️ Archon")
except Exception as e:
    st.sidebar.title("🧙‍♂️ Archon")
    logger.error(f"Could not load logo: {e}")

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
from streamlit_pages.future_enhancements import future_enhancements_tab
from streamlit_pages.logs import logs_tab, capture_terminal_output

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client, supabase = get_clients()

# Load custom CSS styles
load_css()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

def run_process_with_live_output(cmd: list, cwd: str) -> queue.Queue:
    """Run a process and return a queue with its output."""
    # Use the capture_terminal_output function from logs.py to ensure logs are captured
    # in the system logs as well as returned in the queue
    return capture_terminal_output(cmd, cwd)

def create_workbench_zip() -> tuple[bytes, str]:
    """Create a zip file of the workbench directory."""
    workbench_dir = os.path.join(os.getcwd(), "workbench")

    # Create a BytesIO object to store the zip file
    zip_io = io.BytesIO()

    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the workbench directory
        for root, _, files in os.walk(workbench_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate arc_name (path within the zip file)
                arc_name = os.path.relpath(file_path, workbench_dir)
                zip_file.write(file_path, arc_name)

    # Get the current timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_filename = f"workbench_{timestamp}.zip"

    return zip_io.getvalue(), zip_filename

def format_log_message(message: str) -> str:
    """Format log messages for better readability with syntax highlighting cues."""
    # Add HTML classes for styling in the UI
    # Highlight timestamps
    message = re.sub(r'(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])',
                     r'<span class="token timestamp">\1</span>',
                     message)

    # Highlight success messages
    if "✅" in message or "success" in message.lower() or "completed" in message.lower():
        message = f'<span class="token success">{message}</span>'

    # Highlight error messages
    elif "❌" in message or "error" in message.lower() or "failed" in message.lower() or "exception" in message.lower():
        message = f'<span class="token error">{message}</span>'

    return message

def display_formatted_logs(logs: list) -> str:
    """Convert log messages to formatted HTML."""
    formatted_logs = []
    for log in logs:
        formatted_logs.append(format_log_message(log))

    # Return raw HTML that will be displayed
    return '\n'.join(formatted_logs)

def display_workbench_code():
    """Display the generated code from the workbench folder."""
    st.write("### Generated Agent Code")

    # Get the workbench directory path
    workbench_dir = os.path.join(os.getcwd(), "workbench")

    if not os.path.exists(workbench_dir):
        st.warning("No generated code found. The workbench directory does not exist.")
        return

    # Get all files in the workbench directory
    files = sorted([f for f in os.listdir(workbench_dir) if os.path.isfile(os.path.join(workbench_dir, f))])

    if not files:
        st.warning("No files found in the workbench directory.")
        return

    # Initialize session state for logs if not present
    if "log_output" not in st.session_state:
        st.session_state.log_output = []

    # Initialize tab selection state if not present
    if "code_tab" not in st.session_state:
        st.session_state.code_tab = "Files"

    # Initialize terminal state if not present
    if "terminal_initialized" not in st.session_state:
        st.session_state.terminal_initialized = False

    # Initialize run_command state if not present
    if "run_command" not in st.session_state:
        st.session_state.run_command = None

    # Add download button for the entire workbench folder
    col_stats, col_download, col_run = st.columns([2, 1, 1])
    with col_stats:
        total_size = sum(os.path.getsize(os.path.join(workbench_dir, f)) for f in files)
        st.write(f"📁 Total files: {len(files)}")
        st.write(f"💾 Total size: {total_size / 1024:.1f} KB")

    with col_download:
        try:
            zip_data, zip_filename = create_workbench_zip()
            st.download_button(
                label="📥 Download All Files",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip",
                help="Download all generated files as a ZIP archive",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")

    with col_run:
        if st.button("▶️ Run Code", use_container_width=True, help="Run the generated code"):
            # Check if main.py exists
            main_py_path = os.path.join(workbench_dir, "main.py")
            if not os.path.exists(main_py_path):
                st.error("main.py not found in workbench directory!")
                return

            # Clear previous logs
            st.session_state.log_output = []
            # Switch to Logs tab
            st.session_state.code_tab = "Logs"

            # Create containers for output
            log_container = st.container()
            timer_placeholder = st.empty()

            # Record start time
            start_time = time.time()

            # Add timestamp to the terminal output
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.log_output.append(f"[{timestamp}] Starting code execution...\n")

            # Display the initial message
            with log_container:
                st.code('\n'.join(st.session_state.log_output), language='bash')

            try:
                # Get Python executable path
                python_exe = sys.executable

                # Create a unique terminal ID for this run
                terminal_id = f"code_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Start the process
                output_queue = run_process_with_live_output(
                    [python_exe, "main.py"],
                    cwd=workbench_dir
                )

                # Also create an entry in terminal_processes for interactive use
                if "terminal_processes" not in st.session_state:
                    st.session_state.terminal_processes = {}

                # Start the process for interactive terminal
                try:
                    process = subprocess.Popen(
                        [python_exe, "main.py"],
                        cwd=workbench_dir,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        universal_newlines=True
                    )

                    # Create a new entry in terminal_processes
                    st.session_state.terminal_processes[terminal_id] = {
                        "process": process,
                        "command": f"{python_exe} main.py",
                        "cwd": workbench_dir,
                        "output": [f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started code execution: {python_exe} main.py\n"],
                        "start_time": time.time()
                    }

                    # Set as active terminal
                    st.session_state.active_terminal = terminal_id

                    # Start a thread to read output
                    from streamlit_pages.logs import read_process_output
                    threading.Thread(
                        target=read_process_output,
                        args=(process, terminal_id),
                        daemon=True
                    ).start()

                    # Add a message and button to switch to the Logs tab for interactive terminal
                    st.info("Process started in interactive terminal.")
                    if st.button("Go to Interactive Terminal", key="goto_terminal_button"):
                        st.session_state.selected_tab = "Logs"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error creating interactive terminal: {str(e)}")

                # Process the output
                return_code = None
                while return_code is None:
                    try:
                        msg_type, msg = output_queue.get(timeout=0.1)
                        if msg_type == 'output':
                            # Add timestamp to the output
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_message = f"[{timestamp}] {msg.strip()}"
                            st.session_state.log_output.append(log_message)
                            # Also add to system logs
                            if "system_logs" in st.session_state:
                                st.session_state.system_logs.append(log_message)
                            # Update the display
                            with log_container:
                                st.code('\n'.join(st.session_state.log_output), language='bash')

                            # Update elapsed time
                            elapsed = time.time() - start_time
                            timer_placeholder.markdown(f'<div class="timer-display">Elapsed time: {int(elapsed//60):02d}:{int(elapsed%60):02d}</div>', unsafe_allow_html=True)

                        elif msg_type == 'return_code':
                            return_code = msg
                        elif msg_type == 'error':
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            error_message = f"[{timestamp}] ❌ ERROR: {msg}"
                            st.session_state.log_output.append(error_message)
                            # Also add to system logs
                            if "system_logs" in st.session_state:
                                st.session_state.system_logs.append(error_message)
                            with log_container:
                                st.code('\n'.join(st.session_state.log_output), language='bash')
                            return_code = -1
                    except queue.Empty:
                        # Update elapsed time
                        elapsed = time.time() - start_time
                        timer_placeholder.markdown(f'<div class="timer-display">Elapsed time: {int(elapsed//60):02d}:{int(elapsed%60):02d}</div>', unsafe_allow_html=True)
                        continue

                # Add final status
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

                if return_code == 0:
                    completion_message = f"[{timestamp}] ✅ Code execution completed successfully! Total time: {elapsed_str}"
                    st.session_state.log_output.append(completion_message)
                    # Also add to system logs
                    if "system_logs" in st.session_state:
                        st.session_state.system_logs.append(completion_message)
                else:
                    failure_message = f"[{timestamp}] ❌ Code execution failed with return code {return_code}. Total time: {elapsed_str}"
                    st.session_state.log_output.append(failure_message)
                    # Also add to system logs
                    if "system_logs" in st.session_state:
                        st.session_state.system_logs.append(failure_message)

                with log_container:
                    st.code('\n'.join(st.session_state.log_output), language='bash')

                # Final elapsed time
                timer_placeholder.markdown(f'<div class="timer-display">Total time: {elapsed_str}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error running code: {str(e)}")

    # Create tabs for Files, Logs, and Terminal
    files_tab, logs_tab, terminal_tab = st.tabs(["Files", "Logs", "Terminal"])

    # Set the active tab based on session state
    if st.session_state.log_output and st.session_state.code_tab == "Logs":
        # Auto-select the Logs tab if there are logs and we were previously on the Logs tab
        st.session_state.code_tab = "Logs"
    elif st.session_state.code_tab == "Terminal":
        # Keep Terminal tab selected if it was previously selected
        st.session_state.code_tab = "Terminal"
    else:
        # Otherwise default to Files tab
        st.session_state.code_tab = "Files"

    # Files Tab Content
    with files_tab:
        if files_tab.checkbox("Select this tab", value=(st.session_state.code_tab == "Files"), key="select_files_tab"):
            st.session_state.code_tab = "Files"

        # Create columns for better organization
        col1, col2 = st.columns([1, 3])

        # File selector in the first column
        with col1:
            st.write("#### Files")
            selected_file = st.selectbox(
                "Select a file to view:",
                files,
                format_func=lambda x: f"📄 {x}"
            )

            if selected_file:
                # Add individual file download button
                file_path = os.path.join(workbench_dir, selected_file)
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    st.download_button(
                        label="📥 Download This File",
                        data=file_content,
                        file_name=selected_file,
                        mime="text/plain",
                        help=f"Download {selected_file}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")

        # File content in the second column
        with col2:
            if selected_file:
                file_path = os.path.join(workbench_dir, selected_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    st.write(f"#### {selected_file}")

                    # Add copy button
                    if st.button("📋 Copy Code", key=f"copy_{selected_file}"):
                        st.toast(f"Copied {selected_file} to clipboard!")
                        st.write(f"<div style='position: absolute; left: -9999px;'>{content}</div>", unsafe_allow_html=True)
                        st.markdown(f"<script>navigator.clipboard.writeText(`{content}`);</script>", unsafe_allow_html=True)

                    # Show the code with syntax highlighting
                    file_extension = os.path.splitext(selected_file)[1].lower()
                    if file_extension in ['.py', '.pyw']:
                        st.code(content, language='python')
                    elif file_extension == '.json':
                        st.code(content, language='json')
                    elif file_extension == '.md':
                        st.markdown(content)
                    else:
                        st.code(content)

                    # Show file info
                    st.write("---")
                    file_stats = os.stat(file_path)
                    st.write(f"File size: {file_stats.st_size} bytes")
                    st.write(f"Last modified: {time.ctime(file_stats.st_mtime)}")
                except Exception as e:
                    st.error(f"Error reading file {selected_file}: {str(e)}")

    # Logs Tab Content
    with logs_tab:
        if logs_tab.checkbox("Select this tab", value=(st.session_state.code_tab == "Logs"), key="select_logs_tab"):
            st.session_state.code_tab = "Logs"

        if st.session_state.log_output:
            st.write("### Log Output")

            # Add custom CSS for log styling (without fullscreen)
            st.markdown("""
            <style>
            /* Log entries styling */
            .logs-container {
                background-color: #1e1e1e;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                max-height: 600px;
                overflow-y: auto;
            }

            .log-entry {
                background-color: #2d2d2d;
                border-radius: 4px;
                margin-bottom: 6px;
                padding: 10px 15px;
                color: #ffffff;
                font-size: 14px;
                line-height: 1.5;
                border-left: 3px solid #4CAF50;
                position: relative;
                overflow: hidden;
            }

            .timestamp {
                color: #4CAF50;
                font-weight: normal;
            }

            .log-content {
                color: #f0f0f0;
                white-space: pre-wrap;
                word-break: break-word;
                margin-left: 5px;
            }

            /* Success and error messages */
            .success-message {
                color: #4CAF50;
            }

            .error-message {
                color: #FF5252;
            }
            </style>
            """, unsafe_allow_html=True)

            # First display logs in a code block for potential copying
            with st.expander("Raw Logs (for copying)", expanded=False):
                st.code('\n'.join(st.session_state.log_output), language='bash')

            # Display logs with nice formatting
            st.markdown("<div class='logs-container'>", unsafe_allow_html=True)

            for log in st.session_state.log_output:
                # Extract timestamp if present
                timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', log)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    message = log.replace(f"[{timestamp}]", "").strip()

                    # Determine message type
                    css_class = "log-content"
                    if "✅" in message or "success" in message.lower() or "completed" in message.lower():
                        css_class += " success-message"
                    elif "❌" in message or "error" in message.lower() or "failed" in message.lower() or "exception" in message.lower():
                        css_class += " error-message"

                    st.markdown(f"""
                    <div class="log-entry">
                        <span class="timestamp">[{timestamp}]</span>
                        <span class="{css_class}">{message}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # If no timestamp, just display the message
                    st.markdown(f"""
                    <div class="log-entry">
                        <span class="log-content">{log}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No logs available. Run the code to see execution logs here.")

    # Terminal Tab Content
    with terminal_tab:
        if terminal_tab.checkbox("Select this tab", value=(st.session_state.code_tab == "Terminal"), key="select_terminal_tab"):
            st.session_state.code_tab = "Terminal"

        st.write("### Interactive Terminal")
        st.write("Use this terminal to run commands, execute Python scripts, and interact with your code.")

        # Add custom CSS for terminal styling
        st.markdown("""
        <style>
        /* Terminal styling */
        .terminal-container {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        /* Make the terminal input field more prominent */
        .terminal-container .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: #f0f0f0;
            border: 1px solid #4CAF50;
            border-radius: 4px;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        }

        /* Style the terminal output */
        .terminal-container pre {
            background-color: #2d2d2d;
            color: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a container for the terminal with a nice height
        with st.container():
            st.markdown("<div class='terminal-container'>", unsafe_allow_html=True)

            # Add a description and instructions
            st.write("Type commands like `python main.py`, `ls`, or any other shell command.")

            # Create the terminal
            # If we have a command to run, pass it to the terminal
            command = st.session_state.run_command if st.session_state.run_command else ""

            # Clear the run_command after using it
            if st.session_state.run_command:
                # We'll clear it after the rerun
                pass

            # Create the terminal with the command
            welcome_message = "Welcome to the Archon Terminal! You can run commands here to interact with your code."

            # Use the appropriate terminal implementation
            if HAS_STREAMLIT_TERMINAL:
                try:
                    _ = st_terminal(
                        key="workbench_terminal",
                        height=400,
                        command=command,
                        show_welcome_message=True,
                        welcome_message=welcome_message
                    )
                except Exception as e:
                    st.error(f"Error initializing terminal: {str(e)}")
                    st.info("Falling back to custom terminal implementation...")
                    _ = st_custom_terminal(
                        key="workbench_terminal",
                        command=command,
                        height=400,
                        show_welcome_message=True,
                        welcome_message=welcome_message
                    )
            else:
                # Use our custom terminal implementation
                _ = st_custom_terminal(
                    key="workbench_terminal",
                    command=command,
                    height=400,
                    show_welcome_message=True,
                    welcome_message=welcome_message
                )

            # Clear the run_command after the terminal is created
            if st.session_state.run_command:
                st.session_state.run_command = None

            st.markdown("</div>", unsafe_allow_html=True)

            # Add quick command buttons
            st.write("### Quick Commands")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("List Files", key="btn_ls"):
                    # We can't directly run the command, but we can set a flag to run it
                    st.session_state.run_command = "ls -la"
                    st.rerun()

            with col2:
                if st.button("Check Python Version", key="btn_python_version"):
                    st.session_state.run_command = "python --version"
                    st.rerun()

            with col3:
                if st.button("Check Node Version", key="btn_node_version"):
                    st.session_state.run_command = "node --version"
                    st.rerun()

            # Second row of buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("List Python Packages", key="btn_pip_list"):
                    st.session_state.run_command = "pip list"
                    st.rerun()

            with col2:
                if st.button("Run Pydantic AI", key="btn_pydantic_ai"):
                    st.session_state.run_command = "python -m pydantic_ai.cli run"
                    st.rerun()

            with col3:
                if st.button("Check npm Version", key="btn_npm_version"):
                    st.session_state.run_command = "npm --version"
                    st.rerun()

            # Third row of buttons - MCP Agents
            st.write("### Run MCP Agents")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Run Spotify Agent (Terminal)", key="btn_spotify_agent"):
                    st.session_state.run_command = "cd spotify_agent && python main.py"
                    st.rerun()

            with col2:
                if st.button("Run GitHub Agent", key="btn_github_agent"):
                    st.session_state.run_command = "cd github_agent && python main.py"
                    st.rerun()

            with col3:
                if st.button("Run Spotify Streamlit App", key="btn_spotify_streamlit"):
                    st.session_state.run_command = "streamlit run spotify_streamlit_app.py"
                    st.rerun()

            # Add some helpful examples
            with st.expander("Example Commands"):
                st.markdown("""
                - `ls -la` - List all files in the current directory
                - `python main.py` - Run the main.py script
                - `python -m pydantic_ai.cli run` - Run the Pydantic AI CLI
                - `npm --version` - Check npm version
                - `node --version` - Check Node.js version
                - `pip list` - List installed Python packages
                - `cd spotify_agent && python main.py` - Run the Spotify agent
                - `cd github_agent && python main.py` - Run the GitHub agent
                """)

async def main():
    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        if tab_name in ["Home", "Chat", "Agent Runner", "Template Browser", "Workbench", "Code Editor", "No-Code Builder", "Terminal", "Generated Code", "Environment", "Database", "Documentation", "Agent Service", "MCP", "Logs", "Future Enhancements"]:
            st.session_state.selected_tab = tab_name

    # Add sidebar navigation
    with st.sidebar:
        st.image("public/ArchonLightGrey.png", width=1000)

        # Navigation options with vertical buttons
        st.write("### Navigation")

        # Initialize session state for selected tab if not present
        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Home"

        # Initialize system logs if not present
        if "system_logs" not in st.session_state:
            st.session_state.system_logs = []

        # Vertical navigation buttons
        intro_button = st.button("Home", use_container_width=True, key="home_button")
        chat_button = st.button("Chat", use_container_width=True, key="chat_button")
        agent_runner_button = st.button("Agent Runner", use_container_width=True, key="agent_runner_button")
        template_browser_button = st.button("Template Browser", use_container_width=True, key="template_browser_button")
        workbench_button = st.button("Workbench", use_container_width=True, key="workbench_button")
        code_editor_button = st.button("Code Editor", use_container_width=True, key="code_editor_button")
        nocode_builder_button = st.button("No-Code Builder", use_container_width=True, key="nocode_builder_button")
        generated_code_button = st.button("Generated Code", use_container_width=True, key="generated_code_button")
        env_button = st.button("Environment", use_container_width=True, key="env_button")
        db_button = st.button("Database", use_container_width=True, key="db_button")
        docs_button = st.button("Documentation", use_container_width=True, key="docs_button")
        service_button = st.button("Agent Service", use_container_width=True, key="service_button")
        mcp_button = st.button("MCP", use_container_width=True, key="mcp_button")
        logs_button = st.button("Logs", use_container_width=True, key="logs_button")
        terminal_button = st.button("Terminal", use_container_width=True, key="terminal_button")
        future_enhancements_button = st.button("Future Enhancements", use_container_width=True, key="future_enhancements_button")

        # Update selected tab based on button clicks
        if intro_button:
            st.session_state.selected_tab = "Home"
        elif chat_button:
            st.session_state.selected_tab = "Chat"
        elif agent_runner_button:
            st.session_state.selected_tab = "Agent Runner"
        elif template_browser_button:
            st.session_state.selected_tab = "Template Browser"
        elif workbench_button:
            st.session_state.selected_tab = "Workbench"
        elif code_editor_button:
            st.session_state.selected_tab = "Code Editor"
        elif nocode_builder_button:
            st.session_state.selected_tab = "No-Code Builder"
        elif generated_code_button:
            st.session_state.selected_tab = "Generated Code"
        elif env_button:
            st.session_state.selected_tab = "Environment"
        elif db_button:
            st.session_state.selected_tab = "Database"
        elif docs_button:
            st.session_state.selected_tab = "Documentation"
        elif service_button:
            st.session_state.selected_tab = "Agent Service"
        elif mcp_button:
            st.session_state.selected_tab = "MCP"
        elif logs_button:
            st.session_state.selected_tab = "Logs"
        elif terminal_button:
            st.session_state.selected_tab = "Terminal"
        elif future_enhancements_button:
            st.session_state.selected_tab = "Future Enhancements"

    # Display the selected tab
    if st.session_state.selected_tab == "Home":
        st.title("Archon - Home")
        home_tab()
    elif st.session_state.selected_tab == "Agent Runner":
        st.title("Archon - Agent Runner")
        agent_runner_tab()
    elif st.session_state.selected_tab == "Template Browser":
        st.title("Archon - Template Browser")
        template_browser_tab()
    elif st.session_state.selected_tab == "Workbench":
        st.title("Archon - Workbench")
        workbench_tab()
    elif st.session_state.selected_tab == "Code Editor":
        st.title("Archon - Code Editor")
        nocode_editor_tab()
    elif st.session_state.selected_tab == "No-Code Builder":
        st.title("Archon - No-Code Builder")
        nocode_builder_tab()
    elif st.session_state.selected_tab == "Chat":
        st.title("Archon - Agent Builder")
        await chat_tab()
    elif st.session_state.selected_tab == "Terminal":
        st.title("Archon - Terminal")
        terminal_tab()
    elif st.session_state.selected_tab == "Generated Code":
        st.title("Archon - Generated Code")
        display_workbench_code()
    elif st.session_state.selected_tab == "MCP":
        st.title("Archon - MCP Configuration")
        mcp_tab()
    elif st.session_state.selected_tab == "Environment":
        st.title("Archon - Environment Configuration")
        environment_tab()
    elif st.session_state.selected_tab == "Agent Service":
        st.title("Archon - Agent Service")
        agent_service_tab()
    elif st.session_state.selected_tab == "Database":
        st.title("Archon - Database Configuration")
        database_tab(supabase)
    elif st.session_state.selected_tab == "Documentation":
        st.title("Archon - Documentation")
        documentation_tab(supabase)
    elif st.session_state.selected_tab == "Logs":
        st.title("Archon - System Logs")
        logs_tab()
    elif st.session_state.selected_tab == "Future Enhancements":
        st.title("Archon - Future Enhancements")
        future_enhancements_tab()

if __name__ == "__main__":
    asyncio.run(main())
