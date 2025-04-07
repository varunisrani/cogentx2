import streamlit as st
import subprocess
import sys
import os
import threading
import queue
from datetime import datetime
import time
import re
import html

# Try to import streamlit_ace, but provide a fallback if it's not available
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False

# Try to import rich for ANSI color handling
try:
    from rich.console import Console
    from io import StringIO
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Try to import streamlit-terminal
try:
    from streamlit_terminal import st_terminal
    TERMINAL_AVAILABLE = True
except ImportError:
    TERMINAL_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Interactive Terminal", layout="wide")

st.title("Interactive Terminal")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'full_logs' not in st.session_state:
    st.session_state.full_logs = []
if 'current_dir' not in st.session_state:
    st.session_state.current_dir = os.getcwd()
if 'process' not in st.session_state:
    st.session_state.process = None
if 'output_queue' not in st.session_state:
    st.session_state.output_queue = queue.Queue()
if 'command_history' not in st.session_state:
    st.session_state.command_history = []
if 'history_index' not in st.session_state:
    st.session_state.history_index = -1
if 'current_command' not in st.session_state:
    st.session_state.current_command = ""
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'terminal_id' not in st.session_state:
    st.session_state.terminal_id = 0

def run_command(command):
    # Add command to history with shell prompt
    shell_prompt = f"{os.getlogin()}@{os.uname().nodename}:{os.path.basename(st.session_state.current_dir)}$ "

    if command.strip() and (not st.session_state.command_history or st.session_state.command_history[-1] != command):
        st.session_state.command_history.append(command)
        st.session_state.history_index = -1

    # Handle cd command specially
    if command.strip().startswith('cd '):
        try:
            path = command.strip()[3:].strip()
            if not os.path.isabs(path):
                path = os.path.join(st.session_state.current_dir, path)
            path = os.path.normpath(path)
            if os.path.isdir(path):
                st.session_state.current_dir = path
                st.session_state.history.append(f"{shell_prompt}{command}")
                st.session_state.history.append(f"Changed directory to {path}")
                st.session_state.full_logs.append(f"{shell_prompt}{command}")
                st.session_state.full_logs.append(f"Changed directory to {path}")
                return
            else:
                error_msg = f"cd: {path}: No such file or directory"
                st.session_state.history.append(f"{shell_prompt}{command}")
                st.session_state.history.append(error_msg)
                st.session_state.full_logs.append(f"{shell_prompt}{command}")
                st.session_state.full_logs.append(error_msg)
                return
        except Exception as e:
            error_msg = f"cd: error: {str(e)}"
            st.session_state.history.append(f"{shell_prompt}{command}")
            st.session_state.history.append(error_msg)
            st.session_state.full_logs.append(f"{shell_prompt}{command}")
            st.session_state.full_logs.append(error_msg)
            return

    # For other commands
    try:
        st.session_state.is_running = True
        st.session_state.history.append(f"{shell_prompt}{command}")
        st.session_state.full_logs.append(f"{shell_prompt}{command}")

        # Create a new stream session for this command
        if 'stream_sessions' not in st.session_state:
            st.session_state.stream_sessions = {}

        # Configure process with unbuffered output for real-time streaming
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Force Python to run unbuffered

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for better streaming
            universal_newlines=True,
            cwd=st.session_state.current_dir,
            env=env
        )

        # Increment terminal ID for tracking
        st.session_state.terminal_id += 1
        process_id = st.session_state.terminal_id

        # Initialize stream session
        st.session_state.stream_sessions[process_id] = {
            'command': command,
            'start_time': datetime.now(),
            'output': [],
            'status': 'running',
            'return_code': None
        }

        def reader():
            for line in iter(process.stdout.readline, ''):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

                # Process ANSI color codes if present
                processed_line = line.rstrip()
                if '\x1b[' in processed_line and RICH_AVAILABLE:
                    # Use rich to handle ANSI codes if available
                    console = Console(file=StringIO(), color_system="truecolor")
                    console.print(processed_line, end="")
                    processed_line = console.file.getvalue()

                # Store raw output for stream session
                st.session_state.stream_sessions[process_id]['output'].append((timestamp, processed_line))

                # Send to output queue for display
                st.session_state.output_queue.put(('output', processed_line, timestamp, process_id))

            # Process has finished
            process.stdout.close()
            return_code = process.wait()

            # Update stream session status
            st.session_state.stream_sessions[process_id]['status'] = 'completed'
            st.session_state.stream_sessions[process_id]['return_code'] = return_code
            st.session_state.stream_sessions[process_id]['end_time'] = datetime.now()

            # Send completion status
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if return_code != 0:
                st.session_state.output_queue.put(('return_code', return_code, timestamp, process_id))

            st.session_state.is_running = False
            # Add new prompt after command completes
            st.session_state.output_queue.put(('prompt', shell_prompt, timestamp, process_id))

        thread = threading.Thread(target=reader)
        thread.daemon = True
        thread.start()

        st.session_state.process = process
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.history.append(error_msg)
        st.session_state.full_logs.append(error_msg)
        st.session_state.is_running = False
        # Add new prompt after error
        st.session_state.history.append(shell_prompt)
        st.session_state.full_logs.append(shell_prompt)

def display_output():
    last_update = time.time()
    shell_prompt = f"{os.getlogin()}@{os.uname().nodename}:{os.path.basename(st.session_state.current_dir)}$ "

    # Create a container for the streaming terminal with VS Code-like styling
    terminal_html = """
    <div style="background-color: #1E1E1E; color: #CCCCCC; font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
              padding: 10px; border-radius: 5px; height: 400px; overflow-y: auto; line-height: 1.4;"
         id="terminal-output">
    </div>
    <script>
        // Auto-scroll function
        function scrollTerminalToBottom() {
            const terminal = document.getElementById('terminal-output');
            if (terminal) {
                terminal.scrollTop = terminal.scrollHeight;
            }
        }
        // Call initially and set up a MutationObserver to watch for changes
        scrollTerminalToBottom();
        const observer = new MutationObserver(scrollTerminalToBottom);
        const terminal = document.getElementById('terminal-output');
        if (terminal) {
            observer.observe(terminal, { childList: true, subtree: true });
        }
    </script>
    """

    # Initialize the terminal container
    terminal_container = st.empty()
    terminal_container.markdown(terminal_html, unsafe_allow_html=True)

    # Function to update the terminal display with VS Code-like styling
    def update_terminal_display():
        # Format the output with proper styling
        formatted_output = []
        for entry in st.session_state.history[-100:]:  # Show last 100 lines for performance
            if '@' in entry and entry.endswith('$ '):  # Shell prompt
                formatted_output.append(f'<span style="color: #569CD6; font-weight: bold;">{html.escape(entry)}</span>')
            elif entry.startswith('$'):
                # Command styling
                formatted_output.append(f'<span style="color: #569CD6; font-weight: bold;">{html.escape(entry)}</span>')
            elif 'Command failed with exit code' in entry:
                # Error styling
                formatted_output.append(f'<span style="color: #F14C4C;">{html.escape(entry)}</span>')
            elif 'Error:' in entry:
                # Error styling
                formatted_output.append(f'<span style="color: #F14C4C;">{html.escape(entry)}</span>')
            else:
                # Regular output styling - detect common output patterns
                if entry.strip().startswith(('import ', 'from ', 'def ', 'class ')) and entry.strip().endswith((':')):
                    # Python code
                    formatted_output.append(f'<span style="color: #C586C0;">{html.escape(entry)}</span>')
                elif re.search(r'\b(True|False|None|\d+)\b', entry):
                    # Python values
                    formatted_output.append(f'<span style="color: #4EC9B0;">{html.escape(entry)}</span>')
                elif re.search(r'\[.*\]|\{.*\}|\(.*\)', entry):
                    # Data structures
                    formatted_output.append(f'<span style="color: #CE9178;">{html.escape(entry)}</span>')
                else:
                    # Regular output
                    formatted_output.append(f'<span style="color: #CCCCCC;">{html.escape(entry)}</span>')

        # Join with line breaks and update the terminal
        terminal_html = """
        <div style="background-color: #1E1E1E; color: #CCCCCC; font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                  padding: 10px; border-radius: 5px; height: 400px; overflow-y: auto; line-height: 1.4;"
             id="terminal-output">
            {}
        </div>
        <script>
            // Auto-scroll function
            function scrollTerminalToBottom() {
                const terminal = document.getElementById('terminal-output');
                if (terminal) {
                    terminal.scrollTop = terminal.scrollHeight;
                }
            }
            // Call the function
            scrollTerminalToBottom();
        </script>
        """.format('<br>'.join(formatted_output))

        terminal_container.markdown(terminal_html, unsafe_allow_html=True)

    while st.session_state.is_running:
        try:
            msg_type, msg, timestamp, process_id = st.session_state.output_queue.get(timeout=0.1)

            if msg_type == 'output':
                # Add timestamp if provided
                if timestamp:
                    log_entry = f"{msg}"
                else:
                    log_entry = msg

                st.session_state.history.append(log_entry)
                st.session_state.full_logs.append(log_entry)
            elif msg_type == 'return_code':
                if msg != 0:
                    error_msg = f"Command failed with exit code {msg}"
                    st.session_state.history.append(error_msg)
                    st.session_state.full_logs.append(error_msg)

                # Add execution time information if available
                if 'stream_sessions' in st.session_state and process_id in st.session_state.stream_sessions:
                    session = st.session_state.stream_sessions[process_id]
                    if 'start_time' in session and 'end_time' in session:
                        execution_time = (session['end_time'] - session['start_time']).total_seconds()
                        time_info = f"Execution time: {execution_time:.3f} seconds"
                        st.session_state.history.append(time_info)
                        st.session_state.full_logs.append(time_info)

                st.session_state.history.append(shell_prompt)
                st.session_state.full_logs.append(shell_prompt)
                break
            elif msg_type == 'prompt':
                st.session_state.history.append(msg)
                st.session_state.full_logs.append(msg)
                break

            # Update display periodically to avoid too many refreshes
            current_time = time.time()
            if current_time - last_update > 0.1:  # Update every 100ms for more responsive feel
                update_terminal_display()
                last_update = current_time

        except queue.Empty:
            continue

    # Final update after process completes
    update_terminal_display()

# Create tabs for different views
if TERMINAL_AVAILABLE:
    tab1, tab2, tab3 = st.tabs(["Terminal", "Full Logs", "Streamlit Terminal"])
else:
    tab1, tab2 = st.tabs(["Terminal", "Full Logs"])

with tab1:
    # Command input section
    st.subheader("Command Input")

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        # Use a text input for simple commands
        simple_command = st.text_input(
            "Quick Command:",
            value=st.session_state.current_command,
            key="command_text",
            placeholder="Type command here and press Enter or click Execute"
        )

        # Use st_ace for multi-line commands if available, otherwise use a text area
        if ACE_AVAILABLE:
            advanced_command = st_ace(
                value="",
                language="sh",  # Using 'sh' instead of 'bash' as it's more commonly available
                theme="monokai",
                font_size=14,
                show_gutter=False,
                show_print_margin=False,
                wrap=True,
                auto_update=True,
                height=100,
                key="advanced_command"
            )
        else:
            advanced_command = st.text_area(
                "Multi-line Command:",
                height=100,
                key="advanced_command"
            )

        # Store current command in session state
        if simple_command:
            st.session_state.current_command = simple_command

    with col2:
        execute_simple = st.button("Execute Command")
        execute_script = st.button("Run Script")

    with col3:
        if st.button("Clear Logs"):
            st.session_state.history = []
            st.session_state.full_logs = []
            st.experimental_rerun()

    # Execute command when button is clicked or Enter is pressed
    command_to_run = None
    if execute_simple and simple_command.strip():
        command_to_run = simple_command
    elif execute_script and advanced_command.strip():
        command_to_run = advanced_command

    if command_to_run:
        run_command(command_to_run)
        # Clear the command input after execution
        st.session_state.current_command = ""

    # Terminal output display
    st.subheader("Terminal Output")

    # If we have history but no active process, show a static view
    if st.session_state.history and not st.session_state.is_running:
        # Format the output with proper styling
        formatted_output = []
        for entry in st.session_state.history[-100:]:  # Show last 100 lines for performance
            if entry.startswith('$'):
                # Command styling
                formatted_output.append(f'<span style="color: #569CD6; font-weight: bold;">{html.escape(entry)}</span>')
            elif '\u274c' in entry:
                # Error styling
                formatted_output.append(f'<span style="color: #F14C4C;">{html.escape(entry)}</span>')
            elif '\u2705' in entry:
                # Success styling
                formatted_output.append(f'<span style="color: #6A9955;">{html.escape(entry)}</span>')
            else:
                # Regular output styling
                formatted_output.append(f'<span style="color: #CCCCCC;">{html.escape(entry)}</span>')

        # Join with line breaks and display the terminal
        terminal_html = """
        <div style="background-color: #1E1E1E; color: #CCCCCC; font-family: 'Courier New', monospace;
                  padding: 10px; border-radius: 5px; height: 400px; overflow-y: auto;"
             id="terminal-output">
            {}
        </div>
        <script>
            // Auto-scroll function
            function scrollTerminalToBottom() {
                const terminal = document.getElementById('terminal-output');
                if (terminal) {
                    terminal.scrollTop = terminal.scrollHeight;
                }
            }
            // Call the function
            scrollTerminalToBottom();
        </script>
        """.format('<br>'.join(formatted_output))

        st.markdown(terminal_html, unsafe_allow_html=True)

    # Interactive input for running processes
    if st.session_state.is_running and st.session_state.process and st.session_state.process.poll() is None:
        st.subheader("Process Input")
        col1, col2 = st.columns([4, 1])
        with col1:
            process_input = st.text_input("Send input to process:", key="process_input")
        with col2:
            send_button = st.button("Send")

        if send_button and process_input:
            try:
                st.session_state.process.stdin.write(process_input + "\n")
                st.session_state.process.stdin.flush()
                timestamp = datetime.now().strftime("%H:%M:%S")
                input_entry = f"[INPUT] {process_input}"
                st.session_state.history.append(input_entry)
                st.session_state.full_logs.append(input_entry)
            except Exception as e:
                st.error(f"Failed to send input: {str(e)}")

    # Add keyboard shortcuts for command history
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        const input = document.querySelector('#command_text');
        if (!input) return;

        if (e.key === 'ArrowUp') {
            e.preventDefault();
            const event = new CustomEvent('command_history_prev');
            document.dispatchEvent(event);
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            const event = new CustomEvent('command_history_next');
            document.dispatchEvent(event);
        }
    });
    </script>
    """, unsafe_allow_html=True)

    # Handle command history navigation
    if st.session_state.history_index == -1:
        st.session_state.history_index = len(st.session_state.command_history)

    # Add keyboard shortcut handlers
    if st.session_state.get('command_history_prev', False):
        if st.session_state.history_index > 0:
            st.session_state.history_index -= 1
            st.session_state.current_command = st.session_state.command_history[st.session_state.history_index]
        st.session_state.command_history_prev = False

    if st.session_state.get('command_history_next', False):
        if st.session_state.history_index < len(st.session_state.command_history):
            st.session_state.history_index += 1
            if st.session_state.history_index == len(st.session_state.command_history):
                st.session_state.current_command = ""
        else:
                st.session_state.current_command = st.session_state.command_history[st.session_state.history_index]
        st.session_state.command_history_next = False

    # Update command input style to look more like a terminal
    st.markdown("""
    <style>
        .stTextInput input {
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace !important;
            font-size: 14px !important;
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
            border: 1px solid #3c3c3c !important;
            padding: 8px 12px !important;
        }

        .stTextInput input:focus {
            border-color: #0078d4 !important;
            box-shadow: none !important;
        }

        /* Hide the label */
        .stTextInput label {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

with tab2:
    # Full logs display
    st.subheader("Complete Command History")
    if st.session_state.full_logs:
        # Create a downloadable version of the logs
        log_text = "\n".join(st.session_state.full_logs)
        st.download_button(
            label="Download Logs",
            data=log_text,
            file_name="terminal_logs.txt",
            mime="text/plain",
        )

        # Display logs with syntax highlighting
        st.markdown("""
        <style>
        .terminal-logs {
            background-color: #1E1E1E;
            color: #CCCCCC;
            font-family: 'Courier New', monospace;
            padding: 10px;
            border-radius: 5px;
            height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .command-line { color: #569CD6; font-weight: bold; }
        .error-line { color: #F14C4C; }
        .success-line { color: #6A9955; }
        .timestamp { color: #CE9178; }
        </style>
        """, unsafe_allow_html=True)

        # Format logs with HTML styling
        formatted_logs = []
        for log in st.session_state.full_logs:
            if log.startswith('$'):
                formatted_logs.append(f'<div class="command-line">{html.escape(log)}</div>')
            elif '\u274c' in log:  # Error
                formatted_logs.append(f'<div class="error-line">{html.escape(log)}</div>')
            elif '\u2705' in log:  # Success
                formatted_logs.append(f'<div class="success-line">{html.escape(log)}</div>')
            else:
                # Extract timestamp if present
                timestamp_match = re.match(r'\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]', log)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    rest_of_log = log[len(timestamp_match.group(0)):]
                    formatted_logs.append(f'<div><span class="timestamp">[{timestamp}]</span>{html.escape(rest_of_log)}</div>')
                else:
                    formatted_logs.append(f'<div>{html.escape(log)}</div>')

        # Display the formatted logs
        st.markdown(f'<div class="terminal-logs">{"".join(formatted_logs)}</div>', unsafe_allow_html=True)
    else:
        st.info("No command history yet. Execute commands to see logs here.")

# Add streamlit-terminal tab if available
if TERMINAL_AVAILABLE and 'tab3' in locals():
    with tab3:
        st.subheader("Streamlit Terminal Component")

        # Add comparison information
        st.info("""
        **Streamlit-Terminal vs Archon Terminal**

        **Streamlit-Terminal:**
        - Good for simple integration where you need a basic terminal
        - Useful when you want a clean component-based approach
        - Easier to use as a dependency in other Streamlit apps

        **Archon Terminal:**
        - More feature-rich with advanced styling and command history
        - Better for complex terminal operations and scripting
        - Includes full logs and detailed output processing
        """)

        # Create tabs for different examples
        term_tab1, term_tab2, term_tab3 = st.tabs(["Basic Usage", "Custom Command", "Colorful Output"])

        with term_tab1:
            st.markdown("### Basic Terminal")
            st.write("Use this terminal to execute commands. Type commands after the `>` prompt and press `Enter`.")

            # Basic terminal with welcome message
            st_terminal(
                key="basic_terminal",
                height=400,
                show_welcome_message=True
            )

            st.markdown("""
            ```python
            # Source code for basic terminal
            from streamlit_terminal import st_terminal

            st_terminal(
                key="basic_terminal",
                height=400,
                show_welcome_message=True
            )
            ```""")

        with term_tab2:
            st.markdown("### Custom Command Terminal")
            st.write("This terminal runs a specific command that you provide.")

            # Command input
            custom_cmd = st.text_input(
                "Enter command to run:",
                value="ls -la",
                key="custom_command_input"
            )

            # Terminal height control
            term_height = st.slider("Terminal Height", min_value=200, max_value=800, value=400, step=50, key="custom_term_height")

            # Custom command terminal
            full_outputs, updated_outputs = st_terminal(
                key="custom_terminal",
                command=custom_cmd,
                height=term_height,
                disable_input=True  # Disable direct input since we're providing the command
            )

            # Show outputs in expandable section
            with st.expander("Terminal Outputs"):
                st.subheader("Full Outputs")
                st.write(full_outputs)
                st.subheader("Updated Outputs")
                st.write(updated_outputs)

            st.markdown("""
            ```python
            # Source code for custom command terminal
            from streamlit_terminal import st_terminal

            custom_cmd = st.text_input("Enter command to run:", value="ls -la")
            term_height = st.slider("Terminal Height", min_value=200, max_value=800, value=400, step=50)

            full_outputs, updated_outputs = st_terminal(
                key="custom_terminal",
                command=custom_cmd,
                height=term_height,
                disable_input=True
            )
            ```""")

        with term_tab3:
            st.markdown("### Colorful Output Terminal")
            st.write("This terminal demonstrates ANSI color support in the terminal output.")

            # Colorful command with ANSI escape codes
            colorful_cmd = st.text_area(
                "Colorful command with ANSI codes:",
                value=r'''echo -e "\x1b[31mError: Something went wrong\x1b[0m" &&
echo -e "\x1b[33mWarning: Check your configuration\x1b[0m" &&
echo -e "\x1b[32mSuccess: Operation completed successfully\x1b[0m"''',
                height=100,
                key="colorful_command_input"
            )

            # Colorful output terminal
            st_terminal(
                key="colorful_terminal",
                command=colorful_cmd,
                height=400,
                max_height=600
            )

            st.markdown("""
            ```python
            # Source code for colorful output terminal
            from streamlit_terminal import st_terminal

            colorful_cmd = st.text_area(
                "Colorful command with ANSI codes:",
                value=r'''echo -e "\x1b[31mError: Something went wrong\x1b[0m" &&
echo -e "\x1b[33mWarning: Check your configuration\x1b[0m" &&
echo -e "\x1b[32mSuccess: Operation completed successfully\x1b[0m"'''
            )

            st_terminal(
                key="colorful_terminal",
                command=colorful_cmd,
                height=400
            )
            ```""")

# Process output in real-time
if st.session_state.is_running and st.session_state.process:
    display_output()

# Sidebar with info
st.sidebar.markdown(f"""
## Current Directory
```bash
{st.session_state.current_dir}
```
## Instructions
- Type commands in the Quick Command field for simple commands
- Use the script editor for multi-line commands or scripts
- Switch to Full Logs tab to see complete history
- Use Clear Logs button to reset the terminal
- For long-running processes, you can send input in the Process Input field
""")

# Add system info
system_info = f"Python {sys.version.split()[0]} on {sys.platform}"
st.sidebar.markdown(f"""
## System Info
```
{system_info}
```
""")

# Add stream session info if available
if 'stream_sessions' in st.session_state and st.session_state.stream_sessions:
    st.sidebar.markdown("## Recent Commands")
    for session_id, session in sorted(st.session_state.stream_sessions.items(), reverse=True)[:5]:  # Show last 5 sessions
        status_emoji = "🟢" if session['status'] == 'running' else "✅" if session.get('return_code') == 0 else "❌"
        command_short = session['command'][:30] + "..." if len(session['command']) > 30 else session['command']

        # Format execution time
        if 'start_time' in session and 'end_time' in session and session['status'] == 'completed':
            exec_time = (session['end_time'] - session['start_time']).total_seconds()
            time_info = f"({exec_time:.2f}s)"
        elif 'start_time' in session and session['status'] == 'running':
            exec_time = (datetime.now() - session['start_time']).total_seconds()
            time_info = f"(running: {exec_time:.2f}s)"
        else:
            time_info = ""

        st.sidebar.markdown(f"{status_emoji} `{command_short}` {time_info}")

# Add command examples
st.sidebar.markdown("""
## Example Commands
```bash
# List files
ls -la

# Check Python version
python --version

# Run a simple Python script
python -c "print('Hello from Python!')"

# Show current directory
pwd
```
""")

# Add requirements info
if not ACE_AVAILABLE:
    st.sidebar.warning("For enhanced code editing, install streamlit-ace: `pip install streamlit-ace`")

st.sidebar.markdown("""
## Requirements
Install required packages:
```bash
pip install streamlit
pip install streamlit-ace  # Optional, for enhanced code editing
pip install streamlit-terminal  # Optional, for the streamlit-terminal component
```
""")

# Add information about streamlit-terminal if available
if TERMINAL_AVAILABLE:
    st.sidebar.markdown("""
    ## Streamlit Terminal
    The Streamlit Terminal tab uses the `streamlit-terminal` component, which provides:
    - A clean, component-based terminal interface
    - Easy integration with other Streamlit apps
    - Simple API for running commands

    Try it out in the "Streamlit Terminal" tab!
    """)

# Add custom CSS for terminal styling
st.markdown("""
<style>
    .stCode {
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
        padding: 16px !important;
    }

    /* VS Code-like scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background: #3e3e3e;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Terminal blinking cursor effect */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }

    .cursor {
        display: inline-block;
        width: 8px;
        height: 16px;
        background-color: #d4d4d4;
        animation: blink 1s step-end infinite;
        vertical-align: middle;
        margin-left: 2px;
    }
</style>
""", unsafe_allow_html=True)