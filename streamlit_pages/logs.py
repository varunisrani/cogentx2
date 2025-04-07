import streamlit as st
import time
import re
from datetime import datetime
import queue
import threading
import subprocess
import os
import uuid

def logs_tab():
    """Display the logs tab with terminal output in a dedicated section"""
    st.header("System Logs")

    # Initialize session state for logs if not present
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []

    # Initialize session state for log process if not present
    if "log_process" not in st.session_state:
        st.session_state.log_process = None

    # Initialize session state for active terminal if not present
    if "active_terminal" not in st.session_state:
        st.session_state.active_terminal = None

    # Initialize session state for terminal processes if not present
    if "terminal_processes" not in st.session_state:
        st.session_state.terminal_processes = {}

    # Create columns for control buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔄 Refresh Logs", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("🧹 Clear Logs", use_container_width=True):
            st.session_state.system_logs = []
            st.rerun()

    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    # Add custom CSS for log styling
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

    # Create tabs for different log categories
    log_categories = st.tabs(["System Logs", "Interactive Terminal", "Application Logs", "Error Logs"])

    # System Logs Tab
    with log_categories[0]:
        if not st.session_state.system_logs:
            st.info("No system logs available. Run operations to see logs here.")
        else:
            # First display logs in a code block for potential copying
            with st.expander("Raw Logs (for copying)", expanded=False):
                st.code('\n'.join(st.session_state.system_logs), language='bash')

            # Display logs with nice formatting
            st.markdown("<div class='logs-container'>", unsafe_allow_html=True)

            for log in st.session_state.system_logs:
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

    # Interactive Terminal Tab
    with log_categories[1]:
        st.subheader("Interactive Terminal")

        # Display active terminals or option to create a new one
        if not st.session_state.terminal_processes:
            st.info("No active terminals. Start a process from the Generated Code page or create a new terminal below.")
        else:
            # Create a dropdown to select active terminal
            terminal_options = list(st.session_state.terminal_processes.keys())
            selected_terminal = st.selectbox(
                "Select Terminal",
                terminal_options,
                index=terminal_options.index(st.session_state.active_terminal) if st.session_state.active_terminal in terminal_options else 0
            )

            # Update active terminal
            st.session_state.active_terminal = selected_terminal

            # Get the process info
            process_info = st.session_state.terminal_processes[selected_terminal]

            # Display terminal output
            if "output" in process_info:
                # Display in a scrollable text area
                st.text_area(
                    "Terminal Output",
                    value="".join(process_info["output"]),
                    height=400,
                    disabled=True,
                    key=f"terminal_output_{selected_terminal}"
                )

                # Input field for sending commands to the terminal
                user_input = st.text_input("Enter command:", key=f"terminal_input_{selected_terminal}")

                # Send button
                if st.button("Send", key=f"send_button_{selected_terminal}"):
                    if user_input:
                        # Add the command to the output display
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        process_info["output"].append(f"[{timestamp}] $ {user_input}\n")

                        # Send the command to the process
                        if "process" in process_info and process_info["process"].poll() is None:
                            try:
                                process_info["process"].stdin.write(f"{user_input}\n")
                                process_info["process"].stdin.flush()
                                st.rerun()
                            except Exception as e:
                                process_info["output"].append(f"[{timestamp}] ❌ Error sending command: {str(e)}\n")
                                st.error(f"Error sending command: {str(e)}")
                        else:
                            process_info["output"].append(f"[{timestamp}] ❌ Process is not running\n")
                            st.error("Process is not running")

            # Buttons for terminal control
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Terminal", key=f"clear_terminal_{selected_terminal}"):
                    process_info["output"] = []
                    st.rerun()

            with col2:
                if st.button("Kill Process", key=f"kill_process_{selected_terminal}"):
                    if "process" in process_info and process_info["process"].poll() is None:
                        try:
                            process_info["process"].terminate()
                            time.sleep(0.5)
                            if process_info["process"].poll() is None:
                                process_info["process"].kill()

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            process_info["output"].append(f"[{timestamp}] Process terminated\n")
                            st.success("Process terminated")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error terminating process: {str(e)}")
                    else:
                        st.info("Process is not running")

        # Create a new terminal
        st.divider()
        st.subheader("Create New Terminal")

        # Input for command
        new_command = st.text_input("Command to run:", placeholder="e.g., python -i")

        # Input for working directory
        new_cwd = st.text_input("Working directory:", value=os.getcwd())

        # Create button
        if st.button("Create Terminal", key="create_new_terminal"):
            if new_command:
                try:
                    # Create a unique ID for this terminal
                    terminal_id = f"terminal_{uuid.uuid4().hex[:8]}_{time.time()}"

                    # Start the process
                    process = subprocess.Popen(
                        new_command.split(),
                        cwd=new_cwd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        universal_newlines=True,
                        shell=False
                    )

                    # Create a new entry in terminal_processes
                    st.session_state.terminal_processes[terminal_id] = {
                        "process": process,
                        "command": new_command,
                        "cwd": new_cwd,
                        "output": [f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started terminal with command: {new_command}\n"],
                        "start_time": time.time()
                    }

                    # Set as active terminal
                    st.session_state.active_terminal = terminal_id

                    # Start a thread to read output
                    threading.Thread(
                        target=read_process_output,
                        args=(process, terminal_id),
                        daemon=True
                    ).start()

                    st.success(f"Terminal created with command: {new_command}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating terminal: {str(e)}")
            else:
                st.warning("Please enter a command to run")

    # Application Logs Tab
    with log_categories[2]:
        # Get the workbench directory path
        workbench_dir = os.path.join(os.getcwd(), "workbench")
        log_path = os.path.join(workbench_dir, "logs.txt")

        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    app_logs = f.readlines()

                if app_logs:
                    # Display logs with nice formatting
                    st.markdown("<div class='logs-container'>", unsafe_allow_html=True)

                    for log in app_logs:
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
                    st.info("No application logs available.")
            except Exception as e:
                st.error(f"Error reading application logs: {str(e)}")
        else:
            st.info("No application logs file found.")

    # Error Logs Tab
    with log_categories[2]:
        # Filter error logs from system logs
        error_logs = [log for log in st.session_state.system_logs if "error" in log.lower() or "❌" in log or "exception" in log.lower() or "failed" in log.lower()]

        if not error_logs:
            st.info("No error logs found.")
        else:
            # Display logs with nice formatting
            st.markdown("<div class='logs-container'>", unsafe_allow_html=True)

            for log in error_logs:
                # Extract timestamp if present
                timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', log)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    message = log.replace(f"[{timestamp}]", "").strip()

                    st.markdown(f"""
                    <div class="log-entry">
                        <span class="timestamp">[{timestamp}]</span>
                        <span class="log-content error-message">{message}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # If no timestamp, just display the message
                    st.markdown(f"""
                    <div class="log-entry">
                        <span class="log-content error-message">{log}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # Auto-refresh if enabled
    if auto_refresh:
        time.sleep(1)  # Small delay to prevent excessive CPU usage
        st.rerun()

def capture_terminal_output(cmd: list, cwd: str) -> queue.Queue:
    """Run a process and capture its output to the system logs."""
    output_queue = queue.Queue()

    def reader(pipe, queue):
        try:
            with pipe:
                for line in iter(pipe.readline, b''):
                    decoded_line = line.decode('utf-8')
                    # Add timestamp to the output
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"[{timestamp}] {decoded_line.strip()}"

                    # Add to system logs
                    if "system_logs" in st.session_state:
                        st.session_state.system_logs.append(log_line)

                    # Also put in queue for other uses
                    queue.put(('output', log_line))
        finally:
            queue.put(('done', None))

    def run_process():
        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=False
            )

            # Start thread to read output
            thread = threading.Thread(target=reader, args=(process.stdout, output_queue))
            thread.daemon = True
            thread.start()

            # Wait for process to complete
            process.wait()

            # Put return code in queue
            output_queue.put(('return_code', process.returncode))

            # Add completion message to logs
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if process.returncode == 0:
                st.session_state.system_logs.append(f"[{timestamp}] ✅ Process completed successfully with return code 0")
            else:
                st.session_state.system_logs.append(f"[{timestamp}] ❌ Process failed with return code {process.returncode}")

        except Exception as e:
            # Add error message to logs
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.system_logs.append(f"[{timestamp}] ❌ Error executing process: {str(e)}")

            output_queue.put(('error', str(e)))
            output_queue.put(('return_code', -1))

    # Start process in a thread
    process_thread = threading.Thread(target=run_process)
    process_thread.daemon = True
    process_thread.start()

    return output_queue

def read_process_output(process, terminal_id):
    """Read output from a process and add it to the terminal output."""
    try:
        for line in iter(process.stdout.readline, ""):
            if terminal_id in st.session_state.terminal_processes:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.terminal_processes[terminal_id]["output"].append(f"[{timestamp}] {line}")
            else:
                # Terminal was removed, stop reading
                break
    except Exception as e:
        if terminal_id in st.session_state.terminal_processes:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.terminal_processes[terminal_id]["output"].append(f"[{timestamp}] ❌ Error reading output: {str(e)}\n")

def add_log_entry(message: str):
    """Add a log entry to the system logs."""
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.system_logs.append(f"[{timestamp}] {message}")
