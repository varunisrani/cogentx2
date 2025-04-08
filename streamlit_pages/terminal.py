import streamlit as st
import os
import sys
import subprocess
import queue
import threading
import time
from typing import List, Optional, Tuple, Dict, Any

# Try to import the streamlit_terminal package, fall back to custom implementation
try:
    from streamlit_terminal import st_terminal
    HAS_STREAMLIT_TERMINAL = True
except ImportError:
    HAS_STREAMLIT_TERMINAL = False
    from custom_terminal import st_custom_terminal

def terminal_tab():
    """Terminal tab for executing commands in the workbench environment."""
    st.title("Terminal Runner")
    
    # Initialize session state variables
    if "terminal_history" not in st.session_state:
        st.session_state.terminal_history = []
    
    if "current_process" not in st.session_state:
        st.session_state.current_process = None
    
    if "output_queue" not in st.session_state:
        st.session_state.output_queue = queue.Queue()
    
    if "command_history" not in st.session_state:
        st.session_state.command_history = []
    
    if "history_index" not in st.session_state:
        st.session_state.history_index = -1
        
    if "working_directory" not in st.session_state:
        # Set initial working directory to workbench
        workbench_dir = os.path.join(os.getcwd(), "workbench")
        if os.path.exists(workbench_dir):
            st.session_state.working_directory = workbench_dir
        else:
            st.session_state.working_directory = os.getcwd()
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Terminal output area
        terminal_placeholder = st.empty()
        
        if HAS_STREAMLIT_TERMINAL:
            # Use the streamlit_terminal component if available
            with terminal_placeholder.container():
                st_terminal(
                    key="terminal_component",
                    height=400,
                    show_welcome_message=True,
                    working_directory=st.session_state.working_directory
                )
        else:
            # Use our custom terminal implementation
            st_custom_terminal(
                terminal_placeholder, 
                st.session_state.terminal_history,
                on_run_command
            )
        
        # Text input for commands
        command_col1, command_col2 = st.columns([5, 1])
        with command_col1:
            command = st.text_input(
                "Command:",
                key="command_input",
                placeholder="Enter command to run (e.g. python main.py)",
                help="Enter a command to run in the workbench directory"
            )
        
        with command_col2:
            run_button = st.button("Run", use_container_width=True)
        
        if run_button and command:
            run_command(command)
    
    with col2:
        with st.container(border=True):
            st.subheader("Environment")
            
            # Display and allow changing the working directory
            st.text_input(
                "Working Directory:",
                value=st.session_state.working_directory,
                key="new_working_dir"
            )
            
            if st.button("Change Directory", use_container_width=True):
                new_dir = st.session_state.new_working_dir
                if os.path.exists(new_dir) and os.path.isdir(new_dir):
                    st.session_state.working_directory = new_dir
                    add_to_history(f"Changed directory to: {new_dir}")
                    # Run 'ls' to show directory contents
                    run_command("ls")
                else:
                    st.error(f"Directory does not exist: {new_dir}")
            
            # Quick directory navigation
            st.subheader("Quick Navigation")
            
            if st.button("Workbench", use_container_width=True):
                workbench_dir = os.path.join(os.getcwd(), "workbench")
                if os.path.exists(workbench_dir):
                    st.session_state.working_directory = workbench_dir
                    st.session_state.new_working_dir = workbench_dir
                    add_to_history(f"Changed directory to: {workbench_dir}")
                    run_command("ls")
                else:
                    st.error("Workbench directory does not exist")
            
            if st.button("Project Root", use_container_width=True):
                root_dir = os.getcwd()
                st.session_state.working_directory = root_dir
                st.session_state.new_working_dir = root_dir
                add_to_history(f"Changed directory to: {root_dir}")
                run_command("ls")
        
        # Command history
        with st.container(border=True):
            st.subheader("Command History")
            
            # Display and allow rerunning previous commands
            for i, cmd in enumerate(st.session_state.command_history[::-1][:10]):  # Last 10 commands, reversed
                cols = st.columns([4, 1])
                with cols[0]:
                    st.code(cmd, language="bash")
                with cols[1]:
                    if st.button("Run", key=f"rerun_{i}"):
                        run_command(cmd)
        
        # Common commands
        with st.container(border=True):
            st.subheader("Common Commands")
            
            if st.button("List Files (ls)", use_container_width=True):
                run_command("ls -la")
            
            if st.button("Run Python Script", use_container_width=True):
                # Get Python files in the working directory
                python_files = [f for f in os.listdir(st.session_state.working_directory) 
                              if f.endswith('.py') and os.path.isfile(os.path.join(st.session_state.working_directory, f))]
                
                if python_files:
                    file_to_run = st.selectbox("Select Python file to run:", python_files)
                    if file_to_run:
                        run_command(f"python {file_to_run}")
                else:
                    st.error("No Python files found in the current directory")
            
            if st.button("Install Requirements", use_container_width=True):
                req_file = os.path.join(st.session_state.working_directory, "requirements.txt")
                if os.path.exists(req_file):
                    run_command("pip install -r requirements.txt")
                else:
                    st.error("No requirements.txt file found in the current directory")

def execute_command(command: str) -> str:
    """Execute a command and return the output."""
    if not command.strip():
        return ""
    
    add_to_history(f"$ {command}")
    
    try:
        # Execute the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=st.session_state.working_directory
        )
        
        # Store command in history
        if command not in st.session_state.command_history:
            st.session_state.command_history.append(command)
        
        # Read output
        stdout, stderr = process.communicate()
        output = stdout
        if stderr:
            output += f"\nError: {stderr}"
        
        add_to_history(output)
        return output
    
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        add_to_history(error_msg)
        return error_msg

def on_run_command(command: str):
    """Callback for running a command from the custom terminal."""
    run_command(command)

def run_command(command: str):
    """Run a command and update the terminal history."""
    if not command.strip():
        return
    
    # Add command to history if it's new
    if command not in st.session_state.command_history:
        st.session_state.command_history.append(command)
    
    # Reset history index
    st.session_state.history_index = -1
    
    # Execute the command
    add_to_history(f"$ {command}")
    
    try:
        # Start a new process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=st.session_state.working_directory
        )
        
        st.session_state.current_process = process
        
        # Start threads to read output
        def read_output(pipe, queue):
            for line in iter(pipe.readline, ''):
                queue.put(line)
            pipe.close()
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, st.session_state.output_queue))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, st.session_state.output_queue))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Process output from the queue
        output = []
        def process_output():
            while True:
                try:
                    line = st.session_state.output_queue.get_nowait()
                    output.append(line)
                    add_to_history(line.strip())
                except queue.Empty:
                    break
        
        # Check process status and output
        while process.poll() is None:
            process_output()
            time.sleep(0.1)
        
        # Get any remaining output
        process_output()
        
        # Check return code
        return_code = process.poll()
        if return_code != 0:
            add_to_history(f"Command exited with code {return_code}")
        
        # Clean up
        st.session_state.current_process = None
        
    except Exception as e:
        add_to_history(f"Error executing command: {str(e)}")

def add_to_history(text: str):
    """Add text to the terminal history."""
    if not text:
        return
    
    # Split multi-line text
    lines = text.split('\n')
    for line in lines:
        if line.strip():  # Only add non-empty lines
            st.session_state.terminal_history.append(line)
    
    # Limit history length
    max_history = 500
    if len(st.session_state.terminal_history) > max_history:
        st.session_state.terminal_history = st.session_state.terminal_history[-max_history:]
    
    # Trigger rerun to update display
    st.rerun() 