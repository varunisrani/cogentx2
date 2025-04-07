"""
Custom terminal implementation that avoids OpenAI voice helpers dependency issues
"""

import streamlit as st
import subprocess
import threading
import shlex
import sys
import os
import logging
from queue import Queue
import time
from typing import List, Callable, Any

class CustomTerminal:
    """A custom terminal implementation for Streamlit"""
    
    def __init__(self, key):
        """Initialize the terminal"""
        self.__key = key
        self.__process = None
        self.__outputs = []
        self.__run_count = 0
        self.__queue = Queue()
        self.__stdout_thread = None
        self.__stderr_thread = None
        
    def run(self, cmd):
        """Run a command in the terminal"""
        logging.debug(f"Running subprocess: {cmd}")

        # Check if process is running
        if self.__process is not None:
            logging.debug(f"Terminating existing process {self.__process}")
            if self.__process.poll() is None:
                logging.debug("Process is running")
        else:
            logging.debug("No existing process")

        if type(cmd) == list:
            cmd = " ".join(cmd)

        self.__outputs.append({
            "type": "command",
            "value": cmd,
        })
        self.__run_count += 1

        # Start new process
        try:
            if sys.platform == 'win32':
                self.__process = subprocess.Popen(
                    shlex.split(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
            else:
                self.__process = subprocess.Popen(
                    shlex.split(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
        except Exception as e:
            logging.error(f"Error starting process: {e}")
            self.__outputs.append({
                "type": "stderr",
                "value": str(e)
            })
            return
            
        # Start threads to read stdout and stderr
        self._start_watch_stdout_stderr()
    
    def _start_watch_stdout_stderr(self):
        """Start threads to watch stdout and stderr"""
        if self.__process is None:
            return
            
        self.__stdout_thread = threading.Thread(
            target=self._read_output,
            args=("stdout", self.__process.stdout),
            daemon=True
        )
        self.__stderr_thread = threading.Thread(
            target=self._read_output,
            args=("stderr", self.__process.stderr),
            daemon=True
        )
        
        self.__stdout_thread.start()
        self.__stderr_thread.start()
    
    def _read_output(self, output_type, stream):
        """Read output from stdout or stderr"""
        while self.__process and self.__process.poll() is None:
            line = stream.readline()
            if line:
                self.__outputs.append({
                    "type": output_type,
                    "value": line.rstrip('\n')
                })
                
        # Read any remaining output
        remaining = stream.read()
        if remaining:
            for line in remaining.splitlines():
                self.__outputs.append({
                    "type": output_type,
                    "value": line
                })
    
    def send_input(self, input_text):
        """Send input to the process"""
        if self.__process and self.__process.poll() is None:
            try:
                self.__process.stdin.write(input_text + '\n')
                self.__process.stdin.flush()
                self.__outputs.append({
                    "type": "input",
                    "value": input_text
                })
                return True
            except Exception as e:
                self.__outputs.append({
                    "type": "stderr",
                    "value": f"Error sending input: {str(e)}"
                })
                return False
        return False
    
    def terminate(self):
        """Terminate the process"""
        if self.__process and self.__process.poll() is None:
            self.__process.terminate()
            return True
        return False
    
    @property
    def is_running(self):
        """Check if the process is running"""
        return self.__process is not None and self.__process.poll() is None
    
    @property
    def outputs(self):
        """Get the outputs"""
        return self.__outputs
    
    def clear_outputs(self):
        """Clear the outputs"""
        self.__outputs = []
    
    def component(self, command=""):
        """Render the terminal component"""
        # Container for the terminal
        st.markdown("""
        <style>
        .custom-terminal {
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .terminal-output {
            background-color: #2d2d2d;
            color: #f0f0f0;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            padding: 10px;
            border-radius: 5px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .command-line {
            color: #4CAF50;
        }
        .error-line {
            color: #FF5252;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Terminal output
        with st.container():
            # Display the outputs
            if self.__outputs:
                output_text = ""
                for output in self.__outputs:
                    if output["type"] == "command":
                        output_text += f"$ {output['value']}\n"
                    elif output["type"] == "stderr":
                        output_text += f"{output['value']}\n"
                    else:
                        output_text += f"{output['value']}\n"
                
                st.code(output_text, language=None)
        
        # Command input
        col1, col2 = st.columns([4, 1])
        with col1:
            input_text = st.text_input(
                "Command",
                value=command,
                key=f"{self.__key}_input",
                placeholder="Type command and press Enter"
            )
        
        with col2:
            run_button = st.button(
                "Run" if not self.is_running else "Running...",
                key=f"{self.__key}_run",
                disabled=self.is_running and input_text != command
            )
        
        # Run the command if the button is clicked or Enter is pressed
        if (run_button or (input_text and input_text != command)) and not self.is_running:
            self.run(input_text)
        
        # Add controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Terminal", key=f"{self.__key}_clear"):
                self.clear_outputs()
                st.rerun()
        
        with col2:
            if st.button("Terminate Process", key=f"{self.__key}_terminate", disabled=not self.is_running):
                self.terminate()
                st.rerun()
        
        return self.__outputs

def st_custom_terminal(key="terminal", command="", height=400, show_welcome_message=False, welcome_message="Welcome to the terminal!"):
    """
    Create a custom terminal component
    
    Parameters:
    -----------
    key : str
        A unique key for the terminal
    command : str
        The command to run
    height : int
        The height of the terminal
    show_welcome_message : bool
        Whether to show a welcome message
    welcome_message : str
        The welcome message to show
        
    Returns:
    --------
    list
        The terminal outputs
    """
    # Initialize the terminal if it doesn't exist
    if f"{key}_terminal" not in st.session_state:
        st.session_state[f"{key}_terminal"] = CustomTerminal(key)
        if show_welcome_message:
            st.session_state[f"{key}_terminal"].outputs.append({
                "type": "stdout",
                "value": welcome_message
            })
    
    # Get the terminal
    terminal = st.session_state[f"{key}_terminal"]
    
    # Render the terminal component
    outputs = terminal.component(command)
    
    return outputs
