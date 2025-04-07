

import streamlit as st
import subprocess
import threading
import shlex
import sys
import os
import logging
from queue import Queue
from .npm_utils import get_nvm_env, create_nvm_command, is_nvm_installed, is_npm_installed

class Terminal:
    static_instance_id = 0

    def __init__(self, key) -> None:
        logging.debug("IIIIIIIIIIIIIIII Initializing Terminal instance")
        self.__key = key
        self.__process = None
        self.__queue = None
        self.__outputs = []
        self.__threads = []
        self.cmd = ""
        self.__command_hashs = set()
        self.__run_count = 0

        Terminal.static_instance_id += 1
        self.__id = Terminal.static_instance_id

        # https://github.com/streamlit/streamlit/issues/2838#issuecomment-1738983577
        # This is it!
        # get_browser_session_id needs to be run on the relevant script thread,
        # then you can call the rest of this on other threads.
        from .utils import get_browser_session_id, find_streamlit_main_loop, get_streamlit_session, notify
        self.streamlit_loop = find_streamlit_main_loop()
        self.streamlit_session = get_streamlit_session(get_browser_session_id())

        # This can be called on any thread you want.
        # self.streamlit_loop.call_soon_threadsafe(notify)

    def __del__(self):
        logging.debug("DDDDDDDDDDDDDDD Deleting Terminal instance")
        for _th in self.__threads:
            _th.join()

    def _read_stdbuffer(self, which_buf, q, process):
        if which_buf == "stdout":
            stdbuf = process.stdout
        elif which_buf == "stderr":
            stdbuf = process.stderr
        else:
            raise ValueError("Invalid buffer")

        logging.debug(f"Start _read_stdbuffer for {which_buf} {process}, {q}")
        while process.poll() is None:

            logging.debug(f"Polling {which_buf} for process {process}")
            logging.debug(f"Current qsize for process {process} is {q.qsize()}, {q}")
            stdbuf.flush()
            out = stdbuf.readline()
            if out:
                # Clean the output - remove any trailing newlines
                out = out.rstrip('\n')
                logging.debug(f"{process}: {which_buf.upper()}: {out}")
                # Use the same type for both stdout and stderr to avoid special formatting
                # This makes the output consistent with VS Code terminal
                q.put({"type": "output", "value": out, "raw": True})
                self.notify()

        # Read remaining
        out = stdbuf.read()
        if out:
            logging.debug(f"{process}: {which_buf.upper()}(remaining): {out}")
            for o in out.splitlines():
                clean_o = o.rstrip('\n')
                q.put({"type": "output", "value": clean_o, "raw": True})
        logging.debug(f"Finished thread _read_stdbuffer for {which_buf} finished {process}")
        self.notify()

    def _watch_queue(self):
        logging.debug(f"Start watching queue for process {self.__process}, Queue: {self.__queue}")
        while self.__process.poll() is None:
            if self.__queue.qsize() > 0:
                logging.debug(f"Notify Queue: size: {self.__queue.qsize()}")
                # self.__outputs.append(self.__queue.get_nowait())
                self.notify()
        logging.debug(f"Thread _watch_queue finished {self.__process}")


    def _start_watch_stdout_stderr(self):
        self.__queue = Queue()

        # Start reading stdout
        logging.debug(f"Starting reading stdout for process {self.__process}, Queue: {self.__queue}")
        self.__threads = []
        self.__threads.append(
            threading.Thread(target=self._read_stdbuffer,
                             args=("stdout",
                                   self.__queue,
                                   self.__process,))
        )

        # Start reading stderr
        self.__threads.append(
             threading.Thread(target=self._read_stdbuffer,
                             args=("stderr",
                                   self.__queue,
                                   self.__process,))
        )


        # Watch queue
        # self.__threads.append(
        #     threading.Thread(target=self._watch_queue)
        # )

        for _th in self.__threads:
            _th.start()


    def run(self, cmd):
        logging.debug(f"Running subprocess: {cmd}")

        # Check if process is running
        if self.__process is not None:
            logging.debug(f"Terminating existing process {self.__process}")
            if self.__process.poll() is None:
                logging.debug("Process is running")
                # If this is a running MCP server, we'll send the command as input
                if self.is_mcp_server_running():
                    self.send_input_to_process(cmd)
                    return
        else:
            logging.debug("No existing process")

        if type(cmd) == list:
            cmd = " ".join(cmd)

        self.__outputs.append({
            "type": "command",
            "value": cmd,
        })
        self.__run_count += 1

        # Check if this is an npm or node command that should use nvm
        use_nvm = False
        if any(cmd.startswith(prefix) for prefix in ['npm ', 'node ', 'npx ', 'nvm ']):
            use_nvm = True
            logging.debug("Using NVM environment for npm/node command")

        # Check if this is a Spotify or GitHub agent command
        is_mcp_server = False
        if 'python' in cmd and ('spotify_agent/main.py' in cmd or 'github_agent/main.py' in cmd):
            is_mcp_server = True
            logging.debug("Running MCP server command")
            # Add --streamlit flag for Spotify agent
            if 'spotify_agent/main.py' in cmd and '--streamlit' not in cmd:
                cmd = cmd.replace('spotify_agent/main.py', 'spotify_agent/streamlit_main.py --streamlit')

        # Start new process
        try:
            env = os.environ.copy()

            # If using npm/node with nvm, set up the environment
            if use_nvm and is_nvm_installed():
                env = get_nvm_env()
                # For nvm commands, we need to modify the command to source nvm.sh first
                if cmd.startswith('nvm '):
                    cmd = create_nvm_command(cmd)

            # Determine if this is an interactive process that needs stdin
            is_interactive = 'spotify_agent/streamlit_main.py' in cmd or 'github_agent/main.py' in cmd

            # Handle platform-specific command execution
            if sys.platform == 'win32':
                # Windows execution
                if use_nvm:
                    # For nvm commands on Windows, we need to use PowerShell
                    self.__process = subprocess.Popen(
                        ['powershell.exe', '-Command', cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if is_interactive else None,
                        text=True,
                        bufsize=1,
                        env=env
                    )
                else:
                    # Regular command on Windows
                    self.__process = subprocess.Popen(
                        shlex.split(cmd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if is_interactive else None,
                        text=True,
                        bufsize=1,
                        env=env
                    )
            else:
                # Unix-like systems (macOS, Linux)
                if use_nvm:
                    # For nvm commands, use bash -c
                    self.__process = subprocess.Popen(
                        ['/bin/bash', '-c', cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if is_interactive else None,
                        text=True,
                        bufsize=1,
                        env=env
                    )
                else:
                    # Regular command on Unix
                    self.__process = subprocess.Popen(
                        shlex.split(cmd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if is_interactive else None,
                        text=True,
                        bufsize=1,
                        env=env
                    )
        except Exception as e:
            logging.error(f"Error starting process: {e}")
            self.__outputs.append({
                "type": "stderr",
                "value": str(e)
            })
            self.notify()
            return

        # self.__outputs = []

        self._start_watch_stdout_stderr()

    # def attach(self, pid):
    #     logging.debug(f"Attaching to process {pid}")
    #     self.__process = psutil.Process(pid)
    #     logging.debug(f"Attached to process {self.__process}")
    #     self._start_watch_stdout_stderr()

    def getUpdatedOutputs(self):
        outs = []
        if self.__queue is not None:
            logging.debug(f"Getting updated outputs for process {self.__process}, queue: {self.__queue}, size: {self.__queue.qsize()}")
            while self.__queue is not None and not self.__queue.empty():
                out = self.__queue.get_nowait()
                logging.debug(f"Getting updated outputs: {out}")
                outs += [out]
        self.__outputs += outs
        return outs

    def add_not_run_command(self, cmd):
        logging.debug(f"Adding not run command: {cmd}")
        self.__outputs.append({
            "type": "command",
            "not_run": True,
            "value": cmd
        })
        self.notify()

    def _generateHashFromMsg(self, msg):
        keys = list(msg.keys())
        keys.sort()
        h = ""
        for k in keys:
            h += str(f"{k}:{msg[k]}")
        return hash(h)

    def checkIfCommandAlreadyRun(self, msg):
        msg_hash = self._generateHashFromMsg(msg)
        return msg_hash in self.__command_hashs

    def requestRunCommand(self, msg):
        msg_hash = self._generateHashFromMsg(msg)
        if msg_hash not in self.__command_hashs:
            self.__command_hashs.add(msg_hash)
            cmd = msg["args"][0].split(" ")
            self.run(cmd)
            return True
        return False

    def addCommandHash(self, msg):
        msg_hash = self._generateHashFromMsg(msg)
        is_new_msg = msg_hash not in self.__command_hashs
        self.__command_hashs.add(msg_hash)
        return is_new_msg

    def procMsg(self, msg):
        try:
            command = msg["command"]
            args = msg["args"]
            kwargs = msg["kwargs"]
        except:
            logging.error("Invalid message received")
            return msg

        is_new_msg = self.addCommandHash(msg)
        if not is_new_msg:
            logging.debug(f"Command already run: {msg}")
            return {}

        if command == "initialized":
            pass
        elif command == "run_command":
            logging.debug(f"Running command: {args[0]}")
            self.run(args[0])
        elif command == "terminate_process":
            if self.process:
                logging.debug(f"Terminating process {self.process}")
                self.process.terminate()
        elif command == "add_not_run_command":
            self.add_not_run_command(args[0])
        else:
            logging.error(f"Invalid command: {command}")

        return msg

    def notify(self):
        from .utils import notify
        self.streamlit_loop.call_soon_threadsafe(notify, self.streamlit_session)


    @st.fragment
    def component(self, value, key=None):
        if key is None:
            key = self.__key

        is_running = self.__process is not None and self.__process.poll() is None

        logging.debug(f"Rendering component for Terminal instance {self.__process}, Queue: {self.__queue}")

        # Create a container for the terminal output with fixed height and scrolling
        with st.container(height=300):
            # Display the terminal output with proper formatting
            if self.__outputs:
                # Join outputs with newlines and display as preformatted text
                output_text = "\n".join([o.get('value', '') if isinstance(o, dict) else str(o) for o in self.__outputs])
                st.code(output_text, language=None)

        # Interactive command input with Enter key support
        is_mcp_server_running = self.is_mcp_server_running()
        input_label = "Query" if is_mcp_server_running else "Command"
        input_placeholder = "Type your query and press Enter" if is_mcp_server_running else "Type command and press Enter"
        button_text = "Send" if is_mcp_server_running else ("Run" if not is_running else "Running...")

        col1, col2 = st.columns([4, 1])
        with col1:
            self.cmd = st.text_input(input_label, value=value, key=key+"_text_input_cmd",
                                    placeholder=input_placeholder)

        with col2:
            # Run/Send button
            button_disabled = is_running and not is_mcp_server_running
            if st.button(button_text, key=key+"_button_run", disabled=button_disabled) and self.cmd is not None:
                logging.debug(f"Running command/sending input: {self.cmd}")
                self.run(self.cmd)

        # If Enter was pressed in the text input, also run the command or send input
        if self.cmd and self.cmd != value:
            if is_mcp_server_running or not is_running:
                logging.debug(f"Running command/sending input from Enter key: {self.cmd}")
                self.run(self.cmd)

        # Advanced options in an expander
        with st.expander("Terminal Options"):
            # Terminate button
            if st.button("Terminate Process", key=key+"_button_terminate", disabled=not is_running) and self.__process:
                if self.__process is not None and self.__process.poll() is None:
                    logging.debug(f"Terminating process {self.__process}")
                    self.__process.terminate()
                    st.rerun(scope="fragment")

            # Clear output button
            if st.button("Clear Terminal", key=key+"_button_clear"):
                self.__outputs = []
                st.rerun(scope="fragment")

        # Get stdout/stderr
        if self.__queue is not None and self.__queue.qsize() > 0:
            updated_outputs = []
            while not self.__queue.empty():
                out = self.__queue.get_nowait()
                if isinstance(out, dict) and 'value' in out:
                    updated_outputs.append(out)

            if updated_outputs:
                self.__outputs.extend(updated_outputs)
                st.rerun(scope="fragment")


    @property
    def process(self):
        return self.__process

    @property
    def queue(self):
        return self.__queue

    @property
    def outputs(self):
        return self.__outputs

    @property
    def id(self):
        return str(self.__id)

    def is_mcp_server_running(self):
        """Check if the current process is a running MCP server"""
        if self.__process is None or self.__process.poll() is not None:
            return False

        # Check if this is a Spotify or GitHub agent
        for output in self.__outputs:
            if isinstance(output, dict) and output.get('value'):
                value = output.get('value', '')
                if '🎵 Enter your Spotify query:' in value or 'SPOTIFY MCP AGENT' in value:
                    return True
                if 'GITHUB MCP AGENT' in value or 'Enter your GitHub query:' in value:
                    return True
        return False

    def send_input_to_process(self, input_text):
        """Send input to the running process"""
        if self.__process is None or self.__process.poll() is not None:
            logging.error("No running process to send input to")
            return False

        try:
            # Add the input to the outputs for display
            self.__outputs.append({
                "type": "input",
                "value": input_text
            })

            # Send the input to the process
            self.__process.stdin.write(input_text + '\n')
            self.__process.stdin.flush()
            logging.debug(f"Sent input to process: {input_text}")
            self.notify()
            return True
        except Exception as e:
            logging.error(f"Error sending input to process: {e}")
            self.__outputs.append({
                "type": "stderr",
                "value": f"Error sending input: {str(e)}"
            })
            self.notify()
            return False

    @property
    def is_running(self):
        return self.__process is not None and self.__process.poll() is None

    @property
    def outputs(self):
        return self.__outputs

    @property
    def run_count(self):
        return self.__run_count


