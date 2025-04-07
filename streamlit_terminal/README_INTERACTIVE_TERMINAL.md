# Interactive Terminal for MCP Servers

This document explains how to use the enhanced interactive terminal features for MCP servers in the Streamlit Terminal component.

## Features

- Interactive input for MCP servers like Spotify and GitHub agents
- Automatic detection of running MCP servers
- Context-aware input field that changes based on the running process
- Proper handling of stdin/stdout for interactive processes
- VS Code-like terminal experience with consistent formatting

## Usage Examples

### Running Spotify MCP Server

```python
from streamlit_terminal import st_terminal

# Create a terminal for the Spotify MCP server
spotify_terminal = st_terminal(
    key="terminal_spotify", 
    height=500, 
    show_welcome_message=True,
    welcome_message="Spotify MCP Server Terminal"
)

# Run the Spotify agent with the --streamlit flag
# This will start the server in a mode that works with the Streamlit terminal
spotify_terminal.run("python spotify_agent/streamlit_main.py --streamlit")

# After the server starts, you can interact with it by typing queries
# in the input field and pressing Enter or clicking the Send button
```

### Running GitHub MCP Server

```python
from streamlit_terminal import st_terminal

# Create a terminal for the GitHub MCP server
github_terminal = st_terminal(
    key="terminal_github", 
    height=500, 
    show_welcome_message=True,
    welcome_message="GitHub MCP Server Terminal"
)

# Run the GitHub agent
github_terminal.run("python github_agent/main.py")

# After the server starts, you can interact with it by typing queries
# in the input field and pressing Enter or clicking the Send button
```

## How It Works

The interactive terminal works by:

1. Detecting when an MCP server is running based on its output
2. Changing the input field label and placeholder text to match the context
3. Enabling the Send button even when a process is running (for MCP servers)
4. Sending input to the process's stdin instead of starting a new process
5. Properly formatting the output to match VS Code's terminal experience

## Implementation Details

The implementation includes:

1. A modified version of the Spotify agent's main.py file (streamlit_main.py) that works with the Streamlit terminal
2. Enhanced Terminal class with methods for detecting MCP servers and sending input
3. Improved component method for better UI/UX with context-aware input
4. Proper handling of stdin/stdout/stderr for interactive processes

## Troubleshooting

If you encounter issues with the interactive terminal:

1. Make sure you're using the --streamlit flag when running the Spotify agent
2. Check that the terminal is properly detecting the MCP server
3. Try clearing the terminal output if it becomes cluttered
4. If input isn't being sent, check that the process is still running
