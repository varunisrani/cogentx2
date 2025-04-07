# Streamlit Terminal Improvements

This document outlines the improvements made to the Streamlit Terminal component to enhance its functionality and user experience.

## Features Added

### 1. npm and nvm Support
- Added support for running npm and nvm commands
- Automatic detection of npm and nvm installations
- Environment variable handling for npm and nvm
- Platform-specific command execution (Windows vs. Unix-like systems)

### 2. Improved Terminal Display
- VS Code-like terminal experience
- Consistent output formatting for both stdout and stderr
- Removed error prefixes (❌) for cleaner output
- Fixed-height scrollable terminal output container
- Syntax highlighting for better readability

### 3. Enhanced Input Handling
- Support for Enter key to execute commands
- Better command history management
- Improved input field with placeholder text
- Run button for executing commands

### 4. Terminal Controls
- Added Clear Terminal button to reset output
- Improved Terminate Process button
- Terminal Options in a collapsible expander
- Better process management

### 5. MCP Server Support
- Dedicated terminal for running MCP servers
- Proper handling of interactive input for MCP servers
- Consistent output formatting for MCP server logs

## Usage Examples

### Basic Terminal
```python
from streamlit_terminal import st_terminal

# Create a basic terminal
st_terminal(key="my_terminal")
```

### Running npm/nvm Commands
```python
from streamlit_terminal import st_terminal

# Run npm version
st_terminal(command="npm --version")

# Run node version
st_terminal(command="node --version")

# List installed Node.js versions with nvm
st_terminal(command="nvm list")
```

### Running MCP Servers
```python
from streamlit_terminal import st_terminal

# Create a terminal for running MCP servers
mcp_terminal = st_terminal(
    key="terminal_mcp", 
    height=500, 
    show_welcome_message=True,
    welcome_message="MCP Server Terminal - Run your MCP servers here and interact with them."
)
```

## Implementation Details

The improvements were implemented by:

1. Adding npm_utils.py for npm and nvm support
2. Enhancing the Terminal class to better handle input and output
3. Improving the component method for better UI/UX
4. Adding platform-specific command execution
5. Fixing output formatting to match VS Code terminal experience

## Troubleshooting

If you encounter issues with the terminal:

1. Check the terminal output for error messages
2. Use the Terminal Options expander to terminate stuck processes
3. Clear the terminal output if it becomes cluttered
4. For npm/nvm issues, verify that they are properly installed on your system
