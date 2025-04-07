# npm and nvm Support in Streamlit Terminal

This document explains how to use npm and nvm support in the Streamlit Terminal component.

## Features

- Run npm commands directly in the terminal
- Use nvm to manage Node.js versions
- Automatic detection of npm and nvm installations
- Support for running npm scripts
- Environment variable handling for npm and nvm

## Usage Examples

### Basic npm Commands

```python
from streamlit_terminal import st_terminal

# Run npm version
st_terminal(command="npm --version")

# Run node version
st_terminal(command="node --version")

# List installed packages
st_terminal(command="npm list --depth=0")
```

### nvm Commands

```python
from streamlit_terminal import st_terminal

# List installed Node.js versions
st_terminal(command="nvm list")

# Use a specific Node.js version
st_terminal(command="nvm use 16")

# Install a new Node.js version
st_terminal(command="nvm install 18")
```

### Checking npm and nvm Status

```python
from streamlit_terminal import check_npm_nvm_status

# Get npm and nvm status
status = check_npm_nvm_status()
print(f"npm installed: {status['npm_installed']}")
print(f"nvm installed: {status['nvm_installed']}")
print(f"Node.js versions: {status['node_versions']}")
```

## Implementation Details

The npm and nvm support is implemented by:

1. Detecting npm and nvm installations on the system
2. Setting up the correct environment variables for npm and nvm commands
3. Using shell commands to properly source nvm.sh before running nvm commands
4. Handling platform-specific differences (Windows vs. Unix-like systems)

## Requirements

- For npm support: npm must be installed on the system
- For nvm support: nvm must be installed on the system
- The terminal component will work even if npm or nvm is not installed, but npm/nvm-specific commands will fail

## Troubleshooting

If you encounter issues with npm or nvm commands:

1. Make sure npm and/or nvm is installed on your system
2. Check that the paths to npm and nvm are in your system's PATH environment variable
3. For nvm issues, verify that the nvm.sh script is in the expected location
4. Try running the commands directly in your system's terminal to verify they work outside of Streamlit
