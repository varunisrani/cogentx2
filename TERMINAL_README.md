# Archon Terminal Implementations

This project includes different terminal implementations for Streamlit applications:

1. **Basic Terminal**: A simple terminal using the streamlit-terminal component
2. **Archon Terminal**: A full-featured terminal with advanced capabilities
3. **Terminal Comparison**: A side-by-side comparison of both implementations

## Quick Start

Run the terminal launcher script to choose which implementation to use:

```bash
python run_terminal.py [basic|archon|compare]
```

By default, it will run the comparison page if no argument is provided.

## Terminal Implementations

### Basic Terminal (streamlit-terminal)

A lightweight terminal component that can be easily integrated into any Streamlit app.

**Features:**
- Simple integration
- Component-based approach
- Easy to use as a dependency
- Real-time command output

**Best for:**
- When you need a simple terminal interface in your Streamlit app
- When you want a clean, component-based approach
- When you prefer a lightweight dependency

### Archon Terminal

The full-featured Archon terminal implementation with advanced features.

**Features:**
- Command history
- Directory tracking (cd command support)
- Process management
- Multi-line command input
- Input for long-running processes
- Code editor integration
- Downloadable logs
- Detailed process information

**Best for:**
- When you need a full-featured terminal experience
- When you require command history and directory tracking
- When you want detailed process information and management
- When you need support for multi-line commands and script execution

## Installation

To use these terminal implementations, you need to install the required packages:

```bash
pip install -r requirements.txt
```

Or manually install the individual components:

```bash
pip install streamlit
pip install streamlit-terminal  # For the basic terminal component
pip install streamlit-ace  # For enhanced code editing (optional)
pip install rich  # For enhanced terminal output (optional)
```

## Usage Examples

### Basic Terminal (streamlit-terminal)

```python
import streamlit as st
from streamlit_terminal import st_terminal

# Basic usage
st_terminal(key="terminal")

# With custom command
st_terminal(key="terminal", command="echo 'Hello, World!'")

# Get outputs
full_outputs, updated_outputs = st_terminal(key="terminal")
```

### Archon Terminal

The Archon terminal is implemented as a standalone Streamlit application. Run it with:

```bash
streamlit run terminal_app.py
```

## Integration in Your Projects

### Using Basic Terminal in Your Streamlit App

```python
# Add to your app
from streamlit_terminal import st_terminal

# Create terminal component
terminal_outputs, terminal_updates = st_terminal(key="my_terminal")

# Use the outputs in your app
st.write("Last command output:", terminal_updates)
```

### Using Archon Terminal Features

You can borrow functionality from the Archon terminal implementation to enhance your own applications. See `terminal_app.py` for implementation details.

## Differences Between Implementations

| Feature | Streamlit-Terminal | Archon Terminal |
|---------|-------------------|----------------|
| Interactive Terminal | ✅ | ✅ |
| Command Execution | ✅ | ✅ |
| Real-time Output | ✅ | ✅ |
| Command History | ❌ | ✅ |
| Directory Tracking | ❌ | ✅ |
| Process Management | Limited | ✅ |
| Multi-line Commands | ❌ | ✅ |
| Support for stdin | Limited | ✅ |
| Color Output | ✅ | ✅ |
| Code Editor Integration | ❌ | ✅ |
| Downloadable Logs | ❌ | ✅ |
| Installation Complexity | Simple | Moderate |

## Contributing

Contributions to either terminal implementation are welcome! Please check the project repositories for contribution guidelines:

- streamlit-terminal: [GitHub](https://github.com/akipg/streamlit-terminal)
- Archon: [GitHub](https://github.com/your-username/Archon) 