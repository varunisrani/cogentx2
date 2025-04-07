# Terminal Integration for Streamlit UI

This document explains how the terminal functionality has been integrated into the Streamlit UI and how to handle any dependency issues.

## Overview

The terminal integration adds a new "Terminal" tab in the "Generated Code" section of the Streamlit UI. This terminal allows users to run commands, execute Python scripts, and interact with the codebase directly from the Streamlit interface.

## Features

- Interactive terminal with command history
- Support for running npm and nvm commands
- Quick command buttons for common operations
- Support for running MCP agents (Spotify, GitHub)
- Fallback mechanism for handling dependency issues

## Implementation Details

The terminal integration uses two different implementations:

1. **Primary Implementation**: Uses the `streamlit_terminal` package, which provides a full-featured terminal experience with interactive input/output.

2. **Fallback Implementation**: Uses a custom terminal implementation (`custom_terminal.py`) that avoids the dependency on OpenAI's voice helpers, which can cause issues with the `sounddevice` package.

## Dependency Issues and Solutions

### The sounddevice Dependency Issue

The `streamlit_terminal` package indirectly triggers a dependency check for OpenAI's voice helpers, which requires the `sounddevice` package. If this package is not installed, you'll see an error like:

```
MissingDependencyError: OpenAI error: missing `sounddevice`
This feature requires additional dependencies: $ pip install openai[voice_helpers]
```

### Solutions

There are three ways to resolve this issue:

1. **Install the sounddevice package**:
   ```
   pip install sounddevice
   ```

2. **Install OpenAI with voice helpers**:
   ```
   pip install openai[voice_helpers]
   ```

3. **Use the fallback implementation**:
   The code automatically falls back to the custom terminal implementation if there are any issues with the primary implementation.

## Usage

### Running Commands

1. Navigate to the "Generated Code" tab in the Streamlit UI
2. Click on the "Terminal" subtab
3. Type your command in the input field and press Enter or click "Run"

### Using Quick Commands

The terminal interface includes buttons for common commands:

- **List Files**: Runs `ls -la` to list all files in the current directory
- **Check Python Version**: Runs `python --version`
- **Check Node Version**: Runs `node --version`
- **List Python Packages**: Runs `pip list`
- **Run Pydantic AI**: Runs `python -m pydantic_ai.cli run`
- **Check npm Version**: Runs `npm --version`

### Running MCP Agents

There are dedicated buttons for running the MCP agents:

- **Run Spotify Agent (Terminal)**: Runs the Spotify agent in terminal mode
- **Run GitHub Agent**: Runs the GitHub agent
- **Run Spotify Streamlit App**: Runs the Spotify agent in Streamlit mode

## Troubleshooting

If you encounter issues with the terminal:

1. **Missing sounddevice error**:
   - Install the sounddevice package: `pip install sounddevice`
   - The terminal will automatically fall back to the custom implementation

2. **Terminal not responding**:
   - Click the "Terminate Process" button to stop any running processes
   - Click the "Clear Terminal" button to reset the terminal

3. **Commands not executing**:
   - Make sure you're in the correct directory
   - Check that the command is valid
   - Try running the command in a regular terminal to see if it works
