# Serper-Spotify Agent

A powerful AI agent that combines Google Search (via Serper) and Spotify functionalities to answer questions and play music.

## Overview

This agent integrates two powerful MCP (Machine Callable Programs) servers:
1. **Serper MCP Server**: Provides web search capabilities using the Serper API (Google Search)
2. **Spotify MCP Server**: Enables interaction with the Spotify API to search and retrieve music information

## Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- Serper API key (optional, but required for web search)
- Spotify API key (optional, but required for music functionality)

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   SPOTIFY_API_KEY=your_spotify_api_key_here
   MODEL_CHOICE=gpt-4-0125-preview  # or another compatible model
   ```
4. Make sure Node.js dependencies are installed automatically or run:
   ```bash
   npm install -g @smithery/cli
   ```

## Running the Agent

To start the agent, run:

```bash
python main.py
```

The agent will attempt to initialize both the Serper and Spotify MCP servers based on your provided API keys. If at least one server initializes successfully, the agent will start and you can interact with it.

## Troubleshooting MCP Server Issues

If you experience problems with the MCP servers, you can run the diagnostic tool:

```bash
python test_mcp_revised.py
```

This script will:
1. Check for required environment variables
2. Test Node.js and npm availability
3. Attempt to create both MCP servers
4. Inspect the server objects for necessary attributes and methods
5. Test server initialization using multiple methods
6. Try to list available tools from each server

The diagnostic output will help identify any issues with your setup.

### Common Problems and Solutions

1. **Missing API Keys**: Make sure your `.env` file contains valid API keys
   ```
   SERPER_API_KEY=your_serper_api_key
   SPOTIFY_API_KEY=your_spotify_api_key
   ```

2. **Node.js/npm Issues**: 
   - Ensure Node.js 16+ is installed (`node --version`)
   - Ensure npm is installed (`npm --version`)
   - Try reinstalling @smithery/cli: `npm install -g @smithery/cli`

3. **MCP Server Initialization Failures**:
   - Check that your API keys have proper permissions
   - Make sure your Node.js environment can access the internet
   - Try running with verbose logging to see detailed error messages

4. **"No method available to start server" Error**:
   - The agent now includes multiple methods to initialize the servers
   - Different versions of pydantic_ai use different methods
   - The code includes compatibility enhancements for various scenarios

## Recent Enhancements

The agent has been significantly improved with:

1. **Robust Error Handling**: Better handling of initialization failures and runtime errors
2. **Multi-Method Server Initialization**: Multiple approaches to start servers based on available methods
3. **Manual Session Management**: Compatibility fixes for different versions of pydantic_ai
4. **Comprehensive Logging**: Detailed error messages and diagnostic information
5. **Graceful Degradation**: The agent will work with at least one functioning MCP server
6. **Adaptive System Prompts**: Modifies system prompts based on available functionality

## Architecture

The codebase is organized as follows:

- `main.py`: Entry point and interactive loop
- `agent.py`: Agent setup and configuration
- `tools.py`: MCP server creation and initialization
- `test_mcp_revised.py`: Diagnostic tools for troubleshooting

The agent uses `pydantic_ai` to manage the LLM (Large Language Model) interactions and the MCP servers for tool functionality.

## Extending with New MCP Servers

To add additional MCP servers:

1. Create a new function in `tools.py` similar to `create_serper_mcp_server()`
2. Update the `setup_agent()` function in `agent.py` to initialize the new server
3. Modify the system prompt in `main.py` to include the new capabilities

## License

[Insert your license information here] 