# Serper MCP Agent

A modular Python client for interacting with Serper search API using the Pydantic AI framework.

## Overview

This package provides a command-line interface to interact with Serper's search capabilities through the MCP (Model-Control-Protocol) standard. It uses the Pydantic AI framework for handling structured data and OpenAI's models for natural language processing.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install pydantic-ai openai colorlog python-dotenv
   ```
3. Install required npm packages:
   ```
   npm install -g @smithery/cli
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
LLM_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
MODEL_CHOICE=gpt-4o-mini  # optional, default is gpt-4o-mini
BASE_URL=https://api.openai.com/v1  # optional
```

You can get a Serper API key by signing up at [https://serper.dev/](https://serper.dev/).

## Usage

Run the agent with:

```
python -m serper_agent.main
```

For verbose logging:

```
python -m serper_agent.main --verbose
```

To specify a custom log file:

```
python -m serper_agent.main --log-file ./logs/custom_log.log
```

## Features

- Natural language interface for web searches
- Integration with OpenAI's language models
- Colorized console output
- Comprehensive logging
- Clean, modular codebase

## Search Examples

- "What is the current population of Tokyo?"
- "Find the latest news about climate change"
- "Search for traditional Italian pasta recipes"
- "Look up the latest iPhone reviews"
- "Find information about quantum computing breakthroughs in 2024"

## Project Structure

- `models.py`: Configuration and data models
- `tools.py`: MCP tool integration and display
- `agent.py`: Agent setup and query processing
- `main.py`: Command-line interface and main execution
- `__init__.py`: Package initialization

## License

MIT 