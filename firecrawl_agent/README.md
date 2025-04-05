# Firecrawl MCP Agent

A modular Python client for interacting with Firecrawl web crawling services using the Pydantic AI framework.

## Overview

This package provides a command-line interface to interact with Firecrawl's web crawling capabilities through the MCP (Model-Control-Protocol) standard. It uses the Pydantic AI framework for handling structured data and OpenAI's models for natural language processing.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install pydantic-ai openai colorlog python-dotenv
   ```
3. Install Firecrawl MCP:
   ```
   npm install -g firecrawl-mcp
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
LLM_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
MODEL_CHOICE=gpt-4o-mini  # optional, default is gpt-4o-mini
BASE_URL=https://api.openai.com/v1  # optional

# Optional Firecrawl settings
FIRECRAWL_RETRY_MAX_ATTEMPTS=5
FIRECRAWL_RETRY_INITIAL_DELAY=2000
FIRECRAWL_RETRY_MAX_DELAY=30000
FIRECRAWL_RETRY_BACKOFF_FACTOR=3
FIRECRAWL_CREDIT_WARNING_THRESHOLD=2000
FIRECRAWL_CREDIT_CRITICAL_THRESHOLD=500
```

## Usage

Run the agent with:

```
python -m firecrawl_agent.main
```

For verbose logging:

```
python -m firecrawl_agent.main --verbose
```

To specify a custom log file:

```
python -m firecrawl_agent.main --log-file ./logs/custom_log.log
```

## Features

- Natural language interface for web search and crawling queries
- Integration with OpenAI's language models
- Colorized console output
- Comprehensive logging
- Error handling and retry mechanisms
- Credit usage monitoring

## Project Structure

- `models.py`: Configuration and data models
- `tools.py`: MCP tool integration and display
- `agent.py`: Agent setup and query processing
- `main.py`: Command-line interface and main execution
- `__init__.py`: Package initialization

## License

MIT 