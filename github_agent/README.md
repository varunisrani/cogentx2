# GitHub MCP Agent

A modular Python client for interacting with GitHub API using the Pydantic AI framework.

## Overview

This package provides a command-line interface to interact with GitHub's API through the MCP (Model-Control-Protocol) standard. It uses the Pydantic AI framework for handling structured data and OpenAI's models for natural language processing.

## Features

- Natural language interface for GitHub operations
- Integration with OpenAI's language models
- Detailed tracking and display of tool usage
- Colorized console output
- Comprehensive logging
- Clean, modular codebase

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install pydantic-ai openai colorlog python-dotenv
   ```
3. Install required npm packages:
   ```
   npm install -g @modelcontextprotocol/server-github
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
LLM_API_KEY=your_openai_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_personal_access_token
MODEL_CHOICE=gpt-4o-mini  # optional, default is gpt-4o-mini
BASE_URL=https://api.openai.com/v1  # optional
```

### GitHub Personal Access Token

To create a GitHub Personal Access Token:

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token"
3. Give your token a description
4. Select the appropriate scopes (at minimum: `repo`, `user`)
5. Click "Generate token"
6. Copy the token and add it to your `.env` file

## Usage

Run the agent with:

```
python -m github_agent.main
```

For verbose logging:

```
python -m github_agent.main --verbose
```

To specify a custom log file:

```
python -m github_agent.main --log-file ./logs/github_agent.log
```

## Example Queries

- "List my repositories"
- "Find open issues in repository XYZ"
- "Show pull request #123 in repository XYZ"
- "Get information about user johnsmith"
- "Search for repositories about machine learning"
- "Create a new issue in repository XYZ with title 'Bug report'"

## Tool Usage Tracking

The agent tracks which GitHub API tools are used to answer your queries and displays them after each request. This provides transparency about how the agent is interacting with GitHub on your behalf.

## Project Structure

- `models.py`: Configuration and data models
- `tools.py`: MCP tool integration and display
- `agent.py`: Agent setup and query processing with tool tracking
- `main.py`: Command-line interface and main execution
- `__init__.py`: Package initialization

## License

MIT 