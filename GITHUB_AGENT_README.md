# GitHub Agent for Archon

This module provides GitHub API integration for the Archon system, allowing it to interact with GitHub repositories through a subagent architecture.

## Setup

1. Copy the `.env.example` file to `.env` if you haven't already set up your environment variables.
2. Add your GitHub Personal Access Token to the `.env` file:
   ```
   GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
   ```
   
   Note: The provided token `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` has been included in the .env file already.

3. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

## Usage

The GitHub agent can be used in multiple ways:

### 1. Interactive Mode

Run the agent in interactive command-line mode:

```bash
python run.py interactive
```

Or simply:

```bash
python run.py
```

This will allow you to input commands in natural language:

```
Enter your command: create issue in owner/repo with title Bug Report and body This is a critical bug
```

### 2. API Server Mode

Run the agent as a REST API server:

```bash
python run.py api
```

This starts a FastAPI server on port 8000. You can interact with it using HTTP requests:

```bash
# Create an issue with structured data
curl -X POST http://localhost:8000/github/issue \
  -H "Content-Type: application/json" \
  -d '{"repo":"owner/repo","title":"API Test","body":"Testing from API"}'

# Create an issue with natural language
curl -X POST http://localhost:8000/github/parse \
  -H "Content-Type: application/json" \
  -d '{"prompt":"create issue in owner/repo with title API Test and body Testing via natural language"}'
```

### 3. MCP Integration

Run the agent as an MCP service:

```bash
python run.py mcp
```

You can also use the MCP-specific tools:

- `mcp_github_tool.py` - Standalone MCP tool for GitHub integration
- `github_mcp_tool.py` - More detailed MCP tool with Pydantic models

### 4. Integration with Archon

The `GitHubSubagent` class can be imported and used within your Archon workflows:

```python
from agent import GitHubSubagent

# Create an instance of the GitHub subagent
github_agent = GitHubSubagent()

# Use it to create issues
async def create_github_issue(repo, title, body):
    result = await github_agent.create_issue(repo, title, body)
    return result
```

## Quick Test

To quickly test GitHub issue creation:

```bash
python run.py test
```

This will prompt you for repository, title, and body, then attempt to create an issue.

## Features

Currently, the GitHub agent supports:

- Creating issues in repositories
- Natural language processing for issue creation
- REST API for programmatic access
- MCP integration for use with AI assistants

Future extensions could include:
- Creating pull requests
- Managing repository workflows
- Commenting on issues and pull requests
- Repository statistics and metrics

## Security Notes

- Never hardcode your GitHub token in source code
- Use environment variables or secure secret management
- The GitHub token provided has specific permissions - review them before using for sensitive operations 