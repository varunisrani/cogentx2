```markdown
# MCP Tools Package

## Overview

The MCP Tools package provides a set of utilities designed to facilitate integration with various APIs. This package currently includes two tools: **Serper** and **GitHub**. The Serper tool enables users to perform web searches and content scraping through the Serper API, while the GitHub tool allows users to interact with the GitHub API for repository management tasks. This package aims to simplify the process of retrieving information and managing GitHub repositories programmatically.

## Setup Instructions

To get started with the MCP Tools package, follow these instructions:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/mcp-tools.git
   cd mcp-tools
   ```

2. **Install Required Packages:**
   Ensure you have Python installed (version 3.7 or higher). Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Keys:**
   - **Serper API Key:** To use the Serper tool, you will need to sign up at [Serper](https://serper.dev/) and obtain your API key. 
   - **GitHub Personal Access Token:** For GitHub integration, create a personal access token by going to your GitHub settings under Developer settings -> Personal access tokens. Ensure you grant the necessary permissions (like `repo` scope) for the tasks you intend to perform.

4. **Configuration:**
   Place your API keys in a configuration file or set them as environment variables for security:
   ```bash
   export SERPER_API_KEY='your_serper_api_key'
   export GITHUB_TOKEN='your_github_token'
   ```

## Directory Structure

The MCP Tools package follows this directory structure:

```
mcp-tools/
│
├── __init__.py          # Package initialization file
├── tool.py              # Contains the main logic for the MCP tools
├── verification.json     # Sample input/output for verification purposes
├── example.py           # Example usage of the tools
├── connection.py        # Handles API connections and authentication
└── requirements.txt     # Lists the required Python packages
```

## Direct Usage Examples

### Using the Serper Tool

To perform a web search using the Serper tool, you can use the following code in your Python script:

```python
from tool import Serper

# Initialize Serper with your API key
serper = Serper(api_key='your_serper_api_key')

# Perform a search query
results = serper.search("latest technology news")

# Print the results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Link: {result['link']}")
```

### Using the GitHub Tool

To interact with the GitHub API, you can create a new repository or search for existing ones using the GitHub tool. Here’s how:

```python
from tool import GitHub

# Initialize GitHub with your token
github = GitHub(token='your_github_token')

# Create a new repository
new_repo = github.create_repository("my-new-repo", description="A new repository for testing")

print(f"Created Repository: {new_repo['html_url']}")

# Search for repositories
search_results = github.search_repositories("machine learning")

for repo in search_results:
    print(f"Repository Name: {repo['name']}, URL: {repo['html_url']}")
```

## Troubleshooting Tips

- **Invalid API Key:** If you encounter errors related to API authentication, double-check your API keys and ensure they are correctly set in your environment or configuration file.
- **Network Issues:** Ensure you have an active internet connection, as both tools require connectivity to their respective APIs.
- **Dependency Errors:** If you face issues while installing dependencies, make sure you are using a compatible version of Python and that you have the necessary permissions to install packages.
- **API Rate Limits:** Be mindful of the rate limits imposed by the Serper and GitHub APIs to avoid being temporarily blocked from making requests.

For further assistance, refer to the official documentation of the respective APIs.
```