from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import Field
import os
import requests
import json
import asyncio
import mcp
from mcp.client.websocket import websocket_client
try:
    import smithery
except ImportError:
    print("Smithery package not found. Some tools might not work properly.")

# CrewAI-compatible tools collection


```python
import os
import json
import asyncio
import smithery
import mcp
from mcp.client.websocket import websocket_client
from typing import Optional, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GitHubRepositoryParams(BaseModel):
    """Parameters for GitHub repository operations."""
    query: Optional[str] = Field(default=None, description="The search query for repositories")
    name: Optional[str] = Field(default=None, description="The name of the repository")
    description: Optional[str] = Field(default="", description="The description of the repository")
    private: bool = Field(default=False, description="Whether the repository is private")
    auto_init: bool = Field(default=False, description="Whether to initialize the repository with a README")
    page: int = Field(default=1, description="Page number for pagination")
    per_page: int = Field(default=10, description="Number of results per page")

class GitHubMCPTool(BaseTool):
    """
    A CrewAI tool for interacting with GitHub API via Smithery MCP server.
    
    This tool allows searching for GitHub repositories and creating new repositories.
    
    Example usage:
    ```python
    tool = GitHubMCPTool()
    search_results = tool._run("search_repositories", {"query": "user:varunisrani"})
    print(search_results)
    
    create_result = tool._run("create_repository", {"name": "test-repo", "description": "A test repo"})
    print(create_result)
    ```
    """
    
    name: str = "github_mcp_tool"
    description: str = "Interact with GitHub repositories through search and creation operations."
    url: Optional[str] = None
    
    def __init__(self):
        super().__init__()
        # Get GitHub token from environment variables
        self.github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub Personal Access Token not found in environment variables")
        
        # Create Smithery URL with GitHub server endpoint
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@smithery-ai/github/ws", 
            {
                "githubPersonalAccessToken": self.github_token
            }
        )
    
    async def _arun(self, operation: str, params: Dict[str, Any]) -> str:
        """
        Asynchronously run the GitHub tool with the specified operation and parameters.
        
        Args:
            operation: The GitHub operation to perform.
            params: Parameters for the operation.
            
        Returns:
            String result from the operation.
        """
        valid_operations = ["search_repositories", "create_repository", "list_tools"]
        if operation not in valid_operations:
            return f"Invalid operation: {operation}. Supported operations are {', '.join(valid_operations)}."
        
        try:
            async with websocket_client(self.url) as streams:
                async with mcp.ClientSession(*streams) as session:
                    if operation == "list_tools":
                        return await self._list_tools(session)
                    elif operation == "search_repositories":
                        return await self._search_repositories(session, params)
                    elif operation == "create_repository":
                        return await self._create_repository(session, params)
        
        except Exception as e:
            return f"Error connecting to the GitHub Smithery server: {str(e)}"
    
    def _run(self, operation: str, params: Dict[str, Any] = None) -> str:
        """
        Run the GitHub tool with the specified operation and parameters.
        
        Args:
            operation: The GitHub operation to perform.
            params: Parameters for the operation.
            
        Returns:
            String result from the operation.
        """
        if params is None:
            params = {}
        return asyncio.run(self._arun(operation, params))
    
    async def _list_tools(self, session) -> str:
        """List available GitHub tools."""
        try:
            tools_result = await session.list_tools()
            result_parts = ["# Available GitHub Tools\n"]
            for i, tool in enumerate(tools_result.tools, 1):
                result_parts.append(f"## {i}. {tool.name}")
                result_parts.append(f"Description: {tool.description}\n")
            return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error listing GitHub tools: {str(e)}"
    
    async def _search_repositories(self, session, params: Dict[str, Any]) -> str:
        """Search for GitHub repositories."""
        query = params.get("query")
        if not query:
            return "Error: 'query' parameter is required for search_repositories operation"
        
        page = params.get("page", 1)
        per_page = params.get("per_page", 10)
        
        try:
            search_result = await session.call_tool(
                "search_repositories",
                {
                    "query": query,
                    "page": page,
                    "perPage": per_page
                }
            )
            
            if hasattr(search_result, 'content') and search_result.content:
                repos_data = json.loads(search_result.content[0].text)
                return self._format_repository_search_results(repos_data)
            else:
                return "No search results returned or error in response"
                
        except Exception as e:
            return f"Error searching repositories: {str(e)}"
    
    async def _create_repository(self, session, params: Dict[str, Any]) -> str:
        """Create a GitHub repository."""
        name = params.get("name")
        if not name:
            return "Error: 'name' parameter is required for create_repository operation"
        
        description = params.get("description", "")
        private = params.get("private", False)
        auto_init = params.get("auto_init", False)
        
        try:
            result = await session.call_tool(
                "create_repository",
                {
                    "name": name,
                    "description": description,
                    "private": private,
                    "autoInit": auto_init
                }
            )
            
            if hasattr(result, 'content') and result.content:
                repo_data = json.loads(result.content[0].text)
                return self._format_repository_creation_result(repo_data)
            else:
                return "No repository data returned or error in response"
                
        except Exception as e:
            return f"Error creating repository: {str(e)}"
    
    def _format_repository_search_results(self, repos_data: Dict[str, Any]) -> str:
        """Format repository search results into a structured string."""
        result_parts = ["# GitHub Repository Search Results\n"]
        total_count = repos_data.get('total_count', 0)
        result_parts.append(f"Found {total_count} repositories.\n")
        
        items = repos_data.get('items', [])
        if items:
            for i, repo in enumerate(items, 1):
                result_parts.append(f"## {i}. {repo.get('name', 'Unnamed repository')}")
                result_parts.append(f"- **Owner**: {repo.get('owner', {}).get('login', 'Unknown')}")
                result_parts.append(f"- **Description**: {repo.get('description', 'No description')}")
                result_parts.append(f"- **URL**: {repo.get('html_url', 'No URL')}")
                result_parts.append(f"- **Stars**: {repo.get('stargazers_count', 0)}")
                result_parts.append(f"- **Forks**: {repo.get('forks_count', 0)}")
                result_parts.append(f"- **Language**: {repo.get('language', 'Not specified')}")
                result_parts.append(f"- **Created**: {repo.get('created_at', 'Unknown date')}\n")
        else:
            result_parts.append("No repositories found matching the query.")
        
        return "\n".join(result_parts)
    
    def _format_repository_creation_result(self, repo_data: Dict[str, Any]) -> str:
        """Format repository creation result into a structured string."""
        result_parts = ["# GitHub Repository Created\n"]
        result_parts.append(f"## {repo_data.get('name', 'Unnamed repository')}")
        result_parts.append(f"- **Full Name**: {repo_data.get('full_name', 'Unknown')}")
        result_parts.append(f"- **Description**: {repo_data.get('description', 'No description')}")
        result_parts.append(f"- **URL**: {repo_data.get('html_url', 'No URL')}")
        result_parts.append(f"- **Private**: {str(repo_data.get('private', False))}")
        result_parts.append(f"- **Default Branch**: {repo_data.get('default_branch', 'main')}")
        result_parts.append(f"- **Clone URL**: {repo_data.get('clone_url', 'No URL')}")
        result_parts.append(f"- **Created at**: {repo_data.get('created_at', 'Unknown date')}")
        
        return "\n".join(result_parts)
```
