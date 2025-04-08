import os
import json
import asyncio
from typing import Dict, Any, Optional, List

import smithery
from mcp.client.websocket import websocket_client
import mcp

from crewai.tools.tool import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'github_agent', '.env')
load_dotenv(dotenv_path)

class SearchRepositoriesParams(BaseModel):
    query: str = Field(description="The search query for repositories")
    page: int = Field(default=1, description="Page number for pagination")
    per_page: int = Field(default=10, description="Number of results per page")

class CreateRepositoryParams(BaseModel):
    name: str = Field(description="The name of the repository")
    description: str = Field(default="", description="The description of the repository")
    private: bool = Field(default=False, description="Whether the repository is private")
    auto_init: bool = Field(default=False, description="Whether to initialize the repository with a README")

class CreateIssueParams(BaseModel):
    title: str = Field(description="The title of the issue")
    body: str = Field(default="", description="The body of the issue")
    repo: str = Field(description="The repository to create the issue in")

class MCPGitHubTool(BaseTool):
    name: str = "github_tool"
    description: str = "Interact with GitHub repositories through search, creation, and issue tracking operations"
    args_schema: Dict = {
        "operation": {
            "type": "string",
            "enum": ["search_repositories", "create_repository", "list_tools", "create_issue"],
            "description": "The GitHub operation to perform"
        },
        "params": {
            "type": "object",
            "description": "Parameters for the operation"
        }
    }
    
    def __init__(self):
        super().__init__()
        self.github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub Personal Access Token not found in environment variables")
        
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@smithery-ai/github/ws", 
            {
                "githubPersonalAccessToken": self.github_token
            }
        )
    
    def _run(self, operation: str, params: Dict[str, Any] = None) -> str:
        if params is None:
            params = {}
        result = asyncio.run(self._run_async(operation, params))
        return result
    
    async def _run_async(self, operation: str, params: Dict[str, Any]) -> str:
        valid_operations = ["search_repositories", "create_repository", "list_tools", "create_issue"]
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
                    elif operation == "create_issue":
                        return await self._create_issue(session, params)
        
        except Exception as e:
            return f"Error connecting to the GitHub Smithery server: {str(e)}"
    
    async def _list_tools(self, session) -> str:
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
    
    async def _create_issue(self, session, params: Dict[str, Any]) -> str:
        title = params.get("title")
        repo = params.get("repo")
        if not title or not repo:
            return "Error: 'title' and 'repo' parameters are required for create_issue operation"
        
        body = params.get("body", "")
        
        try:
            result = await session.call_tool(
                "create_issue",
                {
                    "title": title,
                    "body": body,
                    "repo": repo
                }
            )
            
            if hasattr(result, 'content') and result.content:
                issue_data = json.loads(result.content[0].text)
                return self._format_issue_creation_result(issue_data)
            else:
                return "No issue data returned or error in response"
                
        except Exception as e:
            return f"Error creating issue: {str(e)}"
    
    def _format_repository_search_results(self, repos_data: Dict[str, Any]) -> str:
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

    def _format_issue_creation_result(self, issue_data: Dict[str, Any]) -> str:
        result_parts = ["# GitHub Issue Created\n"]
        result_parts.append(f"## {issue_data.get('title', 'Unnamed issue')}")
        result_parts.append(f"- **Issue Number**: {issue_data.get('number', 'Unknown')}")
        result_parts.append(f"- **Body**: {issue_data.get('body', 'No body')}")
        result_parts.append(f"- **URL**: {issue_data.get('html_url', 'No URL')}")
        result_parts.append(f"- **State**: {issue_data.get('state', 'Unknown')}")
        
        return "\n".join(result_parts)

class GitHubSearchTool(MCPGitHubTool):
    name: str = "github_search_tool"
    description: str = "Search for GitHub repositories by user, topic, or other criteria"
    
    def _run(self, query: str) -> str:
        return super()._run("search_repositories", {"query": query})

class GitHubCreateTool(MCPGitHubTool):
    name: str = "github_create_tool"
    description: str = "Create a new GitHub repository with specified parameters"
    
    def _run(self, name: str, description: str = "", private: bool = False, auto_init: bool = False) -> str:
        return super()._run("create_repository", {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init
        })

class GitHubIssueTool(MCPGitHubTool):
    name: str = "github_issue_tool"
    description: str = "Create a new issue in a specified GitHub repository"
    
    def _run(self, title: str, repo: str, body: str = "") -> str:
        return super()._run("create_issue", {
            "title": title,
            "repo": repo,
            "body": body
        })