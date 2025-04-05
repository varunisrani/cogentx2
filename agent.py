from __future__ import annotations as _annotations

import os
import asyncio
from contextlib import AsyncExitStack
import httpx
import json
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Simple MCP Server communication implementation
class MCPServerStdio:
    """Simple implementation of MCP Server communication via STDIO"""
    
    async def send_ready(self):
        """Send ready message to MCP server"""
        print(json.dumps({"ready": True}), flush=True)
    
    async def send_thinking(self, message: str):
        """Send thinking message to MCP server"""
        print(json.dumps({"thinking": message}), flush=True)
    
    async def send_response(self, response: dict):
        """Send response to MCP server"""
        print(json.dumps({"response": response}), flush=True)
    
    async def receive_request(self):
        """Receive request from MCP server"""
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, input)
            if not line:
                return None
            return json.loads(line)
        except EOFError:
            return None
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}

class GitHubSubagent:
    """
    GitHub Subagent to manage GitHub API interactions.

    This subagent handles operations such as creating issues, 
    pulling requests, and other GitHub-related tasks.
    """

    def __init__(self):
        self.api_url = "https://api.github.com"
        self.token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("GitHub Personal Access Token not found in environment variables")

    async def create_issue(self, repo: str, title: str, body: str) -> dict:
        """Create a new issue in the specified GitHub repository."""
        url = f"{self.api_url}/repos/{repo}/issues"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                json={"title": title, "body": body}, 
                headers=headers
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                raise Exception(f"Failed to create issue: {response.status_code} - {response.text}")

class PrimaryAgent:
    """
    Primary AI Agent to manage tasks across different services.

    Use this agent to call subagents:
    - GitHubSubagent for GitHub operations, such as creating issues.
    """

    def __init__(self, mcp_stdio=None):
        """Initialize the primary agent with optional MCP connection"""
        self.mcp_stdio = mcp_stdio
        
    async def call_github_subagent(self, repo: str, title: str, body: str):
        """
        Call the GitHub subagent to create an issue.

        Args:
            repo (str): The repository name in the format 'owner/repo'.
            title (str): The title for the new issue.
            body (str): The body text for the new issue.
        """
        async with AsyncExitStack() as stack:
            github_agent = GitHubSubagent()
            result = await github_agent.create_issue(repo, title, body)
            return result

    async def process_user_input(self, user_input: str):
        """
        Process user input to determine what GitHub action to take.
        
        Format expected: "create issue in {repo} with title {title} and body {body}"
        """
        # Send message to MCP if available
        if self.mcp_stdio:
            await self.mcp_stdio.send_thinking(f"Processing request: {user_input}")
        
        # Simple parsing - in production would use a proper NLP approach or pydantic models
        if "create issue" in user_input.lower():
            try:
                # Very basic parsing - in production use more robust parsing
                parts = user_input.split(" in ")
                if len(parts) < 2:
                    return {"error": "Invalid format. Please specify repository with 'in repo_name'"}
                    
                repo_and_rest = parts[1].split(" with title ")
                if len(repo_and_rest) < 2:
                    return {"error": "Invalid format. Please specify title with 'with title your_title'"}
                    
                repo = repo_and_rest[0].strip()
                
                title_and_body = repo_and_rest[1].split(" and body ")
                if len(title_and_body) < 2:
                    return {"error": "Invalid format. Please specify body with 'and body your_body'"}
                    
                title = title_and_body[0].strip()
                body = title_and_body[1].strip()
                
                # Send progress to MCP if available
                if self.mcp_stdio:
                    await self.mcp_stdio.send_thinking(f"Creating issue in {repo} with title: {title}")
                
                # Call the GitHub subagent
                result = await self.call_github_subagent(repo, title, body)
                return {
                    "success": True,
                    "message": f"Issue created successfully in {repo}",
                    "issue_url": result.get("html_url", ""),
                    "issue_number": result.get("number", "")
                }
            except Exception as e:
                error_msg = f"Error creating issue: {str(e)}"
                if self.mcp_stdio:
                    await self.mcp_stdio.send_thinking(error_msg)
                return {"error": error_msg}
        else:
            return {
                "error": "Unsupported action. Currently I can only 'create issue in repo with title X and body Y'"
            }

async def setup_mcp_agent():
    """Setup and run the agent with MCP integration"""
    mcp_stdio = MCPServerStdio()
    primary_agent = PrimaryAgent(mcp_stdio)
    
    # Initialize communication with MCP
    await mcp_stdio.send_ready()
    
    # Main loop to handle requests
    while True:
        request = await mcp_stdio.receive_request()
        if not request:
            # Empty request, could be a signal to exit
            break
            
        user_input = request.get("input", "")
        if not user_input:
            await mcp_stdio.send_response({"error": "No input provided"})
            continue
            
        result = await primary_agent.process_user_input(user_input)
        await mcp_stdio.send_response(result)

async def main():
    """Main entry point that checks if running in MCP mode or interactive mode"""
    # Check if running as MCP agent
    if os.environ.get("MCP_MODE") == "1":
        await setup_mcp_agent()
        return
        
    # Interactive mode for direct testing
    primary_agent = PrimaryAgent()
    
    print("GitHub Issue Creator - Interactive Mode")
    print("-" * 50)
    print("Example: create issue in owner/repo with title Test Issue and body This is a test")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter your command: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        try:
            result = await primary_agent.process_user_input(user_input)
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"SUCCESS: {result['message']}")
                if "issue_url" in result:
                    print(f"Issue URL: {result['issue_url']}")
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main()) 