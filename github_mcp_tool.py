import os
import json
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class GithubIssueRequest(BaseModel):
    """Request format for creating a GitHub issue"""
    repo: str = Field(..., description="Repository in format 'owner/repo'")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue content/description")

class GithubIssueResponse(BaseModel):
    """Response format for GitHub issue creation"""
    issue_url: str = Field(..., description="URL of the created issue")
    issue_number: int = Field(..., description="Number of the created issue")
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field(..., description="Success message")

class GithubIssueTool:
    """MCP Tool to create GitHub issues"""
    
    async def create_issue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an issue in GitHub"""
        from agent import PrimaryAgent
        
        try:
            # Extract parameters
            repo = params.get("repo")
            title = params.get("title")
            body = params.get("body")
            
            # Validate parameters
            if not all([repo, title, body]):
                return {
                    "success": False,
                    "error": "Missing required parameters. Please provide repo, title, and body."
                }
            
            # Create the agent and call it
            agent = PrimaryAgent()
            result = await agent.call_github_subagent(repo, title, body)
            
            # Format the response
            return {
                "success": True,
                "message": f"Issue created successfully in {repo}",
                "issue_url": result.get("html_url", ""),
                "issue_number": result.get("number", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating GitHub issue: {str(e)}"
            }

def register_tool():
    """Register this tool with MCP"""
    # This function would register the tool with MCP's tool registry
    # The implementation depends on how MCP tool registration works
    pass

async def main():
    """Test the tool directly"""
    tool = GithubIssueTool()
    test_params = {
        "repo": "your-username/your-repo",
        "title": "Test Issue from MCP Tool",
        "body": "This is a test issue created by the GitHub MCP Tool."
    }
    
    result = await tool.create_issue(test_params)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # When run directly, test the tool
    asyncio.run(main()) 