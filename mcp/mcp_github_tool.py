import os
import json
import asyncio
from typing import Dict, Any, Optional
from agent import PrimaryAgent

# This is the format an MCP tool expects
MCP_TOOL_DEFINITION = {
    "name": "github_issue_creator",
    "description": "Create issues in GitHub repositories",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "The repository name in format 'owner/repo'"
            },
            "title": {
                "type": "string",
                "description": "The title of the issue"
            },
            "body": {
                "type": "string",
                "description": "The body/description of the issue"
            }
        },
        "required": ["repo", "title", "body"]
    }
}

class MCPGithubTool:
    """
    MCP Tool for GitHub operations
    
    This tool follows the MCP protocol for tool execution
    """
    
    @staticmethod
    async def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the GitHub issue creation
        
        Args:
            params: Dictionary containing repo, title, and body
            
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            repo = params.get("repo")
            title = params.get("title")
            body = params.get("body")
            
            if not all([repo, title, body]):
                return {
                    "status": "error",
                    "message": "Missing required parameters. Please provide repo, title, and body."
                }
            
            # Create the agent and call it
            agent = PrimaryAgent()
            result = await agent.call_github_subagent(repo, title, body)
            
            # Format the result
            return {
                "status": "success",
                "message": f"Issue created successfully in {repo}",
                "data": {
                    "issue_url": result.get("html_url", ""),
                    "issue_number": result.get("number", 0)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating GitHub issue: {str(e)}"
            }

def get_tool_definition():
    """Return the tool definition for MCP registration"""
    return MCP_TOOL_DEFINITION

async def run_from_mcp(input_str: str) -> str:
    """
    Parse and execute a request from MCP
    
    Args:
        input_str: JSON string with parameters
        
    Returns:
        JSON string with results
    """
    try:
        # Parse the input
        params = json.loads(input_str)
        
        # Execute the tool
        result = await MCPGithubTool.execute(params)
        
        # Return JSON string
        return json.dumps(result)
    except json.JSONDecodeError:
        return json.dumps({
            "status": "error",
            "message": "Invalid JSON input"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        })

async def test_tool():
    """Test the tool directly"""
    test_params = {
        "repo": "your-username/your-repo",
        "title": "Test Issue from MCP Tool",
        "body": "This is a test issue created by the GitHub MCP Tool."
    }
    
    result = await MCPGithubTool.execute(test_params)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # When run directly, test the tool
    asyncio.run(test_tool()) 