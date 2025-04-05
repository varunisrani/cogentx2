from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Dict, Any, Optional
from agent import PrimaryAgent

# Define API models
class GithubIssueRequest(BaseModel):
    repo: str
    title: str
    body: str

class GithubDirectRequest(BaseModel):
    prompt: str

app = FastAPI(
    title="GitHub API Agent",
    description="API for GitHub operations using an agent architecture",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "GitHub API Agent is running"}

@app.post("/github/issue")
async def create_github_issue(request: GithubIssueRequest):
    """
    Create a GitHub issue with structured data
    
    Args:
        request: Contains repo, title, and body for the issue
        
    Returns:
        Issue creation response with URL and issue number
    """
    try:
        agent = PrimaryAgent()
        result = await agent.call_github_subagent(
            request.repo, 
            request.title, 
            request.body
        )
        
        return {
            "success": True,
            "message": f"Issue created successfully in {request.repo}",
            "issue_url": result.get("html_url", ""),
            "issue_number": result.get("number", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating issue: {str(e)}")

@app.post("/github/parse")
async def parse_github_command(request: GithubDirectRequest):
    """
    Parse a natural language command to create GitHub issues
    
    Example: "create issue in owner/repo with title Bug Report and body Found a critical bug"
    
    Args:
        request: Contains the natural language prompt
        
    Returns:
        Result of the parsed command execution
    """
    try:
        agent = PrimaryAgent()
        result = await agent.process_user_input(request.prompt)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing command: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with usage information"""
    return {
        "message": "GitHub API Agent",
        "usage": {
            "create_issue": "POST /github/issue with repo, title, and body",
            "parse_command": "POST /github/parse with prompt"
        },
        "example": {
            "parse_command": {
                "prompt": "create issue in owner/repo with title Test Issue and body This is a test"
            }
        }
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000) 