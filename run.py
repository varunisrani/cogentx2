#!/usr/bin/env python3
"""
Run script for GitHub agent

Usage:
  python run.py [mode]

Modes:
  interactive    Run the agent in interactive command-line mode (default)
  api            Run the agent as a REST API server
  mcp            Run the agent as an MCP service
  test           Run a quick test of GitHub issue creation
"""

import os
import sys
import asyncio
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_interactive():
    """Run the agent in interactive mode"""
    from agent import main as agent_main
    await agent_main()

async def run_test():
    """Run a quick test of GitHub issue creation"""
    from agent import PrimaryAgent
    
    agent = PrimaryAgent()
    repo = input("Enter repository (format: owner/repo): ")
    title = input("Enter issue title: ")
    body = input("Enter issue body: ")
    
    try:
        result = await agent.call_github_subagent(repo, title, body)
        print("\nSuccess! Issue created:")
        print(f"URL: {result.get('html_url', 'N/A')}")
        print(f"Number: {result.get('number', 'N/A')}")
    except Exception as e:
        print(f"\nError creating issue: {e}")

def run_api():
    """Run the agent as a REST API server"""
    # Import here to avoid loading FastAPI unnecessarily
    from api_server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)

async def run_mcp():
    """Run the agent as an MCP service"""
    os.environ["MCP_MODE"] = "1"  # Set MCP mode flag
    from agent import setup_mcp_agent
    await setup_mcp_agent()

def main():
    """Main entry point for the run script"""
    # Default mode is interactive
    mode = "interactive"
    
    # Check if a mode was specified
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    # Run the specified mode
    if mode == "interactive":
        asyncio.run(run_interactive())
    elif mode == "api":
        run_api()
    elif mode == "mcp":
        asyncio.run(run_mcp())
    elif mode == "test":
        asyncio.run(run_test())
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main() 