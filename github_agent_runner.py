#!/usr/bin/env python3
"""
GitHub Agent Runner
------------------
A simple script to run the GitHub Pydantic AI agent.
Takes user input as GitHub URL and query, processes it, and displays results.
"""
import os
import sys
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Use importlib to import the module directly from its file path
import importlib.util
import dataclasses
from typing import Any

# Path to the GitHub agent module
agent_path = Path(__file__).parent / "agent-resources" / "examples" / "pydantic_github_agent.py"

# Check if the file exists
if not agent_path.exists():
    print(f"Error: Could not find the GitHub agent at {agent_path}")
    sys.exit(1)

# Import the module dynamically
spec = importlib.util.spec_from_file_location("pydantic_github_agent", agent_path)
github_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(github_agent_module)

# Get GitHub agent and dependencies from the module
github_agent = github_agent_module.github_agent
GitHubDeps = github_agent_module.GitHubDeps

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get GitHub token from environment
    github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not github_token:
        print("Warning: No GitHub token found in .env file. API rate limits may apply.")
    
    print("GitHub Repository Explorer")
    print("Enter a GitHub repo URL and your query, or 'quit' to exit")
    
    # Create HTTP client
    async with httpx.AsyncClient() as client:
        # Create dependencies
        deps = GitHubDeps(client=client, github_token=github_token)
        
        while True:
            # Get user input
            user_input = input("\nEnter GitHub URL and query (e.g., https://github.com/user/repo What files are in this repo?): ")
            
            if user_input.lower() == 'quit':
                break
                
            # Split input into GitHub URL and query
            parts = user_input.split(' ', 1)
            if len(parts) == 0 or not parts[0].startswith('https://github.com/'):
                print("Error: Please enter a valid GitHub URL followed by your query.")
                continue
            
            github_url = parts[0]
            query = parts[1] if len(parts) > 1 else "Tell me about this repository"
            
            print(f"\nProcessing query for {github_url}...")
            
            try:
                # Run the agent with the dependencies
                result = await github_agent.run(f"For the repository {github_url}: {query}", deps=deps)
                print(f"\n{result.data}")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main()) 