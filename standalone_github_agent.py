#!/usr/bin/env python3
"""
Standalone GitHub Agent
----------------------
A simplified version of the GitHub agent that can be run directly.
"""
import asyncio
import os
import re
import httpx
from dataclasses import dataclass
from typing import Any, Dict
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()

def get_model():
    """Get the language model from environment variables."""
    llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', os.getenv('OPENAI_API_KEY'))

    # Initialize the OpenAI client
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return OpenAIModel(llm, client=client)
    except:
        # Fallback to standard initialization
        return OpenAIModel(llm)

@dataclass
class GitHubDeps:
    """Dependencies for the GitHub agent."""
    client: httpx.AsyncClient
    github_token: str | None = None

system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the repository with the provided tools before answering the user's question unless you have already.

When answering a question about the repo, always start your answer with the full repo URL in brackets and then give your answer on a newline. Like:

[Using https://github.com/[repo URL from the user]]

Your answer here...
"""

# Initialize the GitHub agent
github_agent = Agent(
    get_model(),
    system_prompt=system_prompt,
    deps_type=GitHubDeps,
    retries=2
)

@github_agent.tool
async def get_repo_info(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get repository information including size and description using GitHub API.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Repository information as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository info: {response.text}"
    
    data = response.json()
    size_mb = data['size'] / 1024
    
    return (
        f"Repository: {data['full_name']}\n"
        f"Description: {data['description']}\n"
        f"Size: {size_mb:.1f}MB\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Language: {data['language']}\n"
        f"Created: {data['created_at']}\n"
        f"Last Updated: {data['updated_at']}"
    )

@github_agent.tool
async def get_repo_structure(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get the directory structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Directory structure as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get repository structure: {response.text}"
    
    data = response.json()
    tree = data['tree']
    
    # Build directory structure
    structure = []
    for item in tree:
        if not any(excluded in item['path'] for excluded in ['.git/', 'node_modules/', '__pycache__/']):
            structure.append(f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}")
    
    return "\n".join(structure)

@github_agent.tool
async def get_file_content(ctx: RunContext[GitHubDeps], github_url: str, file_path: str) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        file_path: Path to the file within the repository.

    Returns:
        str: File content as a string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get file content: {response.text}"
    
    return response.text

async def main():
    """Main function to run the GitHub agent."""
    # Get GitHub token
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