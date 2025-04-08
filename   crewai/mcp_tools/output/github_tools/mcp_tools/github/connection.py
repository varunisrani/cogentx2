```python
import os
import asyncio
import traceback
import json
import smithery
import mcp
from mcp.client.websocket import websocket_client
from dotenv import load_dotenv

# Load environment variables from github_agent/.env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'github_agent', '.env')
load_dotenv(dotenv_path)

# Get GitHub token from environment variables
github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
if not github_token:
    raise ValueError("GitHub Personal Access Token not found in github_agent/.env file")

print(f"Using GitHub token: {github_token[:5]}...{github_token[-5:]}")  # Only show first and last 5 chars for security

# Create Smithery URL with server endpoint
url = smithery.create_smithery_url(
    "wss://server.smithery.ai/@smithery-ai/github/ws", 
    {
        "githubPersonalAccessToken": github_token
    }
)
print(f"Created Smithery URL (with token redacted): {url.replace(github_token, '[REDACTED]')}")

async def main():
    try:
        # Connect to the server using websocket client
        print("\nConnecting to Smithery server...")
        async with websocket_client(url) as streams:
            print("Connected successfully! Creating MCP session...")
            async with mcp.ClientSession(*streams) as session:
                # List available tools
                print("Listing available GitHub tools...\n")
                tools_result = await session.list_tools()
                
                # Extract and display tool names and descriptions
                tools = []
                for tool in tools_result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description
                    })
                
                # Print tools in a readable format
                print(f"Found {len(tools)} GitHub tools:")
                print("-" * 80)
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool['name']}")
                    print(f"   Description: {tool['description']}")
                    print("-" * 80)
                
                # Search for top 5 repositories from varunisrani
                print("\n🔍 Searching for top repositories from varunisrani...")
                try:
                    search_result = await session.call_tool(
                        "search_repositories", 
                        {
                            "query": "user:varunisrani",
                            "page": 1,
                            "perPage": 5
                        }
                    )
                    
                    if hasattr(search_result, 'content') and search_result.content:
                        # Parse the JSON response
                        repos_data = json.loads(search_result.content[0].text)
                        
                        # Display repositories information
                        print(f"\n📚 Top {min(5, repos_data['total_count'])} repositories from varunisrani:")
                        print("=" * 80)
                        
                        for i, repo in enumerate(repos_data.get('items', [])[:5], 1):
                            print(f"{i}. {repo.get('name', 'Unnamed repo')}")
                            print(f"   Description: {repo.get('description', 'No description')}")
                            print(f"   URL: {repo.get('html_url', 'N/A')}")
                            print(f"   Stars: {repo.get('stargazers_count', 0)} | Forks: {repo.get('forks_count', 0)}")
                            print(f"   Language: {repo.get('language', 'Not specified')}")
                            print(f"   Created: {repo.get('created_at', 'N/A')}")
                            print("-" * 80)
                    else:
                        print("No repositories found or error in response")
                        
                except Exception as e:
                    print(f"Error searching repositories: {e}")
                    traceback.print_exc()
                
                # Create a repository example
                print("\nCreating a GitHub repository...")
                repo_name = f"test-smithery-repo-{int(asyncio.get_event_loop().time())}"
                
                try:
                    result = await session.call_tool(
                        "create_repository", 
                        {
                            "name": repo_name,
                            "description": "A test repository created via Smithery and MCP",
                            "private": True,
                            "autoInit": True
                        }
                    )
                    print(f"Repository created successfully!")
                    
                    # Extract the repository information from the content
                    if hasattr(result, 'content') and result.content:
                        # The first content item contains the repository data as JSON text
                        text_content = result.content[0].text
                        try:
                            # Parse the JSON text into a Python dictionary
                            repo_data = json.loads(text_content)
                            
                            # Display repository information
                            print("\nRepository Details:")
                            print("-" * 80)
                            print(f"Name: {repo_data.get('name', 'N/A')}")
                            print(f"Full Name: {repo_data.get('full_name', 'N/A')}")
                            print(f"URL: {repo_data.get('html_url', 'N/A')}")
                            print(f"Description: {repo_data.get('description', 'N/A')}")
                            print(f"Private: {repo_data.get('private', 'N/A')}")
                            print(f"Clone URL: {repo_data.get('clone_url', 'N/A')}")
                            print(f"Default Branch: {repo_data.get('default_branch', 'N/A')}")
                            print(f"Created at: {repo_data.get('created_at', 'N/A')}")
                            print("-" * 80)
                            
                            print(f"\nYou can view the repository at: {repo_data.get('html_url', 'URL not available')}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing repository data: {e}")
                    else:
                        print("No content returned in the response")
                        
                except Exception as e:
                    print(f"Error creating repository: {e}")
                    traceback.print_exc()
                
    except Exception as e:
        print(f"Error connecting to the Smithery server: {e}")
        print("Detailed error information:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 
```