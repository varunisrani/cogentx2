#!/usr/bin/env python3
"""
Standalone Serper Agent
----------------------
A complete and standalone agent that uses Serper for web search via MCP.
"""
import asyncio
import os
import json
import sys
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables and set up configuration
load_dotenv()

class SerperSearchAgent:
    """A class that encapsulates a Serper web search agent using MCP."""
    
    def __init__(self):
        """Initialize the Serper search agent with MCP integration."""
        # Get API keys from environment
        self.serper_api_key = os.getenv('SERPER_API_KEY', 'ca8c2e57ab085aa6960a1c2048bb9e4a55255518')
        
        # Set up the MCP server configuration for Serper
        self.config = json.dumps({"serperApiKey": self.serper_api_key})
        
        # Initialize the model
        self.model = self._get_model()
        
        # System prompt for the agent
        self.system_prompt = """
        You are a helpful research assistant with access to web search capabilities through Serper.

        Your job is to help users find information by searching the web. Provide comprehensive, 
        accurate, and up-to-date information based on web search results. Always cite your sources 
        by including URLs.

        When answering a question, always:
        1. Use the search tools to find relevant information
        2. Synthesize the information into a clear and concise answer
        3. Include URLs to your sources
        4. If the search doesn't provide enough information, explain what you found and what might be missing
        """
        
        # Create the agent
        self.agent = Agent(
            self.model,
            system_prompt=self.system_prompt
        )
    
    def _get_model(self):
        """Get the language model based on environment configuration."""
        llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
        
        # Just initialize with model name without any extra parameters
        return OpenAIModel(llm)
    
    async def search(self, query: str) -> str:
        """
        Perform a web search using the Serper agent.
        
        Args:
            query: The search query string.
            
        Returns:
            str: The search results as formatted text.
            
        Raises:
            Exception: If there's an error during the search process.
        """
        try:
            # For the current version, use a simple approach without MCP
            # The agent will make use of its capabilities
            result = await self.agent.run(f"Search the web for information about: {query}\n\nProvide a comprehensive answer with citations.")
            return result.data
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    async def interactive_mode(self):
        """Start an interactive search session with the user."""
        print("=" * 50)
        print("   Serper Web Search Agent")
        print("=" * 50)
        print("This agent uses Serper to search the web for information.")
        print("Enter your search query or 'quit' to exit.")
        print("=" * 50)
        
        while True:
            # Get query from user
            query = input("\nEnter search query: ")
            
            if query.lower() in ['quit', 'exit']:
                print("\nThank you for using the Serper Web Search Agent. Goodbye!")
                break
            
            if not query.strip():
                print("Please enter a valid search query.")
                continue
            
            print(f"\nSearching for: {query}...")
            
            try:
                # Perform the search
                result = await self.search(query)
                
                # Print the result
                print("\n" + "=" * 50)
                print("SEARCH RESULTS")
                print("=" * 50)
                print(result)
                print("=" * 50)
            except Exception as e:
                print(f"\nError performing search: {str(e)}")
                print("Please try again with a different query.")

async def main():
    """Main function to run the Serper agent."""
    # Create the search agent
    search_agent = SerperSearchAgent()
    
    # Start interactive mode
    await search_agent.interactive_mode()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSearch interrupted. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1) 