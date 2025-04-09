#!/usr/bin/env python3
"""
SerperMCPTool - A CrewAI tool for performing web searches using Serper API through MCP.
This file serves as a reference implementation for the MCP tool coder.
"""

import os
import json
import asyncio
import smithery
import mcp

from mcp.client.websocket import websocket_client
from typing import Optional, Dict, Any, List
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SerperMCPTool(BaseTool):
    """
    A CrewAI tool for performing web searches using Serper API through MCP.
    
    This tool connects to the Serper API via Smithery's Model Context Protocol (MCP)
    to perform web searches and return structured results.
    """
    
    name: str = "SerperMCPTool"
    description: str = "Search the web for current information using Serper API via MCP."
    serper_api_key: str = "ca8c2e57ab085aa6960a1c2048bb9e4a55255518"  # Default API key
    url: Optional[str] = None
    
    def __init__(self, **data):
        # Initialize with API key
        super().__init__(**data)
        # Use API key from environment if available
        self.serper_api_key = os.environ.get("SERPER_API_KEY", self.serper_api_key)
        # Create Smithery URL with Serper server endpoint
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@marcopesani/mcp-server-serper/ws", 
            {
                "serperApiKey": self.serper_api_key
            }
        )
    
    async def _arun(self, query: str, num_results: int = 5, 
                   country_code: str = "us", language: str = "en") -> str:
        """
        Asynchronously run the search query against Serper API via MCP.
        
        Args:
            query: The search query.
            num_results: Maximum number of results to return.
            country_code: Country code for localized results (e.g., "us", "uk").
            language: Language code for results (e.g., "en").
            
        Returns:
            A string with formatted search results.
        """
        try:
            # Connect to the server using websocket client
            async with websocket_client(self.url) as streams:
                async with mcp.ClientSession(*streams) as session:
                    # Perform the search query
                    search_result = await session.call_tool(
                        "google_search",
                        {
                            "q": query,
                            "gl": country_code,
                            "hl": language
                        }
                    )
                    
                    if hasattr(search_result, 'content') and search_result.content:
                        # Parse the JSON response
                        search_data = json.loads(search_result.content[0].text)
                        
                        # Format the results
                        results = []
                        
                        # Add organic results
                        if 'organic' in search_data:
                            for i, result in enumerate(search_data['organic'][:num_results], 1):
                                title = result.get('title', 'No title')
                                link = result.get('link', 'No link')
                                snippet = result.get('snippet', 'No snippet')
                                
                                results.append(f"Result {i}:\nTitle: {title}\nURL: {link}\nSnippet: {snippet}\n")
                        
                        # Add knowledge graph if available
                        if 'knowledgeGraph' in search_data:
                            kg = search_data['knowledgeGraph']
                            results.append(f"\nKnowledge Graph:\nTitle: {kg.get('title', 'N/A')}\nType: {kg.get('type', 'N/A')}\nDescription: {kg.get('description', 'N/A')}\n")
                        
                        # Add "People Also Ask" if available
                        if 'peopleAlsoAsk' in search_data:
                            results.append("\nPeople Also Ask:")
                            for i, question in enumerate(search_data['peopleAlsoAsk'][:3], 1):
                                results.append(f"Question {i}: {question.get('question', 'No question')}\nAnswer: {question.get('answer', 'No answer')}\n")
                        
                        return "\n".join(results)
                    else:
                        return "No search results found."
                        
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def _run(self, query: str, num_results: int = 5, 
            country_code: str = "us", language: str = "en") -> str:
        """
        Run the search query against Serper API via MCP (synchronous wrapper).
        
        This method is a synchronous wrapper around _arun to allow both
        sync and async usage of this tool.
        
        Args:
            query: The search query.
            num_results: Maximum number of results to return.
            country_code: Country code for localized results (e.g., "us", "uk").
            language: Language code for results (e.g., "en").
            
        Returns:
            A string with formatted search results.
        """
        return asyncio.run(self._arun(query, num_results, country_code, language))

# Example of how to use the tool
if __name__ == "__main__":
    tool = SerperMCPTool()
    print("Searching for 'Smithery MCP protocol'...")
    results = tool._run("what is Smithery MCP protocol")
    print(results) 