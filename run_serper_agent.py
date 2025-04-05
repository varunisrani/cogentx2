#!/usr/bin/env python3
"""
Serper Agent Runner
------------------
A simple script to run the Serper web search agent.
Takes user queries and returns search results.
"""
import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

try:
    # Try to import directly
    from pydantic_serper_agent import serper_agent, search_web
except ImportError:
    print("Could not import Serper agent directly. Make sure pydantic_serper_agent.py is in the same directory.")
    sys.exit(1)

async def main():
    """Main function to run the Serper agent."""
    # Load environment variables
    load_dotenv()
    
    # Check if Serper API key is available
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        print("Warning: No Serper API key found in .env file. Using default key from the script.")
    
    print("=" * 50)
    print("   Serper Web Search Agent")
    print("=" * 50)
    print("This agent uses Serper to search the web for information.")
    print("Enter your search query or 'quit' to exit.")
    print("=" * 50)
    
    while True:
        # Get user input
        query = input("\nEnter search query: ")
        
        if query.lower() in ['quit', 'exit']:
            print("\nThank you for using the Serper Web Search Agent. Goodbye!")
            break
        
        if not query.strip():
            print("Please enter a valid search query.")
            continue
        
        print(f"\nSearching for: {query}...")
        
        try:
            # Run the search
            result = await search_web(query)
            
            # Print the results with formatting
            print("\n" + "=" * 50)
            print("SEARCH RESULTS")
            print("=" * 50)
            print(result)
            print("=" * 50)
        except Exception as e:
            print(f"\nError performing search: {str(e)}")
            print("Please try again with a different query.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSearch interrupted. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1) 