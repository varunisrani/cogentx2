#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any
from openai import AsyncOpenAI
from supabase import create_client
from dotenv import load_dotenv


class AgentTemplateSearch:
    """Utility to search for agent templates by similarity"""
    
    def __init__(self):
        """Initialize the OpenAI and Supabase clients"""
        load_dotenv()
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
            
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            print("Error: SUPABASE_URL and/or SUPABASE_SERVICE_KEY environment variables not set")
            sys.exit(1)
            
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            sys.exit(1)
    
    async def search_by_text(self, query_text: str, threshold: float = 0.5, limit: int = 5) -> List[Dict]:
        """Search for agent templates similar to the provided text query"""
        # Get embedding for the query text
        print(f"Generating embedding for query: '{query_text}'")
        query_embedding = await self.get_embedding(query_text)
        
        # Search using embedding similarity
        print(f"Searching with similarity threshold: {threshold}, limit: {limit}")
        return await self.search_by_embedding(query_embedding, threshold, limit)
    
    async def search_by_embedding(self, query_embedding: List[float], 
                                 threshold: float = 0.5, limit: int = 5) -> List[Dict]:
        """Search for agent templates using embedding similarity"""
        try:
            # Search using the search_agent_embeddings function
            response = self.supabase_client.rpc(
                'search_agent_embeddings',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': threshold,
                    'match_count': limit
                }
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error searching templates: {response.error}")
                return []
            
            return response.data
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    async def fetch_template_by_id(self, template_id: int) -> Dict:
        """Fetch a complete template by ID"""
        try:
            response = self.supabase_client.table("agent_embeddings") \
                .select("*") \
                .eq("id", template_id) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                print(f"Error fetching template: {response.error}")
                return {}
                
            if not response.data:
                print(f"No template found with ID: {template_id}")
                return {}
                
            return response.data[0]
                
        except Exception as e:
            print(f"Error fetching template: {e}")
            return {}


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search for agent templates by similarity")
    parser.add_argument("query", help="Text query to search for similar templates")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Similarity threshold (0.0 to 1.0, default: 0.5)")
    parser.add_argument("--limit", type=int, default=5, 
                        help="Maximum number of results to return (default: 5)")
    parser.add_argument("--fetch", type=int, help="Fetch complete template by ID")
    parser.add_argument("--output", help="Output file for full template data (JSON format)")
    args = parser.parse_args()
    
    # Initialize search utility
    search_utility = AgentTemplateSearch()
    
    # Fetch template by ID if specified
    if args.fetch:
        print(f"Fetching template with ID: {args.fetch}")
        template = await search_utility.fetch_template_by_id(args.fetch)
        
        if template:
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    json.dump(template, f, indent=2)
                print(f"Template data saved to: {args.output}")
            else:
                # Print summary
                print("\nTemplate Summary:")
                print(f"ID: {template['id']}")
                print(f"Folder: {template['folder_name']}")
                print(f"Purpose: {template['purpose']}")
                print(f"Created: {template['created_at']}")
                print("\nFile sizes:")
                print(f"  agents.py: {len(template['agents_code']) if template['agents_code'] else 0} bytes")
                print(f"  main.py: {len(template['main_code']) if template['main_code'] else 0} bytes")
                print(f"  models.py: {len(template['models_code']) if template['models_code'] else 0} bytes")
                print(f"  tools.py: {len(template['tools_code']) if template['tools_code'] else 0} bytes")
                print(f"  mcp.json: {len(template['mcp_json']) if template['mcp_json'] else 0} bytes")
                
                print("\nTo save the complete template data, use the --output option.")
        
        return
    
    # Search by text query
    results = await search_utility.search_by_text(args.query, args.threshold, args.limit)
    
    # Display results
    if results:
        print(f"\nFound {len(results)} similar templates:")
        for idx, result in enumerate(results):
            print(f"\n{idx+1}. {result['folder_name']} (ID: {result['id']})")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Purpose: {result['purpose']}")
            
            # Extract and show some metadata if available
            if result['metadata']:
                if 'agents' in result['metadata'] and result['metadata']['agents']:
                    print(f"   Agents: {', '.join(result['metadata']['agents'])}")
                if 'capabilities' in result['metadata'] and result['metadata']['capabilities']:
                    print(f"   Capabilities: {', '.join(result['metadata']['capabilities'][:5])}" + 
                          ("..." if len(result['metadata']['capabilities']) > 5 else ""))
            
            print(f"   To fetch full template: python search_templates.py --fetch {result['id']}")
    else:
        print("No similar templates found.")


if __name__ == "__main__":
    asyncio.run(main()) 