import sys
import os
import asyncio
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.utils import get_clients, write_to_log
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps

async def test_search_templates():
    print("Starting environment test...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    print("Initializing clients...")
    embedding_client, supabase = get_clients()
    
    # Check client status
    print(f"Embedding client initialized: {embedding_client is not None}")
    print(f"Supabase client initialized: {supabase is not None}")
    print(f"Embedding API key valid: {embedding_client.api_key != 'no-api-key-provided'}")
    
    # Initialize dependencies
    print("Creating PydanticAIDeps...")
    deps = PydanticAIDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        reasoner_output="",
        advisor_output=""
    )
    
    # Test search_agent_templates function
    try:
        print("Testing search_agent_templates...")
        result = await pydantic_ai_coder.run(
            "Search for agent templates related to spotify", 
            deps=deps
        )
        print("Search completed successfully!")
        print(f"Result data: {result.data[:500]}...")
    except Exception as e:
        print(f"Error during search: {str(e)}")
    
    print("\nEnvironment test completed.")

if __name__ == "__main__":
    asyncio.run(test_search_templates()) 