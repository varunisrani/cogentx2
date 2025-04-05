#!/usr/bin/env python3
"""
Simple test script to verify that the merge_agent_templates function now works with the fix.
"""

import asyncio
import logging
from pydantic_ai.models.openai import OpenAIModel
from archon.pydantic_ai_coder import merge_agent_templates
from dataclasses import dataclass
from openai import AsyncOpenAI
from supabase import Client, create_client
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class TestContext:
    model: OpenAIModel
    deps: 'TestDeps'

@dataclass
class TestDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    reasoner_output: str = ""
    advisor_output: str = ""

async def main():
    load_dotenv()
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        logging.error("No API key found in environment. Set OPENAI_API_KEY or LLM_API_KEY.")
        return
    
    # Initialize model
    model = OpenAIModel("gpt-4o-mini")
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logging.error("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY.")
        return
    
    supabase_client = create_client(supabase_url, supabase_key)
    
    # Initialize embedding client
    embedding_client = AsyncOpenAI(api_key=api_key)
    
    # Create test context
    deps = TestDeps(supabase=supabase_client, embedding_client=embedding_client)
    ctx = TestContext(model=model, deps=deps)
    
    # Test with GitHub and Spotify templates (IDs 2 and 4)
    logging.info("Starting test merge of GitHub and Spotify templates")
    
    try:
        result = await merge_agent_templates(
            ctx=ctx,
            template_ids=[2, 4],  # GitHub and Spotify
            custom_name="github_spotify_test",
            custom_description="A test agent combining GitHub and Spotify functionality"
        )
        
        if result.get("success"):
            logging.info("Merge successful!")
            logging.info(f"Generated files: {result.get('generated_files')}")
            logging.info(f"Output directory: {result.get('output_directory')}")
        else:
            logging.error(f"Merge failed: {result.get('error')}")
            
    except Exception as e:
        logging.error(f"Error during merge test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 