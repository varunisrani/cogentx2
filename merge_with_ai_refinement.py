"""
Example script showing how to use the AI merge helper with merge_agent_templates.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from archon.pydantic_ai_coder import merge_agent_templates, PydanticAIDeps
from archon.ai_merge_helper import ai_refine_merged_code

# Load environment variables
load_dotenv()

async def merge_templates_with_ai_refinement(query="", template_ids=None, custom_name="", custom_description=""):
    """
    Merge templates and then refine the merged code using AI.
    
    Args:
        query: Query to search for relevant templates (if template_ids not provided)
        template_ids: List of template IDs to merge
        custom_name: Custom name for the merged template
        custom_description: Custom description for the merged template
    
    Returns:
        Result of the merge and refinement
    """
    # Create OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create OpenAI model
    model = OpenAIModel(
        client=openai_client,
        model="gpt-4-turbo",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Create dependencies
    deps = PydanticAIDeps(
        supabase=None,  # We're not using Supabase in this example
        embedding_client=None,  # We're not using embeddings in this example
        model=model
    )
    
    # Create context
    ctx = RunContext(deps=deps)
    
    # Step 1: Perform the rule-based merge
    print(f"Step 1: Performing rule-based merge of templates...")
    merge_result = await merge_agent_templates(
        ctx=ctx,
        query=query,
        template_ids=template_ids,
        custom_name=custom_name,
        custom_description=custom_description
    )
    
    if not merge_result.get("success", False):
        print(f"Error during merge: {merge_result.get('error', 'Unknown error')}")
        return merge_result
    
    # Step 2: Perform AI refinement of the merged code
    print(f"Step 2: Performing AI refinement of the merged code...")
    workbench_dir = merge_result.get("output_directory")
    output_paths = {filename: os.path.join(workbench_dir, filename) for filename in merge_result.get("generated_files", [])}
    template_names = [f"Template {tid}" for tid in template_ids] if template_ids else ["Unknown template"]
    
    # Refine the merged code using AI
    refined_output_paths = await ai_refine_merged_code(model, workbench_dir, output_paths, template_names)
    
    # Update the merge result with the refined output paths
    merge_result["generated_files"] = list(refined_output_paths.keys())
    merge_result["ai_refined"] = True
    merge_result["message"] += " with AI refinement."
    
    return merge_result

async def main():
    """Run the example."""
    # Example: Merge templates with IDs 1 and 2
    result = await merge_templates_with_ai_refinement(
        template_ids=[1, 2],
        custom_name="AI-Refined Merged Agent",
        custom_description="An agent with capabilities merged and refined by AI"
    )
    
    print(f"\nMerge result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
