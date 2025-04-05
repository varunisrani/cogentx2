import os
import sys
import asyncio
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIModel

# Add the current directory to the path
sys.path.append('.')

# Import the merge function
from archon.pydantic_ai_coder import merge_agent_templates

async def main():
    # Create a model
    model = OpenAIModel("gpt-4o")
    
    # Create a context with the model
    ctx = RunContext(
        deps={},
        model=model,
        usage={},
        prompt=""
    )
    
    # Run the merge
    result = await merge_agent_templates(
        ctx=ctx,
        query="spotify_agent, github_agent",
        use_ai_merge=True
    )
    
    print(f"Merge result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
