"""
Test script for the AI merge helper functionality.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from archon.ai_merge_helper import refine_code_with_ai, refine_merged_files, apply_refined_files

# Load environment variables
load_dotenv()

# Create a simple test model
class TestModel:
    async def generate(self, prompt):
        print(f"Generating response for prompt: {prompt[:100]}...")
        # In a real test, this would call the actual model
        return "# Refined code\ndef hello_world():\n    print('Hello, world!')"

async def test_refine_code_with_ai():
    """Test the refine_code_with_ai function."""
    # Create a test file
    test_file = "test_file.py"
    with open(test_file, "w") as f:
        f.write("# Original code\ndef hello():\n    print('Hello')")
    
    # Test the refine_code_with_ai function
    model = TestModel()
    template_names = ["template1", "template2"]
    
    try:
        refined_file = await refine_code_with_ai(model, test_file, template_names)
        print(f"Refined file: {refined_file}")
        
        # Check if the refined file exists
        if os.path.exists(refined_file):
            with open(refined_file, "r") as f:
                content = f.read()
                print(f"Refined content: {content}")
            
            # Clean up
            os.remove(refined_file)
        
        # Clean up the original file
        os.remove(test_file)
        
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

async def main():
    """Run the tests."""
    await test_refine_code_with_ai()

if __name__ == "__main__":
    asyncio.run(main())
