import os
import re
import uuid
import time
import logging
import json
import requests
import streamlit as st

# Set up logging
logger = logging.getLogger(__name__)

# Default MCP Cogentx server URL
MCP_COGENTX_URL = os.environ.get("MCP_COGENTX_URL", "http://localhost:8008")

def mcp_cogentx_create_thread(random_string=None):
    """
    Create a new conversation thread for Cogentx.
    Always call this function before invoking Cogentx for the first time in a conversation.
    
    Returns:
        str: A unique thread ID for the conversation
    """
    try:
        # Generate a unique thread ID
        thread_id = str(uuid.uuid4())
        
        # Log the thread creation
        logger.info(f"Created new Cogentx thread with ID: {thread_id}")
        
        return thread_id
    except Exception as e:
        logger.error(f"Error creating Cogentx thread: {str(e)}")
        return None

def mcp_cogentx_run_agent(thread_id, user_input):
    """
    Run the Cogentx agent with user input.
    Only use this function after you have called create_thread to get a unique thread ID.
    
    Args:
        thread_id: The conversation thread ID
        user_input: The user's message to process
    
    Returns:
        str: The agent's response which generally includes the code
    """
    try:
        # For now, we'll implement a mock version that uses the Claude integration
        # This will be replaced with actual Cogentx API calls in the future
        
        # Log the request
        logger.info(f"Running Cogentx agent with thread ID: {thread_id}")
        
        # Check if Claude is available
        try:
            import anthropic
            claude_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "NOT_PROVIDED")
            )
            
            # If we have an API key, use Claude
            if os.environ.get("ANTHROPIC_API_KEY"):
                response = claude_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    temperature=0.2,
                    system="You are an expert Python programmer assisting with code errors. Your job is to analyze error messages and fix code issues. Provide ONLY the fixed code with no explanations, no markdown, just the raw Python code.",
                    messages=[
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                )
                
                # Extract response
                if response and response.content:
                    for content_block in response.content:
                        if content_block.type == "text":
                            # Process and clean the response
                            code_response = content_block.text
                            
                            # Strip any potential markdown code blocks
                            code_response = re.sub(r'^```python\s*', '', code_response, flags=re.MULTILINE)
                            code_response = re.sub(r'^```\s*', '', code_response, flags=re.MULTILINE)
                            code_response = re.sub(r'```$', '', code_response, flags=re.MULTILINE)
                            
                            return code_response.strip()
                
                # If we reach here, something went wrong with the response
                return "# Failed to parse Claude response"
            
        except (ImportError, Exception) as e:
            logger.warning(f"Could not use Claude: {str(e)}")
        
        # If Claude is not available, try a fallback to OpenAI
        try:
            from openai import OpenAI
            openai_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "NOT_PROVIDED")
            )
            
            # If we have an API key, use OpenAI
            if os.environ.get("OPENAI_API_KEY"):
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert Python programmer assisting with code errors. Your job is to analyze error messages and fix code issues. Provide ONLY the fixed code with no explanations, no markdown, just the raw Python code."
                        },
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ],
                    max_tokens=4000
                )
                
                if response and response.choices:
                    # Extract and clean the code response
                    code_response = response.choices[0].message.content
                    
                    # Strip any potential markdown code blocks
                    code_response = re.sub(r'^```python\s*', '', code_response, flags=re.MULTILINE)
                    code_response = re.sub(r'^```\s*', '', code_response, flags=re.MULTILINE)
                    code_response = re.sub(r'```$', '', code_response, flags=re.MULTILINE)
                    
                    return code_response.strip()
                    
                # If we reach here, something went wrong with the response
                return "# Failed to parse OpenAI response"
                
        except (ImportError, Exception) as e:
            logger.warning(f"Could not use OpenAI: {str(e)}")
        
        # If we reach here, neither Claude nor OpenAI worked
        # Implement basic error fixes for common patterns
        
        # Extract the error type from the input
        error_match = re.search(r"([\w\.]+Error|Exception): (.*?)(\n|$)", user_input)
        if error_match:
            error_type = error_match.group(1)
            error_message = error_match.group(2).strip()
            
            # Extract code from the user input
            code_match = re.search(r"```python\s*(.*?)\s*```", user_input, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                
                # Apply simple fixes based on error type
                if "ModuleNotFoundError" in error_type:
                    module_match = re.search(r"No module named '([\w\._]+)'", error_message)
                    if module_match:
                        module_name = module_match.group(1)
                        # Add import at the top
                        return f"import {module_name}\n\n{code}"
                        
                elif "ImportError" in error_type:
                    import_match = re.search(r"cannot import name '([\w\._]+)' from '([\w\._]+)'", error_message)
                    if import_match:
                        name = import_match.group(1)
                        module = import_match.group(2)
                        # Comment out the problematic import
                        pattern = re.compile(rf"from\s+{module}\s+import.*?{name}")
                        return pattern.sub(f"# Commented out problematic import: from {module} import {name}", code)
                
                elif "SyntaxError" in error_type:
                    # Check for missing closing parentheses, brackets, or braces
                    if code.count('(') > code.count(')'):
                        return code + ')'
                    elif code.count('[') > code.count(']'):
                        return code + ']'
                    elif code.count('{') > code.count('}'):
                        return code + '}'
                
                # Return the original code if no specific fix was applied
                return code
        
        # If nothing else worked, return a message
        return "# Unable to fix the error automatically. Please check the error message and fix manually."
            
    except Exception as e:
        logger.error(f"Error running Cogentx agent: {str(e)}")
        return f"# Error: {str(e)}"

def mcp_cogentx_generate_mcp_tools(user_request, output_dir="output"):
    """
    Generate MCP tools based on user requirements.
    This tool identifies and creates appropriate MCP tools for the user's request.
    
    Args:
        user_request: The user's requirements for the MCP tools
        output_dir: Directory to save the generated tools
        
    Returns:
        Dict containing information about the generated tools
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, this would call the Cogentx API
        
        # Log the request
        logger.info(f"Generating MCP tools with user request: {user_request}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Return a placeholder response
        return {
            "status": "success",
            "message": f"Generated MCP tools based on user request: {user_request}",
            "output_dir": output_dir,
            "tools_created": ["tool1.py", "tool2.py"]
        }
    except Exception as e:
        logger.error(f"Error generating MCP tools: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

def mcp_cogentx_analyze_tool_requirements(query):
    """
    Analyze and extract structured requirements for MCP tools.
    This tool helps understand what specific tools and functionalities are needed.
    
    Args:
        query: The user's request for tools analysis
        
    Returns:
        Dict containing structured requirements for tools
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, this would call the Cogentx API
        
        # Log the request
        logger.info(f"Analyzing tool requirements with query: {query}")
        
        # Return a placeholder response
        return {
            "status": "success",
            "requirements": {
                "tools": ["file_manipulation", "data_processing"],
                "complexity": "medium",
                "estimated_development_time": "2 hours"
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing tool requirements: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        } 