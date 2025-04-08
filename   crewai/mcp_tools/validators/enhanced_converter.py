#!/usr/bin/env python
"""
Enhanced CrewAI Tool Converter

This module provides an enhanced version of the convert_to_crewai_tool function
with built-in error prevention specifically for YouTube transcript tools.
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("enhanced_converter")

# Error prevention template for YouTube transcript tools
YOUTUBE_TRANSCRIPT_ERROR_PREVENTION = """
# Error prevention for YouTube transcript tools
# The following alias class ensures backward compatibility with code expecting YouTubeTranscriptMCPTool

class YouTubeTranscriptMCPTool(YouTubeTranscriptTool):
    \"\"\"Alias for YouTubeTranscriptTool to maintain backward compatibility.\"\"\"
    pass
"""

class EnhancedToolConverter:
    """Enhanced convert_to_crewai_tool with built-in error prevention."""
    
    @staticmethod
    async def convert_to_crewai_tool(
        ctx: Any,
        original_code: str,
        purpose: str,
        connection_code: str = "",
        crewai_requirements: Optional[Dict[str, Any]] = None,
        reference_keys: Optional[List[str]] = None,
        original_converter_func: callable = None
    ) -> str:
        """
        Enhanced wrapper for convert_to_crewai_tool that includes error prevention
        
        Args:
            ctx: Run context with dependencies
            original_code: Original tool code
            purpose: Tool purpose description
            connection_code: Connection code if any
            crewai_requirements: CrewAI-specific requirements
            reference_keys: Reference keys for examples
            original_converter_func: The original function to call for conversion
            
        Returns:
            Converted CrewAI tool code with error prevention
        """
        try:
            # If no original converter function provided, return a basic implementation
            if not original_converter_func:
                logger.warning("No original converter function provided, using basic implementation")
                converted_code = original_code
            else:
                # Call the original converter
                converted_code = await original_converter_func(
                    ctx, 
                    original_code, 
                    purpose,
                    connection_code,
                    crewai_requirements,
                    reference_keys
                )
            
            # Check if this is a YouTube transcript tool
            is_youtube_transcript_tool = False
            purpose_lower = purpose.lower()
            if "youtube" in purpose_lower and "transcript" in purpose_lower:
                is_youtube_transcript_tool = True
                logger.info("Detected YouTube transcript tool, adding error prevention")
                
            # Apply specific error prevention for YouTube transcript tools
            if is_youtube_transcript_tool:
                # Check if the code has YouTubeTranscriptTool but not YouTubeTranscriptMCPTool
                if "class YouTubeTranscriptTool" in converted_code and "class YouTubeTranscriptMCPTool" not in converted_code:
                    # Find the end of the class
                    class_start = converted_code.find("class YouTubeTranscriptTool")
                    if class_start > -1:
                        # Find the next class or EOF
                        rest_of_content = converted_code[class_start:]
                        next_class_match = re.search(r'class\s+([a-zA-Z0-9_]+)', rest_of_content[30:])
                        
                        if next_class_match:
                            # Insert before the next class
                            insertion_point = class_start + rest_of_content.find("class " + next_class_match.group(1))
                            converted_code = converted_code[:insertion_point] + YOUTUBE_TRANSCRIPT_ERROR_PREVENTION + converted_code[insertion_point:]
                        else:
                            # Add at the end
                            converted_code += YOUTUBE_TRANSCRIPT_ERROR_PREVENTION
                            
                        logger.info("Added YouTubeTranscriptMCPTool alias class for error prevention")
            
            # Add error prevention comment at the top
            header = """# Enhanced CrewAI Tool with error prevention
# This tool includes automatic error prevention for common issues

"""
            # Add the header if it doesn't already have one
            if not converted_code.startswith("# Enhanced"):
                converted_code = header + converted_code
                
            return converted_code
            
        except Exception as e:
            logger.error(f"Error in enhanced conversion: {str(e)}")
            
            # Fallback to original code if our enhancement fails
            return original_code

def create_converter_patcher():
    """
    Creates a patcher function to apply our enhanced converter
    
    Returns:
        Patcher function that monkey-patches the original converter
    """
    try:
        # Import the original function to patch
        from archon.mcp_tools.mcp_tool_coder import convert_to_crewai_tool as original_convert_to_crewai_tool
        
        # Create a patched version
        async def patched_convert_to_crewai_tool(ctx, original_code, purpose, connection_code="", crewai_requirements=None, reference_keys=None):
            logger.info("Using enhanced tool converter with error prevention")
            return await EnhancedToolConverter.convert_to_crewai_tool(
                ctx, 
                original_code, 
                purpose,
                connection_code,
                crewai_requirements,
                reference_keys,
                original_convert_to_crewai_tool
            )
        
        # Function to apply the patch
        def apply_patch():
            import archon.mcp_tools.mcp_tool_coder
            logger.info("Applying patch to mcp_tool_coder.convert_to_crewai_tool")
            archon.mcp_tools.mcp_tool_coder.convert_to_crewai_tool = patched_convert_to_crewai_tool
            logger.info("Successfully patched tool converter")
            
        return apply_patch
    except ImportError:
        logger.error("Could not import original convert_to_crewai_tool function")
        return lambda: None

if __name__ == "__main__":
    print("Enhanced CrewAI Tool Converter")
    print("This module is designed to be imported and used to patch the convert_to_crewai_tool function")
    print("It automatically adds error prevention for common issues in YouTube transcript tools") 