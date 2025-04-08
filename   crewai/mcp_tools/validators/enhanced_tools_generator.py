#!/usr/bin/env python
"""
Enhanced Tools Generator

This module provides an enhanced version of the tools.py generator with built-in
error prevention and validation checks.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import re
import sys

# Add path to import validate_tools module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the validator module
try:
    from validate_tools import (
        check_and_fix_tools_py,
        enhance_tools_generation_prompt,
        validate_cross_file_consistency
    )
except ImportError:
    # If validate_tools.py doesn't exist yet, define minimal versions of required functions
    def check_and_fix_tools_py(tools_py_content, related_files=None):
        return tools_py_content, {"class_naming": {"has_naming_inconsistencies": False}}
        
    def enhance_tools_generation_prompt(prompt):
        return prompt
        
    def validate_cross_file_consistency(files_content):
        return {"inconsistencies": [], "has_inconsistencies": False}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("enhanced_tools_generator")

class EnhancedToolsGenerator:
    """Enhanced tools.py generator with error prevention."""
    
    @staticmethod
    async def generate_tools_py(
        ctx: Any,
        tools_data: Dict[str, Any],
        output_dir: str,
        original_generator_func: callable
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced wrapper for generate_tools_py that includes validation
        
        Args:
            ctx: Run context with dependencies
            tools_data: Information about the tools being used
            output_dir: Directory to save the file
            original_generator_func: The original function to call for generation
            
        Returns:
            Tuple of (tools_py_content, validation_results)
        """
        logger.info("Starting enhanced tools.py generation with validation")
        
        try:
            # Step 1: Check for YouTube transcript tools
            is_youtube_tool = False
            if "tools" in tools_data and isinstance(tools_data["tools"], list):
                for tool in tools_data["tools"]:
                    purpose = tool.get("purpose", "").lower()
                    if "youtube" in purpose and "transcript" in purpose:
                        is_youtube_tool = True
                        logger.info("Detected YouTube transcript tool, enabling specific validation")
                        break
            
            # Step 2: Enhance the prompt before calling the original generator
            if "prompt" in tools_data:
                tools_data["prompt"] = enhance_tools_generation_prompt(tools_data["prompt"])
                logger.info("Enhanced tools.py generation prompt with error prevention guidelines")
            
            # Step 3: Use the original generator to create tools.py content
            tools_py_content = await original_generator_func(ctx, tools_data, output_dir)
            
            # Step 4: Validate and fix the generated content
            logger.info("Validating generated tools.py content")
            
            # Check for other related files for cross-validation
            related_files = {}
            for filename in ["agents.py", "tasks.py", "crew.py"]:
                file_path = os.path.join(output_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            related_files[filename] = f.read()
                    except Exception as e:
                        logger.warning(f"Error reading {filename}: {str(e)}")
            
            # Validate and fix the content
            updated_content, validation_results = check_and_fix_tools_py(
                tools_py_content, 
                related_files
            )
            
            # Step 5: If validation found and fixed issues, write the updated content
            if updated_content != tools_py_content:
                logger.info("Fixed issues in tools.py, writing updated content")
                tools_file_path = os.path.join(output_dir, "tools.py")
                with open(tools_file_path, "w") as f:
                    f.write(updated_content)
                logger.info(f"Saved validated and fixed tools.py to {tools_file_path}")
                
                # Create a validation report
                report_path = os.path.join(output_dir, "tools_validation_report.md")
                with open(report_path, "w") as f:
                    f.write("# Tools Validation Report\n\n")
                    
                    # Class naming section
                    f.write("## Class Naming Validation\n\n")
                    if validation_results["class_naming"]["has_naming_inconsistencies"]:
                        f.write("### Issues Detected and Fixed:\n\n")
                        for issue in validation_results["class_naming"]["inconsistencies"]:
                            f.write(f"- {issue}\n")
                        f.write("\n### Applied Corrections:\n\n")
                        for correction in validation_results["class_naming"]["corrections"]:
                            f.write(f"- {correction}\n")
                    else:
                        f.write("No class naming inconsistencies detected.\n")
                    
                    # Cross-file section
                    f.write("\n## Cross-File Consistency\n\n")
                    if validation_results["cross_file"]["has_inconsistencies"]:
                        f.write("### Issues Detected:\n\n")
                        for issue in validation_results["cross_file"]["inconsistencies"]:
                            f.write(f"- {issue}\n")
                        if validation_results["cross_file"]["corrections"]:
                            f.write("\n### Recommended Corrections:\n\n")
                            for correction in validation_results["cross_file"]["corrections"]:
                                f.write(f"- {correction}\n")
                    else:
                        f.write("No cross-file inconsistencies detected.\n")
                
                logger.info(f"Generated validation report at {report_path}")
                return updated_content, validation_results
            
            # If no issues found, return the original content
            logger.info("No issues found in generated tools.py")
            return tools_py_content, validation_results
            
        except Exception as e:
            logger.error(f"Error in enhanced tools.py generation: {str(e)}", exc_info=True)
            
            # Fallback to original generator if our enhancement fails
            try:
                tools_py_content = await original_generator_func(ctx, tools_data, output_dir)
                return tools_py_content, {"error": str(e)}
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {str(fallback_error)}")
                return f"# Error generating tools.py: {str(e)}\n# Fallback also failed: {str(fallback_error)}", {"error": str(e)}

def create_tools_py_patcher():
    """
    Creates a patcher function to apply our enhanced tools generator
    
    Returns:
        Patcher function that monkey-patches the original generator
    """
    try:
        # Import the original function to patch
        from archon.mcp_tools.mcp_tool_coder import generate_tools_py as original_generate_tools_py
        
        # Create a patched version
        async def patched_generate_tools_py(ctx, tools_data, output_dir):
            logger.info("Using enhanced tools.py generator with error prevention")
            return await EnhancedToolsGenerator.generate_tools_py(
                ctx, tools_data, output_dir, original_generate_tools_py
            )
        
        # Function to apply the patch
        def apply_patch():
            import archon.mcp_tools.mcp_tool_coder
            logger.info("Applying patch to mcp_tool_coder.generate_tools_py")
            archon.mcp_tools.mcp_tool_coder.generate_tools_py = patched_generate_tools_py
            logger.info("Successfully patched tools.py generator")
            
        return apply_patch
    except ImportError:
        logger.error("Could not import original generate_tools_py function")
        return lambda: None

def add_youtube_transcript_mcp_tool_alias(tools_py_path: str) -> bool:
    """
    Adds YouTubeTranscriptMCPTool alias to an existing tools.py file if needed
    
    Args:
        tools_py_path: Path to the tools.py file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        if not os.path.exists(tools_py_path):
            logger.error(f"tools.py file not found at {tools_py_path}")
            return False
            
        with open(tools_py_path, 'r') as f:
            content = f.read()
            
        # Check if already has YouTubeTranscriptMCPTool
        if "class YouTubeTranscriptMCPTool" in content:
            logger.info("tools.py already has YouTubeTranscriptMCPTool class")
            return False
            
        # Check if has YouTubeTranscriptTool but not YouTubeTranscriptMCPTool
        if "class YouTubeTranscriptTool" in content and "class YouTubeTranscriptMCPTool" not in content:
            # Validate and fix
            updated_content, validation_results = check_and_fix_tools_py(content)
            
            if updated_content != content:
                logger.info("Adding YouTubeTranscriptMCPTool alias to tools.py")
                with open(tools_py_path, 'w') as f:
                    f.write(updated_content)
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error adding YouTubeTranscriptMCPTool alias: {str(e)}")
        return False

if __name__ == "__main__":
    # If executed directly, check if path is provided as argument
    if len(sys.argv) > 1:
        tools_py_path = sys.argv[1]
        
        # Check if the file exists
        if os.path.exists(tools_py_path):
            logger.info(f"Validating tools.py at {tools_py_path}")
            
            # Read the file
            with open(tools_py_path, 'r') as f:
                content = f.read()
                
            # Check if related files exist in the same directory
            dir_path = os.path.dirname(tools_py_path)
            related_files = {}
            for filename in ["agents.py", "tasks.py", "crew.py"]:
                file_path = os.path.join(dir_path, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            related_files[filename] = f.read()
                    except Exception as e:
                        logger.warning(f"Error reading {filename}: {str(e)}")
            
            # Validate and fix
            updated_content, validation_results = check_and_fix_tools_py(content, related_files)
            
            # If validation found issues, write the updated content
            if updated_content != content:
                logger.info("Fixed issues in tools.py")
                
                # Create backup
                backup_path = tools_py_path + ".bak"
                with open(backup_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created backup at {backup_path}")
                
                # Write updated content
                with open(tools_py_path, 'w') as f:
                    f.write(updated_content)
                logger.info(f"Saved fixed tools.py to {tools_py_path}")
                
                # Print the validation results
                print("Fixed the following issues:")
                if validation_results["class_naming"]["has_naming_inconsistencies"]:
                    for issue in validation_results["class_naming"]["inconsistencies"]:
                        print(f"- {issue}")
            else:
                logger.info("No issues found in tools.py")
                print("No issues found in tools.py")
        else:
            logger.error(f"File not found: {tools_py_path}")
            print(f"Error: File not found: {tools_py_path}")
    else:
        print("Usage: python enhanced_tools_generator.py path/to/tools.py") 