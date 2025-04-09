#!/usr/bin/env python
"""
YouTube Tool Validator

This module provides validation functions to check for and prevent common errors
in the tools.py and related files before they occur.
"""

import re
import os
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("validate_tools")

def validate_class_naming(tools_py_content: str) -> Dict[str, Any]:
    """
    Validate and detect inconsistent class naming in tools.py
    
    Args:
        tools_py_content: Content of the tools.py file
        
    Returns:
        Dictionary with validation results and recommendations
    """
    import re
    
    # Extract all class definitions
    class_pattern = r'class\s+([a-zA-Z0-9_]+)(?:\s*\([^)]*\))?:'
    class_matches = re.findall(class_pattern, tools_py_content)
    
    # Look for classes that end with "Tool"
    tool_classes = [c for c in class_matches if c.endswith("Tool")]
    non_tool_classes = [c for c in class_matches if not c.endswith("Tool")]
    
    # Look for MCPTool naming pattern
    mcp_tool_classes = [c for c in class_matches if "MCPTool" in c]
    plain_tool_classes = [c for c in class_matches if "Tool" in c and "MCPTool" not in c]
    
    # Identify potential naming inconsistencies
    inconsistencies = []
    corrections = []
    
    # Check for YouTube-specific naming issues (the most common error)
    youtube_variants = [c for c in class_matches if "YouTube" in c or "Youtube" in c]
    
    # Specifically check for YouTubeTranscriptTool vs YouTubeTranscriptMCPTool issue
    has_transcript_tool = any("YoutubeTranscriptTool" in c or "YouTubeTranscriptTool" in c for c in class_matches)
    has_transcript_mcp_tool = any("YoutubeTranscriptMCPTool" in c or "YouTubeTranscriptMCPTool" in c for c in class_matches)
    
    if has_transcript_tool and not has_transcript_mcp_tool:
        inconsistencies.append("YouTubeTranscriptMCPTool referenced but not defined")
        corrections.append("Add YouTubeTranscriptMCPTool as alias for YouTubeTranscriptTool")
    
    # Check for MCP vs non-MCP naming inconsistencies
    if plain_tool_classes and mcp_tool_classes:
        # If we have both naming patterns, flag potential issues
        logger.info(f"Multiple tool naming patterns found: {plain_tool_classes} and {mcp_tool_classes}")
        inconsistencies.append("Mixed tool naming patterns (both 'Tool' and 'MCPTool' suffixes found)")
        
    return {
        "all_classes": class_matches,
        "tool_classes": tool_classes,
        "mcp_tool_classes": mcp_tool_classes,
        "plain_tool_classes": plain_tool_classes,
        "inconsistencies": inconsistencies,
        "corrections": corrections,
        "has_naming_inconsistencies": len(inconsistencies) > 0
    }

def fix_class_naming_inconsistencies(tools_py_content: str) -> str:
    """
    Fix common class naming inconsistencies in tools.py
    
    Args:
        tools_py_content: Content of the tools.py file
        
    Returns:
        Updated tools.py content with fixed class names
    """
    # Validate first
    validation = validate_class_naming(tools_py_content)
    
    if not validation["has_naming_inconsistencies"]:
        return tools_py_content
    
    updated_content = tools_py_content
    
    # Add YouTubeTranscriptMCPTool alias if needed
    if "YouTubeTranscriptMCPTool referenced but not defined" in validation["inconsistencies"]:
        logger.info("Adding YouTubeTranscriptMCPTool alias to fix naming inconsistency")
        
        # Check if YouTubeTranscriptTool exists
        if any("YouTubeTranscriptTool" in c for c in validation["all_classes"]):
            # Find the end of the YouTubeTranscriptTool class
            class_start = updated_content.find("class YouTubeTranscriptTool")
            if class_start > -1:
                # Find the next class definition after YouTubeTranscriptTool
                rest_of_content = updated_content[class_start:]
                next_class_match = re.search(r'class\s+([a-zA-Z0-9_]+)', rest_of_content[30:])
                
                if next_class_match:
                    # Insert the alias class before the next class
                    insertion_point = class_start + rest_of_content.find("class " + next_class_match.group(1))
                    
                    alias_class = r"""

class YouTubeTranscriptMCPTool(YouTubeTranscriptTool):
    '''Alias for YouTubeTranscriptTool to maintain backward compatibility.'''
    pass

"""
                    # Insert the alias class
                    updated_content = updated_content[:insertion_point] + alias_class + updated_content[insertion_point:]
                else:
                    # Add at the end if no next class found
                    alias_class = r"""

class YouTubeTranscriptMCPTool(YouTubeTranscriptTool):
    '''Alias for YouTubeTranscriptTool to maintain backward compatibility.'''
    pass
"""
                    updated_content += alias_class
                    
                logger.info("Added YouTubeTranscriptMCPTool alias class")
    
    return updated_content

def extract_imports_from_file(file_content: str) -> List[str]:
    """
    Extract import statements from a file
    
    Args:
        file_content: Content of the file
        
    Returns:
        List of import lines
    """
    import_lines = []
    for line in file_content.split('\n'):
        if line.strip().startswith(('import ', 'from ')):
            import_lines.append(line.strip())
    return import_lines

def validate_cross_file_consistency(files_content: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate consistency across files (tools.py, agents.py, tasks.py, crew.py)
    
    Args:
        files_content: Dictionary mapping file names to their content
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "inconsistencies": [],
        "has_inconsistencies": False,
        "corrections": []
    }
    
    # Extract tool class names from tools.py
    if "tools.py" not in files_content:
        results["inconsistencies"].append("tools.py missing")
        results["has_inconsistencies"] = True
        return results
    
    tools_py = files_content["tools.py"]
    
    # Extract class names from tools.py
    class_pattern = r'class\s+([a-zA-Z0-9_]+)(?:\s*\([^)]*\))?:'
    tool_classes = re.findall(class_pattern, tools_py)
    
    # Check for imports in other files that don't match tools.py classes
    for filename, content in files_content.items():
        if filename == "tools.py":
            continue
            
        # Look for imports from tools
        import_pattern = r'from\s+tools\s+import\s+([^#\n]+)'
        imports_matches = re.findall(import_pattern, content)
        
        for imports in imports_matches:
            # Split imports by comma
            for imported_item in [i.strip() for i in imports.split(',')]:
                if imported_item == '*':
                    continue  # Wildcard import
                    
                if imported_item not in tool_classes and imported_item != "":
                    results["inconsistencies"].append(
                        f"In {filename}: Imports '{imported_item}' from tools.py but it's not defined there"
                    )
                    
                    # Check for similar names (e.g., YouTubeTranscriptTool vs YouTubeTranscriptMCPTool)
                    close_matches = []
                    for cls in tool_classes:
                        # Common patterns to check
                        if imported_item.replace("MCPTool", "Tool") == cls:
                            close_matches.append(cls)
                        elif imported_item + "MCPTool" == cls:
                            close_matches.append(cls)
                        elif imported_item == cls + "MCPTool":
                            close_matches.append(cls)
                            
                    if close_matches:
                        results["corrections"].append(
                            f"In {filename}: Consider changing import '{imported_item}' to '{close_matches[0]}'"
                        )
                        
                    results["has_inconsistencies"] = True
    
    return results

def check_and_fix_tools_py(
    tools_py_content: str,
    related_files: Optional[Dict[str, str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Comprehensive validation and fixing of tools.py content
    
    Args:
        tools_py_content: Content of tools.py
        related_files: Optional dictionary of related files for cross-validation
        
    Returns:
        Tuple of (updated_content, validation_results)
    """
    validation_results = {
        "class_naming": validate_class_naming(tools_py_content),
        "cross_file": {"inconsistencies": [], "has_inconsistencies": False} 
    }
    
    # If we have related files, validate cross-file consistency
    if related_files:
        all_files = {"tools.py": tools_py_content}
        all_files.update(related_files)
        validation_results["cross_file"] = validate_cross_file_consistency(all_files)
    
    # Fix the content if needed
    updated_content = tools_py_content
    if validation_results["class_naming"]["has_naming_inconsistencies"]:
        updated_content = fix_class_naming_inconsistencies(tools_py_content)
        
    return updated_content, validation_results

def enhance_tools_generation_prompt(prompt: str) -> str:
    """
    Enhance a tools.py generation prompt with error prevention guidelines
    
    Args:
        prompt: Original generation prompt
        
    Returns:
        Enhanced prompt
    """
    error_guidelines = """
IMPORTANT ERROR PREVENTION GUIDELINES:

1. If implementing a YouTube transcript tool, ensure consistent class naming:
   - If using 'YouTubeTranscriptTool', add an alias class 'YouTubeTranscriptMCPTool'
   - Both class names should be usable interchangeably for backward compatibility

2. Maintain consistent naming patterns:
   - Class names should be clear and descriptive
   - Follow existing code conventions
   - If existing code refers to tool with 'MCPTool' suffix, maintain this in your implementation

3. Ensure parameter consistency:
   - Standardize parameter formats
   - Use consistent parameter names across all methods

4. Method naming consistency:
   - Ensure method names match exactly in tool definition and method calls
   - Cross-validate method names with any existing Factory classes
    """
    
    # Add error prevention guidelines to the prompt
    if "IMPORTANT" not in prompt and "ERROR PREVENTION" not in prompt:
        enhanced_prompt = prompt + "\n\n" + error_guidelines
        return enhanced_prompt
    
    return prompt

if __name__ == "__main__":
    # Simple test
    test_content = '''
import os
import logging
from langchain.tools import BaseTool

class YouTubeTranscriptTool(BaseTool):
    """A tool for getting YouTube transcripts."""
    
    def _run(self, url: str) -> str:
        # Implementation here
        pass
        
    def get_transcript(self, url: str) -> str:
        # Implementation
        pass
'''
    
    result = validate_class_naming(test_content)
    print(result)
    
    if result["has_naming_inconsistencies"]:
        fixed_content = fix_class_naming_inconsistencies(test_content)
        print("\nFixed content:")
        print(fixed_content) 