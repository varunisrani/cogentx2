#!/usr/bin/env python
"""
YouTube Tool Validation Installer

This module installs and applies the error prevention patches for YouTube tools.
"""

import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("tool_validator_installer")

def install_patches():
    """
    Install all error prevention patches
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the patchers
        from archon.mcp_tools.validators.enhanced_tools_generator import create_tools_py_patcher
        from archon.mcp_tools.validators.enhanced_converter import create_converter_patcher
        
        # Create and apply the patchers
        tools_patcher = create_tools_py_patcher()
        converter_patcher = create_converter_patcher()
        
        # Apply the patches
        tools_patcher()
        converter_patcher()
        
        logger.info("Successfully installed all error prevention patches")
        
        # Force reimport of patched modules
        importlib.reload(importlib.import_module('archon.mcp_tools.mcp_tool_coder'))
        
        return True
    except Exception as e:
        logger.error(f"Error installing patches: {str(e)}")
        return False

if __name__ == "__main__":
    success = install_patches()
    if success:
        print("Successfully installed all error prevention patches!")
        print("YouTube tool error prevention is now active.")
    else:
        print("Failed to install some patches. See log for details.")
