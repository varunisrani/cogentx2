"""
YouTube Tool Validation Modules

This package provides validation and error prevention for tools.py generation,
particularly addressing common issues with YouTube transcript tools.
"""

from .validate_tools import (
    check_and_fix_tools_py,
    validate_class_naming,
    validate_cross_file_consistency,
    enhance_tools_generation_prompt
)

from .enhanced_tools_generator import (
    EnhancedToolsGenerator,
    create_tools_py_patcher,
    add_youtube_transcript_mcp_tool_alias
)

from .enhanced_converter import (
    EnhancedToolConverter,
    create_converter_patcher
)

__all__ = [
    'check_and_fix_tools_py',
    'validate_class_naming',
    'validate_cross_file_consistency',
    'enhance_tools_generation_prompt',
    'EnhancedToolsGenerator',
    'create_tools_py_patcher',
    'add_youtube_transcript_mcp_tool_alias',
    'EnhancedToolConverter',
    'create_converter_patcher'
]
