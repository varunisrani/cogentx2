"""
MCP Tools Package - Model Context Protocol tool integration

This package provides tools for finding, analyzing, and generating customized MCP tools
using Adaptive Code Synthesis - a sophisticated approach that deeply understands
code and adapts it to specific user requirements.
"""

# Import key functions from mcp_tool_coder
from .mcp_tool_coder import (
    mcp_tool_agent,
    MCPToolDeps,
    find_relevant_mcp_tools,
    integrate_mcp_tool_with_code,
    analyze_tool_code,
    customize_tool_implementation,
    verify_tool_integration,
    create_mcp_context
)

# Import from mcp_tool_graph
from .mcp_tool_graph import (
    mcp_tool_flow,
    get_tool_name_from_purpose,
    adaptive_code_synthesis_flow,
    combined_adaptive_flow
)

# Import from our selector
from .mcp_tool_selector import (
    get_required_tools,
    filter_tools_by_user_needs,
    extract_structured_requirements,
    rank_tools_by_requirement_match,
    UserRequirements,
    ToolRequirement
)

# Define public API

# Import validators for error prevention
try:
    from .validators.install import install_patches
    # Auto-install patches if possible
    install_patches()
except ImportError:
    pass  # Validators not installed

__all__ = [
    # From mcp_tool_coder
    'mcp_tool_agent',
    'MCPToolDeps',
    'find_relevant_mcp_tools',
    'integrate_mcp_tool_with_code',
    'analyze_tool_code',
    'customize_tool_implementation',
    'verify_tool_integration',
    'create_mcp_context',
    
    # From mcp_tool_graph
    'mcp_tool_flow',
    'get_tool_name_from_purpose',
    'adaptive_code_synthesis_flow',
    'combined_adaptive_flow',
    
    # From mcp_tool_selector
    'get_required_tools',
    'filter_tools_by_user_needs',
    'extract_structured_requirements',
    'rank_tools_by_requirement_match',
    'UserRequirements',
    'ToolRequirement'
]
