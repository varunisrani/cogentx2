#!/usr/bin/env python3
"""
Demo for Multi-MCP Template Features
====================================

This script demonstrates the enhanced template visualization and merging features
for multi-MCP agents. It covers:

1. Creating a multi-MCP template example (Serper + Spotify)
2. Displaying template components before merging
3. Previewing and merging templates with the improved workflow

Usage:
    python demo_multi_mcp.py
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.utils import get_clients, ensure_workbench_dir
from archon.pydantic_ai_coder import create_multi_mcp_template_example, display_template_components, merge_agent_templates, display_template_files
from pydantic_ai import RunContext
from archon.pydantic_ai_coder import PydanticAIDeps

async def demo_create_multi_mcp_template():
    """Create a multi-MCP template example"""
    print("\n" + "="*80)
    print("CREATING MULTI-MCP TEMPLATE EXAMPLE")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return None
    
    # Create context with dependencies
    ctx = RunContext(
        deps=PydanticAIDeps(
            supabase=supabase,
            embedding_client=embedding_client
        )
    )
    
    # Create the multi-MCP template example
    template_name = "serper_spotify_agent"
    sample_purpose = "Combined agent for web search and music control"
    
    try:
        result = await create_multi_mcp_template_example(ctx, template_name, sample_purpose)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        print(f"\nSuccess! Created multi-MCP template example:")
        print(f"  ID: {result['id']}")
        print(f"  Name: {template_name}")
        print(f"  Purpose: {sample_purpose}")
        
        return result['id']
    except Exception as e:
        print(f"Error creating multi-MCP template: {str(e)}")
        return None

async def demo_display_template_components(template_id: int):
    """Display template components for a single template"""
    print("\n" + "="*80)
    print("DISPLAYING TEMPLATE COMPONENTS")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Create context with dependencies
    ctx = RunContext(
        deps=PydanticAIDeps(
            supabase=supabase,
            embedding_client=embedding_client
        )
    )
    
    # Display template components
    try:
        # Display all component types
        components = ["config", "mcp_servers", "tools", "system_prompt", "dependencies", "imports"]
        result = await display_template_components(ctx, [template_id], components)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        templates = result.get("templates", [])
        if not templates:
            print("No template components found")
            return
        
        template = templates[0]
        name = template.get("name", f"template_{template_id}")
        
        print(f"\nTemplate: {name} (ID: {template_id})")
        
        # Display each component
        for component_name, component_content in template.get("components", {}).items():
            print(f"\n{'-'*80}")
            print(f"COMPONENT: {component_name.upper()}")
            print(f"{'-'*80}")
            print(component_content)
    
    except Exception as e:
        print(f"Error displaying template components: {str(e)}")

async def demo_multi_template_comparison():
    """Compare components from multiple templates"""
    print("\n" + "="*80)
    print("COMPARING MULTIPLE TEMPLATES")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Create context with dependencies
    ctx = RunContext(
        deps=PydanticAIDeps(
            supabase=supabase,
            embedding_client=embedding_client
        )
    )
    
    # Ask for template IDs to compare
    try:
        template_ids_str = input("Enter template IDs to compare (comma-separated): ")
        template_ids = [int(id.strip()) for id in template_ids_str.split(",")]
        
        # Display template components
        components = ["config", "mcp_servers"]
        result = await display_template_components(ctx, template_ids, components)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        templates = result.get("templates", [])
        if not templates:
            print("No template components found")
            return
        
        # Display template summary
        print(f"\nComparing {len(templates)} templates:")
        for template in templates:
            print(f"  - {template.get('name', 'Unknown')} (ID: {template.get('id', 'Unknown')})")
        
        # Display components side by side (for "config" and "mcp_servers")
        for component_name in result.get("compared_components", []):
            print(f"\n{'-'*80}")
            print(f"COMPONENT COMPARISON: {component_name.upper()}")
            print(f"{'-'*80}")
            
            for template in templates:
                name = template.get("name", f"template_{template.get('id', 'Unknown')}")
                print(f"\n## {name} ##\n")
                component_content = template.get("components", {}).get(component_name, "Not found")
                print(component_content)
                print("\n")
    
    except Exception as e:
        print(f"Error comparing templates: {str(e)}")

async def demo_preview_and_merge():
    """Preview templates before merging them"""
    print("\n" + "="*80)
    print("PREVIEWING AND MERGING TEMPLATES")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Create context with dependencies
    ctx = RunContext(
        deps=PydanticAIDeps(
            supabase=supabase,
            embedding_client=embedding_client
        )
    )
    
    # Ask for template IDs to merge
    try:
        template_ids_str = input("Enter template IDs to merge (comma-separated): ")
        template_ids = [int(id.strip()) for id in template_ids_str.split(",")]
        
        # First, preview the templates
        print(f"\nPreviewing templates before merging...")
        preview_result = await merge_agent_templates(ctx, "", template_ids, preview_before_merge=True)
        
        if "error" in preview_result:
            print(f"Error: {preview_result['error']}")
            return
        
        if preview_result.get("status") == "preview":
            print(f"\n{preview_result.get('message')}")
            
            # Display summary of templates
            summary = preview_result.get("preview", {}).get("summary", {})
            template_names = summary.get("template_names", [])
            
            print(f"\nTemplates to merge:")
            for i, name in enumerate(template_names, 1):
                print(f"  {i}. {name}")
            
            # Ask if user wants to proceed with the merge
            proceed = input("\nDo you want to proceed with the merge? (y/n): ")
            
            if proceed.lower() != 'y':
                print("Merge cancelled.")
                return
            
            # Proceed with the merge
            print(f"\nMerging templates...")
            merge_result = await merge_agent_templates(ctx, "", template_ids, 
                                                    custom_name="demo_merged_template",
                                                    custom_description="Merged template from demo",
                                                    preview_before_merge=False)
            
            if "error" in merge_result:
                print(f"Error: {merge_result['error']}")
                return
            
            print(f"\nSuccessfully merged templates!")
            print(f"  ID: {merge_result.get('id')}")
            print(f"  Name: {merge_result.get('folder_name')}")
            
            # Show file sizes
            print(f"\nMerged files:")
            for file_name, file_size in merge_result.get("files", {}).items():
                print(f"  {file_name}: {file_size} bytes")
    
    except Exception as e:
        print(f"Error previewing and merging templates: {str(e)}")

async def main():
    """Main function"""
    print("Multi-MCP Template Demo")
    print("======================")
    print("1. Create multi-MCP template example")
    print("2. Display template components")
    print("3. Compare multiple templates")
    print("4. Preview and merge templates")
    print("5. Run full demo workflow")
    print("q. Quit")
    
    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        await demo_create_multi_mcp_template()
    elif choice == "2":
        template_id = input("Enter template ID: ")
        try:
            template_id = int(template_id)
            await demo_display_template_components(template_id)
        except ValueError:
            print("Invalid input. Please enter a valid template ID.")
    elif choice == "3":
        await demo_multi_template_comparison()
    elif choice == "4":
        await demo_preview_and_merge()
    elif choice == "5":
        # Run full workflow
        template_id = await demo_create_multi_mcp_template()
        if template_id:
            await demo_display_template_components(template_id)
            print("\nNow let's compare with another template...")
            await demo_multi_template_comparison()
            print("\nFinally, let's preview and merge templates...")
            await demo_preview_and_merge()
    elif choice.lower() == "q":
        print("Exiting.")
    else:
        print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    asyncio.run(main()) 