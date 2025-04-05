#!/usr/bin/env python3
import sys
import os
import asyncio
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.utils import get_clients, ensure_workbench_dir
from archon.agent_tools import search_agent_templates_tool, fetch_template_by_id_tool

async def fetch_and_display_template(template_id: int):
    """Fetch a template by ID and display its contents"""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    print("Initializing clients...")
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Ensure workbench directory exists
    workbench_dir = ensure_workbench_dir()
    
    # Fetch template
    print(f"Fetching template with ID {template_id}...")
    result = await fetch_template_by_id_tool(supabase, template_id)
    
    if not result.get("found", False):
        print(f"Error: Template with ID {template_id} not found")
        return
    
    template = result.get("template", {})
    
    # Display template information
    print(f"\n{'='*80}")
    print(f"TEMPLATE INFORMATION")
    print(f"{'='*80}")
    print(f"ID: {template.get('id')}")
    print(f"Folder: {template.get('folder_name')}")
    print(f"Purpose: {template.get('purpose')}")
    
    # Write template files to disk for easier viewing
    output_dir = os.path.join(workbench_dir, f"template_{template_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write agents.py
    if template.get("agents_code"):
        with open(os.path.join(output_dir, "agents.py"), "w") as f:
            f.write(template.get("agents_code"))
        print(f"\nWrote agents.py to {os.path.join(output_dir, 'agents.py')}")
    
    # Write main.py
    if template.get("main_code"):
        with open(os.path.join(output_dir, "main.py"), "w") as f:
            f.write(template.get("main_code"))
        print(f"Wrote main.py to {os.path.join(output_dir, 'main.py')}")
    
    # Write models.py
    if template.get("models_code"):
        with open(os.path.join(output_dir, "models.py"), "w") as f:
            f.write(template.get("models_code"))
        print(f"Wrote models.py to {os.path.join(output_dir, 'models.py')}")
    
    # Write tools.py
    if template.get("tools_code"):
        with open(os.path.join(output_dir, "tools.py"), "w") as f:
            f.write(template.get("tools_code"))
        print(f"Wrote tools.py to {os.path.join(output_dir, 'tools.py')}")
    
    # Write mcp.json
    if template.get("mcp_json"):
        with open(os.path.join(output_dir, "mcp.json"), "w") as f:
            f.write(template.get("mcp_json"))
        print(f"Wrote mcp.json to {os.path.join(output_dir, 'mcp.json')}")
    
    print(f"\nTemplate files written to {output_dir}")
    print(f"You can now view and edit these files directly.")

async def search_templates(query: str, limit: int = 5):
    """Search for templates based on a query"""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    print("Initializing clients...")
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Search for templates
    print(f"Searching for templates matching: '{query}'...")
    result = await search_agent_templates_tool(supabase, embedding_client, query, 0.3, limit)
    
    templates = result.get("templates", [])
    
    if not templates:
        print("No templates found matching your query.")
        return
    
    # Display search results
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS")
    print(f"{'='*80}")
    
    for i, template in enumerate(templates, 1):
        print(f"{i}. {template.get('folder_name')} (ID: {template.get('id')})")
        print(f"   Purpose: {template.get('purpose')[:200]}..." if len(template.get('purpose', '')) > 200 else template.get('purpose', ''))
        print(f"   Similarity: {template.get('similarity', 0):.4f}")
        print()
    
    # Ask which template to view
    while True:
        choice = input("\nEnter template number to view (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(templates):
                template_id = templates[choice_idx].get('id')
                await fetch_and_display_template(template_id)
            else:
                print("Invalid choice. Please enter a valid template number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

async def merge_templates(template_ids: list):
    """Merge multiple templates"""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    print("Initializing clients...")
    embedding_client, supabase = get_clients()
    
    # Check client status
    if not embedding_client or not supabase:
        print("Error: Could not initialize embedding client or Supabase client")
        return
    
    # Ensure workbench directory exists
    workbench_dir = ensure_workbench_dir()
    
    # Fetch and display templates
    templates = []
    template_names = []
    
    for template_id in template_ids:
        result = await fetch_template_by_id_tool(supabase, template_id)
        if result.get("found", False):
            template = result.get("template", {})
            templates.append(template)
            template_names.append(template.get("folder_name", f"template_{template_id}"))
            print(f"Template {template_id} ({template.get('folder_name')}) loaded")
        else:
            print(f"Error: Template with ID {template_id} not found")
    
    if len(templates) < 2:
        print("Error: At least 2 templates are required for merging")
        return
    
    # Create directory for merged output
    merged_name = "_".join(template_names)
    merged_dir = os.path.join(workbench_dir, f"merged_{merged_name}")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Print details of templates to merge
    print(f"\n{'='*80}")
    print(f"MERGING TEMPLATES")
    print(f"{'='*80}")
    
    for i, template in enumerate(templates, 1):
        print(f"{i}. {template.get('folder_name')} (ID: {template.get('id')})")
        print(f"   Purpose: {template.get('purpose')[:100]}..." if len(template.get('purpose', '')) > 100 else template.get('purpose', ''))
    
    print(f"\nMerged files will be written to {merged_dir}")
    
    # Here we would call merge_agent_templates but since we can't directly access it
    # through the agent, we'll just display the templates for manual review
    
    # Write template files to disk for comparing and manual merging
    for i, template in enumerate(templates):
        template_id = template.get('id')
        template_dir = os.path.join(workbench_dir, f"template_{template_id}")
        os.makedirs(template_dir, exist_ok=True)
        
        # Write agents.py
        if template.get("agents_code"):
            with open(os.path.join(template_dir, "agents.py"), "w") as f:
                f.write(template.get("agents_code"))
        
        # Write main.py
        if template.get("main_code"):
            with open(os.path.join(template_dir, "main.py"), "w") as f:
                f.write(template.get("main_code"))
        
        # Write models.py
        if template.get("models_code"):
            with open(os.path.join(template_dir, "models.py"), "w") as f:
                f.write(template.get("models_code"))
        
        # Write tools.py
        if template.get("tools_code"):
            with open(os.path.join(template_dir, "tools.py"), "w") as f:
                f.write(template.get("tools_code"))
        
        # Write mcp.json
        if template.get("mcp_json"):
            with open(os.path.join(template_dir, "mcp.json"), "w") as f:
                f.write(template.get("mcp_json"))
        
        print(f"Template {template_id} files written to {template_dir}")
    
    print("\nTemplate files written to disk.")
    print("Please review the template files and manually merge them, or use the merge_agent_templates tool in the assistant.")

async def main():
    """Main function"""
    print("Template Viewer and Merger")
    print("=========================")
    print("1. Search templates")
    print("2. View template by ID")
    print("3. Merge templates")
    print("q. Quit")
    
    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        query = input("Enter search query: ")
        limit = input("Enter maximum number of results (default: 5): ")
        try:
            limit = int(limit) if limit else 5
        except ValueError:
            limit = 5
        await search_templates(query, limit)
    elif choice == "2":
        template_id = input("Enter template ID: ")
        try:
            template_id = int(template_id)
            await fetch_and_display_template(template_id)
        except ValueError:
            print("Invalid input. Please enter a valid template ID.")
    elif choice == "3":
        template_ids_str = input("Enter template IDs to merge (comma-separated): ")
        try:
            template_ids = [int(id.strip()) for id in template_ids_str.split(",")]
            await merge_templates(template_ids)
        except ValueError:
            print("Invalid input. Please enter valid template IDs separated by commas.")
    elif choice.lower() == "q":
        print("Exiting.")
    else:
        print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    asyncio.run(main()) 