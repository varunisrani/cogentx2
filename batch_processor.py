#!/usr/bin/env python3
import os
import sys
import asyncio
import argparse
from embeddings_processor import AgentEmbeddingProcessor

# Paths to agent folders
AGENT_FOLDERS = [
    "serper_agent",
    "github_agent",
    "firecrawl_agent", 
    "spotify_agent"
]

async def process_all_agents(base_path, skip_existing=False):
    """Process all agent folders and create embeddings"""
    processor = AgentEmbeddingProcessor()
    results = []
    
    print(f"Starting batch processing of {len(AGENT_FOLDERS)} agent folders...")
    
    for folder_name in AGENT_FOLDERS:
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping.")
            continue
            
        # Check if main agent files exist
        required_files = ["agent.py", "main.py", "models.py", "tools.py"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(folder_path, f))]
        
        if missing_files:
            print(f"Warning: Folder {folder_name} is missing required files: {', '.join(missing_files)}, skipping.")
            continue
        
        # Check if this template already exists
        if skip_existing and processor.supabase_client:
            try:
                response = processor.supabase_client.table("agent_embeddings") \
                    .select("id") \
                    .eq("folder_name", folder_name) \
                    .execute()
                    
                if response.data:
                    print(f"Skipping {folder_name}: Template already exists in database.")
                    continue
            except Exception as e:
                print(f"Error checking if template exists: {e}")
        
        print(f"\nProcessing agent folder: {folder_name}")
        try:
            # Process the agent folder
            template = await processor.process_agent_folder(folder_path)
            
            print(f"✅ Template processed: {folder_name}")
            print(f"Purpose: {template.purpose}")
            print(f"Generated embedding with {len(template.embedding)} dimensions")
            
            # Save metadata locally
            output_path = os.path.join(folder_path, "template_data.json")
            await processor.save_template_metadata(template, output_path)
            
            # Insert into database
            if processor.supabase_client:
                print(f"Inserting {folder_name} into database...")
                await processor.insert_template(template)
                print(f"✅ Successfully inserted {folder_name} into database")
            
            results.append({
                "folder_name": folder_name,
                "success": True,
                "purpose": template.purpose
            })
            
        except Exception as e:
            print(f"❌ Error processing {folder_name}: {e}")
            results.append({
                "folder_name": folder_name,
                "success": False,
                "error": str(e)
            })
    
    return results

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process multiple agent folders and create embeddings")
    parser.add_argument("--base-path", help="Base path containing agent folders (default: current directory)", default=".")
    parser.add_argument("--skip-existing", action="store_true", help="Skip agents already in database")
    args = parser.parse_args()
    
    # Get absolute base path
    base_path = os.path.abspath(args.base_path)
    print(f"Using base path: {base_path}")
    
    # Process all agent folders
    results = await process_all_agents(base_path, args.skip_existing)
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Total agent folders processed: {len(results)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed to process: {len(failed)}")
    
    if successful:
        print("\nSuccessfully processed folders:")
        for idx, result in enumerate(successful):
            print(f"{idx+1}. {result['folder_name']}")
            print(f"   Purpose: {result['purpose']}")
    
    if failed:
        print("\nFailed to process folders:")
        for idx, result in enumerate(failed):
            print(f"{idx+1}. {result['folder_name']}")
            print(f"   Error: {result['error']}")
    
    print("\nTo search for templates, use the search_templates.py script.")

if __name__ == "__main__":
    asyncio.run(main()) 