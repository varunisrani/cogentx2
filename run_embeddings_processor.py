"""
Run Embeddings Processor

This script runs the embeddings processor to process agent templates and store them in Supabase.
"""

import asyncio
import logging
import argparse
from batch_process_agents import process_agents
from embeddings_processor import AgentEmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_single_agent(folder_path):
    """Process a single agent folder"""
    processor = AgentEmbeddingProcessor()
    
    logger.info(f"Processing agent folder: {folder_path}")
    
    try:
        # Process the agent folder
        template = await processor.process_agent_folder(folder_path)
        
        logger.info(f"Template purpose determined: {template.purpose}")
        logger.info(f"Generated embedding with {len(template.embedding)} dimensions")
        
        # Save template metadata locally
        output_path = f"{folder_path}/template_data.json"
        await processor.save_template_metadata(template, output_path)
        
        # Insert into database if credentials are available
        if processor.supabase_client:
            logger.info(f"Inserting template into database...")
            await processor.insert_template(template)
            logger.info(f"Completed processing and database insertion")
        else:
            logger.warning("Skipping database insertion due to missing credentials")
            logger.info("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file to enable database storage.")
            
    except Exception as e:
        logger.error(f"Error processing agent: {str(e)}", exc_info=True)

async def search_templates(query):
    """Search for similar templates"""
    processor = AgentEmbeddingProcessor()
    
    if not processor.supabase_client:
        logger.error("Cannot search: Supabase client not initialized")
        logger.info("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file to enable database search.")
        return
    
    logger.info(f"Searching for templates matching: '{query}'")
    
    try:
        similar_templates = await processor.search_similar_templates(query)
        
        logger.info(f"Found {len(similar_templates)} similar templates:")
        for idx, t in enumerate(similar_templates):
            logger.info(f"{idx+1}. {t['folder_name']} (similarity: {t['similarity']:.4f})")
            logger.info(f"   Purpose: {t['purpose']}")
            logger.info("")
            
    except Exception as e:
        logger.error(f"Error searching templates: {str(e)}", exc_info=True)

async def main():
    """Main function to parse arguments and run the appropriate function"""
    parser = argparse.ArgumentParser(description="Process agent templates and store in Supabase")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process a single agent
    process_parser = subparsers.add_parser("process", help="Process a single agent folder")
    process_parser.add_argument("folder_path", help="Path to the agent folder")
    
    # Process all agents
    batch_parser = subparsers.add_parser("batch", help="Process all agent folders")
    
    # Search for templates
    search_parser = subparsers.add_parser("search", help="Search for similar templates")
    search_parser.add_argument("query", help="Search query")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate function
    if args.command == "process":
        await process_single_agent(args.folder_path)
    elif args.command == "batch":
        await process_agents()
    elif args.command == "search":
        await search_templates(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
