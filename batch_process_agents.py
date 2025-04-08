"""
Batch Process Agents

This script processes multiple agent folders and stores their embeddings in Supabase.
It specifically targets time_agent, file_agent, and weather_agent.
"""

import os
import asyncio
import logging
from embeddings_processor import AgentEmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_agents():
    """Process all specified agent folders"""
    # Initialize the processor
    processor = AgentEmbeddingProcessor()
    
    # Define agent folders to process
    agent_folders = [
        "time_agent",
        "file_agent",
        "weather_agent"
    ]
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process each agent folder
    for folder_name in agent_folders:
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            logger.warning(f"Agent folder not found: {folder_path}")
            continue
            
        logger.info(f"Processing agent folder: {folder_path}")
        
        try:
            # Process the agent folder
            template = await processor.process_agent_folder(folder_path)
            
            logger.info(f"Template purpose determined: {template.purpose}")
            logger.info(f"Generated embedding with {len(template.embedding)} dimensions")
            
            # Save template metadata locally
            output_path = os.path.join(folder_path, "template_data.json")
            await processor.save_template_metadata(template, output_path)
            
            # Insert into database if credentials are available
            if processor.supabase_client:
                logger.info(f"Inserting {folder_name} template into database...")
                await processor.insert_template(template)
                logger.info(f"Completed processing and database insertion for {folder_name}")
            else:
                logger.warning("Skipping database insertion due to missing credentials")
                logger.info("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file to enable database storage.")
                
        except Exception as e:
            logger.error(f"Error processing {folder_name}: {str(e)}", exc_info=True)
    
    # Search for similar templates if database is available
    if processor.supabase_client:
        logger.info("\nSearching for similar templates...")
        search_query = "time agent file agent weather agent"
        similar_templates = await processor.search_similar_templates(search_query)
        
        logger.info(f"Found {len(similar_templates)} similar templates:")
        for idx, t in enumerate(similar_templates):
            logger.info(f"{idx+1}. {t['folder_name']} (similarity: {t['similarity']:.4f})")
            logger.info(f"   Purpose: {t['purpose']}")
            logger.info("")

if __name__ == "__main__":
    asyncio.run(process_agents())
