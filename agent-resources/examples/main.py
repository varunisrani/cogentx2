import asyncio
import logging
import os
from dotenv import load_dotenv
from models import SerperConfig
from agents import SerperAgent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> SerperConfig:
    """Load configuration from environment variables"""
    load_dotenv()
    
    config = SerperConfig(
        serper_api_key=os.getenv("SERPER_API_KEY"),
        model_name=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
    )
    logger.info("Configuration loaded successfully")
    logger.debug(f"Using model: {config.model_name}")
    return config

def format_results(result):
    """Format and print search results"""
    print("\n" + "="*50)
    print("SEARCH RESULTS")
    print("="*50)
    print(f"\nSearch Query: {result.query}\n")
    print("Summary:")
    print("-"*50)
    print(result.summary)
    print("\nDetailed Results:")
    print("-"*50)
    for i, r in enumerate(result.results, 1):
        print(f"\n{i}. {r.title}")
        print(f"   {'-'*40}")
        print(f"   {r.snippet}")
        print(f"   URL: {r.link}")
        print()

async def main():
    # Load configuration
    config = load_config()
    
    # Initialize agent
    agent = SerperAgent(config)
    
    try:
        # Example search query
        query = "What are the latest developments in quantum computing?"
        logger.info(f"Starting search for query: {query}")
        result = await agent.search(query)
        
        # Print results in a formatted way
        format_results(result)
            
    except KeyboardInterrupt:
        logger.info("Exiting gracefully...")
        print("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...") 