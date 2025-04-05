import asyncio
import logging
import os
from dotenv import load_dotenv
from src.models import SerperConfig
from src.agents import SerperAgent

def setup_logging():
    """Configure detailed logging with both file and console output"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            # File handler for all logs
            logging.FileHandler('logs/search.log'),
            # Console handler for INFO and above
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for different components
    logging.getLogger('httpx').setLevel(logging.INFO)
    logging.getLogger('httpcore').setLevel(logging.INFO)
    logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)  # Added for MCP logging
    
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully")
    return logger

logger = setup_logging()

def load_config() -> SerperConfig:
    """Load configuration from environment variables"""
    try:
        load_dotenv()
        
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            api_key = input("Please enter your Serper API key: ")
            # Save to .env file for future use
            with open(".env", "a") as f:
                f.write(f"\nSERPER_API_KEY={api_key}")
        
        model_name = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
        
        config = SerperConfig(
            serper_api_key=api_key,
            model_name=model_name,
        )
        logger.info("Configuration loaded successfully")
        logger.debug(f"Using model: {config.model_name}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def format_results(result):
    """Format and print search results"""
    try:
        output = []
        output.append("\n" + "="*50)
        output.append("SEARCH RESULTS")
        output.append("="*50)
        output.append(f"\nSearch Query: {result.query}\n")
        output.append("Summary:")
        output.append("-"*50)
        output.append(result.summary)
        output.append("\nDetailed Results:")
        output.append("-"*50)
        
        for i, r in enumerate(result.results, 1):
            output.append(f"\n{i}. {r.title}")
            output.append(f"   {'-'*40}")
            output.append(f"   {r.snippet}")
            output.append(f"   URL: {r.link}")
            output.append("")
        
        formatted_output = "\n".join(output)
        print(formatted_output)
        logger.debug(f"Formatted results:\n{formatted_output}")
        
    except Exception as e:
        logger.error(f"Failed to format results: {str(e)}")
        print("\nError: Failed to format search results. Please check the logs for details.")

async def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize agent
        logger.info("Initializing Serper agent...")
        agent = SerperAgent(config)
        logger.info("Serper agent initialized successfully")
        
        print("\nWelcome to the Serper Search Agent!")
        print("Enter your search queries below, or type 'quit' to exit.")
        
        while True:
            try:
                # Get search query from user
                query = input("\nEnter your search query (or 'quit' to exit): ")
                if query.lower() in ('quit', 'exit', 'q'):
                    logger.info("User requested to quit")
                    break
                    
                if not query.strip():
                    logger.warning("Empty query received")
                    print("Please enter a valid search query.")
                    continue
                    
                logger.info(f"Starting search for query: {query}")
                result = await agent.search(query)
                
                # Print results in a formatted way
                format_results(result)
                
            except KeyboardInterrupt:
                logger.info("Search interrupted by user")
                print("\nSearch interrupted. Starting new search...")
                continue
                
            except Exception as e:
                logger.error(f"Error during search: {str(e)}", exc_info=True)
                print(f"\nError: Failed to complete search. Please try again.")
                continue
        
    except KeyboardInterrupt:
        logger.info("Exiting gracefully due to keyboard interrupt...")
        print("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...") 