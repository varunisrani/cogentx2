import asyncio
import logging
import os
from dotenv import load_dotenv
from src.models import FireCrawlConfig
from src.agents import FireCrawlAgent

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
            logging.FileHandler('logs/firecrawl.log'),
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

def load_config() -> FireCrawlConfig:
    """Load configuration from environment variables"""
    try:
        load_dotenv()
        
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            api_key = input("Please enter your FireCrawl API key: ")
            # Save to .env file for future use
            with open(".env", "a") as f:
                f.write(f"\nFIRECRAWL_API_KEY={api_key}")
        
        config = FireCrawlConfig(
            api_key=api_key,
            model_name=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            max_retries=int(os.getenv("FIRECRAWL_RETRY_MAX_ATTEMPTS", "5")),
            initial_delay=int(os.getenv("FIRECRAWL_RETRY_INITIAL_DELAY", "2000")),
            max_delay=int(os.getenv("FIRECRAWL_RETRY_MAX_DELAY", "30000")),
            backoff_factor=int(os.getenv("FIRECRAWL_RETRY_BACKOFF_FACTOR", "3")),
            credit_warning=int(os.getenv("FIRECRAWL_CREDIT_WARNING_THRESHOLD", "2000")),
            credit_critical=int(os.getenv("FIRECRAWL_CREDIT_CRITICAL_THRESHOLD", "500"))
        )
        logger.info("Configuration loaded successfully")
        logger.debug(f"Using model: {config.model_name}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def format_results(result):
    """Format and print crawl results"""
    try:
        output = []
        output.append("\n" + "="*50)
        output.append("FIRECRAWL RESULTS")
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
            output.append(f"   URL: {r.url}")
            if r.timestamp:
                output.append(f"   Crawled: {r.timestamp}")
            output.append("")
        
        formatted_output = "\n".join(output)
        print(formatted_output)
        logger.debug(f"Formatted results:\n{formatted_output}")
        
    except Exception as e:
        logger.error(f"Failed to format results: {str(e)}")
        print("\nError: Failed to format crawl results. Please check the logs for details.")

async def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize agent
        logger.info("Initializing FireCrawl agent...")
        agent = FireCrawlAgent(config)
        logger.info("FireCrawl agent initialized successfully")
        
        print("\nWelcome to the FireCrawl Agent!")
        print("Enter your search queries below, or type 'quit' to exit.")
        
        while True:
            try:
                # Get search query from user
                query = input("\nEnter your crawl query (or 'quit' to exit): ")
                if query.lower() in ('quit', 'exit', 'q'):
                    logger.info("User requested to quit")
                    break
                    
                if not query.strip():
                    logger.warning("Empty query received")
                    print("Please enter a valid query.")
                    continue
                    
                logger.info(f"Starting crawl for query: {query}")
                result = await agent.crawl(query)
                
                # Print results in a formatted way
                format_results(result)
                
            except KeyboardInterrupt:
                logger.info("Crawl interrupted by user")
                print("\nCrawl interrupted. Starting new search...")
                continue
                
            except Exception as e:
                logger.error(f"Error during crawl: {str(e)}", exc_info=True)
                print(f"\nError: Failed to complete crawl. Please try again.")
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