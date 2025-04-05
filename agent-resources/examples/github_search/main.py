import asyncio
import logging
import os
from dotenv import load_dotenv
from src.models import GitHubConfig
from src.agents import GitHubAgent

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
            logging.FileHandler('logs/github.log'),
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

def load_config() -> GitHubConfig:
    """Load configuration from environment variables"""
    try:
        load_dotenv()
        
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not token:
            token = input("Please enter your GitHub Personal Access Token: ")
            # Save to .env file for future use
            with open(".env", "a") as f:
                f.write(f"\nGITHUB_PERSONAL_ACCESS_TOKEN={token}")
        
        config = GitHubConfig(
            access_token=token,
            model_name=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            max_results=int(os.getenv("GITHUB_MAX_RESULTS", "10"))
        )
        logger.info("Configuration loaded successfully")
        logger.debug(f"Using model: {config.model_name}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def format_results(result):
    """Format and print GitHub search results"""
    try:
        output = []
        output.append("\n" + "="*50)
        output.append("GITHUB SEARCH RESULTS")
        output.append("="*50)
        output.append(f"\nSearch Query: {result.query}\n")
        
        output.append("Summary:")
        output.append("-"*50)
        output.append(result.summary)
        
        if result.repositories:
            output.append("\nRepositories:")
            output.append("-"*50)
            for i, repo in enumerate(result.repositories, 1):
                output.append(f"\n{i}. {repo.full_name}")
                output.append(f"   {'-'*40}")
                output.append(f"   Description: {repo.description or 'N/A'}")
                output.append(f"   Language: {repo.language or 'N/A'}")
                output.append(f"   Stars: {repo.stars or 0} | Forks: {repo.forks or 0}")
                output.append(f"   URL: {repo.url}")
                if repo.updated_at:
                    output.append(f"   Last Updated: {repo.updated_at}")
                output.append("")
        
        if result.issues:
            output.append("\nIssues:")
            output.append("-"*50)
            for i, issue in enumerate(result.issues, 1):
                output.append(f"\n{i}. #{issue.number}: {issue.title}")
                output.append(f"   {'-'*40}")
                output.append(f"   State: {issue.state}")
                output.append(f"   Created: {issue.created_at}")
                if issue.labels:
                    output.append(f"   Labels: {', '.join(issue.labels)}")
                output.append(f"   URL: {issue.url}")
                if issue.body:
                    output.append(f"   Description: {issue.body[:200]}...")
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
        logger.info("Initializing GitHub agent...")
        agent = GitHubAgent(config)
        logger.info("GitHub agent initialized successfully")
        
        print("\nWelcome to the GitHub Search Agent!")
        print("Enter your search queries below, or type 'quit' to exit.")
        print("Examples:")
        print("- 'python web framework stars:>1000'")
        print("- 'issue:open label:bug language:javascript'")
        
        while True:
            try:
                # Get search query from user
                query = input("\nEnter your GitHub search query (or 'quit' to exit): ")
                if query.lower() in ('quit', 'exit', 'q'):
                    logger.info("User requested to quit")
                    break
                    
                if not query.strip():
                    logger.warning("Empty query received")
                    print("Please enter a valid query.")
                    continue
                    
                logger.info(f"Starting GitHub search for query: {query}")
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