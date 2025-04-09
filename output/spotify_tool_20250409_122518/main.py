#!/usr/bin/env python3
import os
import logging
import datetime
import time
from dotenv import load_dotenv
from crew import SpotifyCrew
import argparse

def setup_main_logging():
    """Set up logging for the main Spotify Agent system"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/spotify_agent_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a specific logger for Spotify agent
    logger = logging.getLogger("spotify_agent")
    
    return logger, log_file

def main():
    """Main function to run the Spotify Agent system"""
    # Set up logging
    logger, log_file = setup_main_logging()
    logger.info("="*80)
    logger.info("Spotify Agent System Starting")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Validate Spotify credentials are available
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not spotify_client_id or not spotify_client_secret:
        logger.error("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET environment variables are missing")
        print("ERROR: Spotify API credentials not found.")
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file.")
        return
    
    logger.info(f"Spotify client ID found: {spotify_client_id[:5]}...{spotify_client_id[-5:] if len(spotify_client_id) > 10 else ''}")
    logger.info(f"Spotify client secret found: {spotify_client_secret[:5]}...{spotify_client_secret[-5:] if len(spotify_client_secret) > 10 else ''}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Spotify Agent System')
    parser.add_argument('query', type=str, nargs='?', help='Spotify-related query to process')
    parser.add_argument('--no-human-input', action='store_true', help='Disable human input during task execution')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory for the crew')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    logger.info(f"Command-line arguments: {args}")
    
    # Create the Spotify crew
    logger.info("Creating Spotify crew")
    spotify_crew = SpotifyCrew(
        verbose=not args.quiet,
        memory=not args.no_memory,
        human_input=not args.no_human_input
    )
    
    if args.query:
        # Process the query
        logger.info("-"*80)
        logger.info(f"QUERY: {args.query}")
        print(f"Processing query: {args.query}")
        
        start_time = time.time()
        try:
            result = spotify_crew.process_query(args.query)
            
            # Log query completion
            execution_time = time.time() - start_time
            logger.info(f"Query completed in {execution_time:.2f} seconds")
            logger.info(f"RESULT LENGTH: {len(result)} characters")
            logger.info(f"RESULT SUMMARY: {result[:100]}..." if len(result) > 100 else result)
            
            print("\nResult:")
            print(result)
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.2f} seconds: {str(e)}")
            print(f"Error processing query: {str(e)}")
    else:
        # Interactive mode
        logger.info("Starting interactive mode")
        print("Spotify Agent System")
        print("------------------")
        print("Enter 'exit' or 'quit' to exit.")
        print("Enter your Spotify-related queries below:")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['exit', 'quit']:
                    logger.info("User requested exit")
                    break
                
                if query:
                    # Log query
                    logger.info("-"*80)
                    logger.info(f"INTERACTIVE QUERY: {query}")
                    
                    # Process query with timing
                    start_time = time.time()
                    result = spotify_crew.process_query(query)
                    
                    # Log query completion
                    execution_time = time.time() - start_time
                    logger.info(f"Query completed in {execution_time:.2f} seconds")
                    logger.info(f"RESULT LENGTH: {len(result)} characters")
                    logger.info(f"RESULT SUMMARY: {result[:100]}..." if len(result) > 100 else result)
                    
                    print("\nResult:")
                    print(result)
            except KeyboardInterrupt:
                logger.info("Interrupted by user (CTRL+C)")
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing interactive query: {str(e)}")
                print(f"Error: {str(e)}")
    
    logger.info("="*80)
    logger.info("Spotify Agent System Shutting Down")
    logger.info("="*80)

if __name__ == "__main__":
    main() 