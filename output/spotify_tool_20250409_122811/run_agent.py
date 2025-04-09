#!/usr/bin/env python3
import os
import logging
import sys
import argparse
from dotenv import load_dotenv
from crew import SpotifyMCPMCPTool

def setup_logging(verbose):
    """Set up logging for the Spotify Recommendation Agent"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = f"logs/spotify_recommendation_agent.log"
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("spotify_recommendation_agent")
    return logger

def get_input_text():
    """Get input text from command line or stdin"""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    else:
        return input("Please enter your request: ")

def main():
    """Main function to run the Spotify Recommendation Agent"""
    parser = argparse.ArgumentParser(description="Run Spotify Recommendation Agent")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('-a', '--agent', default='spotify', help="Specify which agent to run")
    parser.add_argument('-o', '--output', help="Specify output file path")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    logger.info("Spotify Recommendation Agent Starting")

    input_text = get_input_text()
    logger.info(f"User Request: {input_text}")

    try:
        tool = SpotifyMCPMCPTool()
        response = tool.recommend_songs(input_text)  # Assuming this method exists
        logger.info(f"Agent Response: {response}")
        print(f"Recommendations: {response}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(response)
                logger.info(f"Output written to {args.output}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    load_dotenv()
    main()