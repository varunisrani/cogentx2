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
        return ' '.join(sys.argv[1:])
    return input("Enter song preference: ")

def main():
    """Main function to run the Spotify Recommendation Agent system"""
    parser = argparse.ArgumentParser(description="Run Spotify Recommendation Agent")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('-o', '--output', type=str, help="Output file path")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    logger.info("="*80)
    logger.info("Spotify Recommendation Agent Starting")

    try:
        input_text = get_input_text()
        logger.info(f"User input: {input_text}")

        tool = SpotifyMCPMCPTool()
        recommendations = tool.recommend_songs(input_text)

        if args.output:
            with open(args.output, 'w') as f:
                f.write("\n".join(recommendations))
                logger.info(f"Recommendations written to {args.output}")

        print("Recommendations:")
        for song in recommendations:
            print(f"- {song}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    load_dotenv()
    main()