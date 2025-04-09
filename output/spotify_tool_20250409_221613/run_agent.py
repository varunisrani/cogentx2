#!/usr/bin/env python3
import os
import logging
import argparse
import sys
from dotenv import load_dotenv
from crew import SpotifyMCPMCPTool

def setup_logging(verbose):
    """Set up logging for the agent"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("spotify_recommendation_agent")
    return logger

def run_agent(input_text):
    """Run the Spotify recommendation agent"""
    try:
        tool = SpotifyMCPMCPTool()
        recommendations = tool.get_recommendations(input_text)
        return recommendations
    except Exception as e:
        logging.error(f"Error while running the agent: {e}")
        return None

def main():
    """Main function to run the agent"""
    parser = argparse.ArgumentParser(description="Run Spotify Recommendation Agent")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--agent', '-a', default='SpotifyMCPMCPTool', help='Specify which agent to run')
    parser.add_argument('--output', '-o', type=str, help='Specify output file path')
    parser.add_argument('input_text', nargs='?', type=str, help='Input text for recommendations')

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.input_text is None:
        logging.info("No input text provided. Entering interactive mode.")
        input_text = input("Enter your query for song recommendations: ")
    else:
        input_text = args.input_text

    logging.info(f"Running agent: {args.agent} with input: {input_text}")
    recommendations = run_agent(input_text)

    if recommendations:
        output = "\n".join(recommendations)
        print("Recommendations:\n" + output)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
                logging.info(f"Recommendations saved to {args.output}")
    else:
        logging.error("No recommendations generated.")

if __name__ == "__main__":
    load_dotenv()
    main()