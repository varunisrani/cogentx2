#!/usr/bin/env python3
# Simple agent runner for CrewAI project
# Error during generation: Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}

import argparse
import logging
import sys
from agents import create_agent  # Assumes a create_agent function exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("agent_runner")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run a single agent directly")
    parser.add_argument("input", nargs="?", help="Input text for the agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-a", "--agent", default="primary", help="Agent to run (if multiple available)")
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get input from stdin if not provided as argument
    user_input = args.input
    if not user_input:
        print("Enter input for the agent (Ctrl+D to finish):")
        user_input = sys.stdin.read().strip()
    
    try:
        # Create and run the agent
        agent = create_agent(args.agent)
        result = agent.run(user_input)
        print(result)
        return 0
    except Exception as e:
        logger.error(f"Error running agent: Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
