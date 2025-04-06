#!/usr/bin/env python3
"""
Serper-Spotify Agent 
A combined agent with capabilities for web search and Spotify interactions.
"""

import os
import sys
import logging
import asyncio
import traceback
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import custom modules
try:
    from serper_spotify_agent.agent import setup_agent
except ImportError:
    try:
        from agent import setup_agent
    except ImportError as e:
        logging.error(f"Failed to import setup_agent: {e}")
        logging.error("Make sure you're running this script from the correct directory")
        sys.exit(1)

# Main entry point
async def main():
    """Main entry point for the Serper-Spotify Agent."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API keys from environment variables
        serper_api_key = os.environ.get('SERPER_API_KEY')
        spotify_api_key = os.environ.get('SPOTIFY_API_KEY')
        agent_model = os.environ.get('AGENT_MODEL', os.environ.get('MODEL_CHOICE', 'gpt-4-1106-preview'))
        
        # Check if we have at least one API key
        if not serper_api_key and not spotify_api_key:
            logging.error("No API keys found. Please set SERPER_API_KEY and/or SPOTIFY_API_KEY in your .env file.")
            sys.exit(1)
            
        logging.info(f"Setting up the agent with model: {agent_model}")
        
        # Define the system prompt
        system_prompt = """You are a helpful assistant with various capabilities.
        
        Available tools:
        1. Web Search: You can search the web for information using Serper (Google Search API)
        2. Spotify Music: You can search for and display information about music on Spotify
        
        Respond concisely unless the user requests detailed information.
        Always structure your answer to be as helpful and easy to understand as possible.
        """
        
        # Set up the agent
        agent = await setup_agent(
            serper_api_key=serper_api_key,
            spotify_api_key=spotify_api_key,
            agent_model=agent_model,
            system_prompt=system_prompt
        )
        
        # Check if agent was created successfully
        if not agent:
            logging.error("Failed to set up the agent. Please check the logs for details.")
            logging.error("To diagnose issues, run the test_mcp_revised.py script.")
            sys.exit(1)
            
        logging.info("Agent created successfully!")
        
        # Run interactive mode
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                # Check for exit commands
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    logging.info("Exiting the application...")
                    break
                    
                # Generate response
                logging.info("Generating response...")
                response = await agent.generate(user_input)
                
                # Print response
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                logging.info("Interrupted by user. Exiting...")
                break
            except Exception as e:
                logging.error(f"Error during conversation: {e}")
                logging.error(f"Error details: {traceback.format_exc()}")
                print(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return 1
        
    return 0

if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 