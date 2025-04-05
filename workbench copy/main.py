# Merged main.py from templates: spotify_agent, github_agent
# Created at: 2025-04-05T11:13:10.271945

from agent import setup_agent
from agents import spotify_github_agent
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from models import load_config
from tools import run_spotify_query
import argparse
import asyncio
import colorlog
import json
import logging
import os
import sys
import traceback


load_dotenv()  # Load environment variables from .env file

def get_model():
    """Get the LLM model to use based on environment variables."""
    from pydantic_ai.models.openai import OpenAIModel
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    return OpenAIModel(model_name)

async def main():
    # Initialize all required MCP servers

        # Initialize agent with all MCP servers
        model = get_model()
        
        # Create agent with all required MCP servers
        # No MCP servers required
        
        print("Agent initialized! You can now chat with the agent.")
        print("Type 'exit' to quit the conversation.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            try:
                result = await spotify_github_agent.run(user_input)
                print(f"\nAgent: {result.data}")
            except Exception as e:
                print(f"An error occurred: {e}")
        
    if __name__ == "__main__":
        asyncio.run(main())
    