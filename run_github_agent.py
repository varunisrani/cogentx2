#!/usr/bin/env python3
"""
GitHub Agent Runner
------------------
Simple entry point to run the GitHub agent
"""

import asyncio
import sys

if __name__ == "__main__":
    try:
        # Import the main function from the package
        from github_agent.main import main
        
        # Run the agent's main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 