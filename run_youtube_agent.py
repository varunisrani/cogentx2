#!/usr/bin/env python3
"""
Run YouTube Transcript Agent

This script provides a simple way to run the YouTube Transcript Agent.
"""

import os
import sys
import subprocess

def main():
    """Run the YouTube Transcript Agent"""
    # Check if the youtube_transcript_agent directory exists
    if not os.path.isdir("youtube_transcript_agent"):
        print("Error: youtube_transcript_agent directory not found.")
        sys.exit(1)
    
    # Run the YouTube Transcript Agent
    try:
        subprocess.run([sys.executable, "-m", "youtube_transcript_agent.main"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running YouTube Transcript Agent: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nYouTube Transcript Agent stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
