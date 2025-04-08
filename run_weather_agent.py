#!/usr/bin/env python3
"""
Run Weather Agent

This script provides a simple way to run the Weather Agent.
"""

import os
import sys
import subprocess

def main():
    """Run the Weather Agent"""
    # Check if the weather_agent directory exists
    if not os.path.isdir("weather_agent"):
        print("Error: weather_agent directory not found.")
        sys.exit(1)
    
    # Run the Weather Agent
    try:
        subprocess.run([sys.executable, "-m", "weather_agent.main"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Weather Agent: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWeather Agent stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
