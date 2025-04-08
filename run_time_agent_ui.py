#!/usr/bin/env python3
"""
Run Time Agent UI

This script launches the Time Agent with Streamlit UI.
"""

import os
import sys
import subprocess

def main():
    """Run the Time Agent UI with Streamlit"""
    # Check if the time_agent directory exists
    if not os.path.isdir("time_agent"):
        print("Error: time_agent directory not found.")
        sys.exit(1)
    
    # Check if the mcp_server_time.py file exists
    if not os.path.isfile("mcp_server_time.py"):
        print("Error: mcp_server_time.py file not found.")
        sys.exit(1)
    
    # Run the Time Agent with Streamlit
    try:
        subprocess.run(["streamlit", "run", "time_agent/main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Time Agent UI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTime Agent UI stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
