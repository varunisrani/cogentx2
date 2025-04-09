#!/usr/bin/env python3
import os
import shutil
import sys

def copy_tools():
    """
    Copy the generated Spotify MCP tool files to the current directory.
    """
    source_dir = "/Users/varunisrani/cogentx_mcp_output/spotify_agent"
    target_dir = os.getcwd()
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return False
    
    try:
        # Copy the tools.py file
        shutil.copy2(os.path.join(source_dir, "tools.py"), os.path.join(target_dir, "tools.py"))
        print(f"Successfully copied tools.py to {target_dir}")
        
        # Copy any other relevant files
        for file in os.listdir(source_dir):
            if file.endswith(".json"):
                shutil.copy2(os.path.join(source_dir, file), os.path.join(target_dir, file))
                print(f"Successfully copied {file} to {target_dir}")
        
        return True
    except Exception as e:
        print(f"Error copying files: {e}")
        return False

if __name__ == "__main__":
    if copy_tools():
        print("All Spotify MCP tool files have been copied successfully!")
        print("You can now run spotify_agent_example.py to test the agent.")
    else:
        print("Failed to copy one or more files.")
        sys.exit(1) 