"""
Wrapper module for Spotify agent to handle imports correctly
"""

import sys
import os
import importlib.util
import logging

# Get the absolute path to the spotify_agent directory
spotify_agent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spotify_agent')

# Add the spotify_agent directory to the Python path
if spotify_agent_dir not in sys.path:
    sys.path.insert(0, spotify_agent_dir)

# Import the modules we need
def import_module(module_name):
    """Import a module from the spotify_agent directory"""
    module_path = os.path.join(spotify_agent_dir, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the modules
try:
    models_module = import_module("models")
    agent_module = import_module("agent")
    tools_module = import_module("tools")
    
    # Export the functions we need
    load_config = models_module.load_config
    setup_agent = agent_module.setup_agent
    run_spotify_query = tools_module.run_spotify_query
    
    # Export the Config class
    Config = models_module.Config
    
    logging.info("Successfully imported Spotify agent modules")
except Exception as e:
    logging.error(f"Error importing Spotify agent modules: {e}")
    raise
