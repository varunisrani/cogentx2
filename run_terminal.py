#!/usr/bin/env python3
"""
Terminal Launcher Script

This script allows users to choose which terminal implementation to run:
1. Streamlit-Terminal - Simple component-based terminal
2. Archon Terminal - Full-featured terminal implementation
3. Terminal Comparison - Side-by-side comparison of both terminals
"""

import argparse
import os
import subprocess
import sys

def check_requirements():
    """Check if the required packages are installed."""
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    try:
        import streamlit_terminal
    except ImportError:
        print("streamlit-terminal is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit-terminal"])
    
    try:
        import streamlit_ace
    except ImportError:
        print("streamlit-ace is not installed (optional). Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit-ace"])
    
    try:
        from rich.console import Console
    except ImportError:
        print("rich is not installed (optional). Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "rich"])

def run_terminal(terminal_type):
    """
    Run the specified terminal implementation.
    
    Args:
        terminal_type (str): The type of terminal to run ('basic', 'archon', or 'compare')
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if terminal_type == 'basic':
        # Create a simple app for streamlit-terminal if it doesn't exist
        basic_app_path = os.path.join(script_dir, "basic_terminal.py")
        if not os.path.exists(basic_app_path):
            with open(basic_app_path, "w") as f:
                f.write("""
import streamlit as st
from streamlit_terminal import st_terminal

st.set_page_config(page_title="Basic Terminal", layout="wide")

st.title("Basic Terminal Implementation")
st.write("A simple terminal interface using the streamlit-terminal component.")

# Terminal configuration
st.subheader("Terminal Settings")
show_welcome = st.checkbox("Show Welcome Message", value=True)
term_height = st.slider("Terminal Height", min_value=200, max_value=800, value=400, step=50)
custom_cmd = st.text_input("Custom Command (optional)", 
                          placeholder="Leave empty for interactive terminal or enter a command to run")

# Create the terminal
st.subheader("Terminal")
full_outputs, updated_outputs = st_terminal(
    key="terminal",
    command=custom_cmd if custom_cmd else "",
    height=term_height,
    show_welcome_message=show_welcome
)

# Show outputs
with st.expander("Terminal Outputs"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Full Outputs")
        st.write(full_outputs)
    with col2:
        st.subheader("Updated Outputs")
        st.write(updated_outputs)
""")
        subprocess.run(["streamlit", "run", basic_app_path])
    
    elif terminal_type == 'archon':
        # Run the Archon terminal implementation
        archon_app_path = os.path.join(script_dir, "terminal_app.py")
        subprocess.run(["streamlit", "run", archon_app_path])
    
    elif terminal_type == 'compare':
        # Run the comparison app
        compare_app_path = os.path.join(script_dir, "terminal_comparison.py")
        subprocess.run(["streamlit", "run", compare_app_path])
    
    else:
        print(f"Unknown terminal type: {terminal_type}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the appropriate terminal."""
    parser = argparse.ArgumentParser(description="Run different terminal implementations")
    
    parser.add_argument(
        "terminal_type",
        nargs="?",
        default="compare",
        choices=["basic", "archon", "compare"],
        help="Type of terminal to run (basic, archon, or compare)"
    )
    
    args = parser.parse_args()
    
    # Check and install requirements
    check_requirements()
    
    # Run the selected terminal
    run_terminal(args.terminal_type)

if __name__ == "__main__":
    main() 