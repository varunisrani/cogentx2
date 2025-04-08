#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to the project root directory
cd "$DIR/../.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run archon/mcp_tools/streamlit_app.py 