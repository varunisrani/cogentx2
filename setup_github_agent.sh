#!/bin/bash
# Setup script for GitHub MCP Agent

echo "Setting up GitHub MCP Agent..."

# Make sure npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Install required npm packages
echo "Installing required npm packages..."
npm install @modelcontextprotocol/server-github

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error installing npm packages. Please check the error messages above."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r github_agent/requirements.txt

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error installing Python dependencies. Please check the error messages above."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << EOL
# GitHub MCP Agent Environment Variables
LLM_API_KEY=your_openai_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_personal_access_token
MODEL_CHOICE=gpt-4o-mini
BASE_URL=https://api.openai.com/v1
EOL
    echo ".env file created. Please edit it with your API keys."
fi

echo "Setup complete! To run the agent, use: python -m github_agent.main"
echo "Make sure to update your .env file with your API keys if you haven't already." 