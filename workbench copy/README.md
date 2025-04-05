# spotify_github_agent Agent

## Overview
This is a merged agent that combines functionality from the following templates:
spotify_agent, github_agent

Created at: 2025-04-05T11:13:10.276827

## Features
- **Spotify Integration**: Search for music, manage playlists, control playback
- **GitHub Integration**: Manage repositories, issues, pull requests

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js (for MCP servers)
- API keys for required services

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` file to add your actual API keys

### Running the Agent
```bash
python main.py
```

## Available Commands
- **Spotify Commands**: Search tracks, create playlists, etc.
- **GitHub Commands**: List repositories, create issues, etc.
- **Combined Operations**: Perform operations across both services

## Troubleshooting
If you encounter issues:
- Check your API keys in the .env file
- Ensure all dependencies are installed
- Check logs for specific error messages
