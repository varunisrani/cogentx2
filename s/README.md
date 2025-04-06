# Spotify Github

This agent was adapted from the MCP template "spotify_github_agent".

## Description

An AI agent that Create agent that can recommend me songs from Spotify and also create repository and manage repository from Github in this use github_agent and spotify_agent.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your API keys
3. Run the agent: `python main.py`

## Features

- Uses Pydantic AI for robust agent capabilities
- Includes MCP server integrations for external services
- Customized based on the user's request

## Available Files

- agents.py
- models.py
- main.py
- tools.py
- mcp.json

## MCP Servers

- mcp-spotify
- github
