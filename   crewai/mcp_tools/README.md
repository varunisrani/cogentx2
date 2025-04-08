# Spotify MCP Agent

This agent uses the Spotify API through Model Context Protocol (MCP) to search for songs and provide music recommendations.

## Features

- Search for songs, artists, albums, or playlists on Spotify
- Get personalized track recommendations based on seed tracks
- Detailed information about tracks including artists, album, and Spotify links

## Requirements

- Python 3.8+
- Spotify Developer account and API credentials
- CrewAI
- Smithery and MCP packages

## Setup

1. **Create a Spotify Developer Account**:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
   - Create a new application to get your `Client ID` and `Client Secret`

2. **Environment Variables**:
   Create a `.env` file with the following:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

3. **Install Dependencies**:
   ```bash
   pip install crewai dotenv smithery mcp pydantic
   ```

## Usage

The example script `spotify_agent_example.py` demonstrates how to:

1. Initialize the Spotify MCP Tool
2. Create a CrewAI agent for music discovery
3. Define tasks for searching songs and getting recommendations
4. Set up a crew to execute these tasks

### Basic Usage

```python
from tools import SpotifyMCPTool

# Initialize tool with your Spotify credentials
spotify_tool = SpotifyMCPTool(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Search for tracks
results = spotify_tool._run(query="Coldplay Yellow", limit=5)
print(results)

# Get recommendations based on a track ID
recommendations = spotify_tool.get_recommendations(
    seed_tracks="3AJwUDP919kvQ9QcozQPxg", 
    limit=5
)
print(recommendations)
```

## Example Tasks

1. **Search for Songs by Artist**:
   ```
   "query": "Taylor Swift"
   ```

2. **Search for a Specific Track**:
   ```
   "query": "Bohemian Rhapsody Queen"
   ```

3. **Get Recommendations Based on a Track**:
   First search for a track to get its ID, then use that ID as a seed track:
   ```
   "seed_tracks": "3AJwUDP919kvQ9QcozQPxg"
   ```

## Limitations

- Requires valid Spotify API credentials
- Subject to Spotify API rate limits
- The tool uses MCP, which requires an internet connection

## Troubleshooting

- If you get authentication errors, verify your Spotify credentials
- Make sure your Spotify Developer application has the correct redirect URIs set
- Check that all required packages are installed 