# Spotify MCP Agent

A powerful agent for interacting with Spotify API using the Model Context Protocol (MCP) and Pydantic AI.

## Features

- 🎵 Search for songs, albums, and artists
- 🎧 Control playback (play, pause, skip tracks)
- 📋 Create and manage playlists
- 👤 Get user profile information
- 🔍 Get recommendations based on your music taste

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies:
   ```
   cd spotify_agent
   npm install
   ```

## Configuration

Create a `.env` file in the root directory with the following content:

```
LLM_API_KEY=your_openai_api_key
SPOTIFY_API_KEY=your_spotify_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
```

### Obtaining a Spotify API Key

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Create a new application
3. Note your Client ID and Client Secret
4. Use these values to authenticate and obtain an API key

## Usage

Run the Spotify agent:

```
python -m spotify_agent.main
```

You can use the following command-line options:
- `--verbose` or `-v`: Enable verbose logging
- `--log-file [PATH]`: Specify a custom log file path (default: spotify_agent.log)

## Example Queries

- "Search for songs by Taylor Swift"
- "Play the album Thriller by Michael Jackson"
- "Create a playlist called Summer Hits"
- "What are my top playlists?"
- "Skip to the next track"

## Troubleshooting

If you encounter issues:

1. Make sure your Spotify API key has the necessary permissions
2. Run `npm install` to make sure all dependencies are installed
3. Check the log file for detailed error messages
4. Ensure you have Node.js installed (version 16+)

## License

This project is licensed under the ISC License. 