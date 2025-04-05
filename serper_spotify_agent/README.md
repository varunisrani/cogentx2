# Serper-Spotify Combined Agent

This agent combines two powerful capabilities into one:
1. **Web Search** via Serper API
2. **Music Control** via Spotify API

The agent can answer questions about any topic using web search and control your Spotify account to play music, create playlists, and more.

## Features

- Search the web for information, news, and images
- Search for songs, artists, and albums on Spotify
- Create and manage Spotify playlists
- Control music playback (play, pause, skip, etc.)
- Get recommendations and information about your Spotify library
- Seamlessly handle both types of requests in a single conversation

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm
- Serper API key
- Spotify API key
- OpenAI API key

### Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies:
   ```
   npm install @smithery/cli
   ```
4. Create a `.env` file with your API keys:
   ```
   LLM_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   SPOTIFY_API_KEY=your_spotify_api_key
   MODEL_CHOICE=gpt-4o-mini  # optional
   BASE_URL=https://api.openai.com/v1  # optional
   ```

## Usage

Run the agent with:

```
python main.py
```

Add the `--verbose` flag for more detailed logging:

```
python main.py --verbose
```

### Example Queries

#### Web Search Queries
- "What's the latest news about artificial intelligence?"
- "Find information about climate change from scientific sources"
- "Show me images of the Golden Gate Bridge"
- "What are the ingredients in a classic tiramisu?"

#### Spotify Queries
- "Play songs by Taylor Swift"
- "Create a playlist with relaxing music"
- "What are my top playlists?"
- "Skip to the next track"
- "Add this song to my favorites"

#### Mixed Queries
- "Play music by The Beatles and tell me about their history"
- "What are trending songs right now and find news about the music industry"

## Troubleshooting

If you encounter issues:

1. Make sure all API keys are valid and have the necessary permissions
2. Check that Node.js and npm are properly installed
3. Review the log file for detailed error messages
4. For Spotify queries, ensure you have an active Spotify account
5. Check that you're connected to the internet for web search queries
6. If connection issues occur, try running `npx @smithery/cli@latest` to ensure it's properly installed

## Customization

You can modify the system prompt in `agent.py` to adjust how the agent responds or to add special instructions.

## License

MIT 