# YouTube Transcript Agent

A Python agent for extracting and analyzing YouTube video transcripts powered by pydantic-ai and MCP.

## Features

- Extract full transcripts from YouTube videos
- Search within transcripts for specific content or keywords
- Generate summaries of video content
- Identify key points and topics in videos
- Interactive CLI interface for querying transcripts

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies:
   ```
   npm install
   ```

## Configuration

Create a `.env` file in the root directory with the following content:

```
LLM_API_KEY=your_openai_api_key
YOUTUBE_API_KEY=your_youtube_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
```

- `LLM_API_KEY`: Your OpenAI API key
- `YOUTUBE_API_KEY`: Your YouTube API key for transcript access
- `MODEL_CHOICE`: (Optional) The LLM model to use (default: gpt-4o-mini)
- `BASE_URL`: (Optional) Base URL for the LLM provider API

## Usage

Run the agent with:

```
python main.py
```

Use the `--verbose` flag for more detailed logging:

```
python main.py --verbose
```

## Example Queries

Once the agent is running, you can enter queries such as:

1. "Get the transcript for this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
2. "Summarize the content of this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
3. "What are the main topics discussed in this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
4. "Find all mentions of 'artificial intelligence' in this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"

## Troubleshooting

If you encounter issues:

1. Ensure your YouTube API key is valid and has the necessary permissions
2. Check that the target YouTube video has captions/transcripts available
3. Verify that you have Node.js installed (version 16+)
4. Check the log file for detailed error messages

## Dependencies

- Python 3.8+
- Node.js 16+
- npm 7+
- pydantic-ai
- colorlog
- python-dotenv
- @smithery/cli (NPM)
- @kimtaeyoon83/mcp-server-youtube-transcript (NPM)

## License

MIT 