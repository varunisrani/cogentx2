# Spotify Agent Streamlit Interface

This is a dedicated Streamlit interface for the Spotify Agent, providing a user-friendly way to interact with Spotify using natural language queries.

## Features

- **Native Streamlit UI**: Fully integrated with Streamlit's UI components for a seamless experience
- **Query History**: Keep track of all your previous queries and their results
- **Tool Usage Tracking**: See which tools were used to process your queries
- **Logging**: Comprehensive logging system with filtering options
- **Spotify-Inspired Design**: Custom CSS styling that matches Spotify's brand identity

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js and npm
- Spotify API credentials

### Installation

1. Make sure you have all the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Set up your Spotify API credentials in the appropriate configuration file.

3. Run the Streamlit app:
   ```
   streamlit run spotify_streamlit_app.py
   ```

## Usage

1. **Initialize the Agent**: Click the "Initialize Spotify Agent" button in the sidebar to start the agent.

2. **Enter a Query**: Type your query in the text area on the home page. Examples:
   - "Find the top 5 songs by Taylor Swift"
   - "Create a playlist of relaxing jazz music"
   - "What are the most popular songs in the US right now?"

3. **Run the Query**: Click the "Run Query" button to process your query.

4. **View Results**: The results will be displayed below the query input area.

5. **Explore History**: Navigate to the "Query History" page to see all your previous queries and their results.

6. **Check Logs**: Go to the "Logs" page to view detailed logs of the agent's operations.

## Pages

### Home
The main page where you can enter and run queries, and view the current results.

### Query History
A record of all queries you've run, along with their results, tool usage, and elapsed time.

### Logs
Detailed logs of the agent's operations, with filtering options.

### About
Information about the Spotify Agent and how it works.

## How It Works

The Spotify Agent Streamlit Interface uses:

1. **Streamlit**: For the web interface
2. **Pydantic AI**: For the agent framework
3. **Model Context Protocol (MCP)**: For communication between the agent and Spotify
4. **Asyncio**: For handling asynchronous operations
5. **Spotify Web API**: For accessing Spotify data

The app initializes the Spotify Agent, which connects to the Spotify API through MCP. When you enter a query, it's processed by the agent, which uses a large language model to understand your request and translate it into API calls. The results are then displayed in the Streamlit interface.

## Troubleshooting

If you encounter issues:

1. **Agent Initialization Fails**: Make sure Node.js and npm are installed and that your Spotify API credentials are correct.

2. **Queries Not Working**: Check the logs for detailed error messages. Common issues include API rate limits or permission problems.

3. **UI Issues**: If the UI doesn't look right, make sure the CSS file is in the correct location.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
