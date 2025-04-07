# Spotify Streamlit App Instructions

This document provides instructions on how to run the Spotify Streamlit app.

## Prerequisites

1. Make sure you have Python 3.8 or higher installed
2. Make sure you have Node.js and npm installed
3. Make sure you have the required Python packages installed:
   ```
   pip install streamlit nest-asyncio pydantic-ai
   ```

## Running the App

There are two ways to run the Spotify Streamlit app:

### Option 1: Using the launcher script

```bash
python run_spotify_agent.py --mode streamlit
```

### Option 2: Running Streamlit directly

```bash
streamlit run spotify_streamlit_app.py
```

## Troubleshooting

If you encounter the error "No module named 'models'", it means there's an issue with the import paths. Here are some things to try:

1. Make sure you're running the app from the root directory of the project
2. Check that the spotify_wrapper.py file is in the same directory as spotify_streamlit_app.py
3. Make sure the spotify_agent directory is in the same directory as spotify_streamlit_app.py

If you encounter other issues:

1. Check the logs in the "Logs" page of the app
2. Make sure your .env file has the required environment variables:
   - LLM_API_KEY (your OpenAI API key)
   - SPOTIFY_API_KEY (your Spotify API key)

## Using the App

1. Initialize the Spotify Agent using the button in the sidebar
2. Enter your query in the text area on the home page
3. Click "Run Query" to process your query
4. View the results below
5. Check the "Query History" page to see your previous queries
6. Check the "Logs" page to see detailed logs

## Example Queries

- "Find the top 5 songs by Taylor Swift"
- "Create a playlist of relaxing jazz music"
- "What are the most popular songs in the US right now?"
- "Recommend some songs similar to 'Bohemian Rhapsody'"
- "Tell me about the album 'Thriller' by Michael Jackson"
