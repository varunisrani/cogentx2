```markdown
# smithery_spotify_client

## Project Overview

The `smithery_spotify_client` is an MCP tool designed to facilitate interaction with the Spotify API through a structured interface. It allows users to perform various operations such as searching for music, retrieving recommendations, and managing playlists. By utilizing the Model Context Protocol (MCP) system on Smithery, this tool provides a streamlined method for accessing Spotify's functionalities, making it easier for developers to integrate music-related features into their applications.

## Installation Instructions

To get started with the `smithery_spotify_client`, follow these installation instructions:

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or later
- pip (Python package installer)

### Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Setup Instructions

### Spotify API Setup

This project requires Spotify API credentials. Follow these steps to set them up:

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Create a new application
4. Once created, you'll get a Client ID and Client Secret
5. Add these to your `.env` file:

   ```plaintext
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

   **IMPORTANT:** Never commit your `.env` file to version control!

### Running the Project

To run the project, execute the following command:

```bash
python main.py
```

## Usage Examples

Here are some basic usage examples demonstrating how to interact with the Spotify API through the `smithery_spotify_client`:

### Searching for Music

```python
from crew import SpotifyClient

client = SpotifyClient()
results = client.search_music("Imagine Dragons")
print(results)
```

### Getting Recommendations

```python
recommendations = client.get_recommendations(seed_artists=["artist_id"], seed_genres=["genre"])
print(recommendations)
```

### Managing Playlists

```python
playlist = client.create_playlist("My Favorite Songs")
client.add_song_to_playlist(playlist_id=playlist['id'], song_id="song_id")
```

## Project Structure

The project is organized as follows:

```
smithery_spotify_client/
│
├── agents.py          # Contains the agents for interacting with the Spotify API
├── tasks.py           # Defines tasks for various operations
├── crew.py            # Main crew logic and MCP integration
├── main.py            # Entry point for the application
└── run_agent.py       # Script to run the agent
```

## Troubleshooting Tips

- **Invalid API Credentials**: Ensure your `.env` file contains the correct `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.
  
- **Dependencies Issues**: If you encounter issues with dependencies, double-check your Python version and ensure all packages are correctly installed.

- **Network Errors**: Check your internet connection and ensure that the Spotify API is not experiencing downtime.

If you encounter any other issues, please refer to the official [Spotify API documentation](https://developer.spotify.com/documentation/) for more information.
```
