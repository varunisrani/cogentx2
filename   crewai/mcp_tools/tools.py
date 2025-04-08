from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import Field
import os
import requests
import json
import asyncio
import mcp
from mcp.client.websocket import websocket_client
try:
    import smithery
except ImportError:
    print("Smithery package not found. Some tools might not work properly.")

# CrewAI-compatible tools collection


```python
import os
import json
import asyncio
from typing import Optional, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from smithery.url import create_smithery_url
import mcp
from mcp.client.websocket import websocket_client

# Load environment variables
load_dotenv()

class SpotifySearchParams(BaseModel):
    """Parameters for searching tracks on Spotify."""
    query: str = Field(..., description="The search query string")
    type: str = Field("track", description="The type of items to search for: album, artist, playlist, track")
    limit: int = Field(10, description="The maximum number of results to return (1-50)")
    market: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code to limit results")

class SpotifyRecommendationParams(BaseModel):
    """Parameters for getting track recommendations on Spotify."""
    seed_artists: Optional[str] = Field(None, description="Comma-separated list of artist IDs")
    seed_tracks: Optional[str] = Field(None, description="Comma-separated list of track IDs")
    seed_genres: Optional[str] = Field(None, description="Comma-separated list of genre names")
    limit: int = Field(10, description="The maximum number of recommendations to return (1-100)")

class SpotifyMCPTool(BaseTool):
    """
    A CrewAI tool for interacting with the Spotify API via MCP.
    
    This tool allows searching for tracks and getting recommendations based on user preferences.
    """
    
    name: str = "Spotify MCP Tool"
    description: str = "Search for songs and provide recommendations based on user preferences or specific tracks."
    client_id: str = Field(..., description="Spotify Client ID")
    client_secret: str = Field(..., description="Spotify Client Secret")
    url: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.client_id = self.client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = self.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        
        self.url = create_smithery_url(
            "wss://server.smithery.ai/@superseoworld/mcp-spotify/ws", 
            {
                "spotifyClientId": self.client_id,
                "spotifyClientSecret": self.client_secret
            }
        )
    
    async def _arun(self, query: str, limit: int = 10) -> str:
        """
        Asynchronously search for tracks on Spotify.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            
        Returns:
            A string with formatted search results.
        """
        params = SpotifySearchParams(query=query, limit=limit)
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("search", params.dict())
                if hasattr(result, 'content') and result.content:
                    response_data = json.loads(result.content[0].text)
                    return self._format_search_results(response_data)
                return "No search results found."
    
    def _run(self, query: str, limit: int = 10) -> str:
        """
        Synchronously search for tracks on Spotify.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            
        Returns:
            A string with formatted search results.
        """
        return asyncio.run(self._arun(query, limit))
    
    async def _get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> str:
        """
        Asynchronously get track recommendations based on seed tracks.
        
        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).
            
        Returns:
            A string with formatted recommendation results.
        """
        params = SpotifyRecommendationParams(seed_tracks=seed_tracks, limit=limit)
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("recommendations", params.dict())
                if hasattr(result, 'content') and result.content:
                    response_data = json.loads(result.content[0].text)
                    return self._format_recommendation_results(response_data)
                return "No recommendations found."
    
    def get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> str:
        """
        Synchronously get track recommendations based on seed tracks.
        
        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).
            
        Returns:
            A string with formatted recommendation results.
        """
        return asyncio.run(self._get_recommendations(seed_tracks, limit))
    
    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format the search results into a readable string."""
        if 'tracks' in results and 'items' in results['tracks']:
            tracks = results['tracks']['items']
            formatted_results = "# Spotify Search Results\n\n"
            for i, track in enumerate(tracks, 1):
                artists = ", ".join([artist['name'] for artist in track['artists']])
                album = track['album']['name']
                track_id = track['id']
                external_url = track['external_urls']['spotify'] if 'external_urls' in track and 'spotify' in track['external_urls'] else "N/A"
                
                formatted_results += f"## {i}. {track['name']}\n"
                formatted_results += f"- **Artist(s)**: {artists}\n"
                formatted_results += f"- **Album**: {album}\n"
                formatted_results += f"- **Track ID**: {track_id}\n"
                formatted_results += f"- **URL**: {external_url}\n\n"
            return formatted_results
        return "No tracks found matching your query."
    
    def _format_recommendation_results(self, results: Dict[str, Any]) -> str:
        """Format the recommendation results into a readable string."""
        if 'tracks' in results:
            tracks = results['tracks']
            formatted_results = "# Spotify Recommendations\n\n"
            for i, track in enumerate(tracks, 1):
                artists = ", ".join([artist['name'] for artist in track['artists']])
                album = track['album']['name']
                track_id = track['id']
                external_url = track['external_urls']['spotify'] if 'external_urls' in track and 'spotify' in track['external_urls'] else "N/A"
                
                formatted_results += f"## {i}. {track['name']}\n"
                formatted_results += f"- **Artist(s)**: {artists}\n"
                formatted_results += f"- **Album**: {album}\n"
                formatted_results += f"- **Track ID**: {track_id}\n"
                formatted_results += f"- **URL**: {external_url}\n\n"
            return formatted_results
        return "No recommendations found for the provided seeds."
```
