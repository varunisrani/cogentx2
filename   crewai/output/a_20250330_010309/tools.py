from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import Field
import os
import requests
import json
import asyncio
import time
import re
try:
    import mcp
    from mcp.client.websocket import websocket_client
except ImportError:
    print("MCP package not found. Installing required dependencies may be needed.")
try:
    import smithery
except ImportError:
    print("Smithery package not found. Some tools might not work properly.")


# Additional imports from original tools
# CrewAI-compatible tools collection



# Helper functions
    def __init__(self, **data):
        super().__init__(**data)
        self.url = create_smithery_url(
            "wss://server.smithery.ai/@superseoworld/mcp-spotify/ws",
            {
                "spotifyClientId": self.client_id,
                "spotifyClientSecret": self.client_secret
            }
        )

    async def _arun_search(self, query: str, limit: int = 10, market: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously search for tracks on Spotify.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            market: ISO 3166-1 alpha-2 country code to limit results.

        Returns:
            Dictionary containing search results.
        """
        params = {
            "query": query,
            "type": "track",
            "limit": limit,
            "market": market
        }
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("search", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Asynchronously get track recommendations based on seeds.

        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).

        Returns:
            Dictionary containing recommendation results.
        """
        params = {
            "seed_tracks": seed_tracks,
            "limit": limit
        }
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("recommendations", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_artist_info(self, artist_id: str) -> Dict[str, Any]:
        """
        Asynchronously get information about a specific artist.

        Args:
            artist_id: The Spotify ID of the artist.

        Returns:
            Dictionary containing artist information.
        """
        params = {"id": artist_id}
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("get_artist", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Asynchronously get information about a specific track.

        Args:
            track_id: The Spotify ID of the track.

        Returns:
            Dictionary containing track information.
        """
        params = {"id": track_id}
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("get_track", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    def _run_search(self, query: str, limit: int = 10, market: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for searching tracks on Spotify.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            market: ISO 3166-1 alpha-2 country code to limit results.

        Returns:
            Dictionary containing search results.
        """
        return asyncio.run(self._arun_search(query, limit, market))

    def _run_get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting track recommendations on Spotify.

        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).

        Returns:
            Dictionary containing recommendation results.
        """
        return asyncio.run(self._arun_get_recommendations(seed_tracks, limit))

    def _run_get_artist_info(self, artist_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting artist information.

        Args:
            artist_id: The Spotify ID of the artist.

        Returns:
            Dictionary containing artist information.
        """
        return asyncio.run(self._arun_get_artist_info(artist_id))

    def _run_get_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting track information.

        Args:
            track_id: The Spotify ID of the track.

        Returns:
            Dictionary containing track information.
        """
        return asyncio.run(self._arun_get_track_info(track_id))
class SpotifyMCPTool(BaseTool):
    """
    A CrewAI tool for interacting with the Spotify API via MCP.
    
    This tool allows searching for tracks, getting recommendations, and retrieving information about artists and tracks.
    
    Example usage:
    
    spotify_tool = SpotifyMCPTool()
    search_results = spotify_tool._run_search("Coldplay viva la vida")
    recommendations = spotify_tool._run_get_recommendations(seed_tracks="track_id_here")
    artist_info = spotify_tool._run_get_artist_info("artist_id_here")
    track_info = spotify_tool._run_get_track_info("track_id_here")
    
    """

    name: str = "SpotifyMCPTool"
    description: str = "A tool for interacting with the Spotify API via MCP for music discovery and recommendations."
    client_id: str = Field(..., env="SPOTIFY_CLIENT_ID", description="Spotify Client ID")
    client_secret: str = Field(..., env="SPOTIFY_CLIENT_SECRET", description="Spotify Client Secret")
    url: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.url = create_smithery_url(
            "wss://server.smithery.ai/@superseoworld/mcp-spotify/ws",
            {
                "spotifyClientId": self.client_id,
                "spotifyClientSecret": self.client_secret
            }
        )

    async def _arun_search(self, query: str, limit: int = 10, market: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously search for tracks on Spotify.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            market: ISO 3166-1 alpha-2 country code to limit results.

        Returns:
            Dictionary containing search results.
        """
        params = {
            "query": query,
            "type": "track",
            "limit": limit,
            "market": market
        }
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("search", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Asynchronously get track recommendations based on seeds.

        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).

        Returns:
            Dictionary containing recommendation results.
        """
        params = {
            "seed_tracks": seed_tracks,
            "limit": limit
        }
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("recommendations", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_artist_info(self, artist_id: str) -> Dict[str, Any]:
        """
        Asynchronously get information about a specific artist.

        Args:
            artist_id: The Spotify ID of the artist.

        Returns:
            Dictionary containing artist information.
        """
        params = {"id": artist_id}
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("get_artist", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    async def _arun_get_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Asynchronously get information about a specific track.

        Args:
            track_id: The Spotify ID of the track.

        Returns:
            Dictionary containing track information.
        """
        params = {"id": track_id}
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                result = await session.call_tool("get_track", params)
                if hasattr(result, 'content') and result.content:
                    return json.loads(result.content[0].text)
                return {}

    def _run_search(self, query: str, limit: int = 10, market: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for searching tracks on Spotify.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (1-50).
            market: ISO 3166-1 alpha-2 country code to limit results.

        Returns:
            Dictionary containing search results.
        """
        return asyncio.run(self._arun_search(query, limit, market))

    def _run_get_recommendations(self, seed_tracks: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting track recommendations on Spotify.

        Args:
            seed_tracks: Comma-separated list of track IDs.
            limit: Maximum number of recommendations to return (1-100).

        Returns:
            Dictionary containing recommendation results.
        """
        return asyncio.run(self._arun_get_recommendations(seed_tracks, limit))

    def _run_get_artist_info(self, artist_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting artist information.

        Args:
            artist_id: The Spotify ID of the artist.

        Returns:
            Dictionary containing artist information.
        """
        return asyncio.run(self._arun_get_artist_info(artist_id))

    def _run_get_track_info(self, track_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for getting track information.

        Args:
            track_id: The Spotify ID of the track.

        Returns:
            Dictionary containing track information.
        """
        return asyncio.run(self._arun_get_track_info(track_id))

# Export all tools for use in CrewAI
__all__ = [
    "SpotifyMCPTool",
]
