# Tools for CrewAI project
# Generated based on request: Create a CrewAI project for a Spotify playlist generator that analyzes my listening habits and creat...
from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import Field
import os
import requests
import json

class SpotifyAPI:
    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, access_token: str):
        self.access_token = access_token

    def get_current_user_top_tracks(self, limit: int = 20) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        response = requests.get(f"{self.BASE_URL}/me/top/tracks?limit={limit}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching top tracks: {response.status_code} - {response.text}")
        return response.json().get('items', [])

    def create_playlist(self, user_id: str, name: str, description: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "name": name,
            "description": description,
            "public": False
        }
        response = requests.post(f"{self.BASE_URL}/users/{user_id}/playlists", headers=headers, json=data)
        if response.status_code != 201:
            raise Exception(f"Error creating playlist: {response.status_code} - {response.text}")
        return response.json()

    def add_tracks_to_playlist(self, playlist_id: str, track_uris: List[str]) -> None:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "uris": track_uris
        }
        response = requests.post(f"{self.BASE_URL}/playlists/{playlist_id}/tracks", headers=headers, json=data)
        if response.status_code != 201:
            raise Exception(f"Error adding tracks to playlist: {response.status_code} - {response.text}")

class GetTopTracksTool(BaseTool):
    name: str = Field(default="GetTopTracks", description="Fetches the user's top tracks from Spotify.")
    access_token: str

    def _run(self) -> List[Dict[str, Any]]:
        try:
            spotify_api = SpotifyAPI(self.access_token)
            top_tracks = spotify_api.get_current_user_top_tracks()
            return top_tracks
        except Exception as e:
            return {"error": str(e)}

class CreatePlaylistTool(BaseTool):
    name: str = Field(default="CreatePlaylist", description="Creates a new playlist for the user on Spotify.")
    access_token: str
    user_id: str
    playlist_name: str
    playlist_description: str

    def _run(self) -> Dict[str, Any]:
        try:
            spotify_api = SpotifyAPI(self.access_token)
            playlist = spotify_api.create_playlist(self.user_id, self.playlist_name, self.playlist_description)
            return playlist
        except Exception as e:
            return {"error": str(e)}

class AddTracksToPlaylistTool(BaseTool):
    name: str = Field(default="AddTracksToPlaylist", description="Adds tracks to an existing Spotify playlist.")
    access_token: str
    playlist_id: str
    track_uris: List[str]

    def _run(self) -> Dict[str, Any]:
        try:
            spotify_api = SpotifyAPI(self.access_token)
            spotify_api.add_tracks_to_playlist(self.playlist_id, self.track_uris)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}