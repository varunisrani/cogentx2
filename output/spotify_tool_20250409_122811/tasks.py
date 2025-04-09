from crewai import Task, Agent
from typing import Dict, Any, Optional, List, Tuple
import re

class SpotifyTaskType:
    """Enum for Spotify task types"""
    MUSIC_SEARCH = "music_search"
    RECOMMENDATION = "recommendation"
    PLAYLIST_CREATION = "playlist_creation"

class SpotifyTaskFactory:
    """Factory class to create Spotify-related tasks specifically for music recommendations and playlists"""
    
    @staticmethod
    def create_task(query: str, agent: Agent, human_input: bool = False, context: Optional[List[Tuple[str, Any]]] = None) -> Task:
        """Create appropriate task based on the user query for Spotify music recommendations and playlists"""
        if context is None:
            context = []
        
        task_type = SpotifyTaskFactory.determine_task_type(query)
        
        if task_type == SpotifyTaskType.MUSIC_SEARCH:
            return SpotifyTaskFactory.create_music_search_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.RECOMMENDATION:
            return SpotifyTaskFactory.create_recommendation_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.PLAYLIST_CREATION:
            return SpotifyTaskFactory.create_playlist_creation_task(query, agent, human_input, context)
        
        raise ValueError("Invalid task type")

    @staticmethod
    def determine_task_type(query: str) -> SpotifyTaskType:
        """Determine the type of Spotify task based on the user query"""
        search_keywords = ["search", "find", "look for", "track", "song", "album", "artist"]
        recommendation_keywords = ["recommend", "recommendation", "suggest", "similar to"]
        playlist_keywords = ["playlist", "create playlist", "mood", "activity"]

        query_lower = query.lower()
        
        for keyword in search_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.MUSIC_SEARCH
        
        for keyword in recommendation_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.RECOMMENDATION
        
        for keyword in playlist_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.PLAYLIST_CREATION
        
        raise ValueError("Unrecognized task type in query")

    @staticmethod
    def create_music_search_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a task to search for music on Spotify"""
        search_term = SpotifyTaskFactory._extract_search_term(query)
        
        description = f"""
        You are tasked with assisting the user in searching for music on Spotify.
        
        USER REQUEST: {query}
        
        Search details:
        - Search term: {search_term if search_term else 'Not clearly specified'}
        
        Steps to follow:
        1. Analyze the request to determine the specific music the user is searching for.
        2. Execute the search using the Spotify API.
        3. Present the search results clearly, including details like artist names and album titles.
        """
        
        return Task(
            description=description,
            expected_output="A list of search results including track names, artist details, and album titles.",
            agent=agent,
            human_input=human_input,
            context=context
        )

    @staticmethod
    def create_recommendation_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a task to provide music recommendations based on user interests"""
        description = f"""
        You are tasked with providing personalized music recommendations from Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Identify the user's preferences based on the request.
        2. Use the Spotify API to fetch recommendations based on mentioned artists or genres.
        3. Present the recommendations with explanations of why they fit the user's taste.
        """
        
        return Task(
            description=description,
            expected_output="A curated list of music recommendations with explanations regarding user preferences.",
            agent=agent,
            human_input=human_input,
            context=context
        )

    @staticmethod
    def create_playlist_creation_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a task to assist in playlist creation on Spotify"""
        description = f"""
        You are tasked with helping the user create or find a playlist on Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Determine the user's desired mood or activity for the playlist.
        2. Search for relevant playlists that match the user's criteria.
        3. Present the playlists including names, number of tracks, and brief descriptions.
        """
        
        return Task(
            description=description,
            expected_output="A list of playlists tailored to the user's request with relevant details.",
            agent=agent,
            human_input=human_input,
            context=context
        )

    @staticmethod
    def _extract_search_term(query: str) -> str:
        """Extract the search term from the user's query"""
        patterns = [
            r'(?:search|find|looking for)\s+["\']?([^"\']+)["\']?',
            r'(?:song|track|album|artist)\s+["\']?([^"\']+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""