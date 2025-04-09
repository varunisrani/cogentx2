from crewai import Task, Agent
from typing import Dict, Any, Optional, List, Tuple
import re

class SpotifyTaskType:
    """Enum for Spotify task types"""
    TRACK_SEARCH = "track_search"
    SONG_RECOMMENDATION = "song_recommendation"
    PLAYLIST_CREATION = "playlist_creation"

class SpotifyTaskFactory:
    """Factory class to create Spotify-specific tasks"""
    
    @staticmethod
    def create_task(query: str, agent: Agent, human_input: bool = False, context: Optional[List[Tuple[str, Any]]] = None) -> Task:
        """Create appropriate Spotify task based on user query"""
        # Default context if none provided
        if context is None:
            context = []
            
        # Determine task category
        task_type = SpotifyTaskFactory.determine_task_type(query)
        
        # Create appropriate task
        if task_type == SpotifyTaskType.TRACK_SEARCH:
            return SpotifyTaskFactory.create_track_search_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.SONG_RECOMMENDATION:
            return SpotifyTaskFactory.create_song_recommendation_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.PLAYLIST_CREATION:
            return SpotifyTaskFactory.create_playlist_creation_task(query, agent, human_input, context)
    
    @staticmethod
    def determine_task_type(query: str) -> SpotifyTaskType:
        """Determine the type of Spotify task based on the query"""
        # Track search keywords
        track_search_keywords = [
            "search", "find", "look for", "track", "song", "album", "artist", 
            "who sings", "who performs"
        ]
        
        # Recommendation keywords
        recommendation_keywords = [
            "recommend", "recommendation", "suggest", "similar to", "like",
            "similar artists", "more like", "discover"
        ]
        
        # Playlist keywords
        playlist_keywords = [
            "playlist", "playlists", "collection", "mix", "workout playlist", 
            "mood", "activity"
        ]
        
        # Check for matches
        query_lower = query.lower()
        
        for keyword in track_search_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.TRACK_SEARCH
        
        for keyword in recommendation_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.SONG_RECOMMENDATION
        
        for keyword in playlist_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.PLAYLIST_CREATION
        
        # Default to no task match
        return None
    
    @staticmethod
    def create_track_search_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a track search task"""
        search_term = SpotifyTaskFactory._extract_search_term(query)
        
        # Prepare task description
        description = f"""
        You are tasked with assisting the user in searching for tracks on Spotify.
        
        USER REQUEST: {query}
        
        Search details:
        - Search term: {search_term if search_term else 'Not clearly specified - interpret from context'}
        
        Steps to follow:
        1. Analyze the user's request to identify the exact track they are searching for.
        2. Execute a search using the Spotify API with the identified search term.
        3. Present the search results, including artist names, albums, and track popularity.
        """
        
        return Task(
            description=description,
            expected_output="A comprehensive list of search results with relevant track details.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_song_recommendation_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a song recommendation task"""
        description = f"""
        You are tasked with generating personalized song recommendations for the user based on their query.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Analyze the request to understand the type of recommendations the user needs.
        2. Identify any seed artists, tracks, or genres mentioned in the query.
        3. Use the Spotify recommendation API to find tracks aligned with the user's preferences.
        4. Present the recommendations along with details on why they match the user's interests.
        """
        
        return Task(
            description=description,
            expected_output="A tailored list of song recommendations with explanations of their relevance.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_playlist_creation_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a playlist-related task"""
        description = f"""
        You are tasked with assisting the user in creating or discovering playlists on Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Determine the user's intent regarding playlists (creation, discovery, style, etc.).
        2. Search for existing playlists or suggest track collections based on user criteria.
        3. Present the playlists with relevant details such as title, creator, and track count.
        4. Provide brief descriptions of the playlists' themes or moods.
        """
        
        return Task(
            description=description,
            expected_output="A curated list of playlists that match the user's request with descriptions.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def _extract_search_term(query: str) -> str:
        """Extract potential search term from the query"""
        # Define patterns to capture search terms
        patterns = [
            r'(?:search|find|looking for)\s+["\']?([^"\']+)["\']?',
            r'(?:song|track|album|artist)\s+["\']?([^"\']+)["\']?',
            r'who (?:sings|performs)\s+["\']?([^"\']+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return empty string if no match is found
        return ""