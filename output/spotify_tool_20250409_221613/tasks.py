from crewai import Task, Agent
from typing import Dict, Any, Optional, List, Tuple
import re

class SpotifyTaskType:
    """Enum for Spotify task types"""
    MUSIC_DISCOVERY = "music_discovery"
    MOOD_BASED_RECOMMENDATIONS = "mood_based_recommendations"
    PLAYLIST_CREATION = "playlist_creation"

class SpotifyTaskFactory:
    """Factory class to create Spotify-related tasks based on user preferences"""
    
    @staticmethod
    def create_task(query: str, agent: Agent, human_input: bool = False, context: Optional[List[Tuple[str, Any]]] = None) -> Task:
        """Create appropriate task based on the user's request for Spotify recommendations"""
        # Default context if none provided
        if context is None:
            context = []
            
        # Determine task category
        task_type = SpotifyTaskFactory.determine_task_type(query)
        
        # Create appropriate task
        if task_type == SpotifyTaskType.MUSIC_DISCOVERY:
            return SpotifyTaskFactory.create_music_discovery_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.MOOD_BASED_RECOMMENDATIONS:
            return SpotifyTaskFactory.create_mood_based_recommendations_task(query, agent, human_input, context)
        elif task_type == SpotifyTaskType.PLAYLIST_CREATION:
            return SpotifyTaskFactory.create_playlist_creation_task(query, agent, human_input, context)
        else:
            return SpotifyTaskFactory.create_general_task(query, agent, human_input, context)
    
    @staticmethod
    def determine_task_type(query: str) -> SpotifyTaskType:
        """Determine the type of Spotify task based on the user query"""
        # Music discovery keywords
        discovery_keywords = [
            "discover", "find", "search for", "track", "song", "album", "artist", 
            "who sings", "who performs", "what album", "music by"
        ]
        
        # Mood-based recommendation keywords
        mood_keywords = [
            "recommend", "suggest", "similar to", "like", "mood", "vibe",
            "chill", "happy", "sad", "energetic", "relax"
        ]
        
        # Playlist creation keywords
        playlist_keywords = [
            "playlist", "create a playlist", "compile", "mix",
            "activity", "workout", "party", "chill"
        ]
        
        # Check for matches
        query_lower = query.lower()
        
        for keyword in discovery_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.MUSIC_DISCOVERY
        
        for keyword in mood_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.MOOD_BASED_RECOMMENDATIONS
        
        for keyword in playlist_keywords:
            if keyword in query_lower:
                return SpotifyTaskType.PLAYLIST_CREATION
        
        # Default to general Spotify task
        return SpotifyTaskType.MUSIC_DISCOVERY
    
    @staticmethod
    def create_music_discovery_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a music discovery task"""
        search_term = SpotifyTaskFactory._extract_search_term(query)
        
        description = f"""
        You are tasked with assisting the user in discovering music on Spotify.
        
        USER REQUEST: {query}
        
        Discovery details:
        - Search term: {search_term if search_term else 'Not clearly specified - interpret from context'}
        
        Steps to follow:
        1. Analyze the request to determine what the user is looking to discover.
        2. Execute a search using the Spotify API for tracks, albums, or artists.
        3. Present the results in a structured format, including track metadata and audio features.
        """
        
        return Task(
            description=description,
            expected_output="A structured list of discovered music with track metadata and audio features.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_mood_based_recommendations_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a mood-based recommendations task"""
        description = f"""
        You are tasked with providing mood-based music recommendations from Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Analyze the user's mood or activity as indicated in their request.
        2. Use the Spotify API to generate recommendations based on mood or similar tracks.
        3. Present the recommendations with an explanation of why they fit the user's mood.
        """
        
        return Task(
            description=description,
            expected_output="A tailored list of mood-based music recommendations with explanations.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_playlist_creation_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a playlist-related task"""
        description = f"""
        You are tasked with assisting the user in creating or finding playlists on Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Determine if the user is looking for playlists based on specific moods or activities.
        2. Utilize the Spotify API to find or create playlists matching the criteria.
        3. Present the playlists with relevant details like name, creator, and number of tracks.
        """
        
        return Task(
            description=description,
            expected_output="A list of playlists with descriptions and relevant statistics.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_general_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a general task for non-specific Spotify queries"""
        description = f"""
        You are tasked with addressing a general Spotify-related request.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Analyze the request to determine the information or action needed.
        2. Utilize the Spotify API to gather the necessary details.
        3. Provide a comprehensive response addressing the user's needs.
        """
        
        return Task(
            description=description,
            expected_output="A comprehensive response to the user's Spotify-related query.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def _extract_search_term(query: str) -> str:
        """Extract potential search term from the query"""
        patterns = [
            r'(?:search|find|discover)\s+["\']?([^"\']+)["\']?',
            r'(?:song|track|album|artist)\s+["\']?([^"\']+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""