from crewai import Task, Agent
from typing import Dict, Any, Optional, List, Tuple
import re

class TaskType:
    """Enum for task types"""
    MUSIC_SEARCH = "music_search"
    RECOMMENDATIONS = "recommendations"
    PLAYLIST = "playlist"
    GENERAL = "general"

class SpotifyTaskFactory:
    """Factory class to create Spotify-related tasks"""
    
    @staticmethod
    def create_task(query: str, agent: Agent, human_input: bool = False, context: Optional[List[Tuple[str, Any]]] = None) -> Task:
        """Create appropriate task based on the user query"""
        # Default context if none provided
        if context is None:
            context = []
            
        # Determine task category
        task_type = SpotifyTaskFactory.determine_task_type(query)
        
        # Create appropriate task
        if task_type == TaskType.MUSIC_SEARCH:
            return SpotifyTaskFactory.create_music_search_task(query, agent, human_input, context)
        elif task_type == TaskType.RECOMMENDATIONS:
            return SpotifyTaskFactory.create_recommendations_task(query, agent, human_input, context)
        elif task_type == TaskType.PLAYLIST:
            return SpotifyTaskFactory.create_playlist_task(query, agent, human_input, context)
        else:
            return SpotifyTaskFactory.create_general_task(query, agent, human_input, context)
    
    @staticmethod
    def determine_task_type(query: str) -> TaskType:
        """Determine the type of task based on the query"""
        # Music search keywords
        search_keywords = [
            "search", "find", "look for", "track", "song", "album", "artist", 
            "who sings", "who performs", "what album", "music by"
        ]
        
        # Recommendation keywords
        recommendation_keywords = [
            "recommend", "recommendation", "suggest", "similar to", "like",
            "similar artists", "similar songs", "more like", "fans of", "discover"
        ]
        
        # Playlist keywords
        playlist_keywords = [
            "playlist", "playlists", "compilation", "collection", "mix",
            "workout playlist", "party playlist", "mood", "activity"
        ]
        
        # Check for matches
        query_lower = query.lower()
        
        for keyword in search_keywords:
            if keyword in query_lower:
                return TaskType.MUSIC_SEARCH
        
        for keyword in recommendation_keywords:
            if keyword in query_lower:
                return TaskType.RECOMMENDATIONS
        
        for keyword in playlist_keywords:
            if keyword in query_lower:
                return TaskType.PLAYLIST
        
        # Default to generic Spotify task
        return TaskType.GENERAL
    
    @staticmethod
    def create_music_search_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a music search task"""
        # Extract potential search terms
        search_term = SpotifyTaskFactory._extract_search_term(query)
        search_type = SpotifyTaskFactory._extract_search_type(query)
        
        # Prepare task description
        description = f"""
        You are tasked with helping the user search for music on Spotify.
        
        USER REQUEST: {query}
        
        Search details:
        - Search term: {search_term if search_term else 'Not clearly specified - interpret from context'}
        - Type: {search_type if search_type else 'Not specified - default to tracks'}
        
        Steps to follow:
        1. Analyze the request to determine exactly what the user is searching for
        2. Determine the appropriate search parameters (type, limit, etc.)
        3. Execute the search using the Spotify API
        4. Present the search results in a clear, organized manner
        5. Include relevant details like artist names, albums, popularity, etc.
        """
        
        return Task(
            description=description,
            expected_output="A detailed list of search results formatted in a clear, easily readable manner with all relevant music details.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_recommendations_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a recommendations task"""
        description = f"""
        You are tasked with helping the user get music recommendations from Spotify.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Analyze the request to determine what kind of recommendations the user wants
        2. Identify seed artists, tracks, or genres mentioned in the query
        3. If necessary, first search for the mentioned artists/tracks to get their Spotify IDs
        4. Use the recommendation API to find music matching the user's interests
        5. Present the recommendations in a helpful format with all relevant details
        6. Explain why these recommendations might appeal to the user
        
        Focus on quality over quantity, and provide diverse recommendations when appropriate.
        """
        
        return Task(
            description=description,
            expected_output="A detailed list of personalized music recommendations with explanations about why they match the user's interests.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_playlist_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a playlist-related task"""
        description = f"""
        You are tasked with helping the user with Spotify playlists.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Determine if the user is looking for specific playlist categories, moods, or activities
        2. Search for playlists matching the user's criteria
        3. Present the playlists with relevant details (name, creator, number of tracks, follower count)
        4. For each playlist, include a brief description of its content and style
        5. If applicable, suggest multiple playlists for different variations of the request
        
        Aim to provide playlists that best match the user's stated or implied musical preferences.
        """
        
        return Task(
            description=description,
            expected_output="A detailed list of playlists matching the user's request, with descriptions and relevant statistics for each playlist.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_general_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a general Spotify task for queries that don't fit other categories"""
        description = f"""
        You are tasked with helping the user with a general Spotify-related request.
        
        USER REQUEST: {query}
        
        Steps to follow:
        1. Analyze the request to determine what Spotify information or actions are needed
        2. Use your tools to interact with the Spotify API appropriately
        3. Provide a comprehensive response that addresses the user's needs
        4. Include relevant details about tracks, artists, albums, or playlists as appropriate
        
        Your goal is to be as helpful as possible while working within the capabilities
        of the Spotify API. If you cannot complete a request, explain why and suggest alternatives.
        """
        
        return Task(
            description=description,
            expected_output="A helpful response that addresses the user's Spotify-related query, with appropriate music details and suggestions.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def _extract_search_term(query: str) -> str:
        """Extract potential search term from the query"""
        # Patterns for different search term indicators
        patterns = [
            r'(?:search|find|looking for)\s+["\']?([^"\']+)["\']?(?:\s+by|\s+from|\s+in|\s+on|\s*$)',
            r'(?:song|track|album|artist)\s+["\']?([^"\']+)["\']?',
            r'who (?:sings|performs)\s+["\']?([^"\']+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return empty string
        return ""
    
    @staticmethod
    def _extract_search_type(query: str) -> str:
        """Extract the type of search from the query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["song", "track", "music"]):
            return "track"
        elif any(term in query_lower for term in ["artist", "band", "musician", "singer", "performer"]):
            return "artist"
        elif any(term in query_lower for term in ["album", "record", "LP", "EP"]):
            return "album"
        elif any(term in query_lower for term in ["playlist", "compilation", "collection"]):
            return "playlist"
        
        # Default search type
        return "" 