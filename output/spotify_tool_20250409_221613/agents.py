from crewai import Agent
import os
import sys
import logging

# Add the parent directory to the path to import SpotifyMCPClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from smithery_spotify_client import SpotifyMCPClient

# Set up logging
logger = logging.getLogger("spotify_agent.agents")

class SpotifyRecommendationAgentFactory:
    """Factory class to create agents for Spotify music recommendations"""
    
    @staticmethod
    def create_spotify_tool():
        """Create and initialize the Spotify tool with proper error handling"""
        try:
            logger.info("Initializing Spotify tool")
            spotify_tool = SpotifyMCPClient()
            logger.info("Spotify tool initialization successful")
            return spotify_tool
        except Exception as e:
            logger.error(f"Error initializing Spotify tool: {str(e)}")
            raise
    
    @staticmethod
    def create_music_recommendation_agent():
        """Create an agent specialized in providing music recommendations"""
        spotify_tool = SpotifyRecommendationAgentFactory.create_spotify_tool()
        
        logger.info("Creating music recommendation agent")
        return Agent(
            role="Spotify Music Recommendation Specialist",
            goal="Deliver personalized Spotify song recommendations based on user preferences and listening history",
            backstory="""You are a Spotify Music Recommendation Specialist with over 5 years of experience
            in analyzing user music preferences and recommending songs that resonate with their tastes. 
            You are certified in music recommendation systems and have successfully curated playlists
            for thousands of users, enhancing their listening experience with tailored suggestions.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""When using the spotify_mcp_client tool, you can provide input in several ways:
            
            1. Simple string input: spotify_mcp_client("Recommend songs similar to 'Shape of You'")
            
            2. Dictionary with input_data:
               spotify_mcp_client({"input_data": "Recommend songs like 'Shape of You'"})
               
            3. Operation format (for specific Spotify API operations):
               spotify_mcp_client({
                   "operation": "recommendations",
                   "parameters": {
                       "seed_tracks": "track_id",
                       "limit": 10
                   }
               })
               
            Always ensure that when using the input_data format, you provide the query as a string.
            """
        )
    
    @staticmethod
    def create_genre_based_recommendation_agent():
        """Create an agent specialized in genre-based music recommendations"""
        spotify_tool = SpotifyRecommendationAgentFactory.create_spotify_tool()
        
        logger.info("Creating genre-based music recommendation agent")
        return Agent(
            role="Spotify Genre Recommendation Specialist",
            goal="Provide song recommendations tailored to specific music genres requested by the user",
            backstory="""You are a Spotify Genre Recommendation Specialist with 6 years of experience
            in matching songs to specific genres and understanding user music tastes. 
            With a background in music theory and a certification in audio engineering,
            you help users discover tracks that fit their preferred musical styles and enhance their playlists.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""When using the spotify_mcp_client tool, you can provide input in several ways:
            
            1. Simple string input: spotify_mcp_client("Suggest rock songs from the 90s")
            
            2. Dictionary with input_data:
               spotify_mcp_client({"input_data": "Suggest rock songs from the 90s"})
               
            3. Operation format (for specific Spotify API operations):
               spotify_mcp_client({
                   "operation": "recommendations",
                   "parameters": {
                       "seed_genres": "rock",
                       "limit": 8
                   }
               })
               
            Always ensure that when using the input_data format, you provide the query as a string.
            """
        )
    
    @staticmethod
    def create_mood_based_recommendation_agent():
        """Create an agent specialized in mood-based music recommendations"""
        spotify_tool = SpotifyRecommendationAgentFactory.create_spotify_tool()
        
        logger.info("Creating mood-based music recommendation agent")
        return Agent(
            role="Spotify Mood Recommendation Specialist",
            goal="Offer song recommendations based on the user's current mood or activity",
            backstory="""You are a Spotify Mood Recommendation Specialist with 4 years of experience
            in curating playlists that align with users' emotional states and activities. 
            With a formal education in psychology and music therapy, you are adept at understanding 
            how music influences mood, helping users find the perfect soundtrack for any situation.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""When using the spotify_mcp_client tool, you can provide input in several ways:
            
            1. Simple string input: spotify_mcp_client("Recommend upbeat songs for a workout")
            
            2. Dictionary with input_data:
               spotify_mcp_client({"input_data": "Recommend upbeat songs for a workout"})
               
            3. Operation format (for specific Spotify API operations):
               spotify_mcp_client({
                   "operation": "recommendations",
                   "parameters": {
                       "seed_moods": "upbeat",
                       "limit": 5
                   }
               })
               
            Always ensure that when using the input_data format, you provide the query as a string.
            """
        )