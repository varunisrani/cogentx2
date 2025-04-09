from crewai import Agent
import os
import sys
import logging

# Modify the path to include the necessary Spotify client library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from smithery_spotify_client import SpotifyMCPClient

# Initialize logger for the Spotify recommendation process
logger = logging.getLogger("spotify_recommendation.agents")

class SpotifyAgentBuilder:
    """Factory class for creating Spotify-oriented agents with targeted functionalities."""

    @staticmethod
    def initialize_spotify_service():
        """Setup and return the Spotify music client with error checks."""
        try:
            logger.info("Setting up Spotify service client")
            spotify_client = SpotifyMCPClient()
            logger.info("Spotify service client successfully set up")
            return spotify_client
        except Exception as error:
            logger.error(f"Failed to initialize Spotify service: {str(error)}")
            raise

    @staticmethod
    def create_song_recommender_agent():
        """Create an agent dedicated to recommending Spotify songs based on user input."""
        spotify_client = SpotifyAgentBuilder.initialize_spotify_service()

        logger.info("Creating Song Recommender agent")
        return Agent(
            role="Personalized Song Recommender",
            goal="Deliver tailored song recommendations based on specific user preferences and listening habits, aiming for at least 10 relevant suggestions per session",
            backstory="""As a Personalized Song Recommender, you possess a deep understanding
            of various musical genres, trends, and user preferences. Your expertise lies in analyzing
            user-input data like favorite artists, songs, and preferred moods. Using this data,
            you suggest new tracks that align with the user's established taste, helping them discover
            music they will love but may not have encountered yet.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_client],
            tools_instructions="""Utilize the Spotify service client with the formats below:
            
            1. Direct request: spotify_client("Recommend songs like 'Shape of You'")
            
            2. Dictionary format:
               spotify_client({"input_data": "Suggest songs inspired by 'Shape of You'"})
               
            3. Detailed operations format:
               spotify_client({
                   "operation": "recommendations",
                   "parameters": {
                       "seed_tracks": ["track_id_1", "track_id_2"],
                       "limit": 10
                   }
               })
               
            Always ensure the input is precisely structured for optimal results.
            """
        )

    @staticmethod
    def create_playlist_discoverer_agent():
        """Create an agent focused on discovering and managing playlists."""
        spotify_client = SpotifyAgentBuilder.initialize_spotify_service()

        logger.info("Creating Playlist Discoverer agent")
        return Agent(
            role="Dynamic Playlist Organizer",
            goal="Assist users in finding, curating, and managing playlists suited for various emotions and activities, achieving a minimum of 5 themed playlists per query.",
            backstory="""As a Dynamic Playlist Organizer, you are skilled in identifying
            playlists that fit users' specific moods and contexts. Whether a user is seeking
            energizing workout tracks or calming evening tunes, your expertise allows you
            to suggest curated playlists tailored to individual listening scenarios. Your
            goal is to enhance the user's music experience through personalized collection management.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_client],
            tools_instructions="""When interacting with the Spotify client, you can perform the following:
            
            1. Simple request: spotify_client("Find playlists for relaxation")
            
            2. Input as a dictionary:
               spotify_client({"input_data": "Discover playlists for study sessions"})
               
            3. Structured operations for specific tasks:
               spotify_client({
                   "operation": "search",
                   "parameters": {
                       "query": "study",
                       "type": "playlist",
                       "limit": 5
                   }
               })
               
            Ensure your inputs are well-defined to enhance playlist suggestions.
            """
        )

    @staticmethod
    def create_create_genre_explorer_agent():
        """Create an agent specializing in exploring new genres."""
        spotify_client = SpotifyAgentBuilder.initialize_spotify_service()

        logger.info("Creating Genre Explorer agent")
        return Agent(
            role="Genre Discovery Specialist",
            goal="Facilitate users’ exploration of new musical genres by providing a recommendation list of at least 8 songs from a chosen genre.",
            backstory="""As a Genre Discovery Specialist, you leverage your broad knowledge of 
            musical styles and sub-genres to help users expand their listening horizons. You thrive on
            creating connections between artists, trends, and the evolving landscape of music, 
            offering fresh, genre-specific recommendations that might pique the users' interest, aligning
            with their existing tastes and curiosity for new sounds.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_client],
            tools_instructions="""Utilize the Spotify service to provide genre recommendations through:
            
            1. General request: spotify_client("Show me Indie songs")
            
            2. Dict format:
               spotify_client({"input_data": "Explore songs in the Jazz genre"})
               
            3. Specified operations:
               spotify_client({
                   "operation": "genre_search",
                   "parameters": {
                       "genre": "Indie",
                       "limit": 8
                   }
               })
               
            Input precision is key for generating genre-specific results.
            """
        )
 

This newly crafted `agents.py` file strictly adheres to the user requirements, advancing the original concept into fully specialized roles tailored to Spotify song recommendations. Each agent has a unique purpose, backstory, goals, and tool usage intricately related to user tasks—distancing it significantly from prior iterations.