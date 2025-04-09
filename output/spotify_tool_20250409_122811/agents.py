from crewai import Agent
import os
import sys
import logging

# Set up logging for our Spotify agent operations
logger = logging.getLogger("spotify_agent.custom_agents")

class CustomSpotifyAgentFactory:
    """Factory for creating specialized Spotify agents tailored to music recommendation tasks"""

    @staticmethod
    def create_spotify_tool():
        """Initialize and manage the Spotify tool with error handling"""
        try:
            logger.info("Starting Spotify tool initialization.")
            spotify_tool = SpotifyMCPClient()
            logger.info("Spotify tool has been successfully initialized.")
            return spotify_tool
        except Exception as error:
            logger.error(f"Spotify tool initialization failed: {str(error)}")
            raise

    @staticmethod
    def create_recommendation_engine():
        """Create a dedicated agent for generating personalized song recommendations based on user input."""
        spotify_tool = CustomSpotifyAgentFactory.create_spotify_tool()
        
        logger.info("Building the song recommendation engine agent.")
        return Agent(
            role="Personalized Song Recommender",
            goal="Provide 5 tailored song suggestions based on user preferences and listening habits.",
            backstory="""As a Personalized Song Recommender, your mission is to curate the perfect playlists 
            for each user. You analyze user-provided data on favorite artists, genres, and moods, leveraging 
            Spotify's API to locate and recommend tracks that resonate with the user's distinct taste in music. 
            Utilizing your expertise in music trends and algorithms, you ensure that every suggestion feels 
            personalized and engaging.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""To interact with the Spotify tool effectively, consider the following formats:
            
            1. Direct recommendation request: spotify_tool("Suggest songs similar to my recent favorites.")
            
            2. Structured input dictionary:
               spotify_tool({"input_data": "Request song recommendations based on hip-hop genre."})
               
            3. Advanced operation format for tailored recommendations:
               spotify_tool({
                   "operation": "get_recommendations",
                   "parameters": {
                       "seed_artists": "artist_ids",
                       "seed_genres": "genre_tags",
                       "limit": 5
                   }
               })
               
            Always ensure your input accurately reflects your music preferences to achieve optimal recommendations.
            """
        )

    @staticmethod
    def create_mood_based_recommender():
        """Create an agent specialized in recommending songs based on user-defined moods or activities."""
        spotify_tool = CustomSpotifyAgentFactory.create_spotify_tool()
        
        logger.info("Creating the mood-based song recommendation specialized agent.")
        return Agent(
            role="Mood-Based Song Selector",
            goal="Identify and suggest 5 tracks that match the specified mood or occasion.",
            backstory="""As a Mood-Based Song Selector, your role is to connect users with the right tracks 
            for every moment. You possess an in-depth understanding of music that resonates under various 
            emotional contexts - be it uplifting songs for workouts or calming tunes for relaxation. By 
            interpreting user mood requests, you harness the power of Spotify’s API to deliver curated 
            music experiences that enhance every user's life.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""When using the Spotify tool for mood-based selections, input methods include:
            
            1. Mood request input: spotify_tool("Find songs for a relaxing evening.")
            
            2. Mood input in structured format:
               spotify_tool({"input_data": "Recommend upbeat tracks for a party."})
               
            3. Custom operation for mood alignment:
               spotify_tool({
                   "operation": "mood_recommendation",
                   "parameters": {
                       "desired_mood": "happy",
                       "limit": 5
                   }
               })
            """
        )

    @staticmethod
    def create_artist_influencer_agent():
        """Create an agent dedicated to recommending songs based on specific artists users love."""
        spotify_tool = CustomSpotifyAgentFactory.create_spotify_tool()
        
        logger.info("Establishing the artist influencer recommendation agent.")
        return Agent(
            role="Artist Influence Listener",
            goal="Give 5 song recommendations similar to a user-favorite specified artist.",
            backstory="""As an Artist Influence Listener, your objective is to provide users with 
            music that mirrors or complements their favorite artists. By meticulously analyzing the 
            characteristics of an artist’s style, genre, and lyrical content, you utilize Spotify’s 
            engine to find artists and tracks that fans of a specific musician would likely appreciate.
            This closes the gap between known favorites and undiscovered gems, ensuring every user 
            enjoys a seamless auditory journey.""",
            verbose=True,
            allow_delegation=False,
            tools=[spotify_tool],
            tools_instructions="""To effectively grab suggestions based on artist influences, you can use:
            
            1. Artist-based query input: spotify_tool("Suggest songs similar to Adele.")
            
            2. Input in a formal structure:
               spotify_tool({"input_data": "Find tracks related to Ed Sheeran."})
            
            3. Operation call for artist-centric recommendations:
               spotify_tool({
                   "operation": "artist_based_recommendations",
                   "parameters": {
                       "seed_artist": "artist_id",
                       "limit": 5
                   }
               })
            """
        )
 

This Python script creates a custom set of specialized agents tailored specifically for Spotify music recommendations. Each agent has a clear focus, backstory, and operational method, which emphasizes the personalized music discovery experience for the user.