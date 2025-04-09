# A Agent Definitions
from crewai import Agent
from tools import SpotifyMCPTool  # Import tools from generated tools.py

# Initialize tools
spotifymcptool = SpotifyMCPTool()

# Define your agents
data_analyst = Agent(
    role="Data Analyst",
    goal="Use tools to analyze user listening habits and preferences",
    backstory="You are an expert data analyst specialized in music personalization and data analysis.",
    tools=[spotifymcptool],
    verbose=True
)

playlist_curator = Agent(
    role="Playlist Curator",
    goal="Use tools to generate personalized playlists based on analysis",
    backstory="You are an expert playlist curator specialized in music personalization and data analysis.",
    tools=[spotifymcptool],
    verbose=True
)

user_interface_designer = Agent(
    role="User Interface Designer",
    goal="Use tools to design the user interface for the playlist generator",
    backstory="You are an expert user interface designer specialized in music personalization and data analysis.",
    tools=[spotifymcptool],
    verbose=True
)

