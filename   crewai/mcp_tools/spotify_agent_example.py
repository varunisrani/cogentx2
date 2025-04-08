from crewai import Agent, Task, Crew, Process
from tools import SpotifyMCPTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Spotify MCP tool
spotify_tool = SpotifyMCPTool(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
)

# Create an agent for music discovery
music_discovery_agent = Agent(
    role="Music Discovery Expert",
    goal="Find the best music based on user preferences",
    backstory="""
    You are an AI music expert with deep knowledge of songs, artists,
    and music genres. You help users discover new music they might enjoy
    based on their current preferences and listening habits.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[spotify_tool]
)

# Define tasks
search_task = Task(
    description="Search for songs based on user's query",
    agent=music_discovery_agent,
    expected_output="A list of songs matching the search query",
    context={
        "query": "Enter your search query here, e.g., 'Coldplay', 'Dance music', or 'Billie Eilish Happier Than Ever'"
    }
)

recommendation_task = Task(
    description="Recommend songs based on a seed track",
    agent=music_discovery_agent,
    expected_output="A list of song recommendations",
    context={
        "seed_track": "Enter a Spotify track ID here that you want recommendations based on"
    }
)

# Create the crew
music_crew = Crew(
    agents=[music_discovery_agent],
    tasks=[search_task, recommendation_task],
    verbose=2,
    process=Process.sequential
)

# Run the crew to perform the tasks
result = music_crew.kickoff()

print("\n==== Results ====\n")
print(result) 