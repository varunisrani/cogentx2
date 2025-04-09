# A Task Definitions
from crewai import Task
from agents import data_analyst, playlist_curator, user_interface_designer

task_1 = Task(
    description="Collect user listening data",
    agent=data_analyst,
    expected_output="Detailed output with relevant information"
)

task_2 = Task(
    description="Analyze listening habits",
    agent=playlist_curator,
    expected_output="Detailed output with relevant information"
)

task_3 = Task(
    description="Generate personalized playlists",
    agent=user_interface_designer,
    expected_output="Detailed output with relevant information"
)

task_4 = Task(
    description="Design user interface",
    agent=user_interface_designer,
    expected_output="Detailed output with relevant information"
)

task_5 = Task(
    description="Deploy the application",
    agent=user_interface_designer,
    expected_output="Detailed output with relevant information"
,
    context=[task_1]
,
    context=[task_1, task_2]
,
    context=[task_1, task_2, task_3]
,
    context=[task_1, task_2, task_3, task_4]
)

