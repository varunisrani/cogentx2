# A Crew Definition
from crewai import Crew
from tasks import task_1, task_2, task_3, task_4, task_5

# Create the crew to run the tasks
crew = Crew(
    tasks=[task_1, task_2, task_3, task_4, task_5],
    verbose=True,
    process="sequential"  # Tasks will run in sequence
)

# Function to run the crew with a specific query
def run_crew(query):
    result = crew.kickoff(inputs={"query": query})
    return result

# Allow running directly
if __name__ == "__main__":
    result = run_crew("Example query to research")
    print(result)
