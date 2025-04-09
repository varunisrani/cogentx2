from crewai import Crew, Process
from typing import Dict, Any, List, Optional
import logging
import json
import time
from agents import SpotifyRecommendationAgent

# Get logger
logger = logging.getLogger("spotify_recommendation_agent.crew")

class SpotifyRecommendationCrew:
    """Crew for handling Spotify music recommendation operations"""
    
    def __init__(self, verbose: bool = True, memory: bool = True, human_input: bool = True):
        """Initialize the Spotify recommendation crew"""
        self.verbose = verbose
        self.memory = memory
        self.human_input = human_input
        self.agent = SpotifyRecommendationAgent()
        self.tasks = []
        
        logger.info("Initializing Spotify recommendation crew")
        logger.info(f"Settings: verbose={verbose}, memory={memory}, human_input={human_input}")
    
    def process_query(self, user_preferences: Dict[str, Any]) -> List[str]:
        """Process a music recommendation query based on user preferences and return suggested tracks"""
        logger.info("-" * 80)
        logger.info(f"Processing user preferences: {json.dumps(user_preferences, indent=2)}")
        
        # Track execution time
        start_time = time.time()
        
        # Reset tasks
        self.tasks = []
        logger.info("Tasks reset")
        
        # Create a task for music recommendations
        logger.info("Creating recommendation task")
        task_creation_start = time.time()
        task = self.agent.create_recommendation_task(
            user_preferences=user_preferences,
            human_input=self.human_input
        )
        task_creation_time = time.time() - task_creation_start
        logger.info(f"Recommendation task created in {task_creation_time:.2f} seconds")
        
        # Add the task
        self.tasks.append(task)
        logger.info(f"Task added: {task.description[:100]}..." if len(task.description) > 100 else task.description)
        
        # Create the crew
        logger.info("Creating crew with configured agent and task")
        crew = Crew(
            agents=[self.agent],
            tasks=self.tasks,
            verbose=self.verbose,
            memory=self.memory,
            process=Process.sequential
        )
        
        # Run the crew
        logger.info("Starting crew execution")
        crew_start_time = time.time()
        try:
            recommendations = crew.kickoff()
            crew_execution_time = time.time() - crew_start_time
            logger.info(f"Crew execution completed in {crew_execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = recommendations[:200] + "..." if len(recommendations) > 200 else recommendations
            logger.info(f"Recommendations summary: {result_summary}")
            
            # Log total execution time
            total_execution_time = time.time() - start_time
            logger.info(f"Total query processing time: {total_execution_time:.2f} seconds")
            
            return recommendations
        except Exception as e:
            # Log error
            crew_execution_time = time.time() - crew_start_time
            logger.error(f"Crew execution failed after {crew_execution_time:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            raise
    
    def bulk_process(self, user_preferences_list: List[Dict[str, Any]]) -> List[List[str]]:
        """Process multiple music recommendation queries and return the results"""
        logger.info(f"Starting bulk processing of {len(user_preferences_list)} user preferences")
        results = []
        
        for i, user_preferences in enumerate(user_preferences_list, 1):
            logger.info(f"Processing bulk user preferences {i}/{len(user_preferences_list)}")
            try:
                recommendations = self.process_query(user_preferences)
                results.append(recommendations)
                logger.info(f"Bulk user preferences {i} completed successfully")
            except Exception as e:
                logger.error(f"Bulk user preferences {i} failed: {str(e)}")
                results.append([f"Error: {str(e)}"])
        
        logger.info(f"Bulk processing completed. {len(results)}/{len(user_preferences_list)} user preferences processed")
        return results