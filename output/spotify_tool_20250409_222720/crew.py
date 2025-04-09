from crewai import Crew, Process
from typing import Dict, Any, List, Optional
import logging
import json
import time
from agents import SpotifyRecommendationAgent
from tasks import SpotifyRecommendationTask

# Get logger
logger = logging.getLogger("spotify_recommendation_agent.crew")

class SpotifyRecommendationCrew:
    """Crew for handling Spotify song recommendations"""
    
    def __init__(self, verbose: bool = True, memory: bool = True, human_input: bool = True):
        """Initialize the Spotify recommendation crew"""
        self.verbose = verbose
        self.memory = memory
        self.human_input = human_input
        self.agent = SpotifyRecommendationAgent()
        self.tasks = []
        
        logger.info("Initializing Spotify recommendation crew")
        logger.info(f"Settings: verbose={verbose}, memory={memory}, human_input={human_input}")

    def process_query(self, user_preferences: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Process a user query to recommend Spotify songs based on preferences and context"""
        logger.info("-" * 80)
        logger.info(f"Processing user preferences: {user_preferences}")
        
        # Track execution time
        start_time = time.time()
        
        # Reset tasks
        self.tasks = []
        logger.info("Tasks reset")
        
        # Log context if provided
        if context:
            safe_context = self._get_safe_context(context)
            logger.info(f"Context provided: {json.dumps(safe_context, indent=2)}")
        else:
            logger.info("No context provided")
        
        # Create a recommendation task
        logger.info("Creating recommendation task based on user preferences")
        task = SpotifyRecommendationTask(
            user_preferences=user_preferences,
            agent=self.agent,
            human_input=self.human_input,
            context=context
        )
        
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
        logger.info("Starting crew execution for song recommendations")
        crew_start_time = time.time()
        try:
            result = crew.kickoff()
            crew_execution_time = time.time() - crew_start_time
            logger.info(f"Crew execution completed in {crew_execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = result[:200] + "..." if len(result) > 200 else result
            logger.info(f"Result summary: {result_summary}")
            
            # Log total execution time
            total_execution_time = time.time() - start_time
            logger.info(f"Total query processing time: {total_execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            # Log error
            crew_execution_time = time.time() - crew_start_time
            logger.error(f"Crew execution failed after {crew_execution_time:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            raise
    
    def _get_safe_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of context safe for logging (mask sensitive data)"""
        if not context:
            return {}
            
        safe_context = context.copy()
        
        # Mask sensitive fields
        sensitive_fields = ['token', 'password', 'secret', 'key', 'auth', 'client_id', 'client_secret']
        for field in sensitive_fields:
            for key in list(safe_context.keys()):
                if field in key.lower() and isinstance(safe_context[key], str):
                    safe_context[key] = f"{safe_context[key][:3]}...{safe_context[key][-3:]}"
        
        return safe_context