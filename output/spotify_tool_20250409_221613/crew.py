from crewai import Crew, Process
from typing import Dict, Any, List, Optional
import logging
import json
import time
from agents import SpotifyRecommendationAgentFactory
from tasks import SpotifyRecommendationTaskFactory

# Get logger
logger = logging.getLogger("spotify_recommendation_agent.crew")

class SpotifyRecommendationCrew:
    """Crew for managing Spotify song recommendations based on user preferences."""
    
    def __init__(self, verbose: bool = True, memory: bool = True, human_input: bool = True):
        """Initialize the Spotify recommendation crew."""
        self.verbose = verbose
        self.memory = memory
        self.human_input = human_input
        self.agent = None
        self.tasks = []
        
        logger.info("Initializing Spotify recommendation crew")
        logger.info(f"Settings: verbose={verbose}, memory={memory}, human_input={human_input}")
        
        # Create the recommendation agent
        self._create_recommendation_agent()
    
    def _create_recommendation_agent(self):
        """Create the Spotify recommendation agent."""
        logger.info("Creating Spotify recommendation agent")
        self.agent = SpotifyRecommendationAgentFactory.create_recommendation_agent()
        logger.info("Agent created: Spotify Music Recommendation Specialist")
    
    def process_query(self, user_preferences: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process user preferences to recommend songs and return the results."""
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
        
        # Create a recommendation task based on user preferences
        logger.info("Creating recommendation task based on user preferences")
        task_creation_start = time.time()
        task = SpotifyRecommendationTaskFactory.create_recommendation_task(
            user_preferences=user_preferences,
            agent=self.agent,
            human_input=self.human_input,
            context=context
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
            result = crew.kickoff()
            crew_execution_time = time.time() - crew_start_time
            logger.info(f"Crew execution completed in {crew_execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = result[:200] + "..." if len(result) > 200 else result
            logger.info(f"Result summary: {result_summary}")
            
            # Log total execution time
            total_execution_time = time.time() - start_time
            logger.info(f"Total user preference processing time: {total_execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            # Log error
            crew_execution_time = time.time() - crew_start_time
            logger.error(f"Crew execution failed after {crew_execution_time:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            raise
    
    def _get_safe_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of context safe for logging (mask sensitive data)."""
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