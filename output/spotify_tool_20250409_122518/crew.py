from crewai import Crew, Process
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
from agents import SpotifyAgentFactory
from tasks import SpotifyTaskFactory

# Get logger
logger = logging.getLogger("spotify_agent.crew")

class SpotifyCrew:
    """Crew for handling Spotify operations"""
    
    def __init__(self, verbose: bool = True, memory: bool = True, human_input: bool = True):
        """Initialize the Spotify crew"""
        self.verbose = verbose
        self.memory = memory
        self.human_input = human_input
        self.agent = None
        self.tasks = []
        
        logger.info("Initializing Spotify crew")
        logger.info(f"Settings: verbose={verbose}, memory={memory}, human_input={human_input}")
        
        # Create the agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the Spotify agent"""
        logger.info("Creating general Spotify agent")
        self.agent = SpotifyAgentFactory.create_spotify_agent()
        # Log agent type
        logger.info("Agent created: Spotify Music Expert")
    
    def _create_specialized_agent(self, query: str):
        """Create a specialized agent based on the query"""
        logger.info(f"Selecting specialized agent for query: {query[:50]}..." if len(query) > 50 else query)
        
        # Determine the type of agent needed
        agent_type = "general"
        if any(keyword in query.lower() for keyword in ["search", "find", "track", "song", "album", "artist"]):
            logger.info("Selected music search specialist agent based on keywords")
            self.agent = SpotifyAgentFactory.create_music_search_specialist()
            agent_type = "search_specialist"
        elif any(keyword in query.lower() for keyword in ["recommend", "similar", "like", "discover"]):
            logger.info("Selected recommendations specialist agent based on keywords")
            self.agent = SpotifyAgentFactory.create_music_recommendations_specialist()
            agent_type = "recommendations_specialist"
        elif any(keyword in query.lower() for keyword in ["playlist", "collection", "compilation"]):
            logger.info("Selected playlist specialist agent based on keywords")
            self.agent = SpotifyAgentFactory.create_playlist_specialist()
            agent_type = "playlist_specialist"
        else:
            # Default to general Spotify agent
            logger.info("No specialized keywords found, using general Spotify agent")
            self.agent = SpotifyAgentFactory.create_spotify_agent()
            agent_type = "general"
        
        # Log agent type
        logger.info(f"Agent selected: {agent_type.replace('_', ' ').title()}")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a Spotify-related query and return the result"""
        # Log query processing
        logger.info("-" * 80)
        logger.info(f"Processing query: {query}")
        
        # Track execution time
        start_time = time.time()
        
        # Reset tasks
        self.tasks = []
        logger.info("Tasks reset")
        
        # Create a specialized agent based on the query
        self._create_specialized_agent(query)
        
        # Log context if provided
        if context:
            safe_context = self._get_safe_context(context)
            logger.info(f"Context provided: {json.dumps(safe_context, indent=2)}")
        else:
            logger.info("No context provided")
        
        # Create a task based on the query
        logger.info("Creating task for query")
        task_creation_start = time.time()
        task = SpotifyTaskFactory.create_task(
            query=query,
            agent=self.agent,
            human_input=self.human_input,
            context=context
        )
        task_creation_time = time.time() - task_creation_start
        logger.info(f"Task created in {task_creation_time:.2f} seconds")
        
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
            logger.info(f"Total query processing time: {total_execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            # Log error
            crew_execution_time = time.time() - crew_start_time
            logger.error(f"Crew execution failed after {crew_execution_time:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            raise
    
    def bulk_process(self, queries: List[str], context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process multiple Spotify-related queries and return the results"""
        logger.info(f"Starting bulk processing of {len(queries)} queries")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing bulk query {i}/{len(queries)}")
            try:
                result = self.process_query(query, context)
                results.append(result)
                logger.info(f"Bulk query {i} completed successfully")
            except Exception as e:
                logger.error(f"Bulk query {i} failed: {str(e)}")
                results.append(f"Error: {str(e)}")
        
        logger.info(f"Bulk processing completed. {len(results)}/{len(queries)} queries processed")
        return results
    
    def process_complex_request(self, request: str, steps: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """Process a complex Spotify request with multiple steps"""
        logger.info("-" * 80)
        logger.info(f"Processing complex request: {request}")
        logger.info(f"Number of steps: {len(steps)}")
        
        # Track execution time
        start_time = time.time()
        
        # Reset tasks
        self.tasks = []
        logger.info("Tasks reset")
        
        # Create a general Spotify agent
        logger.info("Creating general Spotify agent for complex request")
        self.agent = SpotifyAgentFactory.create_spotify_agent()
        # Log agent type
        logger.info("Agent created: Spotify Music Expert")
        
        # Create tasks for each step
        logger.info("Creating tasks for each step in the complex request")
        for i, step in enumerate(steps, 1):
            logger.info(f"Creating task for step {i}/{len(steps)}: {step}")
            
            step_context = []
            if context:
                # Convert dictionary to list of tuples
                base_context = [(k, v) for k, v in context.items()]
                step_context = base_context + [
                    ("step_number", i),
                    ("total_steps", len(steps)),
                    ("original_request", request)
                ]
                
                # Log context
                safe_context = [(k, v if k.lower() not in ["token", "key", "secret", "password", "auth", "client_id", "client_secret"] else f"{str(v)[:3]}...{str(v)[-3:]}") for k, v in step_context]
                logger.info(f"Step {i} context: {safe_context}")
            else:
                step_context = [
                    ("step_number", i),
                    ("total_steps", len(steps)),
                    ("original_request", request)
                ]
                logger.info(f"Step {i} context: {step_context}")
            
            # Create task
            task_creation_start = time.time()
            task = SpotifyTaskFactory.create_task(
                query=step,
                agent=self.agent,
                human_input=self.human_input,
                context=step_context
            )
            task_creation_time = time.time() - task_creation_start
            logger.info(f"Task for step {i} created in {task_creation_time:.2f} seconds")
            
            self.tasks.append(task)
            logger.info(f"Task added for step {i}")
        
        # Create the crew
        logger.info(f"Creating crew with {len(self.tasks)} tasks")
        crew = Crew(
            agents=[self.agent],
            tasks=self.tasks,
            verbose=self.verbose,
            memory=self.memory,
            process=Process.sequential
        )
        
        # Run the crew
        logger.info("Starting crew execution for complex request")
        crew_start_time = time.time()
        try:
            result = crew.kickoff()
            crew_execution_time = time.time() - crew_start_time
            logger.info(f"Complex request crew execution completed in {crew_execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = result[:200] + "..." if len(result) > 200 else result
            logger.info(f"Complex request result summary: {result_summary}")
            
            # Log total execution time
            total_execution_time = time.time() - start_time
            logger.info(f"Total complex request processing time: {total_execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            # Log error
            crew_execution_time = time.time() - crew_start_time
            logger.error(f"Complex request crew execution failed after {crew_execution_time:.2f} seconds")
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