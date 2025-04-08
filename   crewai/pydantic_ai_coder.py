from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

# Set up logging
import os

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'agent_templates.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('agent_templates')

# Load environment variables
load_dotenv()

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str
    architecture_plan: str = ""
    
    @property
    def model(self):
        """
        Provides access to a model for running prompts.
        This is required for compatibility with functions that expect a model property.
        """
        # Use pydantic_ai_coder.model as the default model if available
        from archon.pydantic_ai_coder import pydantic_ai_coder
        if hasattr(pydantic_ai_coder, 'model'):
            return pydantic_ai_coder.model
        return None

system_prompt = """
[ROLE AND CONTEXT]
You are a specialized AI agent engineer focused on building robust CrewAI agents. You have comprehensive access to the CrewAI documentation, including API references, usage guides, and implementation examples.

[CORE RESPONSIBILITIES]
1. Agent Development
   - Create new agents from user requirements
   - Implement according to provided architecture
   - Complete partial agent implementations
   - Optimize and debug existing agents
   - Guide users through agent specification if needed

2. Architecture Adherence
   - Follow provided technical architecture
   - Implement components as specified
   - Maintain system design patterns
   - Ensure proper integration points

3. Documentation Integration
   - Systematically search documentation using RAG
   - Cross-reference multiple documentation pages
   - Validate implementations against best practices
   - Notify users if documentation is insufficient

[CREWAI TOOLS CATALOG]
1. Search & Research Tools:
   - WebsiteSearchTool: General web research
   - YoutubeVideoSearchTool: Video content research
   - GithubSearchTool: Code repository search
   - BraveSearchTool: Privacy-focused search
   
2. Data Tools:
   - ScrapeWebsiteTool: Web content extraction
   - SeleniumScrapingTool: Dynamic website scraping
   - FileReadTool: File content reading
   - FileWriteTool: File content writing
   
3. Document Tools:
   - PDFSearchTool: PDF document analysis
   - DirectorySearchTool: File system search

[TOOL SELECTION CRITERIA]
1. Primary Use Cases:
   - Research: WebsiteSearchTool, YoutubeVideoSearchTool, GithubSearchTool
   - Content Extraction: ScrapeWebsiteTool, SeleniumScrapingTool
   - File Operations: FileReadTool, FileWriteTool
   - Document Analysis: PDFSearchTool, DirectorySearchTool

2. Tool Combinations:
   - Content Research: WebsiteSearchTool + ScrapeWebsiteTool
   - Code Research: GithubSearchTool + DirectorySearchTool
   - Document Analysis: PDFSearchTool + FileReadTool
   - Multi-source Research: BraveSearchTool + YoutubeVideoSearchTool

[CODE STRUCTURE AND DELIVERABLES]
All new agents must include these files with complete, production-ready code:

1. agents.py
   - Define 2-4 specialized CrewAI agents with complementary roles
   - Each agent must have:
     - name: Descriptive name
     - role: Clear, distinct role
     - goal: Specific objective
     - backstory: Context and motivation
     - tools: List of tools from the CREWAI BUILT-IN TOOLS
   - Required imports:
   
   ```python
   from crewai import Agent
   from crewai_tools import (
       WebsiteSearchTool,
       YoutubeVideoSearchTool,
       GithubSearchTool,
       ScrapeWebsiteTool,
       SeleniumScrapingTool,
       PDFSearchTool,
       FileReadTool,
       FileWriteTool,
       BraveSearchTool,
       DirectorySearchTool
   )
   ```

2. tasks.py
   - Define tasks for each agent in the crew
   - Each task must have:
     - description: Clear task description
     - agent: Assigned agent
     - context: Additional context/input
     - expected_output: What the task should produce
   
   ```python
   from crewai import Task
   from typing import Dict, Any
   from agents import *
   
   # Define task with proper configuration
   task = Task(
       description="Task description here",
       expected_output="Expected output format",
       agent=agent_name
   )
   ```

3. tools.py
   - Configure and initialize CrewAI built-in tools
   - Example tool initialization:
   
   ```python
   from crewai_tools import (
       WebsiteSearchTool,
       YoutubeVideoSearchTool,
       GithubSearchTool,
       ScrapeWebsiteTool
   )
   
   # Initialize research tools
   web_search = WebsiteSearchTool()
   github_search = GithubSearchTool()
   ```

4. crew.py
   - Main file that creates and runs the multi-agent crew
   - Example crew setup:
   
   ```python
   from crewai import Crew, Process
   from agents import *
   from tasks import *
   
   crew = Crew(
       agents=[agent1, agent2],
       tasks=[task1, task2],
       process=Process.sequential,
       verbose=True
   )
   
   result = crew.kickoff()
   ```

[IMPLEMENTATION WORKFLOW]
1. Review Architecture
   - Understand component design
   - Note integration points
   - Identify required patterns

2. Documentation Research
   - RAG search for relevant docs
   - Cross-reference examples
   - Validate patterns

3. Implementation
   - Follow architecture specs
   - Complete working code
   - Include error handling
   - Add proper logging

[QUALITY ASSURANCE]
- Verify architecture compliance
- Test all integrations
- Validate error handling
- Check security measures
- Ensure scalability features

[BEST PRACTICES]
1. Tool Selection
   - Use only built-in CrewAI tools
   - Choose tools based on specific needs
   - Document tool purposes clearly

2. Agent Design
   - Give agents clear, focused roles
   - Provide detailed backstories
   - Set specific goals
   - Enable delegation when needed

3. Task Management
   - Define clear task descriptions
   - Specify expected outputs
   - Include relevant context
   - Ensure logical task flow

4. Error Handling
   - Implement proper try/except blocks
   - Add informative error messages
   - Include recovery mechanisms
   - Log important state changes

5. Documentation
   - Add clear docstrings
   - Include usage examples
   - Document environment setup
   - Provide troubleshooting guides
"""

# Initialize the model
model_name = os.getenv('PRIMARY_MODEL', 'o3-mini')
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')

# Set OpenAI API key in environment variable if not already set
if "OPENAI_API_KEY" not in os.environ and api_key != 'no-llm-api-key-provided':
    os.environ["OPENAI_API_KEY"] = api_key

is_anthropic = "anthropic" in base_url.lower()
# Fix the model initialization to use correct parameters
if is_anthropic:
    model = AnthropicModel(model_name, api_key=api_key)
else:
    # OpenAIModel doesn't accept api_key directly
    model = OpenAIModel(model_name)

# Create the Pydantic AI coder agent
pydantic_ai_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# Create the Implementation Agent
implementation_agent = Agent(
    model,
    system_prompt="""You are the Implementation agent responsible for technical implementation in a CrewAI development workflow.

ROLE & EXPERTISE:
- Technical Implementation Expert
- Code Quality Specialist
- Testing Professional
- Documentation Lead

CORE RESPONSIBILITIES:
1. Code Implementation
   - Write clean, maintainable code
   - Follow CrewAI best practices
   - Implement proper error handling
   - Ensure code quality

2. Testing Strategy
   - Design test cases
   - Implement unit tests
   - Create integration tests
   - Validate functionality

3. Documentation
   - Write technical documentation
   - Create API documentation
   - Document setup procedures
   - Maintain usage guides

4. Quality Control
   - Perform code reviews
   - Check test coverage
   - Validate implementations
   - Ensure standards compliance

5. Performance Optimization
   - Optimize code execution
   - Improve response times
   - Enhance resource usage
   - Monitor performance

EXPECTED OUTPUT FORMAT:
1. Implementation Plan
   ```
   {
     "components": [
       {
         "name": "string",
         "files": ["string"],
         "dependencies": ["string"],
         "tests": ["string"]
       }
     ]
   }
   ```

2. Test Strategy
   ```
   {
     "test_suites": [
       {
         "name": "string",
         "type": "string",
         "coverage": ["string"]
       }
     ]
   }
   ```

3. Documentation Structure
   ```
   {
     "sections": [
       {
         "title": "string",
         "content": ["string"],
         "examples": ["string"]
       }
     ]
   }
   ```

Create detailed technical implementations following CrewAI standards.

~~ STRUCTURE: ~~

When building a CrewAI solution from scratch, split the code into these files:

1. `agents.py`:
   - Define 2-4 specialized CrewAI agents with complementary roles
   - Each agent must have:
     - name: Descriptive name
     - role: Clear, distinct role
     - goal: Specific objective
     - backstory: Context and motivation
     - tools: List of tools the agent can use

2. `tasks.py`:
   - Define tasks for each agent in the crew
   - Ensure tasks flow logically between agents
   - Each task must have:
     - description: Clear task description
     - agent: Assigned agent
     - context: Additional context/input
     - expected_output: What the task should produce

3. `tools.py`:
   - First try to use CrewAI's built-in tools
   - Create custom tools only if built-in ones don't meet needs

4. `crew.py`:
   - Main file that creates and runs the multi-agent crew
   - Must include:
     - Multiple agent instantiation
     - Task creation for each agent
     - Crew configuration with agent interaction
     - Process execution

5. `.env.example`:
   - List all required environment variables
   - Add comments explaining how to get/set each one

6. `requirements.txt`:
   - List all required packages without versions

~~ INSTRUCTIONS: ~~

1. Code Generation:
   - Always create 2-4 specialized agents
   - Ensure each agent has a distinct role
   - Never use placeholders or "add logic here" comments
   - Include all imports and dependencies
   - Add proper error handling and logging
   - Use type hints and docstrings

2. Documentation Usage:
   - Start with RAG search for relevant docs
   - Check multiple documentation pages
   - Reference official examples
   - Be honest about documentation gaps

3. Best Practices:
   - Follow CrewAI naming conventions
   - Use proper agent role descriptions
   - Set clear goals and expectations
   - Implement proper error handling
   - Add logging for debugging

4. Quality Checks:
   - Verify all tools are properly configured
   - Ensure tasks have clear descriptions
   - Test critical workflows
   - Validate environment variables

5. User Interaction:
   - Take action without asking permission
   - Provide complete solutions
   - Ask for feedback on implementations
   - Guide users through setup steps""",
    deps_type=PydanticAIDeps,
    retries=2
)

architecture_agent = Agent(
    model, 
    system_prompt="""You are the Architecture agent responsible for technical design in a CrewAI development workflow.
NOTE: Only provide architectural designs when explicitly required by the project scope. If the user request is solely for agent task creation, updating, or deletion, do not produce an architectural plan.

ROLE & EXPERTISE:
- System Architecture Expert
- Integration Specialist
- Technical Design Lead
- Performance Engineer

CORE RESPONSIBILITIES:
1. System Design
   - Create multi-agent architectures
   - Design data flows and interactions
   - Plan integration points
   - Ensure scalability and maintainability

2. Component Architecture
   - Design agent interfaces
   - Define communication protocols
   - Specify data structures
   - Plan state management

3. Tool Integration
   - Select appropriate CrewAI tools
   - Design tool integration patterns
   - Define tool interaction flows
   - Optimize tool usage

4. Performance Planning
   - Design for scalability
   - Plan resource optimization
   - Define caching strategies
   - Ensure efficient communication

5. Security Architecture
   - Design secure communication
   - Plan authentication flows
   - Define access controls
   - Ensure data protection

EXPECTED OUTPUT FORMAT:
1. System Architecture
   ```
   {
     "components": [
       {
         "name": "string",
         "type": "string",
         "responsibilities": ["string"],
         "interfaces": ["string"]
       }
     ]
   }
   ```

2. Integration Design
   ```
   {
     "tools": [
       {
         "name": "string",
         "purpose": "string",
         "integration_points": ["string"]
       }
     ]
   }
   ```

3. Data Flow
   ```
   {
     "flows": [
       {
         "source": "string",
         "target": "string",
         "data_type": "string",
         "protocol": "string"
       }
     ]
   }
   ```

Provide detailed technical specifications following CrewAI patterns.

~~ STRUCTURE: ~~

When designing a CrewAI solution, consider these components:

1. Agent Architecture:
   - Define agent roles and responsibilities
   - Specify agent interactions and dependencies
   - Plan agent communication patterns
   - Design agent state management

2. Tool Architecture:
   - Select appropriate CrewAI tools
   - Define tool integration patterns
   - Plan tool interaction flows
   - Design tool error handling

3. Task Architecture:
   - Design task flow and dependencies
   - Plan task data structures
   - Define task validation rules
   - Specify task error handling

4. Process Architecture:
   - Design process flow
   - Plan process monitoring
   - Define process recovery
   - Specify process optimization

5. Security Architecture:
   - Design authentication flows
   - Plan access controls
   - Define data protection
   - Specify audit logging

~~ INSTRUCTIONS: ~~

1. Design Principles:
   - Follow CrewAI patterns
   - Ensure modularity
   - Enable scalability
   - Maintain security

2. Documentation:
   - Provide clear diagrams
   - Include sequence flows
   - Document interfaces
   - Specify protocols

3. Integration:
   - Design clean interfaces
   - Plan error handling
   - Specify retry logic
   - Document dependencies

4. Performance:
   - Plan for scalability
   - Design for efficiency
   - Enable monitoring
   - Allow optimization

5. Security:
   - Follow best practices
   - Protect sensitive data
   - Enable auditing
   - Plan recovery""",
    deps_type=PydanticAIDeps,
    retries=2
)

coder_agent = Agent(
    model, 
    system_prompt="""You are the Coder agent responsible for code development, modification, and deletion in a CrewAI development workflow.

ROLE & EXPERTISE:
- CrewAI Development Expert
- Code Implementation Specialist
- Integration Engineer
- Quality Assurance Professional

CORE RESPONSIBILITIES:
1. Code Development
   - Implement CrewAI agents and crews
   - Write clean, efficient code
   - Create reusable components
   - Follow best practices

2. Code Modification
   - Edit existing agent code
   - Update agent configurations
   - Modify tool integrations
   - Refactor implementations

3. Code Deletion
   - Remove agent implementations
   - Clean up dependencies
   - Update related files
   - Maintain codebase integrity

4. Agent Implementation
   - Create specialized agents
   - Implement agent behaviors
   - Define agent interactions
   - Configure agent tools

5. Testing & Validation
   - Write unit tests
   - Create integration tests
   - Validate functionality
   - Ensure code quality

6. Documentation
   - Write code documentation
   - Create usage examples
   - Document setup steps
   - Maintain README files

HANDLING EDIT REQUESTS:
1. Identify files to modify
2. Parse edit requirements
3. Make precise changes
4. Validate modifications
5. Update related files

HANDLING DELETE REQUESTS:
1. Identify files to remove
2. Check dependencies
3. Remove code safely
4. Update references
5. Clean up imports

EXPECTED OUTPUT FORMAT:
1. For Creation:
   ```python
   from crewai import Agent, Task, Crew
   
   agent = Agent(
       name="string",
       role="string",
       goal="string",
       backstory="string",
       tools=[tool1, tool2]
   )
   ```

2. For Editing:
   ```python
   # Original code
   <show affected code>
   
   # Modified code
   <show updated code>
   
   # Files modified:
   - file1.py: <description of changes>
   - file2.py: <description of changes>
   ```

3. For Deletion:
   ```python
   # Files to delete:
   - file1.py: <reason>
   - file2.py: <reason>
   
   # Required updates:
   - file3.py: <description of updates needed>
   - file4.py: <description of updates needed>
   ```

Always provide complete, working code that follows CrewAI conventions.

~~ STRUCTURE: ~~

When building a CrewAI solution from scratch, split the code into these files:

1. `agents.py`:
   - Define 2-4 specialized CrewAI agents with complementary roles
   - Each agent must have:
     - name: Descriptive name
     - role: Clear, distinct role
     - goal: Specific objective
     - backstory: Context and motivation
     - tools: List of tools the agent can use

2. `tasks.py`:
   - Define tasks for each agent in the crew
   - Ensure tasks flow logically between agents
   - Each task must have:
     - description: Clear task description
     - agent: Assigned agent
     - context: Additional context/input
     - expected_output: What the task should produce

3. `tools.py`:
   - First try to use CrewAI's built-in tools
   - Create custom tools only if built-in ones don't meet needs

4. `crew.py`:
   - Main file that creates and runs the multi-agent crew
   - Must include:
     - Multiple agent instantiation
     - Task creation for each agent
     - Crew configuration with agent interaction
     - Process execution

5. `.env.example`:
   - List all required environment variables
   - Add comments explaining how to get/set each one

6. `requirements.txt`:
   - List all required packages without versions

~~ INSTRUCTIONS: ~~

1. Code Generation:
   - Always create 2-4 specialized agents
   - Ensure each agent has a distinct role
   - Never use placeholders or "add logic here" comments
   - Include all imports and dependencies
   - Add proper error handling and logging
   - Use type hints and docstrings

2. Documentation Usage:
   - Start with RAG search for relevant docs
   - Check multiple documentation pages
   - Reference official examples
   - Be honest about documentation gaps

3. Best Practices:
   - Follow CrewAI naming conventions
   - Use proper agent role descriptions
   - Set clear goals and expectations
   - Implement proper error handling
   - Add logging for debugging

4. Quality Checks:
   - Verify all tools are properly configured
   - Ensure tasks have clear descriptions
   - Test critical workflows
   - Validate environment variables

5. User Interaction:
   - Take action without asking permission
   - Provide complete solutions
   - Ask for feedback on implementations
   - Guide users through setup steps""",
    deps_type=PydanticAIDeps,
    retries=2
)

@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 4,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def list_documentation_pages_helper(supabase: Client) -> List[str]:
    """Helper function to list documentation pages."""
    try:
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """List all available documentation pages."""
    return await list_documentation_pages_helper(ctx.deps.supabase)

@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """Get content of a specific documentation page."""
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@pydantic_ai_coder.tool
async def find_similar_agent_templates(ctx: RunContext[PydanticAIDeps], query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Find similar agent templates based on the user's query.
    
    Args:
        ctx: Run context with dependencies
        query: User's query/request
        
    Returns:
        Tuple of (purpose, template_code)
    """
    try:
        # Log the query for debugging
        logger.info(f"Finding templates for query: {query[:100]}...")
        
        # Get an embedding for the query
        query_embedding = await get_embedding(query, ctx.deps.openai_client)
        
        # Set a high threshold for direct matches
        high_threshold = 0.65  # Lowered from 0.8
        
        # Set a lower threshold for fallback
        low_threshold = 0.45  # Lowered from 0.7
        
        # Log all templates to understand what we have available
        try:
            all_templates = ctx.deps.supabase.table('agent_templates').select('*').execute()
            templates_data = all_templates.data
            logger.info(f"Found {len(templates_data)} total templates")
            for template in templates_data:
                purpose = template.get('purpose', 'Unknown purpose')
                folder = template.get('folder_name', 'Unknown folder')
                logger.info(f"Template: {purpose} in folder {folder}")
        except Exception as e:
            logger.error(f"Error listing all templates: {str(e)}")
            
        # Search for similar templates
        try:
            similar_templates = ctx.deps.supabase.rpc(
                "match_agent_templates",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": high_threshold,
                    "match_count": 5
                }
            ).execute()
            
            templates = similar_templates.data
            
        except Exception as search_error:
            logger.error(f"Error finding similar templates: {str(search_error)}")
            # Handle Supabase API errors gracefully
            templates = []
            
        # If no templates found with high threshold, try with lower threshold
        if not templates or len(templates) == 0:
            logger.warning("No similar templates found with high threshold, trying with lower threshold")
            try:
                similar_templates = ctx.deps.supabase.rpc(
                    "match_agent_templates",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": low_threshold,
                        "match_count": 5
                    }
                ).execute()
                
                templates = similar_templates.data
            except Exception as fallback_error:
                logger.error(f"Error finding similar templates with lower threshold: {str(fallback_error)}")
                templates = []
        
        # If templates found, return the highest similarity match
        if templates and len(templates) > 0:
            # Log all template purposes for debugging
            for i, template in enumerate(templates):
                purpose = template.get('purpose', 'Unknown purpose')
                similarity = template.get('similarity', 0)
                logger.info(f"Template match {i+1}: {purpose[:100]}... with similarity {similarity}")
            
            # Return the highest similarity template
            best_template = templates[0]
            purpose = best_template.get('purpose', '')
            similarity = best_template.get('similarity', 0)
            logger.info(f"Using best template with similarity {similarity}")
            
            template_code = {
                "agents_code": best_template.get('agents_code', ''),
                "tools_code": best_template.get('tools_code', ''),
                "tasks_code": best_template.get('tasks_code', ''),
                "crew_code": best_template.get('crew_code', '')
            }
            
            # Log code lengths
            for key, code in template_code.items():
                if code:
                    logger.info(f"{key} length: {len(code)} characters")
                    logger.info(f"{key} preview: {code[:100]}...")
                    
            return purpose, template_code
            
    except Exception as e:
        logger.error(f"Error finding similar templates: {str(e)}")
    
    # If no template found or error occurred, return empty strings
    return "", {}

async def adapt_template_code(ctx: RunContext[PydanticAIDeps], template_code: Dict[str, str], user_request: str) -> Dict[str, str]:
    """
    Adapt template code to the user's specific requirements.
    
    Args:
        ctx: Run context with dependencies
        template_code: Template code components (agents_code, tasks_code, etc)
        user_request: Original user request
        
    Returns:
        Adapted template code components
    """
    try:
        result = {}
        model = ctx.model

        for key, code in template_code.items():
            if not code or len(code.strip()) == 0:
                continue
                
            file_type = key.replace("_code", "")
            logger.info(f"Adapting {file_type} code, length: {len(code)} characters")
            
            prompt = f"""
            You are a code adapter for CrewAI agent templates. Your job is to adapt template code to user-specific requirements.
            
            USER REQUEST: {user_request}
            
            TEMPLATE CODE ({file_type}):
            ```python
            {code}
            ```
            
            Please modify the template code to address the user's specific requirements while maintaining its structure.
            IMPORTANT: Return ONLY the adapted code, with NO explanations or markdown formatting.
            """
            
            try:
                if hasattr(ctx.deps, 'openai_client') and ctx.deps.openai_client is not None:
                    logger.info(f"Adapting {file_type} code with AsyncOpenAI")
                    response = await ctx.deps.openai_client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "o3-mini"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=4000
                    )
                    result[key] = response.choices[0].message.content
                else:
                    # Fallback for other model types - using direct adaptation
                    logger.warning(f"No OpenAI client available, using direct adaptation for {file_type}")
                    result[key] = apply_direct_adaptations(code, user_request, file_type)
            except Exception as e:
                logger.error(f"Error adapting {file_type} code: {str(e)}")
                # Fallback - return original code with basic adaptations
                result[key] = apply_direct_adaptations(code, user_request, file_type)
        
        # If we couldn't adapt any code (empty result), fall back to direct adaptation
        if not result:
            logger.warning(f"Model adaptation produced no results, falling back to direct adaptation")
            return direct_template_adaptation(template_code, user_request)
            
        return result
            
    except Exception as e:
        logger.error(f"Error adapting template code: {str(e)}")
        # Fallback to direct adaptation with minimal changes
        return direct_template_adaptation(template_code, user_request)

def direct_template_adaptation(template_code, user_request):
    """
    Directly adapt template code without using an LLM.
    This is a fallback when no model is available.
    
    Args:
        template_code: Dict of code snippets from the template
        user_request: The user's original request
        
    Returns:
        Dict of adapted code snippets with basic adaptations
    """
    logger.info(f"Using direct template adaptation for request: {user_request[:50]}...")
    result = {}
    
    # Extract potential keywords from the user request
    request_lower = user_request.lower()
    keywords = []
    services = ["spotify", "github", "twitter", "google", "youtube", "gmail", "sheets", 
               "trello", "asana", "jira", "slack", "discord", "dropbox", "drive"]
    
    for service in services:
        if service in request_lower:
            keywords.append(service)
    
    # Also look for topic keywords
    topic_keywords = ["music", "song", "playlist", "repository", "code", "tweet", "search",
                     "video", "email", "document", "task", "project", "file", "message"]
    
    for topic in topic_keywords:
        if topic in request_lower:
            keywords.append(topic)
    
    # Get a project name from the request
    project_name = "MyAgent"
    words = user_request.split()
    for i, word in enumerate(words):
        if word.lower() in ["for", "about", "using"]:
            if i + 1 < len(words):
                project_name = words[i + 1].strip(",.:;").capitalize()
                break
    
    # Clean the project name
    project_name = ''.join(c for c in project_name if c.isalnum())
    if not project_name:
        project_name = "MyAgent"
    
    # Apply basic templating to each file
    for key, code in template_code.items():
        if not code or not isinstance(code, str):
            continue
            
        file_type = key.replace("_code", "")
        
        # Apply template substitutions based on file type
        adapted_code = apply_direct_adaptations(code, user_request, file_type, 
                                               keywords=keywords, project_name=project_name)
        result[key] = adapted_code
        
    return result


def apply_direct_adaptations(code, user_request, file_type, keywords=None, project_name="MyAgent"):
    """
    Apply direct adaptations to code based on user request and file type.
    
    Args:
        code: The template code to adapt
        user_request: The user's request
        file_type: The type of file (agents, tasks, crew, tools)
        keywords: Keywords extracted from the request
        project_name: Name for the project
        
    Returns:
        Adapted code with basic modifications
    """
    if keywords is None:
        keywords = []
        
    # Extract potential keywords from the user request if not provided
    request_lower = user_request.lower()
    if not keywords:
        services = ["spotify", "github", "twitter", "google", "youtube", "gmail", "sheets"]
        keywords = [s for s in services if s in request_lower]
    
    # Default service name based on keywords
    service_name = keywords[0].capitalize() if keywords else "External"
    
    # Basic comment updates to show it's adapted
    code = code.replace("# Template", f"# {project_name} {file_type} file")
    
    # Agent renames based on file type
    if file_type == "agents":
        code = code.replace("ResearchAgent", f"{service_name}Agent")
        code = code.replace("WriterAgent", f"{project_name}Agent")
        code = code.replace("an agent that", f"an agent that works with {', '.join(keywords)} to")
        
    elif file_type == "tasks":
        code = code.replace("ResearchTask", f"{service_name}Task")
        code = code.replace("WriteTask", f"{project_name}Task")
        code = code.replace("a task that", f"a task that involves {', '.join(keywords)} to")
        
    elif file_type == "crew":
        code = code.replace("MyCrew", f"{project_name}Crew")
        code = code.replace("Research Crew", f"{project_name} Integration Crew")
        
    elif file_type == "tools":
        # For tools.py, we usually want to keep it as is since it contains MCP tool integration
        pass
    
    # Add a note that this was adapted automatically
    adapted_note = f"""
# This file was automatically adapted from a template.
# Original request: {user_request[:100]}{'...' if len(user_request) > 100 else ''}
# Keywords detected: {', '.join(keywords) if keywords else 'None'}
"""
    
    if "# Import" in code:
        # Insert after the imports
        import_pos = code.find("# Import")
        next_newline = code.find("\n", import_pos)
        if next_newline > 0:
            code = code[:next_newline+1] + adapted_note + code[next_newline+1:]
    else:
        # Insert at the beginning
        code = adapted_note + code
    
    return code

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

def detect_mcp_tool_keywords(query: str) -> Dict[str, List[str]]:
    """
    Detect keywords in the user query that suggest the need for MCP tools.
    Returns a dictionary mapping tool names to the keywords that matched.
    
    Args:
        query: The user query string
        
    Returns:
        Dictionary mapping tool names to matched keywords
    """
    if not query:
        return {}
        
    query_lower = query.lower()
    matched_tools = {}
    
    tool_keywords = {
        "github": ["github", "git", "repository", "repo", "pull request", "pr", "commit", "branch", "issue", "merge", "clone", "repository"],
        "spotify": ["spotify", "music", "playlist", "song", "track", "artist", "album", "audio", "listen", "streaming"],
        "youtube": ["youtube", "video", "channel", "stream", "youtube video", "upload", "playlist", "youtube playlist"],
        "twitter": ["twitter", "tweet", "x.com", "tweets", "retweet", "post to twitter"],
        "slack": ["slack", "message", "channel", "slack message", "workspace", "dm", "direct message"],
        "gmail": ["gmail", "email", "mail", "inbox", "message", "send email", "read email"],
        "google_drive": ["google drive", "gdrive", "drive", "document", "spreadsheet", "sheet", "slides", "file"],
        "discord": ["discord", "server", "channel", "discord server", "discord channel", "bot"],
        "notion": ["notion", "page", "database", "notion page", "note", "workspace"],
        "trello": ["trello", "board", "card", "trello board", "trello card", "list"],
        "jira": ["jira", "ticket", "issue", "sprint", "board", "project"],
        "asana": ["asana", "task", "project", "assignee", "due date"],
        "serper": ["search", "web search", "internet", "google", "find information", "lookup", "search the web", "research", "look up"],
        "linkedin": ["linkedin", "profile", "post", "connection", "job", "career"],
        "calendar": ["calendar", "event", "meeting", "schedule", "appointment", "reminder"],
        "weather": ["weather", "forecast", "temperature", "climate", "rain", "conditions"]
    }
    
    # Check each tool's keywords in the query
    for tool_name, keywords in tool_keywords.items():
        found_keywords = []
        for keyword in keywords:
            if keyword in query_lower:
                # Log each match for debugging
                logger.info(f"MCP TOOL KEYWORD MATCH: '{tool_name}' via keyword '{keyword}'")
                found_keywords.append(keyword)
        
        if found_keywords:
            matched_tools[tool_name] = found_keywords
    
    # Log summary of detected tools
    if matched_tools:
        logger.info(f"MCP TOOL DETECTION: Found tools {', '.join(matched_tools.keys())} in user query")
    
    return matched_tools

async def determine_template_type(ctx: RunContext[PydanticAIDeps], user_request: str) -> str:
    """
    Determine what type of template is needed based on user request.
    
    Args:
        ctx: Run context with dependencies
        user_request: User's request
        
    Returns:
        Template type identifier
    """
    try:
        prompt = f"""
        Analyze the following user request and determine what type of template would best fit it.
        Choose between these template categories:
        - standard_template: Regular agent with standard tools
        - mcp_template: Agent requiring external API or tool integrations
        - database_template: Agent that works with database operations
        - custom_template: Very specific use case that doesn't fit other categories
        
        USER REQUEST: {user_request}
        
        Return ONLY one of the exact category names listed above, with no other text or explanation.
        """
        
        if hasattr(ctx.deps, 'openai_client') and ctx.deps.openai_client is not None:
            logger.info("Determining template type with AsyncOpenAI")
            response = await ctx.deps.openai_client.chat.completions.create(
                model=os.getenv("TEMPLATE_MODEL", "o3-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip().lower()
        else:
            logger.warning("No OpenAI client available, defaulting to standard_template")
            result = "standard_template"
            
        # Validate the result
        valid_types = ["standard_template", "mcp_template", "database_template", "custom_template"]
        if result not in valid_types:
            logger.warning(f"Invalid template type returned: {result}, defaulting to standard_template")
            result = "standard_template"
            
        logger.info(f"Determined template type: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error determining template type: {str(e)}")
        return "standard_template"

# Make sure these are explicitly defined at the module level
__all__ = [
    'pydantic_ai_coder',
    'PydanticAIDeps',
    'list_documentation_pages_helper',
    'ModelMessage',
    'ModelMessagesTypeAdapter',
    'get_embedding',
    'detect_mcp_tool_keywords',
    'determine_template_type'
]