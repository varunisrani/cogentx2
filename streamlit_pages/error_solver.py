import os
import re
import sys
import json
import time
import logging
import inspect
import streamlit as st
import traceback
import subprocess
import asyncio
import glob
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any, Optional, Set, Tuple, Union, cast
import difflib
import ast
import importlib.util
from pathlib import Path
import functools
import hashlib
import concurrent.futures
import threading

# Import MCP Cogentx functions
try:
    from mcp.cogentx import mcp_cogentx_create_thread, mcp_cogentx_run_agent
except ImportError:
    # If the module doesn't exist yet, define placeholder functions
    def mcp_cogentx_create_thread(random_string: str) -> str:
        logger.info("Creating new Cogentx thread")
        import uuid
        return str(uuid.uuid4())
    
    def mcp_cogentx_run_agent(thread_id: str, user_input: str) -> str:
        logger.info(f"Running agent with thread ID: {thread_id}")
        return f"This is a placeholder response. The real MCP Cogentx integration is not available.\n\nQuery: {user_input}\n\nAgentType: Search Agent"

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import Pydantic models from workbench
try:
    from workbench.models import ErrorDetails, TracebackEntry, FixResult
except ImportError:
    # Define models locally if not available
    from pydantic import BaseModel, Field
    
    class TracebackEntry(BaseModel):
        """Model for a single traceback entry with strong typing"""
        file_path: str
        file_name: str
        line_number: int
        line_content: Optional[str] = None
    
    class ErrorDetails(BaseModel):
        """Structured error information with validation"""
        original_message: str
        error_type: Optional[str] = None
        file_name: Optional[str] = None
        line_number: Optional[int] = None
        error_message: Optional[str] = None
        traceback: Optional[List[TracebackEntry]] = None
        suggested_fixes: List[str] = Field(default_factory=list)
        related_files: List[str] = Field(default_factory=list)
        file_contents: Dict[str, str] = Field(default_factory=dict)
        ai_analysis: Optional[str] = None
        analysis_error: Optional[str] = None
    
    class FixResult(BaseModel):
        """Results from fix attempts with validation"""
        success: bool
        message: str
        fixed_content: Optional[str] = None
        file_name: Optional[str] = None
        file_path: Optional[str] = None
        timestamp: Optional[str] = None

# Create a global instance of the ErrorSolver to prevent multiple instances
_ERROR_SOLVER_INSTANCE = None

# Initialize O3-Mini client
client = OpenAI()

# Initialize Cogentx thread ID
if "cogentx_thread_id" not in st.session_state:
    st.session_state.cogentx_thread_id = None

# Add a simple caching mechanism
class SimpleCache:
    """Simple in-memory cache to reduce redundant API calls and file operations"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove a random item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

# Create global cache instances
_API_CACHE = SimpleCache()
_FILE_CACHE = SimpleCache(max_size=200)

def cached_api_call(func):
    """Decorator for caching expensive API calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function name and arguments
        key_parts = [func.__name__]
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
        
        cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        
        # Check if result is in cache
        cached_result = _API_CACHE.get(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached result for {func.__name__}")
            return cached_result
        
        # Call function and cache result
        result = func(*args, **kwargs)
        _API_CACHE.set(cache_key, result)
        return result
    
    return wrapper

async def create_agent(agent_type: str, description: str) -> Dict[str, Any]:
    """Create a new agent using Cogentx."""
    try:
        # Create a new thread if we don't have one
        if not st.session_state.cogentx_thread_id:
            thread_response = mcp_cogentx_create_thread(random_string="init")
            st.session_state.cogentx_thread_id = thread_response

        # Format the agent creation request
        user_input = f"""Create a new {agent_type} agent with the following description:
{description}

Please implement:
1. Agent setup with proper configuration
2. Error handling and logging
3. Required MCP server integration
4. System prompt and tools
5. Example usage"""

        # Call Cogentx to create the agent
        response = mcp_cogentx_run_agent(
            thread_id=st.session_state.cogentx_thread_id,
            user_input=user_input
        )

        return {
            "success": True,
            "message": "Agent created successfully",
            "response": response
        }
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return {
            "success": False,
            "message": f"Error creating agent: {str(e)}"
        }

class CodeDependencyAnalyzer:
    """Analyze code dependencies and relationships between files"""
    
    def __init__(self):
        self.import_cache = {}
        self.file_ast_cache = {}
        self.dependency_graph = {}
    
    def parse_file(self, file_path: str) -> Optional[ast.Module]:
        """Parse a Python file into an AST"""
        if file_path in self.file_ast_cache:
            return self.file_ast_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                self.file_ast_cache[file_path] = tree
                return tree
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return None
    
    def find_imports(self, file_path: str) -> Dict[str, str]:
        """Extract imports from a Python file"""
        if file_path in self.import_cache:
            return self.import_cache[file_path]
        
        imports = {}
        tree = self.parse_file(file_path)
        if not tree:
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.name] = name.asname or name.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    full_name = f"{module}.{name.name}" if module else name.name
                    imports[full_name] = name.asname or name.name
        
        self.import_cache[file_path] = imports
        return imports
    
    def find_class_definitions(self, file_path: str) -> List[str]:
        """Find class definitions in a file"""
        classes = []
        tree = self.parse_file(file_path)
        if not tree:
            return classes
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    
    def find_related_files(self, file_path: str, search_dirs: List[str], max_depth: int = 3, 
                          current_depth: int = 0, visited: Optional[Set[str]] = None) -> List[str]:
        """Recursively find files related to the given file"""
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or file_path in visited:
            return []
        
        visited.add(file_path)
        related_files = []
        
        # Find imports in the current file
        imports = self.find_imports(file_path)
        if not imports:
            return related_files
        
        # Convert imports to possible file paths
        for import_name in imports:
            # Convert import name to file path
            file_segments = import_name.split('.')
            
            for search_dir in search_dirs:
                # Try as a direct module
                potential_paths = []
                
                # Try as a module
                module_path = os.path.join(search_dir, *file_segments) + '.py'
                potential_paths.append(module_path)
                
                # Try as a package with __init__.py
                package_path = os.path.join(search_dir, *file_segments, '__init__.py')
                potential_paths.append(package_path)
                
                # Check all potential paths
                for path in potential_paths:
                    if os.path.exists(path) and path not in visited:
                        related_files.append(path)
                        # Recursively find related files
                        if current_depth < max_depth - 1:
                            deeper_files = self.find_related_files(
                                path, search_dirs, max_depth, current_depth + 1, visited
                            )
                            related_files.extend(deeper_files)
        
        return related_files

# Add a timeout wrapper for API calls
def timeout(timeout_seconds=30):
    """Decorator to add timeout to a function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            completed = [False]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                    completed[0] = True
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if completed[0]:
                return result[0]
            if error[0]:
                raise error[0]
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        
        return wrapper
    return decorator

# Update get_o3_mini_analysis with timeout
@cached_api_call
@timeout(timeout_seconds=20)  # Set 20 second timeout for API calls
def get_o3_mini_analysis(error_details: ErrorDetails, file_contents: Dict[str, str] = None) -> Dict[str, Any]:
    """Get comprehensive error analysis from OpenAI model with improved context handling and timeout."""
    try:
        # Extract relevant error information
        error_type = error_details.error_type or "Unknown"
        error_message = error_details.error_message or ""
        file_name = error_details.file_name or "Unknown"
        line_number = error_details.line_number or "Unknown"
        
        # Get file contents - either use provided or from error_details
        if file_contents is None:
            file_contents = error_details.file_contents or {}
        
        # Prepare context about the error and files
        context = f"""Error Type: {error_type}
Error Message: {error_message}
File with Error: {file_name}
Line Number: {line_number}

Relevant Files and Their Contents:
"""
        # Start with the main file
        main_file_content = file_contents.get(file_name, "")
        if main_file_content:
            # Limit content to 50 lines max for faster processing
            main_file_lines = main_file_content.splitlines()[:50]
            context += f"\n--- {file_name} (MAIN ERROR FILE) ---\n" + "\n".join(main_file_lines) + "\n"

        # Add context from other related files - limit to 2 files and 30 lines each for performance
        file_count = 0
        MAX_RELATED_FILES = 2
        
        for rel_file, content in file_contents.items():
            if file_count >= MAX_RELATED_FILES:
                break
                
            if rel_file != file_name:
                # Include only first 30 lines for context to keep prompt size reasonable
                short_content = "\n".join(content.splitlines()[:30])
                if short_content.strip():  # Only add if there's actual content
                    context += f"\n--- {rel_file} ---\n" + short_content + "\n"
                    file_count += 1

        # Get analysis from OpenAI with a more concise system prompt
        system_prompt = """You are an expert Python developer specialized in debugging code errors.
Analyze the root cause, understand dependencies, and provide specific fixes.
Keep your analysis brief and actionable."""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000  # Limit response length for faster processing
            )
            
            analysis = response.choices[0].message.content
            
            # Extract suggested fixes from the analysis with improved parsing
            suggested_fixes = []
            if analysis:
                # Better pattern for extracting code blocks and suggestions
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', analysis, re.DOTALL)
                fixes = re.findall(r'(?:^\d+\.|^\-)\s*(.+?)(?=(?:\n\d+\.|\n\-|\n\n|$))', analysis, re.MULTILINE)
                
                # Add code blocks as fixes
                for block in code_blocks:
                    if block.strip():
                        suggested_fixes.append(f"Replace with: {block.strip()}")
                
                # Add bullet points as fixes
                for fix in fixes:
                    fix_text = fix.strip()
                    if fix_text and not any(f.endswith(fix_text) for f in suggested_fixes):
                        suggested_fixes.append(fix_text)
            
            return {
                "success": True,
                "analysis": analysis,
                "suggested_fixes": suggested_fixes
            }
        except TimeoutError:
            logger.error("OpenAI analysis timed out")
            return {"success": False, "error": "Analysis timed out"}
        except Exception as e:
            logger.error(f"Error getting OpenAI analysis: {str(e)}")
            return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error in get_o3_mini_analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def get_related_files(file_path: str, search_dirs: Optional[List[str]] = None) -> List[str]:
    """Enhanced function to get related Python files based on imports and dependencies."""
    if search_dirs is None:
        # Default to common directory locations
        base_dir = os.path.dirname(file_path)
        search_dirs = [
            base_dir,
            os.path.dirname(base_dir),  # Parent directory
            os.path.join(os.path.dirname(base_dir), "workbench"),
            "workbench", 
            "."
        ]
    
    # Use the advanced dependency analyzer
    analyzer = CodeDependencyAnalyzer()
    related_files = analyzer.find_related_files(file_path, search_dirs)
    
    # Add the original file to the list if it exists
    if os.path.exists(file_path) and file_path not in related_files:
        related_files.append(file_path)
    
    return related_files

def get_error_solver():
    """Get or create a singleton instance of ErrorSolver."""
    global _ERROR_SOLVER_INSTANCE
    if _ERROR_SOLVER_INSTANCE is None:
        logger.info("Creating new ErrorSolver instance")
        _ERROR_SOLVER_INSTANCE = ErrorSolver()
    return _ERROR_SOLVER_INSTANCE

# Common errors and their solutions with examples from agent_tools.py
COMMON_ERRORS = {
    "ModuleNotFoundError": {
        "description": "Python cannot find a module that was imported",
        "solutions": [
            "Check if the module is installed in your environment",
            "Make sure the module name is spelled correctly",
            "Verify that the module path is correct if it's a local module",
            "Install the missing module with pip"
        ],
        "examples": [
            {
                "error": "ModuleNotFoundError: No module named 'openai'",
                "context": "In agent_tools.py when using AsyncOpenAI client",
                "fix": "Add 'openai' to requirements.txt and run pip install openai"
            }
        ]
    },
    "ImportError": {
        "description": "Python cannot import a specific function or class from a module",
        "solutions": [
            "Check if the module has the specified function or class",
            "Verify the function or class name is spelled correctly",
            "Make sure the module is properly installed",
            "Check that the import statement follows the correct format"
        ],
        "examples": [
            {
                "error": "ImportError: cannot import name 'AsyncOpenAI' from 'openai'",
                "context": "In agent_tools.py when importing OpenAI client",
                "fix": "Update import to: from openai import AsyncOpenAI"
            }
        ]
    },
    "AttributeError": {
        "description": "Python cannot find an attribute or method on an object",
        "solutions": [
            "Check if the object has the specified attribute or method",
            "Verify the attribute or method name is spelled correctly",
            "Ensure the object is initialized before accessing its attributes",
            "Check the documentation for the correct attribute or method name"
        ],
        "examples": [
            {
                "error": "AttributeError: 'Client' object has no attribute 'from_table'",
                "context": "In agent_tools.py when using Supabase client",
                "fix": "Use client.table('table_name').select() instead of from_table"
            }
        ]
    },
    "SyntaxError": {
        "description": "The code has a syntax error and cannot be parsed",
        "solutions": [
            "Check for missing parentheses, brackets, or quotes",
            "Verify indentation is consistent",
            "Look for missing colons in function definitions, class definitions, or control structures",
            "Ensure that strings are properly closed"
        ]
    },
    "NameError": {
        "description": "A name (variable, function, etc.) is used but not defined",
        "solutions": [
            "Define the variable or function before using it",
            "Check the spelling of the variable or function name",
            "Verify the variable or function is defined in the correct scope",
            "Import any necessary modules or functions"
        ]
    },
    "TypeError": {
        "description": "An operation is applied to a value of the wrong type",
        "solutions": [
            "Check the data types of the values being used",
            "Convert values to the appropriate types before operations",
            "Verify function arguments match the expected types",
            "Check for misuse of operator symbols"
        ],
        "examples": [
            {
                "error": "TypeError: object AsyncOpenAI can't be used in 'await' expression",
                "context": "In agent_tools.py when using async functions",
                "fix": "Ensure the function is called with 'await' and is inside an async function"
            }
        ]
    },
    "IndexError": {
        "description": "An index is out of range for a list, tuple, or other sequence",
        "solutions": [
            "Verify the index is within the valid range",
            "Check if the sequence is empty before accessing elements",
            "Use len() to determine the valid range of indices",
            "Add boundary checks before accessing elements by index"
        ]
    },
    "KeyError": {
        "description": "A key is not found in a dictionary",
        "solutions": [
            "Verify the key exists in the dictionary before accessing it",
            "Use dict.get(key, default) to provide a default value",
            "Check the spelling of the key",
            "Print the dictionary keys to verify available keys"
        ]
    },
    "FileNotFoundError": {
        "description": "Python cannot find the specified file",
        "solutions": [
            "Check if the file exists at the specified path",
            "Verify the file path is correct",
            "Ensure the working directory is correct",
            "Use absolute paths instead of relative paths"
        ]
    },
    "ValueError": {
        "description": "A function receives an argument of the right type but invalid value",
        "solutions": [
            "Check the range or format of input values",
            "Verify string formats match expected patterns",
            "Ensure numerical values are within valid ranges",
            "Add validation before passing values to functions"
        ]
    },
    "ZeroDivisionError": {
        "description": "Division or modulo by zero",
        "solutions": [
            "Add checks to prevent division by zero",
            "Use try/except to handle potential zero division",
            "Verify denominators are non-zero before division",
            "Provide alternative calculations for zero cases"
        ]
    },
    "FileStructureError": {
        "description": "Issues with Python file organization and imports",
        "solutions": [
            "Ensure proper __init__.py files exist in packages",
            "Use correct relative imports",
            "Follow the project's file structure convention",
            "Check import paths match directory structure"
        ],
        "examples": [
            {
                "error": "ModuleNotFoundError: No module named 'models'",
                "context": "Common error when importing models.py",
                "structure": """
                project_root/
                ├── main.py
                ├── models.py
                ├── agents.py
                ├── tools.py
                └── utils/
                    ├── __init__.py
                    └── helpers.py
                """,
                "fix": "Add __init__.py files and use correct import paths"
            }
        ]
    },
    "AsyncError": {
        "description": "Issues with async/await usage in agent code",
        "solutions": [
            "Ensure all async functions are properly awaited",
            "Use async with async functions",
            "Handle coroutines correctly",
            "Check event loop management"
        ],
        "examples": [
            {
                "error": "RuntimeError: Cannot call async function without await",
                "context": "In agent_tools.py async functions",
                "code_example": """
                # Wrong:
                result = get_embedding(text, client)
                
                # Correct:
                result = await get_embedding(text, client)
                """,
                "fix": "Add 'await' keyword before async function calls"
            }
        ]
    },
    "AgentError": {
        "description": "Issues with AI agent implementation and MCP servers",
        "solutions": [
            "Ensure proper agent initialization and configuration",
            "Check MCP server setup and connections",
            "Verify system prompts and model configurations",
            "Handle async operations correctly",
            "Implement proper error handling for agent tools"
        ],
        "examples": [
            {
                "error": "RuntimeError: MCP server initialization failed",
                "context": "When setting up Serper-Spotify agent",
                "code_example": """
# Correct agent setup
async def setup_agent(config: Config) -> Agent:
    try:
        # Create MCP server instances
        serper_server = create_serper_mcp_server(config.SERPER_API_KEY)
        spotify_server = create_spotify_mcp_server(config.SPOTIFY_API_KEY)
        
        # Initialize agent with both servers
        agent = Agent(get_model(config), mcp_servers=[serper_server, spotify_server])
        agent.system_prompt = system_prompt
        return agent
    except Exception as e:
        logging.error(f"Error setting up agent: {str(e)}")
        raise
""",
                "fix": "Properly initialize MCP servers and handle exceptions"
            },
            {
                "error": "TypeError: Object of type MCPServerStdio is not JSON serializable",
                "context": "When creating MCP servers for Serper or Spotify",
                "code_example": """
def create_serper_mcp_server(serper_api_key):
    try:
        config = {"serperApiKey": serper_api_key}
        mcp_args = [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@marcopesani/mcp-server-serper",
            "--config",
            json.dumps(config)
        ]
        return MCPServerStdio("npx", mcp_args)
    except Exception as e:
        logging.error(f"Error creating Serper MCP server: {str(e)}")
        raise
""",
                "fix": "Ensure proper JSON serialization of MCP server configurations"
            },
            {
                "error": "AttributeError: 'Agent' object has no attribute 'system_prompt'",
                "context": "When setting up agent system prompt",
                "code_example": """
# Combined system prompt for Serper-Spotify agent
system_prompt = \"\"\"
You are a powerful assistant with dual capabilities:

1. Web Search (Serper): You can search the web for information using the Serper API
   - Search for any information on the internet
   - Retrieve news articles and general knowledge
   - Find images and news about specific topics

2. Spotify Music: You can control and interact with Spotify
   - Search for songs, albums, and artists
   - Create and manage playlists
   - Control playback (play, pause, skip)
   - Get user library information

IMPORTANT USAGE NOTES:
- For Spotify search operations, a 'market' parameter is required
- For web searches, be specific about what information you're looking for
- Use web search for knowledge questions
- Use Spotify tools for music-related requests
\"\"\"

# Set the system prompt
agent.system_prompt = system_prompt
""",
                "fix": "Initialize agent with proper system prompt and capabilities"
            },
            {
                "error": "RuntimeError: Cannot call async function without await",
                "context": "When running agent queries",
                "code_example": """
async def run_query(agent, user_query: str) -> tuple:
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Execute the query
        result = await agent.run(user_query)
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        return (result, elapsed_time, [])
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise
""",
                "fix": "Use proper async/await syntax for agent operations"
            }
        ]
    },
    "MCPServerError": {
        "description": "Issues with MCP server setup and tool execution",
        "solutions": [
            "Verify MCP server configuration",
            "Check API keys and authentication",
            "Ensure proper tool invocation",
            "Handle server connection issues",
            "Implement proper error handling for MCP tools"
        ],
        "examples": [
            {
                "error": "MCPServerError: Tool execution failed",
                "context": "When executing MCP tools",
                "code_example": """
async def execute_mcp_tool(server: MCPServerStdio, tool_name: str, params: dict = None):
    try:
        logging.info(f"Executing MCP tool: {tool_name}")
        if params:
            logging.info(f"With parameters: {json.dumps(params, indent=2)}")
            
        # Execute through server session
        if hasattr(server, 'session') and server.session:
            result = await server.session.invoke_tool(tool_name, params or {})
        elif hasattr(server, 'invoke_tool'):
            result = await server.invoke_tool(tool_name, params or {})
        else:
            raise Exception(f"Cannot find way to invoke tool {tool_name}")
            
        return result
    except Exception as e:
        logging.error(f"Error executing MCP tool '{tool_name}': {str(e)}")
        raise
""",
                "fix": "Implement proper MCP tool execution with error handling"
            },
            {
                "error": "ConfigError: Missing required API keys",
                "context": "When loading agent configuration",
                "code_example": """
class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SERPER_API_KEY: str
    SPOTIFY_API_KEY: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        load_dotenv()
        
        # Check required environment variables
        missing_vars = []
        for var in ["LLM_API_KEY", "SERPER_API_KEY", "SPOTIFY_API_KEY"]:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        return cls(
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY"),
            SPOTIFY_API_KEY=os.getenv("SPOTIFY_API_KEY")
        )
""",
                "fix": "Implement proper configuration validation and error handling"
            }
        ]
    }
}

# Update safe_read_file to use file caching
def safe_read_file(file_path):
    """Read a file with multiple encoding attempts and error handling, with caching."""
    # Check cache first
    cached_content = _FILE_CACHE.get(file_path)
    if cached_content is not None:
        return cached_content
        
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                # Cache the result
                _FILE_CACHE.set(file_path, content)
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading file {file_path} with {encoding} encoding: {str(e)}")
            break
    
    # If all encodings fail, try binary mode and replace invalid characters
    try:
        with open(file_path, 'rb') as f:
            binary_content = f.read()
            # Try to decode with replacement
            content = binary_content.decode('utf-8', errors='replace')
            # Cache the result
            _FILE_CACHE.set(file_path, content)
            return content
    except Exception as e:
        logger.error(f"Failed to read file {file_path} in binary mode: {str(e)}")
    
    return None  # Return None if all attempts fail

class ErrorSolver:
    """Enhanced class to analyze and fix Python errors with comprehensive dependency analysis."""
    
    def __init__(self, workbench_dir="workbench", search_depth=3):
        """Initialize the error solver with advanced configuration."""
        self.workbench_dir = workbench_dir
        self.search_depth = search_depth
        self.analyzer = CodeDependencyAnalyzer()
        self.visited_files = set()
        
        # Create workbench directory if it doesn't exist
        if not os.path.exists(workbench_dir):
            try:
                os.makedirs(workbench_dir)
                logger.info(f"Created workbench directory: {workbench_dir}")
            except Exception as e:
                logger.error(f"Error creating workbench directory: {str(e)}")

    def analyze_error(self, error_message: str) -> ErrorDetails:
        """Advanced error analysis with optimized file scanning for better performance."""
        # Create cache key for the error message
        cache_key = f"analysis:{hashlib.md5(error_message.encode()).hexdigest()}"
        cached_analysis = _API_CACHE.get(cache_key)
        if cached_analysis:
            logger.info("Using cached error analysis")
            return cached_analysis
            
        # Initialize error details with validated structure
        error_details = ErrorDetails(
            original_message=error_message,
            error_type=None,
            file_name=None,
            line_number=None,
            error_message=None,
            traceback=None,
            suggested_fixes=[],
            related_files=[],
            file_contents={}
        )

        try:
            # If error message is too short or malformed, extract what we can
            if not error_message or len(error_message.strip()) < 10:
                logger.warning(f"Error message too short or empty: '{error_message}'")
                error_details.error_message = error_message
                return error_details
                
            # Extract traceback entries using regex with improved pattern for more flexibility
            traceback_entries = re.findall(r'File [\"\'"]?([^\"\'"\n]+)[\"\'"]?, line (\d+)(?:,? in (\w+))?', error_message)
            logger.info(f"Found {len(traceback_entries)} traceback entries")
            
            # If no entries found, try a simpler pattern
            if not traceback_entries:
                # Try another pattern that might catch different traceback formats
                traceback_entries = re.findall(r'at ([^:]+):(\d+)', error_message)
                logger.info(f"Second pattern found {len(traceback_entries)} traceback entries")
                
                # If still no entries, try to extract error type and message
                if not traceback_entries:
                    # Extract any file references
                    file_refs = re.findall(r'[\'\"]([^\'\"]+\.py)[\'\"]', error_message)
                    if file_refs:
                        file_path = file_refs[0]
                        error_details.file_name = os.path.basename(file_path)
                        error_details.related_files.append(file_path)
                        
                        # Try to load the file content
                        try:
                            if os.path.exists(file_path):
                                content = safe_read_file(file_path)
                                if content:
                                    error_details.file_contents[error_details.file_name] = content
                        except Exception as e:
                            logger.error(f"Error loading referenced file: {str(e)}")
            
            # Store traceback information with enhanced context
            traceback_info = []
            for entry_tuple in traceback_entries:
                # Handle both 2-tuple and 3-tuple formats from different regex patterns
                if len(entry_tuple) >= 3:
                    file_path, line_number, function_name = entry_tuple
                else:
                    file_path, line_number = entry_tuple
                    function_name = None
                    
                file_name = os.path.basename(file_path)
                line_content = None
                
                # Try to extract the line content if the file exists
                try:
                    if os.path.exists(file_path):
                        content = safe_read_file(file_path)
                        if content:
                            lines = content.splitlines()
                            line_num = int(line_number)
                            if 0 < line_num <= len(lines):
                                line_content = lines[line_num - 1].strip()
                except Exception as e:
                    logger.error(f"Error reading line content: {str(e)}")
                
                traceback_info.append(TracebackEntry(
                    file_path=file_path,
                    file_name=file_name,
                    line_number=int(line_number),
                    line_content=line_content
                ))
            
            if traceback_info:
                error_details.traceback = traceback_info
                
                # Set the main file to the last traceback entry (usually the error source)
                error_details.file_name = traceback_info[-1].file_name
                error_details.line_number = traceback_info[-1].line_number
                
                # Only get files directly from traceback - skip deep dependency scanning for speed
                self.visited_files = set()
                for entry in traceback_info:
                    if entry.file_path not in error_details.related_files:
                        error_details.related_files.append(entry.file_path)
            
            # Extract error type and message with improved pattern
            error_match = re.search(r'((?:\w+\.)*\w+Error|Exception): (.+?)(?:\n|$)', error_message)
            if error_match:
                error_details.error_type = error_match.group(1)
                error_details.error_message = error_match.group(2).strip()
                logger.info(f"Identified error type: {error_details.error_type}")
            else:
                # Try alternative patterns for different error formats
                # Look for common error keywords if standard pattern fails
                error_keywords = ["error", "exception", "failed", "failure", "invalid", "not found", "cannot"]
                lines = error_message.split("\n")
                
                for line in lines:
                    line = line.lower().strip()
                    if any(keyword in line for keyword in error_keywords):
                        # Extract the error line and try to determine type
                        if ":" in line:
                            parts = line.split(":", 1)
                            error_type = parts[0].strip().title()
                            error_msg = parts[1].strip()
                            
                            # Only use if it looks like an error type
                            if "error" in error_type.lower() or "exception" in error_type.lower():
                                error_details.error_type = error_type
                                error_details.error_message = error_msg
                                logger.info(f"Identified alternate error type: {error_type}")
                                break
                        else:
                            # If no clear type, use the line as the message
                            if not error_details.error_message:
                                error_details.error_message = line
                                error_details.error_type = "UnknownError"
                                logger.info("Using error line as message")
                                break
            
            # Load main file content with better error handling - limit to just what we need
            if error_details.file_name:
                main_file_path = None
                
                # Try to find the file in the traceback
                if error_details.traceback:
                    for entry in error_details.traceback:
                        if entry.file_name == error_details.file_name:
                            main_file_path = entry.file_path
                        break
                
                # If found, try to load the content
                if main_file_path:
                    try:
                        content = safe_read_file(main_file_path)
                        if content:
                            error_details.file_contents[error_details.file_name] = content
                            logger.info(f"Loaded content for main file: {main_file_path}")
                    except Exception as e:
                        logger.error(f"Error loading main file: {str(e)}")
                        
                        # Try relative path in workbench with multiple attempts - but limit to just a few paths
                        possible_paths = [
                            os.path.join(self.workbench_dir, error_details.file_name),
                            os.path.join(".", error_details.file_name)
                        ]
                        
                        for path in possible_paths:
                            try:
                                if os.path.exists(path):
                                    content = safe_read_file(path)
                                    if content:
                                        error_details.file_contents[error_details.file_name] = content
                                        logger.info(f"Loaded content from alternative path: {path}")
                                        break
                            except Exception as e2:
                                logger.error(f"Error loading from path {path}: {str(e2)}")
            
            # Load content for only essential related files - limit to a maximum of 3 for speed
            MAX_RELATED_FILES = 3
            loaded_files = 0
            
            for file_path in error_details.related_files:
                if loaded_files >= MAX_RELATED_FILES:
                    break
                    
                file_name = os.path.basename(file_path)
                
                # Skip if we already loaded this file
                if file_name in error_details.file_contents:
                    continue
                
                try:
                    if os.path.exists(file_path):
                        content = safe_read_file(file_path)
                        if content:
                            error_details.file_contents[file_name] = content
                            logger.info(f"Loaded content for related file: {file_path}")
                            loaded_files += 1
                except Exception as e:
                    logger.error(f"Error loading related file: {str(e)}")
            
            # If we still don't have sufficient context, scan limited set of files
            if len(error_details.file_contents) < 2:  # We want at least a couple of files for context
                logger.info("Limited file context, scanning project for additional relevant files")
                try:
                    # Limited project file search - only look in key directories
                    project_files = []
                    key_dirs = [self.workbench_dir, "."]
                    
                    # Use a more targeted approach - avoid deep recursive scanning
                    MAX_FILES_TO_SCAN = 50
                    files_scanned = 0
                    
                    for search_dir in key_dirs:
                        if files_scanned >= MAX_FILES_TO_SCAN:
                            break
                            
                        if os.path.exists(search_dir) and os.path.isdir(search_dir):
                            # Avoid deep directory scanning - use glob instead of os.walk
                            py_files = glob.glob(os.path.join(search_dir, "*.py"))
                            py_files.extend(glob.glob(os.path.join(search_dir, "*", "*.py")))
                            
                            for py_file in py_files:
                                if files_scanned >= MAX_FILES_TO_SCAN:
                                    break
                                project_files.append(py_file)
                                files_scanned += 1
                    
                    # Only search a limited set of key files
                    # First try to include key project files that are often relevant
                    key_files = ["__init__.py", "main.py", "app.py", "streamlit_ui.py", "error_solver.py"]
                    additional_files = 0
                    MAX_ADDITIONAL_FILES = 3  # Limit additional files 
                    
                    for key_file in key_files:
                        if additional_files >= MAX_ADDITIONAL_FILES:
                            break
                            
                        for file_path in project_files:
                            if file_path.endswith("/" + key_file) and os.path.basename(file_path) not in error_details.file_contents:
                                try:
                                    content = safe_read_file(file_path)
                                    if content:
                                        file_name = os.path.basename(file_path)
                                        error_details.file_contents[file_name] = content
                                        error_details.related_files.append(file_path)
                                        logger.info(f"Added key project file: {file_path}")
                                        additional_files += 1
                                        if additional_files >= MAX_ADDITIONAL_FILES:
                                            break
                                except Exception as e:
                                    logger.error(f"Error loading key file: {str(e)}")
                except Exception as e:
                    logger.error(f"Error scanning project files: {str(e)}")
            
            # Cache the analysis results
            _API_CACHE.set(cache_key, error_details)
            
            # Now that we have comprehensive file contents, get AI analysis
            if error_details.file_contents:
                analysis_result = get_o3_mini_analysis(error_details, error_details.file_contents)
                if analysis_result.get("success", False):
                    error_details.ai_analysis = analysis_result.get("analysis", "")
                    error_details.suggested_fixes = analysis_result.get("suggested_fixes", [])
                    logger.info(f"Got AI analysis with {len(error_details.suggested_fixes)} suggested fixes")
            
            return error_details
        
        except Exception as e:
            logger.error(f"Error in analyze_error: {str(e)}")
            error_details.analysis_error = str(e)
            return error_details

    def fix_error(self, error_details: ErrorDetails) -> FixResult:
        """Apply fixes to errors based on comprehensive error analysis."""
        if not error_details or not error_details.file_name:
            return FixResult(success=False, message="No file to fix")
        
        try:
            # Get the main file info
            file_name = error_details.file_name
            file_contents = error_details.file_contents
            
            if file_name not in file_contents:
                return FixResult(success=False, message=f"Cannot find content for file {file_name}")
            
            file_content = file_contents[file_name]
            
            # Get the AI fix with enhanced accuracy
            fix_result = self.get_ai_fix(error_details, file_content)
            
            if not fix_result.get("success", False):
                return FixResult(
                    success=False, 
                    message="Failed to generate a fix",
                    file_name=file_name
                )
            
            # Get the fixed content
            fixed_content = fix_result.get("fixed_content")
            
            # Determine file path with robust resolution
            file_path = None
            
            # Try to find the full path from traceback
            if error_details.traceback:
                for entry in error_details.traceback:
                    if entry.file_name == file_name:
                        file_path = entry.file_path
                        break
            
            # Fall back to workbench path
            if not file_path or not os.path.exists(file_path):
                file_path = os.path.join(self.workbench_dir, file_name)
            
            # Create a proper FixResult object
            return FixResult(
                success=True,
                message=fix_result.get("message", "Fixed successfully"),
                fixed_content=fixed_content,
                file_name=file_name,
                file_path=file_path,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.error(f"Error in fix_error: {str(e)}")
            return FixResult(
                success=False, 
                message=f"Error applying fix: {str(e)}",
                file_name=error_details.file_name
            )

    def get_ai_fix(self, error_details: ErrorDetails, file_content: str) -> Dict[str, Any]:
        """Generate AI-assisted fixes for the error with advanced context-awareness and timeout."""
        try:
            error_type = error_details.error_type or "Unknown"
            error_message = error_details.error_message or ""
            line_number = error_details.line_number
            file_name = error_details.file_name or ""
            fixed_content = file_content
            
            logger.info(f"Generating AI fix for {error_type} in {file_name}")
            
            # Check if we already have a cached fix
            cache_key = f"fix:{error_type}:{file_name}:{hashlib.md5(file_content.encode()).hexdigest()[:8]}"
            cached_fix = _API_CACHE.get(cache_key)
            if cached_fix:
                logger.info("Using cached fix")
                return cached_fix
            
            # For all errors, use an advanced AI-powered approach with better context
            try:
                # Limit the number of files sent to the model for faster processing
                MAX_RELATED_FILES = 2
                
                # Prepare context from related files to help the model understand dependencies
                related_context = ""
                related_file_count = 0
                for rel_file, content in error_details.file_contents.items():
                    if rel_file != file_name and content and isinstance(content, str):
                        if related_file_count >= MAX_RELATED_FILES:
                            break
                        # Include a brief summary of each related file - limit to 10 lines for performance
                        file_lines = content.splitlines() if hasattr(content, 'splitlines') else []
                        summary = "\n".join(file_lines[:10]) if file_lines else ""  # Just show the first 10 lines
                        related_context += f"\n--- {rel_file} ---\n{summary}\n...\n"
                        related_file_count += 1
                
                # Create a detailed, focused prompt for the AI model
                prompt = f"""
Fix the following {error_type} in a Python file:

Error Message: {error_message}
File: {file_name}
Line Number: {line_number if line_number else 'Unknown'}

Code with the error:
```python
{file_content}
```

"""
                # Add related files context separately (avoiding backslash in f-string expression)
                if related_context:
                    prompt += f"Related files context (abbreviated):\n{related_context}\n\n"
                
                prompt += """
Apply a precise fix that addresses the root cause of this error.
Return only the complete, fixed code without explanations or markdown.
"""
                
                # Get AI response with improved model selection
                model_choice = "gpt-4o-mini"  # Default to a powerful model for better fixes
                
                try:
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": "You are an expert Python developer specialized in fixing code errors. Return only the fixed code without explanations or markdown formatting."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Get the fixed content from the response
                    ai_fix = response.choices[0].message.content.strip()
                    
                    # If the response is wrapped in code blocks, extract just the code
                    ai_fix = re.sub(r'^```\w*\n', '', ai_fix)
                    ai_fix = re.sub(r'\n```$', '', ai_fix)
                    
                    # Verify the fix isn't just the original code with improved comparison
                    if ai_fix != file_content:
                        # Calculate the difference percentage for better validation
                        import difflib
                        diff = difflib.ndiff(file_content.splitlines(), ai_fix.splitlines())
                        changes = sum(1 for line in diff if line.startswith('+ ') or line.startswith('- '))
                        total_lines = max(len(file_content.splitlines()), len(ai_fix.splitlines()))
                        change_percentage = (changes / total_lines) * 100 if total_lines > 0 else 0
                        
                        # Only accept fixes with meaningful but not excessive changes
                        if 0 < change_percentage < 80:
                            logger.info(f"AI generated a fix for {error_type} with {change_percentage:.1f}% changes")
                            return {
                                "success": True,
                                "message": f"Applied AI-generated fix for {error_type}",
                                "fixed_content": ai_fix
                            }
                        elif change_percentage >= 80:
                            logger.warning(f"AI fix rejected - too many changes ({change_percentage:.1f}%)")
                        else:
                            logger.warning("AI fix rejected - minimal changes detected")
                except Exception as e:
                    logger.error(f"Error generating enhanced AI fix: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in advanced AI fix generation: {str(e)}")
                # Fall back to rule-based fixes for specific error types
                # (Rest of the rule-based fixes code)
                return {
                    "success": False,
                    "message": f"Could not automatically fix {error_type}",
                    "fixed_content": file_content
                }
        except Exception as e:
            logger.error(f"Error in get_ai_fix: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating fix: {str(e)}"
            }

    def extract_traceback_details(self, traceback_entries, content=None, source_file=None):
        file_line_map = {}
        if not traceback_entries:
            logger.warning("No traceback entries found")
            return file_line_map
        
        for entry in traceback_entries:
            file_path = entry.get("file")
            if not file_path or not isinstance(file_path, str):
                continue
                
            line_number = entry.get("line")
            if not line_number or not isinstance(line_number, int) or line_number <= 0:
                continue
                
            # Check if we already have content for this file
            if source_file and file_path == source_file and content:
                file_content = content
            else:
                try:
                    file_content = safe_read_file(file_path)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    continue
            
            if not file_content:
                logger.warning(f"No content found for file {file_path}")
                continue
                
            # Get the specified line and surrounding context
            try:
                lines = file_content.splitlines()
                if line_number <= len(lines):
                    line_content = lines[line_number - 1]
                    
                    # Get context lines
                    start_line = max(0, line_number - 3)
                    end_line = min(len(lines), line_number + 2)
                    context_lines = lines[start_line:end_line]
                    
                    # Store extracted information
                    if file_path not in file_line_map:
                        file_line_map[file_path] = {}
                    
                    file_line_map[file_path][line_number] = {
                        "line": line_content,
                        "context": context_lines,
                        "start_line": start_line + 1  # Convert to 1-indexed
                    }
                else:
                    logger.warning(f"Line number {line_number} exceeds file length ({len(lines)}) for {file_path}")
            except Exception as e:
                logger.error(f"Error extracting line {line_number} from {file_path}: {str(e)}")
        
        return file_line_map

def update_progress(step_name, completed=False):
    """Update the progress tracking with a new step."""
    if "processing_steps" not in st.session_state:
        st.session_state.processing_steps = []
    
    # Add timestamp to the step
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Update existing step if it exists
    for step in st.session_state.processing_steps:
        if step["name"] == step_name:
            step["status"] = "complete" if completed else "running"
            step["time"] = timestamp
            return
    
    # Add new step
    st.session_state.processing_steps.append({
        "name": step_name,
        "status": "complete" if completed else "running",
        "time": timestamp
    })

def clear_progress():
    """Clear all progress tracking."""
    if "processing_steps" in st.session_state:
        st.session_state.processing_steps = []
    if "log_messages" in st.session_state:
        st.session_state.log_messages = []

def error_solver_tab():
    """Enhanced Error Solver UI with improved multi-file handling and interactive workflow."""
    st.title("🛠️ Advanced AI Error Solver")
    
    # Initialize session state variables with better structure
    if "error_text" not in st.session_state:
        st.session_state.error_text = ""
    if "error_analysis" not in st.session_state:
        st.session_state.error_analysis = None
    if "error_fix_results" not in st.session_state:
        st.session_state.error_fix_results = None
    if "error_solver" not in st.session_state:
        st.session_state.error_solver = ErrorSolver()
    if "show_examples" not in st.session_state:
        st.session_state.show_examples = False
    if "recent_fixes" not in st.session_state:
        st.session_state.recent_fixes = []
    if "proposed_changes" not in st.session_state:
        st.session_state.proposed_changes = {}
    if "selected_files_content" not in st.session_state:
        st.session_state.selected_files_content = {}
    if "processing_step" not in st.session_state:
        st.session_state.processing_step = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = 0
    if "processing_times" not in st.session_state:
        st.session_state.processing_times = {}
    if "use_fast_mode" not in st.session_state:
        st.session_state.use_fast_mode = True
    
    def analyze_error():
        """Enhanced error analysis with detailed progress tracking and timing."""
        if st.session_state.error_text.strip():
            # Reset states
            st.session_state.error_analysis = None
            st.session_state.error_fix_results = None
            st.session_state.proposed_changes = {}
            st.session_state.analysis_complete = False
            st.session_state.processing_times = {}
            
            # Track processing steps
            st.session_state.processing_step = "analyzing"
            
            try:
                # Record start time
                start_time = time.time()
                
                with st.spinner("📊 Analyzing error..."):
                    # Perform the analysis with comprehensive detection
                    st.session_state.error_analysis = st.session_state.error_solver.analyze_error(st.session_state.error_text)
                    
                    # Record analysis time
                    analysis_time = time.time() - start_time
                    st.session_state.processing_times["analysis"] = f"{analysis_time:.2f}s"
                    
                    # Store original file contents for comparison and validation
                    if st.session_state.error_analysis and st.session_state.error_analysis.file_contents:
                        st.session_state.selected_files_content = st.session_state.error_analysis.file_contents.copy()
                        
                        # Fast mode - only generate fix for main file
                        if st.session_state.use_fast_mode:
                            # Generate fix proposals for main file only
                            st.session_state.processing_step = "generating_fixes"
                            
                            main_file = st.session_state.error_analysis.file_name
                            if main_file and main_file in st.session_state.selected_files_content:
                                # Record start time for fix generation
                                fix_start_time = time.time()
                                
                                # Generate fix for main file
                                with st.spinner("🔧 Generating fix for main error file..."):
                                    fix_result = st.session_state.error_solver.get_ai_fix(
                                        st.session_state.error_analysis,
                                        st.session_state.selected_files_content[main_file]
                                    )
                                    if fix_result.get("success", False):
                                        st.session_state.proposed_changes[main_file] = fix_result
                                
                                # Record fix generation time
                                fix_time = time.time() - fix_start_time
                                st.session_state.processing_times["fix"] = f"{fix_time:.2f}s"
                        else:
                            # Normal mode - check related files too
                            # Generate fix proposals for all affected files
                            st.session_state.processing_step = "generating_fixes"
                            
                            main_file = st.session_state.error_analysis.file_name
                            if main_file and main_file in st.session_state.selected_files_content:
                                # Record start time for fix generation
                                fix_start_time = time.time()
                                
                                # Generate fix for main file
                                with st.spinner("🔧 Generating fix for main error file..."):
                                    fix_result = st.session_state.error_solver.get_ai_fix(
                                        st.session_state.error_analysis,
                                        st.session_state.selected_files_content[main_file]
                                    )
                                    if fix_result.get("success", False):
                                        st.session_state.proposed_changes[main_file] = fix_result
                                
                                # Check if we need fixes for other related files
                                if st.session_state.error_analysis.ai_analysis:
                                    other_files_mentioned = False
                                    analysis_text = st.session_state.error_analysis.ai_analysis
                                    
                                    # Look for mentions of other files in the analysis
                                    for file_name in st.session_state.selected_files_content.keys():
                                        if file_name != main_file and file_name in analysis_text:
                                            other_files_mentioned = True
                                            with st.spinner(f"🔧 Generating fix for related file: {file_name}..."):
                                                fix_result = st.session_state.error_solver.get_ai_fix(
                                                    st.session_state.error_analysis,
                                                    st.session_state.selected_files_content[file_name]
                                                )
                                                if fix_result.get("success", False):
                                                    st.session_state.proposed_changes[file_name] = fix_result
                                
                                # Record fix generation time
                                fix_time = time.time() - fix_start_time
                                st.session_state.processing_times["fix"] = f"{fix_time:.2f}s"
                    
                    # Record total time
                    total_time = time.time() - start_time
                    st.session_state.processing_times["total"] = f"{total_time:.2f}s"
                    
                    st.session_state.processing_step = None
                    st.session_state.analysis_complete = True
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                logger.error(f"Analysis error: {str(e)}")
                st.session_state.processing_step = None
    
    def apply_fix():
        """Apply fixes with user confirmation and detailed tracking."""
        fixed_files = []
        
        try:
            st.session_state.processing_step = "applying_fixes"
            
            with st.spinner("🔄 Applying selected fixes..."):
                for file_name, fix_info in st.session_state.proposed_changes.items():
                    if fix_info.get("apply", False):  # Only apply fixes that were confirmed
                        original_content = st.session_state.selected_files_content.get(file_name, "")
                        fixed_content = fix_info.get("fixed_content", "")
                        
                        # Skip if no changes
                        if original_content == fixed_content:
                            continue
                        
                        # Determine file path with robust resolution
                        file_path = None
                        
                        # Try to find the full path from the error analysis
                        if (st.session_state.error_analysis and st.session_state.error_analysis.traceback):
                            for entry in st.session_state.error_analysis.traceback:
                                if entry.file_name == file_name:
                                    file_path = entry.file_path
                                    break
                        
                        # If not found, check in workbench
                        if not file_path or not os.path.exists(file_path):
                            workbench_path = os.path.join("workbench", file_name)
                            if os.path.exists(workbench_path):
                                file_path = workbench_path
                        
                        # Apply the fix with careful validation
                        if file_path:
                            try:
                                # Create backup first
                                backup_path = f"{file_path}.bak"
                                try:
                                    content = safe_read_file(file_path)
                                    if content:
                                        with open(backup_path, "w", encoding="utf-8") as dst:
                                            dst.write(content)
                                        logger.info(f"Created backup at {backup_path}")
                                except Exception as e:
                                    logger.error(f"Error creating backup: {str(e)}")
                                
                                # Write the fixed content
                                try:
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        f.write(fixed_content)
                                    
                                    fixed_files.append({
                                        "file_name": file_name,
                                        "file_path": file_path,
                                        "message": fix_info.get("message", "Fixed successfully"),
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "backup_path": backup_path if os.path.exists(backup_path) else None
                                    })
                                    logger.info(f"Applied fix to {file_path}: {fix_info.get('message', 'Fixed successfully')}")
                                except Exception as e:
                                    logger.error(f"Error writing file {file_path}: {str(e)}")
                            except Exception as e:
                                logger.error(f"Error applying fix to {file_path}: {str(e)}")
            
            # Add to recent fixes list with careful management
            if fixed_files:
                st.session_state.recent_fixes.extend(fixed_files)
                # Keep only the latest 10 fixes
                st.session_state.recent_fixes = st.session_state.recent_fixes[-10:]
                # Reset states for next analysis
                st.session_state.error_fix_results = fixed_files
                st.session_state.proposed_changes = {}
            
            st.session_state.processing_step = None
            
        except Exception as e:
            st.error(f"Error applying fixes: {str(e)}")
            logger.error(f"Fix application error: {str(e)}")
            st.session_state.processing_step = None
    
    # UI Layout with enhanced layout and guidance
    st.markdown("""
    <style>
    .diff-added { color: #28a745; font-weight: bold; }
    .diff-removed { color: #dc3545; font-weight: bold; }
    .diff-unchanged { color: #6c757d; }
    .status-success { color: #28a745; font-weight: bold; }
    .status-running { color: #17a2b8; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main UI columns with responsive design
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        # Error input section with better guidance
        st.subheader("🔍 Enter Error Message")
        
        # Toggle for examples with better examples
        if st.checkbox("Show common error examples", value=st.session_state.show_examples):
            st.session_state.show_examples = True
            st.info("""
            **Example 1: NameError**
            ```
            Traceback (most recent call last):
              File "app.py", line 25, in <module>
                result = process_data(data)
              File "app.py", line 10, in process_data
                formatted = print_data(data)
            NameError: name 'print_data' is not defined
            ```
            
            **Example 2: ImportError**
            ```
            Traceback (most recent call last):
              File "ml_model.py", line 3, in <module>
                import tensorflow as tf
            ModuleNotFoundError: No module named 'tensorflow'
            ```
            
            **Example 3: SyntaxError**
            ```
            File "utils.py", line 15
                if x == 5
                        ^
            SyntaxError: invalid syntax
            ```
            
            **Example 4: Multi-file Error**
            ```
            Traceback (most recent call last):
              File "main.py", line 5, in <module>
                from data_processor import process
              File "/Users/dev/project/data_processor.py", line 12, in <module>
                from utils import format_data
              File "/Users/dev/project/utils.py", line 8, in <module>
                class DataFormatter
            SyntaxError: expected ':'
            ```
            """)
        else:
            st.session_state.show_examples = False
        
        # Error text area with better placeholder
        st.text_area(
            "Error Message", 
            key="error_text", 
            height=200, 
            placeholder="Paste the full error message including traceback here for best results...",
            help="For best results, include the complete traceback which shows the chain of function calls that led to the error."
        )
        
        # Processing status indicator
        if st.session_state.processing_step:
            step_text = {
                "analyzing": "🔍 Analyzing error...",
                "generating_fixes": "🔧 Generating fixes...",
                "applying_fixes": "✅ Applying fixes..."
            }.get(st.session_state.processing_step, "Processing...")
            
            st.info(step_text)
        
        # Analyze button with better UX
        analyze_button = st.button(
            "🔍 Analyze Error", 
            on_click=analyze_error, 
            disabled=not st.session_state.error_text.strip() or st.session_state.processing_step is not None,
            help="Analyze the error to identify the cause and potential fixes"
        )
        
        # Analysis Results with tabbed interface for better navigation
        if st.session_state.error_analysis:
            st.subheader("📊 Error Analysis Results")
            
            error_type = st.session_state.error_analysis.error_type or "Unknown Error"
            error_msg = st.session_state.error_analysis.error_message or ""
            file_name = st.session_state.error_analysis.file_name or "Unknown file"
            line_number = st.session_state.error_analysis.line_number or "?"
            
            # Show error details with better formatting
            st.markdown(f"**Error Type:** `{error_type}`")
            st.markdown(f"**Message:** `{error_msg}`")
            st.markdown(f"**Main File:** `{file_name}`")
            st.markdown(f"**Line:** `{line_number if line_number else 'Unknown'}`")
            
            # Display AI analysis if available
            if st.session_state.error_analysis.ai_analysis:
                with st.expander("🤖 AI Analysis", expanded=True):
                    st.markdown(st.session_state.error_analysis.ai_analysis)
            
            # Display traceback info with better visualization
            if st.session_state.error_analysis.traceback:
                with st.expander("🔍 Traceback Information"):
                    for idx, entry in enumerate(st.session_state.error_analysis.traceback):
                        file_info = f"`{entry.file_name}`"
                        line_info = f"line `{entry.line_number}`"
                        content_info = f"```python\n{entry.line_content}\n```" if entry.line_content else ""
                        
                        st.markdown(f"**{idx+1}.** {file_info}, {line_info}")
                        if content_info:
                            st.markdown(content_info)
            
            # Display file contents with improved tabbed interface
            if st.session_state.error_analysis.file_contents:
                st.subheader("📄 File Contents and Proposed Fixes")
                
                # Create tabs for each file
                file_names = list(st.session_state.error_analysis.file_contents.keys())
                if file_names:
                    # Ensure main file is first
                    if file_name in file_names:
                        file_names.remove(file_name)
                        file_names.insert(0, file_name)
                    
                    tabs = st.tabs(file_names)
                    
                    for i, (tab_file_name, tab) in enumerate(zip(file_names, tabs)):
                        with tab:
                            content = st.session_state.error_analysis.file_contents[tab_file_name]
                            is_main_file = tab_file_name == file_name
                            
                            # Display original content with better formatting
                            st.markdown(f"**{tab_file_name}**{' (Main Error File)' if is_main_file else ''}")
                            
                            # Highlight affected line with advanced context
                            if is_main_file and line_number and content and isinstance(content, str):
                                lines = content.split('\n')
                                
                                if 0 < line_number <= len(lines):
                                    # Show context (5 lines before and after) with better highlighting
                                    start_line = max(0, line_number - 6)
                                    end_line = min(len(lines), line_number + 5)
                                    
                                    context_code = []
                                    for idx, line in enumerate(lines[start_line:end_line], start=start_line + 1):
                                        if idx == line_number:
                                            context_code.append(f"➡️ {idx}: {line}")
                                        else:
                                            context_code.append(f"   {idx}: {line}")
                                    
                                    st.code("\n".join(context_code), language="python")
                                    
                                    # Option to view full file
                                    with st.expander("View Full File"):
                                        st.code(content, language="python")
                                else:
                                    st.code(content, language="python")
                            else:
                                # Use an expander for long files
                                if content and isinstance(content, str) and len(content.splitlines()) > 50:
                                    with st.expander("View File Content"):
                                        st.code(content, language="python")
                                else:
                                    st.code(content, language="python")
                            
                            # Show fix proposals with enhanced diff view
                            if tab_file_name in st.session_state.proposed_changes:
                                fix_info = st.session_state.proposed_changes[tab_file_name]
                                
                                st.markdown(f"**Proposed Fix:** {fix_info.get('message', 'Unknown fix')}")
                                
                                # Display diff between original and fixed content with improved visualization
                                with st.expander("View Changes", expanded=True):
                                    original_lines = content.split('\n') if content and isinstance(content, str) else []
                                    fixed_lines = fix_info.get('fixed_content', '').split('\n') if fix_info.get('fixed_content') and isinstance(fix_info.get('fixed_content'), str) else []
                                    
                                    # Use difflib for better diff display
                                    diff_output = []
                                    d = difflib.Differ()
                                    diff = list(d.compare(original_lines, fixed_lines))
                                    
                                    for line in diff:
                                        if line.startswith('- '):
                                            diff_output.append(f"🔴 {line[2:]}")
                                        elif line.startswith('+ '):
                                            diff_output.append(f"🟢 {line[2:]}")
                                        elif line.startswith('  '):
                                            diff_output.append(f"   {line[2:]}")
                                    
                                    st.code("\n".join(diff_output), language="diff")
                                
                                # Checkbox to apply this fix with better confirmation
                                st.session_state.proposed_changes[tab_file_name]["apply"] = st.checkbox(
                                    f"✅ Apply fix to {tab_file_name}", 
                                    value=False,
                                    key=f"apply_{tab_file_name}",
                                    help="Select to apply this fix"
                                )
            
            # Apply Fix Button with improved confirmation
            if st.session_state.proposed_changes:
                has_selected_changes = any(fix.get("apply", False) for fix in st.session_state.proposed_changes.values())
                
                if has_selected_changes:
                    st.warning("⚠️ Applying fixes will modify your files. Backups will be created, but please review changes carefully.")
                
                fix_button = st.button(
                    "✅ Apply Selected Fixes", 
                    on_click=apply_fix,
                    disabled=not has_selected_changes or st.session_state.processing_step is not None,
                    help="Apply the selected fixes to your code files"
                )
    
    with right_col:
        # Help and guidance section first for better UX
        with st.expander("ℹ️ How to Use the Error Solver", expanded=not st.session_state.error_analysis):
            st.markdown("""
            ### How to Fix Errors:
            
            1. **Paste your error**: Include the complete traceback for best results
            2. **Analyze the error**: The AI will identify the problem and search for related files
            3. **Review the AI analysis**: See what the AI believes is causing the issue
            4. **Review suggested fixes**: Carefully examine the proposed changes in each file
            5. **Select and apply fixes**: Choose which fixes to apply by checking the boxes
            6. **Confirm changes**: Click "Apply Selected Fixes" to implement the changes
            
            ### Tips for Best Results:
            
            - Include the **full traceback** from your error, not just the last line
            - The solver can detect issues across **multiple files**
            - Always **review proposed changes** before applying them
            - For complex errors, you may need to fix **multiple files**
            - **Backups** are automatically created when applying fixes
            """)
        
        # Recent fixes list for better user tracking
        st.subheader("📋 Fix History")
        
        if st.session_state.error_fix_results:
            with st.expander("Recently Applied Fixes", expanded=True):
                for fix in st.session_state.error_fix_results:
                    st.success(f"✅ Fixed {fix['file_name']}: {fix['message']}")
                    st.markdown(f"**Path:** `{fix['file_path']}`")
                    st.markdown(f"**Applied at:** {fix['timestamp']}")
                    if fix.get('backup_path'):
                        st.markdown(f"**Backup:** `{fix['backup_path']}`")