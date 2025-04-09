from __future__ import annotations as _annotations

from dataclasses import dataclass
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

# Import template integration module
from .mcp_template_integration import generate_from_template

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'mcp_tools.log')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mcp_tools')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Reference examples of CrewAI-compatible MCP tools
REFERENCE_EXAMPLES = {
    "serper": '''
import os
import asyncio
import json
import smithery
import mcp
import logging
import datetime
import time
from mcp.client.websocket import websocket_client
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure enhanced logging
def setup_logging():
    """Set up detailed logging for Serper operations"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/serper_operations_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a specific logger for Serper operations
    logger = logging.getLogger("serper_operations")
    
    return logger, log_file

# Initialize logger
serper_logger, log_file_path = setup_logging()
serper_logger.info(f"Serper operations log initialized. Log file: {log_file_path}")

class SerperMCPToolParams(BaseModel):
    """Parameters for Serper tool operations"""
    operation: str = Field(
        description="The Serper operation to perform, such as search, scrape, etc."
    )
    parameters: Dict[str, Any] = Field(
        default={},
        description="Parameters for the Serper operation"
    )

class SerperMCPTool(BaseTool):
    """Tool for interacting with Serper API using MCP"""
    name: str = "serper_mcp_tool"
    description: str = "Interact with Serper API to perform search operations, web scraping, and research."
    args_schema: type[BaseModel] = SerperMCPToolParams
    serper_api_key: str = ""
    url: str = ""
    
    def __init__(self):
        super().__init__()
        # Load environment variables
        load_dotenv()
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        if not self.serper_api_key:
            serper_logger.error("Serper API key not found in environment variables")
            raise ValueError("SERPER_API_KEY not found in environment variables")
        
        # Create Smithery URL
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@marcopesani/mcp-server-serper/ws", 
            {
                "serperApiKey": self.serper_api_key
            }
        )
        
        # Cache for available operations
        self._available_operations = None
        
        # Try to preload available operations
        try:
            asyncio.run(self._preload_operations())
        except Exception as e:
            serper_logger.warning(f"Could not preload operations: {str(e)}")
        
        # Log initialization
        serper_logger.info("SerperMCPTool initialized")
        serper_logger.info(f"Serper API key: {self.serper_api_key[:5]}...{self.serper_api_key[-5:]}")
    
    async def _preload_operations(self):
        """Preload available operations for better logging"""
        serper_logger.info("Preloading available Serper operations...")
        async with websocket_client(self.url) as streams:
            serper_logger.info("Connection established for preloading operations")
            async with mcp.ClientSession(*streams) as session:
                self._available_operations = await self._get_available_operations(session)
                serper_logger.info(f"Preloaded {len(self._available_operations)} Serper operations")
                
                # Log all available operations
                serper_logger.info("=" * 80)
                serper_logger.info("AVAILABLE SERPER OPERATIONS:")
                for i, (op_name, op_info) in enumerate(self._available_operations.items(), 1):
                    serper_logger.info(f"{i}. {op_name}: {op_info.get('description', 'No description')}")
                    if op_info.get('required'):
                        serper_logger.info(f"   Required parameters: {', '.join(op_info.get('required', []))}")
                serper_logger.info("=" * 80)
    
    def _run(self, operation: str, parameters: Dict[str, Any] = None) -> str:
        """Run the Serper operation"""
        if parameters is None:
            parameters = {}
        
        # Log operation with distinct formatting
        serper_logger.info("=" * 80)
        serper_logger.info(f"OPERATION CALLED: {operation}")
        serper_logger.info("-" * 80)
        
        # Safely log parameters (remove sensitive data)
        safe_params = self._get_safe_parameters(parameters)
        serper_logger.info(f"PARAMETERS: {json.dumps(safe_params, indent=2)}")
        
        # Track execution time
        start_time = time.time()
        
        # Run the async operation and wait for result
        try:
            result = asyncio.run(self._run_async(operation, parameters))
            
            # Calculate execution time
            execution_time = time.time() - start_time
            serper_logger.info(f"EXECUTION TIME: {execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = self._get_result_summary(result)
            serper_logger.info(f"RESULT SUMMARY: {result_summary}")
            serper_logger.info("=" * 80)
            
            return result
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            serper_logger.error(f"ERROR ({execution_time:.2f}s): {str(e)}")
            serper_logger.info("=" * 80)
            return f"Error executing Serper operation: {str(e)}"
    
    def _get_safe_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of parameters safe for logging (no sensitive data)"""
        safe_params = parameters.copy()
        
        # Mask sensitive fields if present
        sensitive_fields = ['token', 'password', 'secret', 'key', 'auth']
        for field in sensitive_fields:
            for key in list(safe_params.keys()):
                if field in key.lower() and isinstance(safe_params[key], str):
                    safe_params[key] = f"{safe_params[key][:3]}...{safe_params[key][-3:]}"
        
        return safe_params
    
    def _get_result_summary(self, result: str) -> str:
        """Create a summary of the result for logging"""
        if not result:
            return "Empty result"
            
        # For JSON results
        try:
            result_obj = json.loads(result)
            
            if isinstance(result_obj, dict):
                # For dictionary responses
                if 'organic' in result_obj:
                    return f"Search returned {len(result_obj['organic'])} organic results"
                elif 'answerBox' in result_obj:
                    return f"Search returned answer box result"
                else:
                    keys = list(result_obj.keys())
                    return f"JSON object with keys: {', '.join(keys[:5])}" + ("..." if len(keys) > 5 else "")
            
            # For array responses
            elif isinstance(result_obj, list):
                return f"Array with {len(result_obj)} items"
                
        except json.JSONDecodeError:
            # For non-JSON results
            if len(result) > 100:
                return f"Text response ({len(result)} characters)"
            return result
        
        return "Result processed successfully"
    
    async def _run_async(self, operation: str, parameters: Dict[str, Any]) -> str:
        """Run the Serper operation asynchronously"""
        # Log connection attempt
        serper_logger.info("Connecting to Serper API...")
        
        # Connect to the Smithery server
        async with websocket_client(self.url) as streams:
            serper_logger.info("Connection established")
            
            async with mcp.ClientSession(*streams) as session:
                # Get available operations if not cached
                if self._available_operations is None:
                    serper_logger.info("Fetching available Serper operations...")
                    self._available_operations = await self._get_available_operations(session)
                    
                    # Log all available operations
                    serper_logger.info("AVAILABLE SERPER OPERATIONS:")
                    for i, (op_name, op_info) in enumerate(self._available_operations.items(), 1):
                        serper_logger.info(f"{i}. {op_name}: {op_info.get('description', 'No description')}")
                    
                    serper_logger.info(f"Found {len(self._available_operations)} available operations")
                
                # Check if the operation is valid
                if operation not in self._available_operations:
                    available_ops = ", ".join(self._available_operations.keys())
                    error_msg = f"Invalid operation: {operation}. Available operations: {available_ops}"
                    serper_logger.error(error_msg)
                    return error_msg
                
                # Get the operation details
                op_details = self._available_operations[operation]
                serper_logger.info("-" * 60)
                serper_logger.info(f"OPERATION SELECTED: {operation}")
                serper_logger.info(f"Description: {op_details.get('description', 'No description')}")
                
                # Validate parameters
                required_params = op_details.get("required", [])
                serper_logger.info(f"Required parameters: {', '.join(required_params)}")
                
                missing_params = [param for param in required_params if param not in parameters]
                if missing_params:
                    error_msg = f"Missing required parameters for {operation}: {', '.join(missing_params)}"
                    serper_logger.error(error_msg)
                    return error_msg
                
                # Log call
                serper_logger.info(f"Calling Serper API: {operation}")
                call_start_time = time.time()
                
                # Call the operation
                result = await session.call_tool(operation, parameters)
                
                # Log call completion
                call_duration = time.time() - call_start_time
                serper_logger.info(f"API call completed in {call_duration:.2f} seconds")
                
                # Process the result
                if hasattr(result, 'content') and result.content:
                    try:
                        # Parse JSON response
                        response_data = json.loads(result.content[0].text)
                        json_result = json.dumps(response_data, indent=2)
                        
                        # Log result size
                        serper_logger.info(f"Result size: {len(json_result)} characters")
                        
                        return json_result
                    except json.JSONDecodeError:
                        # Return raw text if not JSON
                        raw_text = result.content[0].text
                        serper_logger.info(f"Non-JSON result received ({len(raw_text)} characters)")
                        return raw_text
                else:
                    serper_logger.warning(f"Operation {operation} returned no content")
                    return f"Operation {operation} executed successfully but returned no content."
    
    async def _get_available_operations(self, session) -> Dict[str, Dict]:
        """Get available Serper operations from the API"""
        tools_result = await session.list_tools()
        
        ops_dict = {}
        if hasattr(tools_result, 'tools'):
            for tool in tools_result.tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # Extract tool schema
                    schema = {}
                    if hasattr(tool, 'inputSchema'):
                        schema = tool.inputSchema
                    
                    # Parse required parameters and properties
                    required = []
                    properties = {}
                    if isinstance(schema, dict):
                        required = schema.get('required', [])
                        properties = schema.get('properties', {})
                    
                    ops_dict[tool.name] = {
                        "description": tool.description,
                        "required": required,
                        "properties": properties
                    }
        
        return ops_dict
    
    def get_available_operations(self) -> str:
        """Get a formatted list of available Serper operations"""
        serper_logger.info("Getting formatted list of Serper operations")
        operations = asyncio.run(self.list_available_operations())
        
        # Format the output
        result = "# Available Serper Operations\n\n"
        for op in operations:
            result += f"## {op['name']}\n"
            result += f"Description: {op['description']}\n"
            if op['required_params']:
                result += f"Required parameters: {', '.join(op['required_params'])}\n"
            result += "\n"
        
        serper_logger.info(f"Formatted {len(operations)} operations")
        return result
    
    async def list_available_operations(self) -> List[Dict[str, str]]:
        """List all available Serper operations with descriptions"""
        serper_logger.info("Listing all available Serper operations")
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                ops_dict = await self._get_available_operations(session)
                
                operations = []
                for name, details in ops_dict.items():
                    operations.append({
                        "name": name,
                        "description": details["description"],
                        "required_params": details.get("required", [])
                    })
                
                serper_logger.info(f"Found {len(operations)} available operations")
                
                # Log all available operations
                serper_logger.info("AVAILABLE OPERATIONS:")
                for i, op in enumerate(operations, 1):
                    serper_logger.info(f"{i}. {op['name']}: {op['description']}")
                    if op['required_params']:
                        serper_logger.info(f"   Required parameters: {', '.join(op['required_params'])}")
                
                return operations

# Example implementation using the CrewAI wrapper
class SerperDevTool(SerperMCPTool):
    """
    A wrapper around SerperMCPTool that provides a simplified interface
    specifically for web search functionality, compatible with CrewAI.
    """
    name: str = "serper_dev_tool"
    description: str = "Search the web using Serper API"
    
    def _run(self, query: str) -> str:
        """Simplified interface that just takes a query string"""
        return super()._run("search", {"query": query}) 
    ''',
    "github": '''
import os
import asyncio
import json
import smithery
import mcp
import logging
import datetime
import time
from mcp.client.websocket import websocket_client
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure enhanced logging
def setup_logging():
    """Set up detailed logging for GitHub operations"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/github_operations_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a specific logger for GitHub operations
    logger = logging.getLogger("github_operations")
    
    return logger, log_file

# Initialize logger
github_logger, log_file_path = setup_logging()
github_logger.info(f"GitHub operations log initialized. Log file: {log_file_path}")

class GitHubMCPToolParams(BaseModel):
    """Parameters for GitHub tool operations"""
    operation: str = Field(
        description="The GitHub operation to perform, such as search_repositories, create_repository, etc."
    )
    parameters: Dict[str, Any] = Field(
        default={},
        description="Parameters for the GitHub operation"
    )

class GitHubMCPTool(BaseTool):
    """Tool for interacting with GitHub API using MCP"""
    name: str = "github_mcp_tool"
    description: str = "Interact with GitHub API to perform various operations like searching repositories, creating repositories, managing issues, etc."
    args_schema: type[BaseModel] = GitHubMCPToolParams
    github_token: str = ""
    url: str = ""
    
    def __init__(self):
        super().__init__()
        # Load environment variables
        load_dotenv()
        self.github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        if not self.github_token:
            github_logger.error("GitHub token not found in environment variables")
            raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment variables")
        
        # Create Smithery URL
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@smithery-ai/github/ws",
            {"githubPersonalAccessToken": self.github_token}
        )
        
        # Cache for available tools
        self._available_tools = None
        
        # Try to preload available tools
        try:
            asyncio.run(self._preload_tools())
        except Exception as e:
            github_logger.warning(f"Could not preload tools: {str(e)}")
        
        # Log initialization
        github_logger.info("GitHubMCPTool initialized")
        github_logger.info(f"GitHub token: {self.github_token[:5]}...{self.github_token[-5:]}")
    
    async def _preload_tools(self):
        """Preload available tools for better logging"""
        github_logger.info("Preloading available GitHub tools...")
        async with websocket_client(self.url) as streams:
            github_logger.info("Connection established for preloading tools")
            async with mcp.ClientSession(*streams) as session:
                self._available_tools = await self._get_available_tools(session)
                github_logger.info(f"Preloaded {len(self._available_tools)} GitHub tools")
                
                # Log all available tools
                github_logger.info("=" * 80)
                github_logger.info("AVAILABLE GITHUB TOOLS:")
                for i, (tool_name, tool_info) in enumerate(self._available_tools.items(), 1):
                    github_logger.info(f"{i}. {tool_name}: {tool_info.get('description', 'No description')}")
                    if tool_info.get('required'):
                        github_logger.info(f"   Required parameters: {', '.join(tool_info.get('required', []))}")
                github_logger.info("=" * 80)
    
    def _run(self, operation: str, parameters: Dict[str, Any] = None) -> str:
        """Run the GitHub operation"""
        if parameters is None:
            parameters = {}
        
        # Log operation with distinct formatting
        github_logger.info("=" * 80)
        github_logger.info(f"OPERATION CALLED: {operation}")
        github_logger.info("-" * 80)
        
        # Safely log parameters (remove sensitive data)
        safe_params = self._get_safe_parameters(parameters)
        github_logger.info(f"PARAMETERS: {json.dumps(safe_params, indent=2)}")
        
        # Track execution time
        start_time = time.time()
        
        # Run the async operation and wait for result
        try:
            result = asyncio.run(self._run_async(operation, parameters))
            
            # Calculate execution time
            execution_time = time.time() - start_time
            github_logger.info(f"EXECUTION TIME: {execution_time:.2f} seconds")
            
            # Log result summary
            result_summary = self._get_result_summary(result)
            github_logger.info(f"RESULT SUMMARY: {result_summary}")
            github_logger.info("=" * 80)
            
            return result
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            github_logger.error(f"ERROR ({execution_time:.2f}s): {str(e)}")
            github_logger.info("=" * 80)
            return f"Error executing GitHub operation: {str(e)}"
    
    def _get_safe_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of parameters safe for logging (no sensitive data)"""
        safe_params = parameters.copy()
        
        # Mask sensitive fields if present
        sensitive_fields = ['token', 'password', 'secret', 'key', 'auth']
        for field in sensitive_fields:
            for key in list(safe_params.keys()):
                if field in key.lower() and isinstance(safe_params[key], str):
                    safe_params[key] = f"{safe_params[key][:3]}...{safe_params[key][-3:]}"
        
        return safe_params
    
    def _get_result_summary(self, result: str) -> str:
        """Create a summary of the result for logging"""
        if not result:
            return "Empty result"
            
        # For JSON results
        try:
            result_obj = json.loads(result)
            
            if isinstance(result_obj, dict):
                # For dictionary responses
                if 'total_count' in result_obj:
                    return f"Total count: {result_obj['total_count']}"
                elif 'id' in result_obj:
                    return f"Object ID: {result_obj['id']}"
                elif 'name' in result_obj:
                    return f"Object name: {result_obj['name']}"
                else:
                    keys = list(result_obj.keys())
                    return f"JSON object with keys: {', '.join(keys[:5])}" + ("..." if len(keys) > 5 else "")
            
            # For array responses
            elif isinstance(result_obj, list):
                return f"Array with {len(result_obj)} items"
                
        except json.JSONDecodeError:
            # For non-JSON results
            if len(result) > 100:
                return f"Text response ({len(result)} characters)"
            return result
        
        return "Result processed successfully"
    
    async def _run_async(self, operation: str, parameters: Dict[str, Any]) -> str:
        """Run the GitHub operation asynchronously"""
        # Log connection attempt
        github_logger.info("Connecting to GitHub API...")
        
        # Normalize parameters for specific operations
        parameters = self._normalize_parameters(operation, parameters)
        
        # Connect to the Smithery server
        async with websocket_client(self.url) as streams:
            github_logger.info("Connection established")
            
            async with mcp.ClientSession(*streams) as session:
                # Get available tools if not cached
                if self._available_tools is None:
                    github_logger.info("Fetching available GitHub tools...")
                    self._available_tools = await self._get_available_tools(session)
                    
                    # Log all available tools
                    github_logger.info("AVAILABLE GITHUB TOOLS:")
                    for i, (tool_name, tool_info) in enumerate(self._available_tools.items(), 1):
                        github_logger.info(f"{i}. {tool_name}: {tool_info.get('description', 'No description')}")
                    
                    github_logger.info(f"Found {len(self._available_tools)} available tools")
                
                # Check if the operation is valid
                if operation not in self._available_tools:
                    available_ops = ", ".join(self._available_tools.keys())
                    error_msg = f"Invalid operation: {operation}. Available operations: {available_ops}"
                    github_logger.error(error_msg)
                    return error_msg
                
                # Get the tool details
                tool_details = self._available_tools[operation]
                github_logger.info("-" * 60)
                github_logger.info(f"TOOL SELECTED: {operation}")
                github_logger.info(f"Description: {tool_details.get('description', 'No description')}")
                
                # Validate parameters
                required_params = tool_details.get("required", [])
                github_logger.info(f"Required parameters: {', '.join(required_params)}")
                
                missing_params = [param for param in required_params if param not in parameters]
                if missing_params:
                    error_msg = f"Missing required parameters for {operation}: {', '.join(missing_params)}"
                    github_logger.error(error_msg)
                    return error_msg
                
                # Log call
                github_logger.info(f"Calling GitHub API: {operation}")
                call_start_time = time.time()
                
                # Call the tool
                result = await session.call_tool(operation, parameters)
                
                # Log call completion
                call_duration = time.time() - call_start_time
                github_logger.info(f"API call completed in {call_duration:.2f} seconds")
                
                # Process the result
                if hasattr(result, 'content') and result.content:
                    try:
                        # Parse JSON response
                        response_data = json.loads(result.content[0].text)
                        json_result = json.dumps(response_data, indent=2)
                        
                        # Log result size
                        github_logger.info(f"Result size: {len(json_result)} characters")
                        
                        return json_result
                    except json.JSONDecodeError:
                        # Return raw text if not JSON
                        raw_text = result.content[0].text
                        github_logger.info(f"Non-JSON result received ({len(raw_text)} characters)")
                        return raw_text
                else:
                    github_logger.warning(f"Operation {operation} returned no content")
                    return f"Operation {operation} executed successfully but returned no content."
    
    async def _get_available_tools(self, session) -> Dict[str, Dict]:
        """Get available GitHub tools from the API"""
        tools_result = await session.list_tools()
        
        tools_dict = {}
        if hasattr(tools_result, 'tools'):
            for tool in tools_result.tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # Extract tool schema
                    schema = {}
                    if hasattr(tool, 'inputSchema'):
                        schema = tool.inputSchema
                    
                    # Parse required parameters and properties
                    required = []
                    properties = {}
                    if isinstance(schema, dict):
                        required = schema.get('required', [])
                        properties = schema.get('properties', {})
                    
                    tools_dict[tool.name] = {
                        "description": tool.description,
                        "required": required,
                        "properties": properties
                    }
        
        return tools_dict
    
    def get_authenticated_user(self) -> str:
        """Get the authenticated user's GitHub username"""
        github_logger.info("Getting authenticated GitHub user")
        try:
            # Run this synchronously
            username = asyncio.run(self._get_authenticated_user())
            github_logger.info(f"Authenticated as: {username}")
            return username
        except Exception as e:
            # Fallback to environment variable if available
            github_username = os.getenv('GITHUB_USERNAME')
            if github_username:
                github_logger.info(f"Using username from environment: {github_username}")
                return github_username
            
            github_logger.error(f"Error getting username: {str(e)}")
            return f"Error getting username: {str(e)}"
    
    async def _get_authenticated_user(self) -> str:
        """Get the authenticated user's GitHub username asynchronously"""
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                # Call API to get user info
                github_logger.info("Calling get_authenticated_user API")
                result = await session.call_tool("get_authenticated_user", {})
                
                if hasattr(result, 'content') and result.content:
                    # Parse response
                    try:
                        user_data = json.loads(result.content[0].text)
                        return user_data.get('login', 'unknown')
                    except json.JSONDecodeError:
                        github_logger.warning("Failed to parse user data JSON")
                        return "unknown"
                
                github_logger.warning("No content returned from get_authenticated_user")
                return "unknown"
    
    async def list_available_operations(self) -> List[Dict[str, str]]:
        """List all available GitHub operations with descriptions"""
        github_logger.info("Listing all available GitHub operations")
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                tools_dict = await self._get_available_tools(session)
                
                operations = []
                for name, details in tools_dict.items():
                    operations.append({
                        "name": name,
                        "description": details["description"],
                        "required_params": details.get("required", [])
                    })
                
                github_logger.info(f"Found {len(operations)} available operations")
                
                # Log all available operations
                github_logger.info("AVAILABLE OPERATIONS:")
                for i, op in enumerate(operations, 1):
                    github_logger.info(f"{i}. {op['name']}: {op['description']}")
                    if op['required_params']:
                        github_logger.info(f"   Required parameters: {', '.join(op['required_params'])}")
                
                return operations

    def get_available_operations(self) -> str:
        """Get a formatted list of available GitHub operations"""
        github_logger.info("Getting formatted list of GitHub operations")
        operations = asyncio.run(self.list_available_operations())
        
        # Format the output
        result = "# Available GitHub Operations\n\n"
        for op in operations:
            result += f"## {op['name']}\n"
            result += f"Description: {op['description']}\n"
            if op['required_params']:
                result += f"Required parameters: {', '.join(op['required_params'])}\n"
            result += "\n"
        
        github_logger.info(f"Formatted {len(operations)} operations")
        return result

    def _normalize_parameters(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for specific operations to handle common parameter name mismatches
        """
        # Create a copy to avoid modifying the original
        normalized_params = parameters.copy()
        
        # Handle search_repositories - convert 'q' to 'query'
        if operation == "search_repositories" and "q" in normalized_params and "query" not in normalized_params:
            github_logger.info(f"Converting parameter 'q' to 'query' for {operation}")
            normalized_params["query"] = normalized_params.pop("q")
            
        # Add timestamp to parameters to avoid identical request detection
        normalized_params["_timestamp"] = str(time.time())
            
        return normalized_params 
'''
}

@dataclass
class MCPToolDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str = ""
    architecture_plan: str = ""
    
    def __post_init__(self):
        """Validate dependencies after initialization."""
        if not self.supabase:
            logger.warning("Supabase client not provided in MCPToolDeps")
        if not self.openai_client:
            logger.warning("OpenAI client not provided in MCPToolDeps")

# Initialize the model
model_name = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')

# Set OpenAI API key in environment variable if not already set
if "OPENAI_API_KEY" not in os.environ and api_key != 'no-llm-api-key-provided':
    os.environ["OPENAI_API_KEY"] = api_key

is_anthropic = "anthropic" in os.getenv('BASE_URL', '').lower()
if is_anthropic:
    model = AnthropicModel(model_name, api_key=api_key)
else:
    model = OpenAIModel(model_name)

# Create the MCP Tool agent
mcp_tool_agent = Agent(
    model,
    system_prompt="""
You are a specialized MCP (Model Context Protocol) tool integration expert. Your primary role is to locate, integrate,
and configure MCP tools for CrewAI agent applications.

Key responsibilities:
1. Finding relevant MCP tools based on user requirements
2. Integrating MCP tools with CrewAI agent code
3. Creating connection scripts for API authentication
4. Generating comprehensive documentation for MCP tool integration
5. Providing setup instructions for third-party API credentials

Always ensure proper:
- Import statements for MCP tools
- Tool initialization in CrewAI agents
- Authentication handling for external APIs
- Documentation of API key requirements
- Example usage in README files

You specialize in integrating external services like GitHub, Spotify, YouTube, Twitter, and other APIs
through the Model Context Protocol framework.
""",
    deps_type=MCPToolDeps,
    retries=2
)

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

async def extract_tools_from_query(query: str, openai_client: AsyncOpenAI) -> List[str]:
    """
    Directly ask OpenAI to identify which tools the user needs based on their query.
    More reliable than keyword matching for extracting user intent.
    
    Args:
        query: User's query/request
        openai_client: AsyncOpenAI client
        
    Returns:
        List of tool names the user wants to use
    """
    try:
        logger.info(f"MCP DETECTION: Extracting tool needs from query using OpenAI")
        
        prompt = f"""
        Based on the following user request, identify which specific external tools, services or APIs are needed:
        
        USER REQUEST: "{query}"
        
        Return ONLY a comma-separated list of the specific tools needed (e.g., "github, spotify"). 
        Focus ONLY on these key service names: github, spotify, youtube, twitter, slack, gmail, google_drive, discord, notion, trello, asana, jira, instagram, linkedin, facebook, shopify, stripe, aws.
        
        The response should contain ONLY the 4-5 letter service names in lowercase, separated by commas. No additional text.
        """
        
        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1  # Low temperature for more deterministic answers
        )
        
        tools_text = response.choices[0].message.content.strip().lower()
        
        # Clean up the response
        tools_text = tools_text.replace(".", "").replace("and", ",")
        
        # Parse the comma-separated list
        tools = [tool.strip() for tool in tools_text.split(",") if tool.strip()]
        
        # Log the specific tools found
        if tools:
            logger.info(f"MCP DETECTION SUCCESS: OpenAI identified tools: {', '.join(tools)}")
        else:
            logger.info("MCP DETECTION: No specific tools identified by OpenAI")
        
        return tools
        
    except Exception as e:
        logger.error(f"MCP DETECTION ERROR: Failed to extract tools from query: {e}")
        return []

@mcp_tool_agent.tool
async def find_relevant_mcp_tools(ctx: RunContext[MCPToolDeps], user_query: str) -> Dict[str, Any]:
    """
    Find relevant MCP tools based on the user's query.
    Uses OpenAI to directly identify which tools the user needs.
    Returns a dictionary containing information about multiple relevant MCP tools if found.
    """
    try:
        logger.info(f"MCP TOOL SEARCH: Looking for tools matching query: {user_query[:100]}...")
        
        # Use OpenAI to directly extract which tools the user needs
        mentioned_tools = await extract_tools_from_query(user_query, ctx.deps.openai_client)
        
        if mentioned_tools:
            logger.info(f"MCP TOOL SEARCH: Found specific tools in query: {', '.join(mentioned_tools)}")
        else:
            logger.info("MCP TOOL SEARCH: No specific tools identified by OpenAI, using fallback methods")
            
            # Fall back to basic keyword detection if OpenAI didn't identify anything
            user_query_lower = user_query.lower()
            
            # List of known tools to check for
            known_tools = {
                "github": ["github", "git", "repository", "repo", "pull request", "issue"],
                "spotify": ["spotify", "music", "playlist", "song", "track", "artist", "album"],
                "youtube": ["youtube", "video", "channel", "stream"],
                "twitter": ["twitter", "tweet", "x.com"],
                "slack": ["slack", "message", "channel"],
                "gmail": ["gmail", "email", "mail"],
                "google_drive": ["google drive", "gdrive", "drive"],
                "discord": ["discord", "server"],
                "notion": ["notion", "page", "database"],
                "trello": ["trello", "board", "card"],
                "asana": ["asana", "task"],
                "jira": ["jira", "ticket"],
                "instagram": ["instagram", "post", "story"],
                "linkedin": ["linkedin", "profile", "post"],
                "facebook": ["facebook", "post", "page"],
                "shopify": ["shopify", "store", "product"],
                "stripe": ["stripe", "payment", "invoice"],
                "aws": ["aws", "amazon web services", "s3", "ec2", "lambda"]
            }
            
            # Check for tool mentions in the query as a backup
            for tool_name, keywords in known_tools.items():
                for keyword in keywords:
                    if keyword in user_query_lower:
                        if tool_name not in mentioned_tools:
                            mentioned_tools.append(tool_name)
                            logger.info(f"MCP TOOL KEYWORD FOUND: '{tool_name}' via keyword '{keyword}'")
                        break
            
            if mentioned_tools:
                logger.info(f"MCP TOOL FALLBACK: Found tools via keyword matching: {', '.join(mentioned_tools)}")
        
        # Generate embedding for similarity search (we'll still use this to find the most relevant tools)
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        logger.info("MCP TOOL SEARCH: Query embedding generated successfully")
        
        # Search for similar MCP tools
        result = ctx.deps.supabase.rpc(
            'match_mcp_tools',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': 20  # Increased to get more candidate tools
            }
        ).execute()
        
        if not result.data:
            logger.info("MCP TOOL SEARCH: No matching tools with high threshold, trying lower threshold")
            # Try with lower threshold and even more results
            result = ctx.deps.supabase.rpc(
                'match_mcp_tools',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.3,
                    'match_count': 25  # Increased to get more candidate tools
                }
            ).execute()
        
        if not result.data:
            logger.info("MCP TOOL SEARCH: No relevant tools found in database")
            return {"found": False}
        
        # Get all matching tools that meet the threshold
        tools = result.data
        
        # Filter tools based on mentioned tools if any were detected
        if mentioned_tools:
            filtered_tools = []
            for tool in tools:
                tool_purpose = tool.get('purpose', '').lower()
                
                # Check if this tool matches any of the mentioned tools
                is_matching_tool = False
                for mentioned_tool in mentioned_tools:
                    # Check if mentioned tool name is in the tool purpose
                    if mentioned_tool in tool_purpose:
                        is_matching_tool = True
                        logger.info(f"MCP TOOL MATCH: Tool with purpose '{tool.get('purpose', '')}' matches required service '{mentioned_tool}'")
                        break
                
                if is_matching_tool:
                    filtered_tools.append(tool)
            
            if filtered_tools:
                logger.info(f"MCP TOOL FILTER: Found {len(filtered_tools)} tools matching services: {', '.join(mentioned_tools)}")
                tools = filtered_tools
            else:
                logger.info(f"MCP TOOL FILTER: No tools matched the services {', '.join(mentioned_tools)}, using similarity-based results instead")
        
        tool_count = len(tools)
        logger.info(f"MCP TOOL SEARCH COMPLETE: Found {tool_count} matching tools")
        
        # Log the top matches
        for i, tool in enumerate(tools[:3]):  # Log the top 3 for debugging
            logger.info(f"MCP TOOL RESULT #{i+1}: '{tool.get('purpose', 'Unknown')}' (similarity: {tool.get('similarity', 'N/A')})")
        
        # Return information about all matching tools
        return {
            "found": True,
            "tool_count": tool_count,
            "tools": [
                {
                    "tool_id": tool.get('id'),
                    "purpose": tool.get('purpose', ''),
                    "similarity": tool.get('similarity', 0),
                    "tool_code": tool.get('tool_code', ''),
                    "example_crew_code": tool.get('example_crew_code', ''),
                    "readme_content": tool.get('readme_content', ''),
                    "connection_script": tool.get('connection_script', ''),
                    "requirements": tool.get('requirements', ''),
                    "config": tool.get('config', '')
                }
                for tool in tools
            ],
            "mentioned_tools": mentioned_tools  # Include the list of mentioned tools
        }
        
    except Exception as e:
        logger.error(f"MCP TOOL SEARCH FAILED: {str(e)}", exc_info=True)
        return {"found": False, "error": str(e)}

@mcp_tool_agent.tool
async def integrate_mcp_tool_with_code(ctx: RunContext[MCPToolDeps], 
                                     agent_code: Dict[str, str],
                                     mcp_tools_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Process MCP tools and format them for direct CrewAI usage.
    Returns a single tools.py file with CrewAI integration.
    
    Args:
        ctx: Run context with dependencies
        agent_code: Dictionary containing any existing code (can be empty)
        mcp_tools_data: Dictionary containing the tools data from find_relevant_mcp_tools
        
    Returns:
        Dictionary with the merged tool code
    """
    try:
        # Check if we have multiple tools or just a single tool
        if "tools" in mcp_tools_data and isinstance(mcp_tools_data["tools"], list):
            tools = mcp_tools_data["tools"]
            logger.info(f"MCP TOOL INTEGRATION: Processing {len(tools)} tools into a single file")
        else:
            # Legacy format - single tool
            tools = [mcp_tools_data]
            logger.info("MCP TOOL INTEGRATION: Processing a single tool")
        
        # Get CrewAI requirements if available
        crewai_requirements = mcp_tools_data.get("crewai_requirements", {})
        if crewai_requirements:
            logger.info(f"MCP TOOL INTEGRATION: Using CrewAI requirements for generation")
            
        # Determine which reference examples to use
        # Default to SerperMCPTool, but could be extended based on tool type
        reference_keys = ["serper"]
        
        # Start with required imports for CrewAI
        tools_py_content = """from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import Field
import os
import requests
import json
import asyncio
import time
import re
try:
    import mcp
    from mcp.client.websocket import websocket_client
except ImportError:
    print("MCP package not found. Installing required dependencies may be needed.")
try:
    import smithery
except ImportError:
    print("Smithery package not found. Some tools might not work properly.")

# Common errors that will be validated on tools.py creation:
# 1. Class naming inconsistency (YouTubeTranscriptTool vs YouTubeTranscriptMCPTool)
# 2. Missing tool suffix in class names
# 3. Inconsistent method names across files
# 4. Missing alias classes for backward compatibility

# CrewAI-compatible tools collection
"""
        
        # Keep track of all tool class names for exports
        tool_class_names = []
        
        # Keep track of additional imports we need to add
        additional_imports = set()
        
        # Process each tool in sequence and convert to CrewAI format
        for i, tool_data in enumerate(tools):
            tool_purpose = tool_data.get('purpose', f'Tool {i+1}')
            logger.info(f"MCP TOOL CONVERSION: Converting tool {i+1}/{len(tools)}: {tool_purpose}")
            
            # Extract tool information
            original_tool_code = tool_data.get('tool_code', '')
            if not original_tool_code:
                logger.warning(f"MCP TOOL CONVERSION WARNING: No code found for tool: {tool_purpose}")
                continue
            
            # Extract imports from original code before conversion
            extracted_imports = extract_imports(original_tool_code)
            for imp in extracted_imports:
                additional_imports.add(imp)
                
            # Convert to CrewAI format
            crewai_tool = await convert_to_crewai_tool(
                ctx, 
                original_tool_code, 
                tool_purpose,
                tool_data.get('connection_script', ''),
                crewai_requirements,
                reference_keys
            )
            
            # Extract the tool class name from the converted code
            tool_class_name = extract_class_name(crewai_tool)
            if tool_class_name:
                tool_class_names.append(tool_class_name)
            
            # Add to our combined file
            tools_py_content += f"\n\n{crewai_tool}\n"
            logger.info(f"MCP TOOL CONVERSION SUCCESS: Added tool to tools.py: {tool_purpose}")
        
        # Add additional imports at the top of the file
        import_section = ""
        if additional_imports:
            import_section = "\n# Additional imports from original tools\n"
            for imp in sorted(additional_imports):
                # Skip duplicates of what we already have in the base imports
                if any(base_import in imp for base_import in ["typing", "crewai", "pydantic", "os", "requests", "json", "asyncio", "mcp", "smithery"]):
                    continue
                import_section += f"{imp}\n"
            
        # Insert additional imports after the try-except blocks
        if import_section:
            # Find the position after the try-except blocks
            try_except_end = tools_py_content.find("# CrewAI-compatible tools collection")
            if try_except_end != -1:
                tools_py_content = tools_py_content[:try_except_end] + import_section + tools_py_content[try_except_end:]
        
        # Validate and fix common errors before finalizing
        fixed_content, applied_fixes = validate_tools_py_content(tools_py_content)
        if applied_fixes:
            logger.info(f"MCP TOOL INTEGRATION: Applied {len(applied_fixes)} fixes to tools.py for common errors")
            for fix in applied_fixes:
                logger.info(f"  - {fix['message']}")
            tools_py_content = fixed_content

        # Add exports for all the tools
        if tool_class_names:
            tools_py_content += "\n# Export all tools for use in CrewAI\n__all__ = [\n"
            for name in tool_class_names:
                tools_py_content += f'    "{name}",\n'
            tools_py_content += "]\n"
        
        logger.info(f"MCP TOOL INTEGRATION COMPLETE: Generated tools.py with {len(tools)} tools")
        
        return {"tools_py_content": tools_py_content}
        
    except Exception as e:
        logger.error(f"MCP TOOL INTEGRATION FAILED: {str(e)}", exc_info=True)
        return {"tools_py_content": f"# Error occurred during tool integration: {str(e)}"}

@mcp_tool_agent.tool
async def generate_complete_crewai_project(ctx: RunContext[MCPToolDeps],
                                         tools_data: Dict[str, Any],
                                         output_dir: str,
                                         existing_tools_py: str = None) -> Dict[str, str]:
    """
    Generate a complete CrewAI project with tools, agents, tasks, and crew files.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Information about the tools being used
        output_dir: Output directory for the generated files
        existing_tools_py: Optional existing tools.py content
        
    Returns:
        Dictionary with file contents
    """
    try:
        logger.info(f"Generating complete CrewAI project in {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing files in the output directory
        existing_files = []
        if os.path.exists(output_dir):
            existing_files = os.listdir(output_dir)
        if existing_files:
            logger.info(f"Found existing files in {output_dir}: {', '.join(existing_files)}")
            
        # Generate the .env file first - it's needed for proper setup
        env_content = generate_env_file_content(tools_data)
        env_file_path = os.path.join(output_dir, '.env')
        with open(env_file_path, 'w') as f:
            f.write(env_content)
        logger.info(f"Generated .env file at {env_file_path}")
        
        # Step 1: Generate the tools.py file if it doesn't already exist
        tools_file_path = os.path.join(output_dir, 'tools.py')
        
        # If existing_tools_py is provided, use that content
        if existing_tools_py:
            logger.info("Using provided tools.py content")
            with open(tools_file_path, 'w') as f:
                f.write(existing_tools_py)
            tools_py_content = existing_tools_py
        # If tools.py already exists in the output directory, read it
        elif os.path.exists(tools_file_path):
            logger.info(f"Found existing tools.py at {tools_file_path}")
            with open(tools_file_path, 'r') as f:
                tools_py_content = f.read()
            logger.info("Saved existing tools.py to {tools_file_path}")
        # Otherwise, generate the tools.py file
        else:
            tools_py = await generate_tools_py(ctx, tools_data, output_dir)
            
            # Read the generated tools.py file
            if os.path.exists(tools_file_path):
                with open(tools_file_path, 'r') as f:
                    tools_py_content = f.read()
            else:
                logger.error("Failed to generate tools.py file")
                return {}
        
        # Extract tool class names from tools.py content
        tool_class_names = extract_class_names_from_tools(tools_py_content)
        logger.info(f"Extracted tool class names: {tool_class_names}")
        
        # Generate requirements.txt
        requirements_txt = generate_requirements(tools_data, tool_class_names)
        with open(os.path.join(output_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements_txt)
        logger.info(f"Generated requirements.txt at {os.path.join(output_dir, 'requirements.txt')}")
        
        # Use template-based generation instead of standard generation
        from .mcp_tools.mcp_template_integration import generate_from_template
        
        # Use specified model for template generation
        template_model = os.getenv("TEMPLATE_MODEL", "gpt-4o-mini")
        logger.info(f"Using model {template_model} for template generation")
        
        # Update tools_data with extracted tool class names and other metadata
        tools_data["tool_class_names"] = tool_class_names
        
        # Generate from template directly
        template_results = await generate_from_template(
            tools_data.get("query", ""),
            tools_data,
            tool_class_names,
            output_dir,
            ctx.deps.supabase,
            ctx.deps.openai_client,
            template_model
        )
        
        # Check if template generation was successful
        if template_results and len(template_results) > 0:
            logger.info("Template generation successful! Created CrewAI project files.")
            
            # Generate README.md
            readme_content = await generate_crewai_readme(
                ctx,
                tools_data,
                list(template_results.keys())
            )
            
            with open(os.path.join(output_dir, "README.md"), "w") as f:
                f.write(readme_content)
            logger.info(f"Generated README.md at {os.path.join(output_dir, 'README.md')}")
            
            logger.info(f"Successfully generated complete CrewAI project in {output_dir}")
            return template_results
        
        # If template generation fails, return an error indicator
        logger.error("Template generation failed")
        return {"error": "Template generation failed"}
        
    except Exception as e:
        logger.error(f"Error generating complete CrewAI project: {e}", exc_info=True)
        return {}

def extract_imports(code: str) -> List[str]:
    """
    Extract import statements from the code to preserve them.
    
    Args:
        code: The original tool code
        
    Returns:
        List of import statements
    """
    imports = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports

def clean_tool_code(code: str) -> str:
    """
    Clean up the generated tool code by removing imports and any explanatory text.
    This preserves all functional code.
    
    Args:
        code: The generated tool code from the LLM
        
    Returns:
        Cleaned tool code with the class definition and helper functions
    """
    # Remove markdown code blocks if present
    code = code.replace("```python", "").replace("```", "").strip()
    
    # Remove only imports and unnecessary comments
    cleaned_lines = []
    preserve_mode = False
    
    for line in code.split("\n"):
        # Skip import lines and any explanatory text before class definition
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            continue
        
        # Once we hit a class definition, we're in the actual code
        if line.strip().startswith("class "):
            preserve_mode = True
        
        # Keep lines that are part of the actual code
        if preserve_mode or not line.strip() or line.strip().startswith("def "):
            cleaned_lines.append(line)
    
    # If we have any helper functions before the class, include them too
    has_added_helpers = False
    result = []
    
    for line in code.split("\n"):
        # Include helper functions defined before the class
        if line.strip().startswith("def ") and not has_added_helpers:
            result.append("\n# Helper functions")
            has_added_helpers = True
            result.append(line)
            continue
            
        if has_added_helpers and not line.strip().startswith("class ") and not line.strip().startswith("import ") and not line.strip().startswith("from "):
            # Continue adding lines of the helper function until we hit a class
            if not line.strip().startswith("class "):
                result.append(line)
            
    # Add the class implementation
    result.extend(cleaned_lines)
    
    return "\n".join(result)

@mcp_tool_agent.tool
async def convert_to_crewai_tool(ctx: RunContext[MCPToolDeps], 
                               original_code: str,
                               purpose: str,
                               connection_code: str = "",
                               crewai_requirements: Dict[str, Any] = None,
                               reference_keys: List[str] = None) -> str:
    """
    Convert an MCP tool to a CrewAI-compatible tool class.
    
    Args:
        ctx: Run context with dependencies
        original_code: Original tool code from database
        purpose: Purpose of the tool
        connection_code: Optional connection script code
        crewai_requirements: Optional specific requirements for CrewAI format
        reference_keys: Keys for reference examples to use (defaults to based on purpose)
        
    Returns:
        CrewAI-compatible tool class as string
    """
    try:
        logger.info(f"MCP TOOL CONVERT: Converting tool for purpose: {purpose}")
        
        # Find matching requirements if available
        specific_requirements = {}
        if crewai_requirements and "tools" in crewai_requirements:
            # Find a matching tool in requirements
            purpose_lower = purpose.lower()
            for tool_req in crewai_requirements.get("tools", []):
                tool_name = tool_req.get("name", "").lower()
                if tool_name in purpose_lower or purpose_lower in tool_name:
                    specific_requirements = tool_req
                    logger.info(f"MCP TOOL CONVERT: Found matching requirements for: {tool_name}")
                    break
        
        # Format the specific requirements for the prompt
        requirements_text = ""
        if specific_requirements:
            requirements_text = "\nSPECIFIC REQUIREMENTS:\n"
            for key, value in specific_requirements.items():
                if isinstance(value, list):
                    requirements_text += f"{key}: {', '.join(value)}\n"
                else:
                    requirements_text += f"{key}: {value}\n"
        
        # Determine which reference examples to use based on purpose if not specified
        if reference_keys is None:
            reference_keys = ["serper"]  # Default
            purpose_lower = purpose.lower()
            
            # Select appropriate reference examples based on purpose
            if "github" in purpose_lower or "git" in purpose_lower or "repository" in purpose_lower:
                reference_keys = ["github"]
                logger.info("MCP TOOL CONVERT: Using GitHub reference for tool conversion")
            elif "search" in purpose_lower or "serper" in purpose_lower:
                reference_keys = ["serper"]
                logger.info("MCP TOOL CONVERT: Using Serper reference for tool conversion")
            elif "spotify" in purpose_lower or "music" in purpose_lower:
                # You could add a Spotify reference example in the future
                reference_keys = ["serper"]  # Fallback to serper for now
                logger.info("MCP TOOL CONVERT: Using generic reference for music/Spotify tool")
        
        # Prepare reference examples section
        reference_section = "\nREFERENCE EXAMPLES (follow these patterns closely):\n"
        for key in reference_keys:
            if key in REFERENCE_EXAMPLES:
                reference_section += f"```python\n{REFERENCE_EXAMPLES[key]}\n```\n\n"
                    
        # Use LLM to convert the tool
        prompt = f"""
        Convert this MCP tool code into a CrewAI-compatible tool class.
        
        ORIGINAL TOOL CODE:
        ```python
        {original_code}
        ```
        
        CONNECTION CODE (incorporate this into the tool if relevant):
        ```python
        {connection_code}
        ```
        
        Tool purpose: {purpose}
        {requirements_text}
        
        {reference_section}
        
        REQUIREMENTS:
        1. Create a class that inherits from crewai.tools.BaseTool
        2. Include all authentication and setup within the tool class
        3. If there are multiple functionalities, convert them into methods
        4. Add proper Field annotations and docs
        5. No separate files - everything must be in this one class
        6. Handle credentials through environment variables or pydantic Field parameters
        7. Include example code as a docstring
        8. Make sure the tool follows the CrewAI pattern with a _run method
        9. Follow the same structure and patterns as the reference examples
        10. Implement both _run and _arun methods if applicable
        11. DO NOT include imports in your response - common imports will be added separately
        12. Use descriptive class name that reflects the tool's purpose
        13. IMPORTANT: Preserve ALL functionality from the original code - do not remove any features
        14. If the original code has helper functions, preserve them as methods inside the class
        15. If any code might need to be preserved outside the class, keep it in your response
        
        Return ONLY the CrewAI tool class code with no explanation or imports.
        """
        
        logger.info("MCP TOOL CONVERT: Sending conversion request to LLM")
        
        # Use OpenAI API directly for more control
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.2
            )
            crewai_tool_code = response.choices[0].message.content
            # Clean up any imports or extra text that might have been added, but preserve all functional code
            crewai_tool_code = clean_tool_code(crewai_tool_code)
            logger.info(f"MCP TOOL CONVERT SUCCESS: Generated CrewAI-compatible tool for {purpose}")
            return crewai_tool_code
        else:
            # Fallback to using the model through the Agent interface
            logger.info("MCP TOOL CONVERT: Using fallback LLM method")
            result = await ctx.model.run(prompt)
            # Clean up any imports or extra text, but preserve all functional code
            cleaned_code = clean_tool_code(result.data)
            logger.info(f"MCP TOOL CONVERT SUCCESS: Generated CrewAI-compatible tool for {purpose}")
            return cleaned_code
            
    except Exception as e:
        logger.error(f"MCP TOOL CONVERT FAILED: {str(e)}", exc_info=True)
        return f"""
# Error converting tool: {str(e)}
class ErrorMCPTool(BaseTool):
    \"\"\"This is a placeholder due to an error in tool conversion.\"\"\"
    name: str = "error_mcp_tool"
    description: str = "Error in tool conversion: {str(e)}"
    
    def _run(self, query: str) -> str:
        return f"Tool conversion error: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)
"""

def extract_class_name(code: str) -> str:
    """
    Extract the class name from the generated tool code.
    
    Args:
        code: The generated tool code
        
    Returns:
        The extracted class name or empty string if not found
    """
    for line in code.split("\n"):
        if line.strip().startswith("class "):
            # Extract the class name using a simple pattern
            # Example: "class SpotifyMCPTool(BaseTool):" -> "SpotifyMCPTool"
            parts = line.strip().split("class ")[1].split("(")[0].strip()
            return parts
    return ""

@mcp_tool_agent.tool
async def create_connection_script(ctx: RunContext[MCPToolDeps], 
                                  connection_script_template: str,
                                  service_name: str) -> str:
    """
    Create a connection script for the MCP tool based on template.
    """
    try:
        prompt = f"""
        Create a detailed connection script for {service_name} based on this template:
        
        {connection_script_template}
        
        The script should include:
        1. Clear setup instructions
        2. Environment variable requirements
        3. Authentication steps
        4. Error handling
        5. Example usage
        
        Return ONLY the Python code with no additional text.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            return result.data
        
    except Exception as e:
        logger.error(f"Error creating connection script: {str(e)}", exc_info=True)
        return connection_script_template  # Return the original template on error

@mcp_tool_agent.tool
async def generate_mcp_integration_readme(ctx: RunContext[MCPToolDeps], 
                                         mcp_tool_data: Dict[str, Any],
                                         generated_files: List[str]) -> str:
    """
    Generate a README file for the MCP tools.
    
    Args:
        ctx: Run context with dependencies
        mcp_tool_data: Dictionary containing information about MCP tools
        generated_files: List of generated filenames
        
    Returns:
        README content as a string
    """
    try:
        files_list = "\n".join([f"- {file}" for file in generated_files])
        
        # Check if we have multiple tools
        if "tools" in mcp_tool_data and isinstance(mcp_tool_data["tools"], list):
            tools = mcp_tool_data["tools"]
            tool_count = len(tools)
            
            # Get purposes for all tools
            if "purposes" in mcp_tool_data and isinstance(mcp_tool_data["purposes"], list):
                purposes = mcp_tool_data["purposes"]
            else:
                purposes = [tool.get('purpose', f"Tool {i+1}") for i, tool in enumerate(tools)]
                
            # Create a summary of tools
            if tool_count > 3:
                tools_summary = f"{tool_count} MCP tools including {', '.join(purposes[:3])} and more"
            else:
                tools_summary = f"{tool_count} MCP tools: {', '.join(purposes)}"
                
            prompt = f"""
            Create a detailed README.md file for a package that only contains MCP tools (without CrewAI integration).
            
            The package contains {tools_summary}.
            
            The README should include:
            1. Overview of the tool package and purpose
            2. Setup instructions including any API keys needed
            3. Directory structure explanation
            4. Direct usage examples for each tool (not through CrewAI)
            5. Troubleshooting tips
            
            Generated files:
            {files_list}
            
            Tool details:
            """
            
            # Add information about each tool
            for i, tool in enumerate(tools[:min(5, tool_count)]):  # Limit to 5 tools in the prompt
                purpose = tool.get('purpose', f"Tool {i+1}")
                prompt += f"""
                Tool {i+1}: {purpose}
                """
                
            if tool_count > 5:
                prompt += f"\nAnd {tool_count - 5} more tools...\n"
                
        else:
            # Legacy format - single tool
            purpose = mcp_tool_data.get('purpose', 'an external tool')
            prompt = f"""
            Create a detailed README.md file for a package that only contains a single MCP tool (without CrewAI integration).
            
            The tool purpose: {purpose}
            
            The README should include:
            1. Overview of the tool and purpose
            2. Setup instructions including any API keys needed
            3. Directory structure explanation
            4. Direct usage examples (not through CrewAI)
            5. Troubleshooting tips
            
            Generated files:
            {files_list}
            
            Tool purpose:
            {purpose}
            """
        
        # Add return instruction
        prompt += """
        
        Return ONLY the README.md content with no additional text.
        IMPORTANT: Do not mention CrewAI or agent-based architecture. Focus only on direct tool usage.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            return result.data
        
    except Exception as e:
        logger.error(f"Error generating README: {str(e)}", exc_info=True)
        
        # Create a fallback README
        if "tools" in mcp_tool_data and isinstance(mcp_tool_data["tools"], list):
            tools = mcp_tool_data["tools"]
            tools_text = ", ".join([tool.get('purpose', 'External Tool') for tool in tools[:3]])
            if len(tools) > 3:
                tools_text += f" and {len(tools) - 3} more"
            return f"# MCP Tools Package\n\nThis package provides {tools_text}.\n\nSee individual tool directories for usage instructions."
        else:
            purpose = mcp_tool_data.get('purpose', 'External Tool')
            return f"# MCP Tool: {purpose}\n\nThis package provides a tool for {purpose}.\n\nSee the tool files for usage instructions."

@mcp_tool_agent.tool
async def setup_mcp_tool_structure(ctx: RunContext[MCPToolDeps], 
                                  mcp_tool_data: Dict[str, str],
                                  output_dir: str) -> Dict[str, str]:
    """
    Set up the directory structure for MCP tools.
    Creates only the tool structure without CrewAI agents.
    
    Args:
        ctx: Run context with dependencies
        mcp_tool_data: Dictionary containing information about the MCP tool
        output_dir: Base directory to create files in
        
    Returns:
        Dictionary with path information for created files
    """
    try:
        # Create directories for MCP tools
        os.makedirs(output_dir, exist_ok=True)
        
        # Create MCP tools directory structure
        mcp_tools_dir = os.path.join(output_dir, "mcp_tools")
        os.makedirs(mcp_tools_dir, exist_ok=True)
        
        # Create __init__.py in mcp_tools directory
        with open(os.path.join(mcp_tools_dir, "__init__.py"), "w") as f:
            f.write("# MCP Tools Package\n")
        
        # Determine tool directory name from purpose
        if "spotify" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "spotify"
        elif "github" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "github"
        elif "youtube" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "youtube"
        elif "twitter" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "twitter"
        elif "slack" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "slack"
        elif "gmail" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "gmail"
        elif "google_drive" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "google_drive"
        elif "discord" in mcp_tool_data.get("purpose", "").lower():
            tool_dir_name = "discord"
        else:
            # Default to a sanitized version of the purpose
            tool_dir_name = mcp_tool_data.get("purpose", "").lower().split()[0]
            # Clean up any non-alphanumeric characters
            tool_dir_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in tool_dir_name)
            if not tool_dir_name or tool_dir_name.isdigit():
                tool_dir_name = "tool"
        
        # Create tool-specific directory
        tool_dir = os.path.join(mcp_tools_dir, tool_dir_name)
        os.makedirs(tool_dir, exist_ok=True)
        
        # Create __init__.py in tool directory
        with open(os.path.join(tool_dir, "__init__.py"), "w") as f:
            f.write(f"# {tool_dir_name.capitalize()} MCP Tool\n")
            f.write(f"# Purpose: {mcp_tool_data.get('purpose', 'Integration tool')}\n")
        
        # Write tool code
        tool_code_path = os.path.join(tool_dir, "tool.py")
        with open(tool_code_path, "w") as f:
            f.write(mcp_tool_data.get("tool_code", ""))
        
        # Write a tool example file with standalone usage (no CrewAI)
        example_code_path = os.path.join(tool_dir, "example.py")
        with open(example_code_path, "w") as f:
            f.write(f"""# Example usage of the {tool_dir_name} tool
# This example shows how to use the tool directly without CrewAI

from mcp_tools.{tool_dir_name}.tool import *

def main():
    # Initialize the tool
    # Example based on: {mcp_tool_data.get('purpose', 'MCP Tool')}
    
    # TODO: Add initialization and usage code
    
    print(f"Example usage of {tool_dir_name} tool")
    
if __name__ == "__main__":
    main()
""")
        
        # Write connection script if available
        connection_script = mcp_tool_data.get("connection_script")
        if connection_script:
            connection_path = os.path.join(tool_dir, "connection.py")
            with open(connection_path, "w") as f:
                f.write(connection_script)
        
        # Write requirements
        requirements = mcp_tool_data.get("requirements")
        if requirements:
            req_path = os.path.join(tool_dir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write(requirements)
                
        # Return paths
        return {
            "mcp_tools_dir": mcp_tools_dir,
            "tool_dir": tool_dir,
            "tool_code_path": tool_code_path,
            "example_code_path": example_code_path,
            "tool_name": tool_dir_name
        }
                
    except Exception as e:
        logger.error(f"Error setting up MCP tool structure: {str(e)}", exc_info=True)
        return {}

@mcp_tool_agent.tool
async def analyze_tool_code(ctx: RunContext[MCPToolDeps], 
                           tool_code: str,
                           purpose: str) -> Dict[str, Any]:
    """
    Deeply analyze the provided tool code to understand its structure, 
    functionality, and integration points.
    
    Args:
        ctx: Run context with dependencies
        tool_code: The code to analyze
        purpose: The purpose of the tool
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        logger.info(f"Analyzing tool code for purpose: {purpose}")
        
        prompt = f"""
        Perform a deep analysis of this MCP tool code to understand its structure, 
        functionality, and integration points. The tool purpose is: "{purpose}".
        
        CODE:
        ```python
        {tool_code}
        ```
        
        Provide your analysis in this JSON format:
        {{
            "main_class": "name of the main class",
            "inheritance": ["base classes"],
            "key_methods": [
                {{
                    "name": "method name",
                    "purpose": "what this method does",
                    "parameters": ["param1", "param2"],
                    "modifications_possible": true/false,
                    "modification_points": ["description of possible modifications"]
                }}
            ],
            "authentication_mechanism": "how the tool authenticates",
            "api_endpoints_used": ["endpoint1", "endpoint2"],
            "dependencies": ["library1", "library2"],
            "error_handling": "approach to error handling",
            "customization_points": ["description of parts that can be customized"],
            "architecture_pattern": "description of the code architecture"
        }}
        
        Return ONLY valid JSON without explanations or comments.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.1,
                response_format={"type": "json_object"}  # Explicitly request JSON format
            )
            result_text = response.choices[0].message.content.strip()
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            result_text = result.data
        
        # Extract JSON if it's wrapped in a code block
        if "```json" in result_text:
            result_text = result_text.split("```json", 1)[1]
            if "```" in result_text:
                result_text = result_text.split("```", 1)[0]
        elif "```" in result_text:
            # Generic code block with no language specified
            result_text = result_text.split("```", 1)[1]
            if "```" in result_text:
                result_text = result_text.split("```", 1)[0]
        
        # Parse the JSON response
        try:
            analysis = json.loads(result_text)
            logger.info(f"Successfully analyzed tool code. Main class: {analysis.get('main_class', 'Unknown')}")
            return analysis
        except Exception as parse_error:
            logger.error(f"Error parsing tool analysis JSON: {str(parse_error)}")
            # Return a minimal structure with actual response for debugging
            logger.info(f"Raw response: {result_text[:100]}...")  # Log part of the response for debugging
            return {
                "main_class": "Unknown",
                "inheritance": [],
                "key_methods": [],
                "authentication_mechanism": "Unknown",
                "api_endpoints_used": [],
                "dependencies": [],
                "error_handling": "Unknown",
                "customization_points": [],
                "architecture_pattern": "Unknown",
                "raw_response": result_text[:500]  # Include part of the raw response for debugging
            }
    
    except Exception as e:
        logger.error(f"Error analyzing tool code: {str(e)}", exc_info=True)
        return {"error": f"Error analyzing tool code: {str(e)}"}

@mcp_tool_agent.tool
async def customize_tool_implementation(ctx: RunContext[MCPToolDeps],
                                      reference_code: str,
                                      code_analysis: Dict[str, Any],
                                      requirements: Dict[str, Any],
                                      custom_features: List[str]) -> str:
    """
    Customize a tool implementation based on reference code and specific requirements.
    
    Args:
        ctx: Run context with dependencies
        reference_code: The reference code to adapt
        code_analysis: Analysis of the reference code
        requirements: User requirements for the tool
        custom_features: Specific features to add
        
    Returns:
        Customized tool code
    """
    try:
        logger.info(f"Customizing tool implementation with {len(custom_features)} custom features")
        
        # Create a prompt that includes the reference code, analysis, and requirements
        prompt = f"""
        You are an expert MCP tool developer. Your task is to customize the reference tool 
        implementation to meet specific user requirements.
        
        ### REFERENCE CODE:
        ```python
        {reference_code}
        ```
        
        ### CODE ANALYSIS:
        {json.dumps(code_analysis, indent=2)}
        
        ### USER REQUIREMENTS:
        {json.dumps(requirements, indent=2)}
        
        ### CUSTOM FEATURES TO IMPLEMENT:
        {json.dumps(custom_features, indent=2)}
        
        Adapt the reference code to meet these requirements. Specifically:
        1. Keep the overall architecture and structure of the original tool
        2. Modify the implementation to include all required custom features
        3. Add proper error handling for each new feature
        4. Include appropriate documentation
        5. Maintain compatibility with the MCP framework and CrewAI
        
        Return ONLY the complete customized Python code without explanations or comments outside the code.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.2
            )
            customized_code = response.choices[0].message.content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            customized_code = result.data
        
        # Clean up the response - extract just the code if wrapped in markdown
        if "```python" in customized_code:
            customized_code = customized_code.split("```python", 1)[1]
            if "```" in customized_code:
                customized_code = customized_code.split("```", 1)[0]
        
        logger.info(f"Successfully customized tool implementation: {len(customized_code)} characters")
        return customized_code.strip()
        
    except Exception as e:
        logger.error(f"Error customizing tool implementation: {str(e)}", exc_info=True)
        return reference_code  # Return the original reference code on error

@mcp_tool_agent.tool
async def verify_tool_integration(ctx: RunContext[MCPToolDeps],
                                 customized_code: str,
                                 requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that the customized tool meets all requirements and will function correctly.
    
    Args:
        ctx: Run context with dependencies
        customized_code: The customized tool code
        requirements: User requirements for the tool
        
    Returns:
        Dictionary with verification results
    """
    try:
        logger.info(f"Verifying tool integration")
        
        prompt = f"""
        Verify that this customized MCP tool meets all requirements and will function correctly.
        
        ### CUSTOMIZED CODE:
        ```python
        {customized_code}
        ```
        
        ### REQUIREMENTS TO CHECK:
        {json.dumps(requirements, indent=2)}
        
        Analyze the code for:
        1. Implementation of all required features
        2. Proper error handling
        3. Correct authentication mechanisms
        4. Compatibility with MCP/CrewAI framework
        5. Missing dependencies
        6. Potential runtime issues
        
        Provide your verification in this JSON format:
        {{
            "verification_status": "passed|partial|failed",
            "implemented_features": ["feature1", "feature2"],
            "missing_features": ["feature3"],
            "potential_issues": ["issue1", "issue2"],
            "error_handling_quality": "good|adequate|poor",
            "authentication_properly_implemented": true|false,
            "framework_compatibility": "good|adequate|poor",
            "missing_dependencies": ["dependency1"],
            "suggestions": ["suggestion1", "suggestion2"]
        }}
        
        Return ONLY valid JSON without explanations or comments.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}  # Explicitly request JSON format
            )
            result_text = response.choices[0].message.content.strip()
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            result_text = result.data
        
        # Extract JSON if it's wrapped in a code block
        if "```json" in result_text:
            result_text = result_text.split("```json", 1)[1]
            if "```" in result_text:
                result_text = result_text.split("```", 1)[0]
        elif "```" in result_text:
            # Generic code block with no language specified
            result_text = result_text.split("```", 1)[1]
            if "```" in result_text:
                result_text = result_text.split("```", 1)[0]
        
        # Parse the JSON response
        try:
            verification = json.loads(result_text)
            logger.info(f"Verification status: {verification.get('verification_status', 'Unknown')}")
            return verification
        except Exception as parse_error:
            logger.error(f"Error parsing verification results: {str(parse_error)}")
            # Return a minimal structure with the actual response for debugging
            logger.info(f"Raw response: {result_text[:100]}...")  # Log part of the response for debugging
            return {
                "verification_status": "unknown",
                "implemented_features": [],
                "missing_features": [],
                "potential_issues": ["Failed to parse verification results"],
                "error_handling_quality": "unknown",
                "authentication_properly_implemented": False,
                "framework_compatibility": "unknown",
                "missing_dependencies": [],
                "suggestions": ["Manually verify the tool implementation"],
                "raw_response": result_text[:500]  # Include part of the raw response for debugging
            }
    
    except Exception as e:
        logger.error(f"Error verifying tool integration: {str(e)}", exc_info=True)
        return {
            "verification_status": "error",
            "error": str(e)
        }

# Add this helper function to safely create RunContext for MCP tools
def create_mcp_context(deps: Any, prompt: str = "", usage: Dict[str, Any] = None) -> RunContext:
    """
    Safely create a RunContext for MCP tools, handling potential errors.
    
    Args:
        deps: MCPToolDeps or PydanticAIDeps instance
        prompt: The prompt to include in the context
        usage: Optional usage tracking data
        
    Returns:
        RunContext instance with proper dependencies
    """
    try:
        if usage is None:
            usage = {}
        
        # Handle different types of deps objects
        model_param = None
        
        # Check if deps has openai_client attribute which can be used as model
        if hasattr(deps, 'openai_client'):
            model_param = deps.openai_client
            
        # If deps has a model attribute directly, use that
        if hasattr(deps, 'model'):
            model_param = deps.model
            
        # Create the RunContext with all required parameters
        ctx = RunContext(
            deps=deps,
            usage=usage,
            prompt=prompt,
            model=model_param  # Use the model parameter we determined
        )
        return ctx
    except Exception as e:
        logger.error(f"Error creating MCP RunContext: {str(e)}")
        # Create a minimal context as fallback
        return RunContext(
            deps=deps,
            usage={},
            prompt=prompt,
            model=model_param if 'model_param' in locals() else None  # Use model_param if it was set
        )

@mcp_tool_agent.tool
async def generate_agents_py(ctx: RunContext[MCPToolDeps],
                           tools_data: Dict[str, Any],
                           output_dir: str,
                           tool_class_names: List[str]) -> str:
    """
    Generate an agents.py file that uses the MCP tools from tools.py.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Dictionary containing information about the tools
        output_dir: Directory to save the file
        tool_class_names: List of tool class names from the generated tools.py file
        
    Returns:
        Content of the generated agents.py file
    """
    try:
        logger.info("Generating agents.py file")
        
        # Extract tool purposes for better agent descriptions
        tool_purposes = []
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            tool_purposes = [tool.get('purpose', f"Tool {i+1}") for i, tool in enumerate(tools_data["tools"])]
        elif "purpose" in tools_data:
            tool_purposes = [tools_data.get("purpose")]
            
        # Create summary of what the tools do for better agent context
        tools_summary = ", ".join(tool_purposes[:3])
        if len(tool_purposes) > 3:
            tools_summary += f" and {len(tool_purposes) - 3} more tools"
            
        # Example code for GitHub agents as reference
        github_agents_example = '''
from crewai import Agent
from tools import GitHubMCPTool

class GitHubAgentFactory:
    """Factory class to create GitHub agents"""
    
    @staticmethod
    def create_github_agent():
        """Create a GitHub agent with the ability to interact with GitHub API"""
        github_tool = GitHubMCPTool()
        
        return Agent(
            role="GitHub Operations Expert",
            goal="Help users perform GitHub operations efficiently and accurately",
            backstory="""You are an expert in GitHub operations with years of experience using GitHub's API.
            You understand repository management, issue tracking, code search, and all other 
            GitHub functionalities. You help users interact with GitHub by analyzing their
            requests, determining the appropriate GitHub operations to perform, and executing
            those operations on their behalf.""",
            verbose=True,
            allow_delegation=False,
            tools=[github_tool]
        )
    
    @staticmethod
    def create_github_repository_manager():
        """Create a GitHub agent specializing in repository management"""
        github_tool = GitHubMCPTool()
        
        return Agent(
            role="GitHub Repository Manager",
            goal="Efficiently manage GitHub repositories and their contents",
            backstory="""You are a specialized GitHub expert focused on repository management.
            You excel at creating, updating, and managing repositories, files, branches, and
            other repository-related operations. You understand Git workflows and can help
            users perform repository operations using GitHub's API.""",
            verbose=True,
            allow_delegation=False,
            tools=[github_tool]
        )
    
    @staticmethod
    def create_github_issue_tracker():
        """Create a GitHub agent specializing in issue tracking"""
        github_tool = GitHubMCPTool()
        
        return Agent(
            role="GitHub Issue Tracker",
            goal="Efficiently manage GitHub issues and pull requests",
            backstory="""You are a specialized GitHub expert focused on issue tracking and management.
            You excel at creating, updating, and managing issues, pull requests, comments, and 
            other collaboration features. You understand GitHub project management workflows and
            can help users perform issue-related operations using GitHub's API.""",
            verbose=True,
            allow_delegation=False,
            tools=[github_tool]
        )
    
    @staticmethod
    def create_github_code_searcher():
        """Create a GitHub agent specializing in code search"""
        github_tool = GitHubMCPTool()
        
        return Agent(
            role="GitHub Code Searcher",
            goal="Find and analyze code across GitHub repositories",
            backstory="""You are a specialized GitHub expert focused on code search and analysis.
            You excel at finding code, repositories, and users across GitHub using advanced search
            queries. You can help users discover relevant code, repositories, and developers based
            on their specific needs and interests using GitHub's API.""",
            verbose=True,
            allow_delegation=False,
            tools=[github_tool]
        ) 


'''
            
        prompt = f"""
        Generate a complete agents.py file for a CrewAI project that will use MCP tools.
        
        The file should:
        1. Import the necessary modules and tools
        2. Initialize the tools from tools.py
        3. Create 1-3 specialized agents that use these tools effectively
        
        Available tool classes: {", ".join(tool_class_names)}
        
        Tools summary: {tools_summary}
        
        REFERENCE EXAMPLE (follow this pattern closely):
        ```python
{github_agents_example}
        ```
        
        Return ONLY the complete Python code for agents.py with detailed documentation.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.7
            )
            agents_py_content = response.choices[0].message.content
            
            # Save the file
            agents_file_path = os.path.join(output_dir, "agents.py")
            with open(agents_file_path, "w") as f:
                f.write(agents_py_content)
                
            logger.info(f"Generated and saved agents.py to {agents_file_path}")
            return agents_py_content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            
            # Save the file
            agents_file_path = os.path.join(output_dir, "agents.py")
            with open(agents_file_path, "w") as f:
                f.write(result.data)
                
            logger.info(f"Generated and saved agents.py to {agents_file_path}")
            return result.data
        
    except Exception as e:
        logger.error(f"Error generating agents.py: {str(e)}", exc_info=True)
        return f"# Error generating agents.py: {str(e)}"

@mcp_tool_agent.tool
async def generate_tasks_py(ctx: RunContext[MCPToolDeps],
                          tools_data: Dict[str, Any],
                          output_dir: str,
                          agent_names: List[str]) -> str:
    """
    Generate a tasks.py file for a CrewAI project based on the tools and agents.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Dictionary containing information about the tools
        output_dir: Directory to save the file
        agent_names: List of agent names from the generated agents.py file
        
    Returns:
        Content of the generated tasks.py file
    """
    try:
        logger.info("Generating tasks.py file")
        
        # Extract tool purposes for better task descriptions
        tool_purposes = []
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            tool_purposes = [tool.get('purpose', f"Tool {i+1}") for i, tool in enumerate(tools_data["tools"])]
        elif "purpose" in tools_data:
            tool_purposes = [tools_data.get("purpose")]
            
        # Create summary of what the tools do for better task design
        tools_summary = ", ".join(tool_purposes[:3])
        if len(tool_purposes) > 3:
            tools_summary += f" and {len(tool_purposes) - 3} more tools"
        
        # Example code for GitHub tasks as reference
        github_tasks_example = '''
from crewai import Task, Agent
from typing import Dict, Any, Optional, List, Tuple
import re

class TaskType:
    """Enum for task types"""
    CREATE_REPO = "repository_management"
    ISSUE_MANAGEMENT = "issue_management"
    CODE_SEARCH = "code_search"
    GENERAL = "general"

class GitHubTaskFactory:
    """Factory class to create GitHub-related tasks"""
    
    @staticmethod
    def create_task(query: str, agent: Agent, human_input: bool = False, context: Optional[List[Tuple[str, Any]]] = None) -> Task:
        """Create appropriate task based on the user query"""
        # Default context if none provided
        if context is None:
            context = []
            
        # Determine task category
        task_type = GitHubTaskFactory.determine_task_type(query)
        
        # Create appropriate task
        if task_type == TaskType.CREATE_REPO:
            return GitHubTaskFactory.create_repo_task(query, agent, human_input, context)
        elif task_type == TaskType.ISSUE_MANAGEMENT:
            return GitHubTaskFactory.create_issue_task(query, agent, human_input, context)
        elif task_type == TaskType.CODE_SEARCH:
            return GitHubTaskFactory.create_code_search_task(query, agent, human_input, context)
        else:
            return GitHubTaskFactory.create_general_task(query, agent, human_input, context)
    
    @staticmethod
    def determine_task_type(query: str) -> TaskType:
        """Determine the type of task based on the query"""
        # Repository-related keywords
        repo_keywords = [
            "repository", "repo", "create repo", "fork", "clone", "branch", 
            "commit", "push", "pull", "merge", "create repository"
        ]
        
        # Issue-related keywords
        issue_keywords = [
            "issue", "pull request", "pr", "bug", "feature", "comment",
            "create issue", "close issue", "update issue", "track issue"
        ]
        
        # Code search keywords
        search_keywords = [
            "search", "find", "locate", "code search", "search code",
            "search repository", "search repositories", "find code", "search for"
        ]
        
        # Check for matches
        query_lower = query.lower()
        
        for keyword in repo_keywords:
            if keyword in query_lower:
                return TaskType.CREATE_REPO
        
        for keyword in issue_keywords:
            if keyword in query_lower:
                return TaskType.ISSUE_MANAGEMENT
        
        for keyword in search_keywords:
            if keyword in query_lower:
                return TaskType.CODE_SEARCH
        
        # Default to generic GitHub task
        return TaskType.GENERAL
    
    @staticmethod
    def create_repo_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a repository management task"""
        # Extract repository name if present
        repo_name_pattern = r'(?:repository|repo) (?:named|called) ["|\']?([a-zA-Z0-9_-]+)["|\']?'
        match = re.search(repo_name_pattern, query, re.IGNORECASE)
        repo_name = match.group(1) if match else None
        
        # Prepare task description
        description = f"""
        You are tasked with helping the user with GitHub repository management. 
        
        USER REQUEST: {query}
        
        If the user wants to create a repository:
        1. Extract the repository name from the query: {repo_name if repo_name else 'Not specified - ask the user to provide a name'}
        2. Create the repository with the specified name, description, and other options
        3. Provide the URL to the newly created repository
        
        If the user wants to update a repository:
        1. Search for the repository if necessary
        2. Make the requested changes (description, topics, visibility, etc.)
        3. Confirm the changes were made successfully
        
        If the user wants information about a repository:
        1. Search for the repository by name or description
        2. Retrieve the requested information (stars, forks, issues, etc.)
        3. Present the information in a clear, formatted manner
        """
        
        return Task(
            description=description,
            expected_output="A detailed response to the user's repository-related request, including confirmation of actions taken and any relevant details.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_issue_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create an issue management task"""
        description = f"""
        You are tasked with helping the user with GitHub issue management.
        
        USER REQUEST: {query}
        
        If the user wants to create an issue:
        1. Determine the target repository (ask if not specified)
        2. Extract the issue title and description from the query
        3. Create the issue with appropriate labels, assignees, or milestone if specified
        4. Provide the URL to the newly created issue
        
        If the user wants to update an issue:
        1. Find the issue by number or through search
        2. Make the requested changes (status, labels, assignees, etc.)
        3. Confirm the changes were made successfully
        
        If the user wants information about issues:
        1. Search for issues matching the criteria (open/closed, labels, assignees, etc.)
        2. Retrieve the requested information
        3. Present the information in a clear, formatted manner
        """
        
        return Task(
            description=description,
            expected_output="A detailed response to the user's issue-related request, including confirmation of actions taken and any relevant details.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_code_search_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a code search task"""
        description = f"""
        You are tasked with helping the user search for code on GitHub.
        
        USER REQUEST: {query}
        
        1. Extract the search terms and any filters from the query
        2. Perform the search with appropriate parameters (language, stars, etc.)
        3. Present the most relevant results, including:
           - Repository name and link
           - File path and link
           - Code snippet with context
           - Star count and other relevant metrics
        4. If the user wants more details about a specific result, provide additional information
        
        Be thorough in your search but focus on presenting the most relevant and high-quality results first.
        """
        
        return Task(
            description=description,
            expected_output="A detailed list of code search results formatted in a clear, easily readable manner, with links to the repositories and files.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def create_general_task(query: str, agent: Agent, human_input: bool, context: List[Tuple[str, Any]]) -> Task:
        """Create a general GitHub task for queries that don't fit other categories"""
        description = f"""
        You are tasked with helping the user with a general GitHub-related request.
        
        USER REQUEST: {query}
        
        1. Analyze the request to determine what GitHub information or actions are needed
        2. Use your tools to interact with the GitHub API appropriately
        3. Provide a comprehensive response that addresses the user's needs
        4. Include relevant links, instructions, or follow-up actions as appropriate
        
        Your goal is to be as helpful as possible while working within the capabilities
        of the GitHub API. If you cannot complete a request, explain why and suggest alternatives.
        """
        
        return Task(
            description=description,
            expected_output="A helpful response that addresses the user's GitHub-related query, with appropriate details, links, and suggestions.",
            agent=agent,
            human_input=human_input,
            context=context
        )
    
    @staticmethod
    def _extract_params_from_query(query: str) -> Dict[str, str]:
        """Extract potential parameters from the query"""
        params = {}
        
        # Look for repositories in format "username/repo"
        repo_pattern = r'(\w+)/(\w+)'
        repo_matches = re.findall(repo_pattern, query)
        if repo_matches:
            params["owner"] = repo_matches[0][0]
            params["repo"] = repo_matches[0][1]
        
        # Look for issue numbers
        issue_pattern = r'#(\d+)'
        issue_matches = re.findall(issue_pattern, query)
        if issue_matches:
            params["issue_number"] = issue_matches[0]
        
        # Look for potential repository names
        repo_name_pattern = r'(?:repository|repo) (?:named|called) ["|\']?([a-zA-Z0-9_-]+)["|\']?'
        repo_name_matches = re.findall(repo_name_pattern, query, re.IGNORECASE)
        if repo_name_matches:
            params["name"] = repo_name_matches[0]
        
        return params 
'''
        
        prompt = f"""
        Generate a complete tasks.py file for a CrewAI project.
        
        The file should:
        1. Import the necessary modules
        2. Create 2-4 task definitions that effectively use the agents
        3. Design tasks that make good use of the specialized tools
        
        Available agents: {", ".join(agent_names)}
        
        Tools summary: {tools_summary}
        
        REFERENCE EXAMPLE (follow this pattern closely):
        ```python
{github_tasks_example}
        ```
        
        Return ONLY the complete Python code for tasks.py with detailed documentation.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.7
            )
            tasks_py_content = response.choices[0].message.content
            
            # Save the file
            tasks_file_path = os.path.join(output_dir, "tasks.py")
            with open(tasks_file_path, "w") as f:
                f.write(tasks_py_content)
                
            logger.info(f"Generated and saved tasks.py to {tasks_file_path}")
            return tasks_py_content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            
            # Save the file
            tasks_file_path = os.path.join(output_dir, "tasks.py")
            with open(tasks_file_path, "w") as f:
                f.write(result.data)
                
            logger.info(f"Generated and saved tasks.py to {tasks_file_path}")
            return result.data
        
    except Exception as e:
        logger.error(f"Error generating tasks.py: {str(e)}", exc_info=True)
        return f"# Error generating tasks.py: {str(e)}"

@mcp_tool_agent.tool
async def generate_crew_py(ctx: RunContext[MCPToolDeps],
                         tools_data: Dict[str, Any],
                         output_dir: str,
                         agent_names: List[str],
                         task_names: List[str]) -> str:
    """
    Generate a crew.py file that coordinates agents and tasks.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Dictionary containing information about the tools
        output_dir: Directory to save the file
        agent_names: List of agent names from the generated agents.py file
        task_names: List of task names from the generated tasks.py file
        
    Returns:
        Content of the generated crew.py file
    """
    try:
        logger.info("Generating crew.py file")
        
        # Extract the primary purpose for the crew name
        primary_purpose = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list) and len(tools_data["tools"]) > 0:
            primary_purpose = tools_data["tools"][0].get('purpose', '')
        elif "purpose" in tools_data:
            primary_purpose = tools_data.get("purpose", '')
            
        # Create a descriptive crew name based on the tool purpose
        crew_name = "CrewAI"
        if "spotify" in primary_purpose.lower():
            crew_name = "MusicDiscoveryCrew"
        elif "github" in primary_purpose.lower():
            crew_name = "CodeCollaborationCrew"
        elif "youtube" in primary_purpose.lower():
            crew_name = "VideoAnalysisCrew"
        elif "twitter" in primary_purpose.lower() or "social" in primary_purpose.lower():
            crew_name = "SocialMediaCrew"
        elif "search" in primary_purpose.lower() or "research" in primary_purpose.lower():
            crew_name = "ResearchCrew"
        
        # Example code for GitHub crew as reference
        github_crew_example = '''
from crewai import Crew, Process
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
from agents import GitHubAgentFactory
from tasks import GitHubTaskFactory

# Get logger
logger = logging.getLogger("github_agent.crew")

class GitHubCrew:
    """Crew for handling GitHub operations"""
    
    def __init__(self, verbose: bool = True, memory: bool = True, human_input: bool = True):
        """Initialize the GitHub crew"""
        self.verbose = verbose
        self.memory = memory
        self.human_input = human_input
        self.agent = None
        self.tasks = []
        
        logger.info("Initializing GitHub crew")
        logger.info(f"Settings: verbose={verbose}, memory={memory}, human_input={human_input}")
        
        # Create the agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the GitHub agent"""
        logger.info("Creating general GitHub agent")
        self.agent = GitHubAgentFactory.create_github_agent()
        # Log agent type instead of agent name
        logger.info(f"Agent created: GitHub General Agent")
    
    def _create_specialized_agent(self, query: str):
        """Create a specialized agent based on the query"""
        logger.info(f"Selecting specialized agent for query: {query[:50]}..." if len(query) > 50 else query)
        
        # Determine the type of agent needed
        agent_type = "general"
        if any(keyword in query.lower() for keyword in ["repository", "repo", "create", "fork", "branch"]):
            logger.info("Selected repository manager agent based on keywords")
            self.agent = GitHubAgentFactory.create_github_repository_manager()
            agent_type = "repository_manager"
        elif any(keyword in query.lower() for keyword in ["issue", "pull request", "pr", "bug"]):
            logger.info("Selected issue tracker agent based on keywords")
            self.agent = GitHubAgentFactory.create_github_issue_tracker()
            agent_type = "issue_tracker"
        elif any(keyword in query.lower() for keyword in ["search", "find", "code search"]):
            logger.info("Selected code searcher agent based on keywords")
            self.agent = GitHubAgentFactory.create_github_code_searcher()
            agent_type = "code_searcher"
        else:
            # Default to general GitHub agent
            logger.info("No specialized keywords found, using general GitHub agent")
            self.agent = GitHubAgentFactory.create_github_agent()
            agent_type = "general"
        
        # Log agent type instead of agent name
        logger.info(f"Agent selected: GitHub {agent_type.replace('_', ' ').title()} Agent")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a GitHub-related query and return the result"""
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
        task = GitHubTaskFactory.create_task(
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
        """Process multiple GitHub-related queries and return the results"""
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
        """Process a complex GitHub request with multiple steps"""
        logger.info("-" * 80)
        logger.info(f"Processing complex request: {request}")
        logger.info(f"Number of steps: {len(steps)}")
        
        # Track execution time
        start_time = time.time()
        
        # Reset tasks
        self.tasks = []
        logger.info("Tasks reset")
        
        # Create a general GitHub agent
        logger.info("Creating general GitHub agent for complex request")
        self.agent = GitHubAgentFactory.create_github_agent()
        # Log agent type instead of agent name
        logger.info("Agent created: GitHub General Agent")
        
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
                safe_context = [(k, v if k.lower() not in ["token", "key", "secret", "password", "auth"] else f"{str(v)[:3]}...{str(v)[-3:]}") for k, v in step_context]
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
            task = GitHubTaskFactory.create_task(
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
        sensitive_fields = ['token', 'password', 'secret', 'key', 'auth']
        for field in sensitive_fields:
            for key in list(safe_context.keys()):
                if field in key.lower() and isinstance(safe_context[key], str):
                    safe_context[key] = f"{safe_context[key][:3]}...{safe_context[key][-3:]}"
        
        return safe_context 
'''
        
        prompt = f"""
        Generate a complete crew.py file for a CrewAI project.
        
        The file should:
        1. Import the necessary modules, agents, and tasks
        2. Create a Crew class that effectively coordinates the agents to perform tasks
        3. Include a simple main function that demonstrates how to use the crew
        
        Suggested crew name: {crew_name}
        Available agents: {", ".join(agent_names)}
        Available tasks: {", ".join(task_names)}
        
        REFERENCE EXAMPLE (follow this pattern closely):
        ```python
{github_crew_example}
        ```
        
        Return ONLY the complete Python code for crew.py with detailed documentation.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.7
            )
            crew_py_content = response.choices[0].message.content
            
            # Save the file
            crew_file_path = os.path.join(output_dir, "crew.py")
            with open(crew_file_path, "w") as f:
                f.write(crew_py_content)
                
            logger.info(f"Generated and saved crew.py to {crew_file_path}")
            return crew_py_content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            
            # Save the file
            crew_file_path = os.path.join(output_dir, "crew.py")
            with open(crew_file_path, "w") as f:
                f.write(result.data)
                
            logger.info(f"Generated and saved crew.py to {crew_file_path}")
            return result.data
        
    except Exception as e:
        logger.error(f"Error generating crew.py: {str(e)}", exc_info=True)
        return f"# Error generating crew.py: {str(e)}"

@mcp_tool_agent.tool
async def extract_names_from_file(ctx: RunContext[MCPToolDeps],
                                file_content: str,
                                entity_type: str) -> List[str]:
    """
    Extract entity names (agents, tasks, tools) from a file.
    
    Args:
        ctx: Run context with dependencies
        file_content: Content of the file to analyze
        entity_type: Type of entity to extract ('agent', 'task', 'tool')
        
    Returns:
        List of extracted entity names
    """
    try:
        logger.info(f"Extracting {entity_type} names from file")
        
        prompt = f"""
        Extract all {entity_type} names from this Python code:
        
        ```python
        {file_content}
        ```
        
        Return ONLY a comma-separated list of {entity_type} variable names (e.g., "spotify_agent, github_agent").
        Do not include any additional text, just the variable names.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            names_text = response.choices[0].message.content.strip()
            
            # Parse the comma-separated list
            names = [name.strip() for name in names_text.split(",") if name.strip()]
            logger.info(f"Extracted {len(names)} {entity_type} names: {', '.join(names)}")
            return names
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            names = [name.strip() for name in result.data.split(",") if name.strip()]
            logger.info(f"Extracted {len(names)} {entity_type} names: {', '.join(names)}")
            return names
            
    except Exception as e:
        logger.error(f"Error extracting {entity_type} names: {str(e)}", exc_info=True)
        return []

@mcp_tool_agent.tool
async def generate_crewai_readme(ctx: RunContext[MCPToolDeps],
                               tools_data: Dict[str, Any],
                               generated_files: List[str]) -> str:
    """
    Generate a README.md file for the complete CrewAI project.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Dictionary containing information about the tools
        generated_files: List of generated filenames
        
    Returns:
        README content as a string
    """
    try:
        files_list = "\n".join([f"- {file}" for file in generated_files])
        
        # Extract the primary purpose
        primary_purpose = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list) and len(tools_data["tools"]) > 0:
            primary_purpose = tools_data["tools"][0].get('purpose', '')
        elif "purpose" in tools_data:
            primary_purpose = tools_data.get("purpose", '')
            
        # Determine if we need special setup instructions based on tool type
        purpose_lower = primary_purpose.lower()
        special_instructions = ""
        
        if "github" in purpose_lower or "git" in purpose_lower or "repository" in purpose_lower:
            special_instructions = """
## GitHub Authentication Setup

This project requires a GitHub Personal Access Token to access the GitHub API. Follow these steps to set it up:

1. Go to your GitHub account settings
2. Navigate to Developer Settings > Personal Access Tokens > Tokens (classic)
3. Click "Generate new token" and select "Generate new token (classic)"
4. Give your token a descriptive name
5. Select the following scopes:
   - `repo` (Full control of private repositories)
   - `user` (User profile data)
6. Click "Generate token"
7. Copy the token (you will only see it once!)
8. Add it to your .env file:
   ```
   GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here
   ```

**IMPORTANT:** Never commit your .env file to version control!
"""
        elif "spotify" in purpose_lower or "music" in purpose_lower:
            special_instructions = """
## Spotify API Setup

This project requires Spotify API credentials. Follow these steps to set them up:

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Create a new application
4. Once created, you'll get a Client ID and Client Secret
5. Add these to your .env file:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

**IMPORTANT:** Never commit your .env file to version control!
"""
            
        prompt = f"""
        Create a comprehensive README.md file for a CrewAI project that uses MCP tools.
        
        The project purpose: {primary_purpose}
        
        The README should include:
        1. A clear title and project overview
        2. Installation instructions, including all required dependencies
        3. Setup instructions including any API keys needed
        4. Usage examples with code snippets
        5. Description of the project structure
        6. Troubleshooting tips
        
        Special Setup Instructions:
        {special_instructions}
        
        Generated files:
        {files_list}
        
        Return ONLY the README.md content with professional formatting.
        """
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            return result.data
            
    except Exception as e:
        logger.error(f"Error generating CrewAI README: {str(e)}", exc_info=True)
        
        # Create a fallback README with special instructions if applicable
        purpose_lower = primary_purpose.lower() if 'primary_purpose' in locals() else ""
        readme_content = f"# CrewAI Project: {tools_data.get('purpose', 'CrewAI with MCP Tools')}\n\n"
        readme_content += "This package provides a complete CrewAI project with MCP tools integration.\n\n"
        
        if "github" in purpose_lower:
            readme_content += "## GitHub Authentication\n\nThis project requires a GitHub Personal Access Token.\n"
            readme_content += "Add your token to a .env file:\n```\nGITHUB_PERSONAL_ACCESS_TOKEN=your_token_here\n```\n\n"
        
        readme_content += "See the individual files for usage instructions."
        return readme_content

def extract_class_names_from_tools(tools_py_content: str) -> List[str]:
    """
    Extract tool class names from tools.py content.
    
    Args:
        tools_py_content: Content of the tools.py file
        
    Returns:
        List of tool class names
    """
    try:
        import re
        logger.info("Extracting tool class names from tools.py content")
        
        matches = []
        
        # Pattern 1: Look for class definitions that inherit from BaseTool
        pattern1 = r'class\s+([a-zA-Z0-9_]+)\s*\([^)]*BaseTool[^)]*\)'
        matches1 = re.findall(pattern1, tools_py_content)
        if matches1:
            logger.info(f"Found {len(matches1)} classes inheriting from BaseTool: {', '.join(matches1)}")
            matches.extend(matches1)
        
        # Pattern 2: Look for classes with Tool in the name
        pattern2 = r'class\s+([a-zA-Z0-9_]+Tool)\s*\('
        matches2 = re.findall(pattern2, tools_py_content)
        if matches2:
            logger.info(f"Found {len(matches2)} classes with 'Tool' in name: {', '.join(matches2)}")
            # Add only classes that aren't already found
            for match in matches2:
                if match not in matches:
                    matches.append(match)
        
        # Pattern 3: Look for any class that inherits from Agent or Tool
        pattern3 = r'class\s+([a-zA-Z0-9_]+)\s*\([^)]*(?:Agent|Tool)[^)]*\)'
        matches3 = re.findall(pattern3, tools_py_content)
        if matches3:
            logger.info(f"Found {len(matches3)} classes inheriting from Agent or Tool: {', '.join(matches3)}")
            # Add only classes that aren't already found
            for match in matches3:
                if match not in matches:
                    matches.append(match)
        
        # Pattern 4: Look for tool instantiations (for cases where tools aren't defined in the file)
        pattern4 = r'([a-zA-Z0-9_]+)\s*=\s*Tool\('
        matches4 = re.findall(pattern4, tools_py_content)
        
        # If no class definitions found, try looking for imports
        if not matches:
            # Look for imports that might be tool classes
            import_pattern = r'from\s+[a-zA-Z0-9_.]+\s+import\s+([^#\n]+)'
            import_matches = re.findall(import_pattern, tools_py_content)
            
            if import_matches:
                # Process each import statement
                for import_stmt in import_matches:
                    # Split by comma to get individual imports
                    imports = [i.strip() for i in import_stmt.split(',')]
                    
                    # Filter for likely tool classes (ends with 'Tool' or contains 'Tool')
                    tool_imports = [i for i in imports if i.endswith('Tool') or ('Tool' in i and i[0].isupper())]
                    logger.info(f"Found potential tool imports: {', '.join(tool_imports)}")
                    matches.extend(tool_imports)
        
        # If still no matches found, use a more generic approach to detect class definitions
        if not matches:
            generic_class_pattern = r'class\s+([a-zA-Z0-9_]+)\s*\('
            generic_matches = re.findall(generic_class_pattern, tools_py_content)
            logger.info(f"Falling back to generic class detection, found: {', '.join(generic_matches)}")
            matches.extend(generic_matches)
        
        # Remove any duplicates while preserving order
        unique_matches = []
        for match in matches:
            if match not in unique_matches:
                unique_matches.append(match)
                
        logger.info(f"Final extracted tool class names: {', '.join(unique_matches)}")
        return unique_matches
        
    except Exception as e:
        logger.error(f"Error extracting class names from tools.py: {e}", exc_info=True)
        return []

def generate_env_file_content(tools_data: Dict[str, Any]) -> str:
    """
    Generate .env file content based on the tools data.
    
    Args:
        tools_data: Information about the tools being used
        
    Returns:
        Content for the .env file with appropriate API keys
    """
    try:
        env_content = "# Environment variables for the CrewAI project\n\n"
        
        # Check tool purpose to add relevant API keys
        purpose_text = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list) and len(tools_data["tools"]) > 0:
            purpose_text = " ".join([tool.get('purpose', '') for tool in tools_data["tools"]]).lower()
        elif "purpose" in tools_data:
            purpose_text = tools_data.get("purpose", "").lower()
        
        if "query" in tools_data:
            purpose_text += " " + tools_data.get("query", "").lower()
            
        # Add specific API keys based on the detected services
        if any(keyword in purpose_text for keyword in ["github", "repository", "repo", "git"]):
            env_content += "# GitHub API Token\nGITHUB_PERSONAL_ACCESS_TOKEN=your_token_here\n\n"
            
        if any(keyword in purpose_text for keyword in ["spotify", "music", "playlist", "song"]):
            env_content += "# Spotify API Credentials\nSPOTIFY_CLIENT_ID=your_client_id_here\nSPOTIFY_CLIENT_SECRET=your_client_secret_here\n\n"
            
        if any(keyword in purpose_text for keyword in ["search", "serper", "web", "google"]):
            env_content += "# Serper API Key\nSERPER_API_KEY=your_serper_api_key_here\n\n"
            
        if any(keyword in purpose_text for keyword in ["youtube", "video"]):
            env_content += "# YouTube API Key\nYOUTUBE_API_KEY=your_youtube_api_key_here\n\n"
            
        # Add OpenAI API key for CrewAI (always needed)
        env_content += "# OpenAI API Key for CrewAI\nOPENAI_API_KEY=your_openai_api_key_here\n"
        
        return env_content
        
    except Exception as e:
        logger.error(f"Error generating .env file content: {e}")
        return "# Environment variables for the CrewAI project\n\nOPENAI_API_KEY=your_openai_api_key_here\n"

def generate_requirements(tools_data: Dict[str, Any], tool_class_names: List[str]) -> str:
    """
    Generate requirements.txt content based on the tools data.
    
    Args:
        tools_data: Information about the tools being used
        tool_class_names: Names of tool classes from tools.py
        
    Returns:
        Content for the requirements.txt file
    """
    try:
        # Base requirements for any CrewAI project
        req_content = """# Required packages for this CrewAI project
crewai>=0.28.0
smithery>=0.1.0
mcp>=0.4.0
python-dotenv>=1.0.0
openai>=1.3.0
"""
        
        # Check tool purpose to add extra requirements
        purpose_text = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list) and len(tools_data["tools"]) > 0:
            purpose_text = " ".join([tool.get('purpose', '') for tool in tools_data["tools"]]).lower()
        elif "purpose" in tools_data:
            purpose_text = tools_data.get("purpose", "").lower()
        
        if "query" in tools_data:
            purpose_text += " " + tools_data.get("query", "").lower()
        
        # Add specific packages based on detected services or class names
        if any(keyword in purpose_text for keyword in ["spotify", "music"]) or any("Spotify" in name for name in tool_class_names):
            req_content += "spotipy>=2.23.0\n"
            
        if any(keyword in purpose_text for keyword in ["github", "repository", "git"]) or any("GitHub" in name for name in tool_class_names):
            req_content += "pygithub>=2.1.0\n"
            
        if any(keyword in purpose_text for keyword in ["youtube", "video"]) or any("YouTube" in name for name in tool_class_names):
            req_content += "google-api-python-client>=2.100.0\n"
            
        if any(keyword in purpose_text for keyword in ["twitter", "tweet"]) or any("Twitter" in name for name in tool_class_names):
            req_content += "tweepy>=4.14.0\n"
            
        return req_content
        
    except Exception as e:
        logger.error(f"Error generating requirements.txt content: {e}")
        return "# Required packages for this CrewAI project\ncrewai>=0.28.0\nsmithery>=0.1.0\nmcp>=0.4.0\npython-dotenv>=1.0.0\nopenai>=1.3.0\n"

@mcp_tool_agent.tool
async def generate_tools_py(ctx: RunContext[MCPToolDeps], 
                           tools_data: Dict[str, Any],
                           output_dir: str) -> str:
    """
    Generate a tools.py file for a CrewAI project.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Information about the tools being used
        output_dir: Directory to save the file
        
    Returns:
        Content of the generated tools.py file
    """
    try:
        logger.info("Generating tools.py file")
        
        # Use integrate_mcp_tool_with_code to generate the tools.py content
        tools_result = await integrate_mcp_tool_with_code(ctx, {}, tools_data)
        tools_py_content = tools_result.get("tools_py_content", "")
        
        # Validate and fix common errors in the tools.py content
        fixed_content, applied_fixes = validate_tools_py_content(tools_py_content)
        
        # If fixes were applied, log them
        if applied_fixes:
            logger.info(f"MCP TOOL INTEGRATION: Applied {len(applied_fixes)} fixes to tools.py for common errors")
            for fix in applied_fixes:
                logger.info(f"  - {fix['message']}")
            tools_py_content = fixed_content
        
        # Save the file
        tools_file_path = os.path.join(output_dir, "tools.py")
        with open(tools_file_path, "w") as f:
            f.write(tools_py_content)
            
        # Create an error_list.md file documenting known errors and their fixes
        error_list_path = os.path.join(output_dir, "error_list.md")
        with open(error_list_path, "w") as f:
            f.write("# Common Errors in tools.py and Their Fixes\n\n")
            f.write("This file documents common errors that can occur in tools.py files and their automatic fixes.\n\n")
            
            for error in TOOLS_PY_COMMON_ERRORS:
                f.write(f"## {error['description']}\n\n")
                f.write(f"**Error Type:** {error['error_type']}\n\n")
                f.write(f"**Solution:** {error['error_message']}\n\n")
            
            f.write("## Applied Fixes in this Project\n\n")
            if applied_fixes:
                for fix in applied_fixes:
                    f.write(f"- **{fix['error_type']}**: {fix['message']}\n")
            else:
                f.write("No fixes were needed for this tools.py file.\n")
        
        logger.info(f"Generated and saved tools.py to {tools_file_path}")
        logger.info(f"Generated error_list.md documenting common errors and fixes at {error_list_path}")
        
        return tools_py_content
        
    except Exception as e:
        logger.error(f"Error generating tools.py: {str(e)}", exc_info=True)
        return f"# Error generating tools.py: {str(e)}"

# Make sure these are explicitly defined at the module level
__all__ = [
    'mcp_tool_agent',
    'MCPToolDeps',
    'find_relevant_mcp_tools',
    'integrate_mcp_tool_with_code',
    'create_connection_script',
    'generate_mcp_integration_readme',
    'setup_mcp_tool_structure',
    'analyze_tool_code',
    'customize_tool_implementation',
    'verify_tool_integration',
    'create_mcp_context',
    'generate_complete_crewai_project',
    'generate_agents_py',
    'generate_tasks_py',
    'generate_crew_py',
    'generate_main_py',
    'generate_crewai_readme',
    'extract_names_from_file',
    'generate_tools_py',
    'extract_class_names_from_tools',
    'generate_env_file_content',
    'generate_requirements',
    'adapt_templates_to_user_requirements',
    'extract_specific_requirements',
    'adapt_agents_to_requirements',
    'adapt_tasks_to_requirements',
    'adapt_crew_to_requirements',
    'adapt_tools_to_requirements'
]

@mcp_tool_agent.tool
async def adapt_templates_to_user_requirements(
    ctx: RunContext[MCPToolDeps],
    user_query: str,
    template_results: Dict[str, str],
    tools_data: Dict[str, Any],
    tool_class_names: List[str]
) -> Dict[str, str]:
    """
    Adapt the generated template files to better match specific user requirements.
    This function analyzes the user query and customizes the generated code to fit their needs.
    
    Args:
        ctx: Run context with dependencies
        user_query: The user's original query
        template_results: Dictionary of generated template files
        tools_data: Dictionary containing tool data
        tool_class_names: List of tool class names
        
    Returns:
        Dictionary of adapted template files
    """
    try:
        logger.info("Adapting generated templates to specific user requirements")
        
        # Extract the current code from template results
        agents_code = template_results.get("agents_py", "")
        tasks_code = template_results.get("tasks_py", "")
        crew_code = template_results.get("crew_py", "")
        tools_code = template_results.get("tools_py", "")
        
        if not agents_code or not tasks_code or not crew_code:
            logger.warning("Missing required code for adaptation. Using original template results.")
            return template_results
        
        # Extract specific requirements from user query with enhanced analysis
        user_requirements = await extract_specific_requirements(ctx, user_query, tools_data)
        logger.info(f"Extracted {len(user_requirements)} specific requirements from user query")
        
        # If we have specific requirements, adapt each file
        if user_requirements:
            # First, analyze if we need specific functionality beyond what templates provide
            required_functions = []
            required_agent_types = []
            required_task_types = []
            workflow_requirements = []
            data_flow_requirements = []
            
            # Categorize requirements with better classification
            for req in user_requirements:
                req_lower = req.lower()
                
                # Improved categorization with more specific patterns
                if any(term in req_lower for term in ["function", "method", "tool", "api", "capability", "feature"]):
                    required_functions.append(req)
                elif any(term in req_lower for term in ["agent", "assistant", "specialist", "expert", "analyzer"]):
                    required_agent_types.append(req)
                elif any(term in req_lower for term in ["task", "job", "process", "operation", "activity"]):
                    required_task_types.append(req)
                elif any(term in req_lower for term in ["workflow", "sequence", "pipeline", "flow", "order"]):
                    workflow_requirements.append(req)
                elif any(term in req_lower for term in ["data", "information", "input", "output", "parameter"]):
                    data_flow_requirements.append(req)
                else:
                    # If can't categorize, treat as general workflow requirement
                    workflow_requirements.append(req)
            
            # Get any additional context from the reasoner output if available
            additional_context = ""
            if ctx.deps.reasoner_output:
                additional_context = f"\nAdditional context from reasoning:\n{ctx.deps.reasoner_output}\n"
            
            # Get any architectural plan if available
            architecture_plan = ""
            if ctx.deps.architecture_plan:
                architecture_plan = f"\nArchitectural plan:\n{ctx.deps.architecture_plan}\n"
            
            # Adapt agents.py based on requirements with enhanced instructions
            if required_agent_types or workflow_requirements:
                logger.info("Adapting agents.py to match specific user requirements")
                agents_code = await adapt_agents_to_requirements(
                    ctx, 
                    agents_code, 
                    required_agent_types, 
                    workflow_requirements,
                    tool_class_names,
                    user_query,
                    data_flow_requirements
                )
            
            # Adapt tasks.py based on requirements with enhanced instructions
            if required_task_types or workflow_requirements or data_flow_requirements:
                logger.info("Adapting tasks.py to match specific user requirements")
                # First extract agent names from the adapted agents code
                agent_names = await extract_names_from_file(ctx, agents_code, "agent")
                tasks_code = await adapt_tasks_to_requirements(
                    ctx,
                    tasks_code,
                    agent_names,
                    required_task_types,
                    workflow_requirements,
                    user_query,
                    data_flow_requirements
                )
            
            # Adapt crew.py based on requirements with enhanced workflow support
            if workflow_requirements or data_flow_requirements:
                logger.info("Adapting crew.py to match specific user requirements")
                # Extract names from adapted code
                agent_names = await extract_names_from_file(ctx, agents_code, "agent")
                task_names = await extract_names_from_file(ctx, tasks_code, "task")
                crew_code = await adapt_crew_to_requirements(
                    ctx,
                    crew_code,
                    agent_names,
                    task_names,
                    workflow_requirements,
                    user_query,
                    data_flow_requirements
                )
            
            # Adapt tools.py if needed for new functionality with enhanced features
            if required_functions or data_flow_requirements:
                logger.info("Adapting tools.py to implement required functions")
                tools_code = await adapt_tools_to_requirements(
                    ctx,
                    tools_code or "",  # Handle case when tools_py isn't in template_results
                    agents_code,
                    required_functions,
                    user_query,
                    data_flow_requirements
                )
        
        # Create updated results dictionary
        adapted_results = dict(template_results)
        adapted_results["agents_py"] = agents_code
        adapted_results["tasks_py"] = tasks_code
        adapted_results["crew_py"] = crew_code
        if tools_code:
            adapted_results["tools_py"] = tools_code
        
        logger.info("Successfully adapted template-generated code to user requirements")
        return adapted_results
        
    except Exception as e:
        logger.error(f"Error adapting templates to user requirements: {str(e)}", exc_info=True)
        # Return original results if adaptation fails
        return template_results

@mcp_tool_agent.tool
async def extract_specific_requirements(
    ctx: RunContext[MCPToolDeps],
    user_query: str,
    tools_data: Dict[str, Any]
) -> List[str]:
    """
    Extract specific requirements from the user query.
    This helps understand what the user actually wants to do with the tools.
    
    Args:
        ctx: Run context with dependencies
        user_query: The user's original query/request
        tools_data: Information about the tools
        
    Returns:
        List of specific requirements extracted from the query
    """
    try:
        if not user_query:
            return []
            
        logger.info("Extracting specific requirements from user query")
        
        # Use the LLM to extract requirements
        prompt = f"""
        Analyze this user request and extract specific requirements for a CrewAI project:
        
        USER REQUEST: "{user_query}"
        
        Focus on extracting:
        1. What specific functions/capabilities they need
        2. What specific agents they might need
        3. What specific tasks they want to perform
        4. Any specific workflow or process requirements
        
        Return ONLY a JSON array of requirement strings, with NO explanation.
        Each requirement should be a specific, actionable item that needs to be implemented.
        Example: ["Create an agent that can recommend songs based on genres", "Implement playlist creation functionality"]
        """
        
        # Use OpenAI client if available
        if ctx.deps.openai_client:
            try:
                model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
                response = await ctx.deps.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                # Parse the JSON response
                try:
                    requirements_json = json.loads(response.choices[0].message.content)
                    if isinstance(requirements_json, dict) and "requirements" in requirements_json:
                        requirements = requirements_json["requirements"]
                    elif isinstance(requirements_json, list):
                        requirements = requirements_json
                    else:
                        requirements = []
                        
                    logger.info(f"Extracted {len(requirements)} specific requirements")
                    return requirements
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response for requirements extraction")
                    # Try basic extraction instead
                    content = response.choices[0].message.content
                    if "[" in content and "]" in content:
                        try:
                            # Try to extract the array portion
                            array_text = content[content.find("["):content.rfind("]")+1]
                            requirements = json.loads(array_text)
                            return requirements
                        except:
                            pass
                    
                    # Fallback to basic extraction
                    lines = content.strip().split("\n")
                    requirements = [line.strip().strip('"\'').strip() for line in lines if line.strip()]
                    return [r for r in requirements if len(r) > 10]  # Filter out very short lines
            except Exception as api_error:
                logger.error(f"Error calling OpenAI API for requirements extraction: {str(api_error)}")
        
        # Fallback: Basic requirement extraction
        tools_text = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            for tool in tools_data["tools"]:
                if "purpose" in tool:
                    tools_text += f" {tool['purpose']}"
                    
        # List of common operations/verbs to look for
        operations = [
            "recommend", "create", "find", "search", "analyze", "generate", 
            "retrieve", "compose", "build", "optimize", "summarize"
        ]
        
        # List of common objects in the domains
        objects = [
            "song", "songs", "artist", "artists", "album", "albums", "playlist", "playlists",
            "repository", "repositories", "code", "issue", "issues", "pull request",
            "video", "videos", "channel", "channels", "content"
        ]
        
        # Extract basic requirements based on operations and objects
        requirements = []
        query_lower = user_query.lower()
        
        for operation in operations:
            if operation in query_lower:
                # Find the closest object
                for obj in objects:
                    if obj in query_lower:
                        # Try to extract a more complete phrase
                        start_idx = max(0, query_lower.find(operation) - 10)
                        end_idx = min(len(query_lower), query_lower.find(obj) + len(obj) + 10)
                        phrase = user_query[start_idx:end_idx]
                        requirements.append(f"Implement functionality to {operation} {obj}")
                        break
        
        logger.info(f"Extracted {len(requirements)} basic requirements")
        return requirements
                
    except Exception as e:
        logger.error(f"Error extracting requirements: {str(e)}")
        return []

@mcp_tool_agent.tool
async def adapt_agents_to_requirements(
    ctx: RunContext[MCPToolDeps],
    agents_code: str,
    required_agent_types: List[str],
    workflow_requirements: List[str],
    tool_class_names: List[str],
    user_query: str,
    data_flow_requirements: List[str] = None
) -> str:
    """
    Adapt agents code to meet specific requirements.
    
    Args:
        ctx: RunContext with dependencies
        agents_code: Original agents code
        required_agent_types: List of required agent types
        workflow_requirements: List of workflow requirements
        tool_class_names: List of tool class names
        user_query: Original user query
        data_flow_requirements: Data flow requirements
        
    Returns:
        Adapted agents code
    """
    try:
        # Analyze user query to determine specific needs
        agent_needs = await analyze_agent_requirements(ctx, user_query)
        agent_count = agent_needs.get("agent_count", 0)
        domain_terminology = agent_needs.get("domain_terms", [])
        specific_agent_roles = agent_needs.get("agent_roles", [])
        
        # Check if we should strictly limit the number of agents
        should_limit_agents = agent_count > 0 and agent_count < 3  # If user specifically wants fewer agents
        
        # Identify problematic generic terms that should be replaced
        generic_terms = [
            "YouTube Transcript Expert", 
            "YouTube Transcript Specialist",
            "Video Content Analyst", 
            "Language Specialist", 
            "Educational Content Analyst",
            "YouTube", 
            "Transcript",
            "Video"
        ]
        
        # Count original occurrences to compare later
        original_generic_count = sum(agents_code.count(term) for term in generic_terms)
        
        # Prepare a comprehensive, directive prompt
        prompt = f"""
        # AGENT CODE TRANSFORMATION TASK

        You will COMPLETELY TRANSFORM this agents.py template code into a HIGHLY SPECIALIZED version that 
        meets the exact requirements from the user query. You must make SUBSTANTIAL changes, not minor edits.
        
        ## USER QUERY
        {user_query}
        
        ## AGENT REQUIREMENTS
        Required Agent Types: {', '.join(required_agent_types)}
        Specific Agent Roles: {', '.join(specific_agent_roles)}
        Number of Agents: {"STRICT LIMIT of " + str(agent_count) if should_limit_agents else "Flexible, based on needs"}
        Domain-Specific Terminology: {', '.join(domain_terminology)}
        
        ## WORKFLOW REQUIREMENTS
        {chr(10).join("- " + req for req in workflow_requirements)}
        
        ## AVAILABLE TOOL CLASSES
        {', '.join(tool_class_names)}
        
        ## TRANSFORMATION REQUIREMENTS (MANDATORY)
        
        1. AGENT ROLES AND PURPOSES:
           - COMPLETELY REWRITE all agent roles, goals, and backstories with highly specific language
           - Each agent must have a SPECIALIZED purpose clearly tied to the user's domain
           - Agent descriptions must reference SPECIFIC ENTITIES, TECHNOLOGIES, and CONCEPTS from the user's domain
           - Replace generic descriptions with INDUSTRY-SPECIFIC, DETAILED characterizations
        
        2. AGENT STRUCTURE:
           - If user wants {agent_count} agents, RUTHLESSLY DELETE any unnecessary agent factory methods
           - RENAME agent classes to directly reflect their domain-specific functions (e.g., "FinancialDataAnalysisAgent" not "DataAgent")
           - RESTRUCTURE the agent hierarchy if needed to better match the workflow requirements
        
        3. TOOL ASSIGNMENT:
           - CAREFULLY DISTRIBUTE tools based on each agent's specialized role
           - ENSURE tool assignments make logical sense for each agent's specific responsibilities
           - ADD specialized knowledge about how to use each tool in the agent descriptions
        
        4. CODE CUSTOMIZATION:
           - CHANGE variable names to use domain terminology
           - MODIFY parameter handling to match specific use case requirements
           - ADD any specialized functionality needed for this specific domain
        
        5. DOMAIN SPECIFICITY:
           - All text in the file must use DOMAIN-SPECIFIC LANGUAGE that directly relates to {', '.join(domain_terminology[:3])}
           - DEFAULT values must reflect industry standards for this specific domain
           - ERROR handling must address domain-specific edge cases
           
        6. GENERIC TERMS TO REPLACE:
           - You must REMOVE or REPLACE all occurrences of these generic terms: {', '.join(generic_terms)}
           - Each agent's role title must be domain-specific and NOT contain the words "YouTube" or "Transcript"
           - Every agent backstory must include at least 3 of these domain terms: {', '.join(domain_terminology[:3])}
        
        ## ORIGINAL AGENTS CODE
        ```python
        {agents_code}
        ```
        
        ## IMPORTANT NOTES
        - Your adaptation must be DOMAIN-SPECIFIC, not generic
        - The final code should look SUBSTANTIALLY DIFFERENT from the template
        - Make STRUCTURAL changes, not just text substitutions
        - If the query mentions a specific number of agents, strictly adhere to that
        - Return ONLY the Python code with NO explanations
        - CRITICAL: Every agent function must be substantially rewritten or removed
        """
        
        # Call OpenAI to transform the code
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for more creative customization
            max_tokens=4000
        )
        
        # Extract the transformed code
        transformed_code = completion.choices[0].message.content
        
        # Clean up the code - remove any markdown formatting
        transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
        
        # Verify substantial changes were made
        difference_percentage = calculate_code_difference(agents_code, transformed_code)
        logger.info(f"Initial code difference: {difference_percentage:.2f}%")
        
        # Check for remaining generic terms
        remaining_generic_count = sum(transformed_code.count(term) for term in generic_terms)
        
        # Count occurrences of domain terminology
        domain_term_count = sum(1 for term in domain_terminology if term.lower() in transformed_code.lower())
        
        # If still too generic, try again with stronger prompt
        if difference_percentage < 40 or remaining_generic_count > original_generic_count * 0.3 or domain_term_count < min(3, len(domain_terminology)):
            logger.info(f"Initial transformation not substantial enough: diff={difference_percentage:.2f}%, generic_terms={remaining_generic_count}, domain_terms={domain_term_count}")
            
            # Try again with a more forceful prompt
            enhanced_prompt = f"""
            Your previous adaptation was NOT TRANSFORMATIVE ENOUGH. You must make MUCH MORE EXTENSIVE CHANGES.
            
            {prompt}
            
            ## ADDITIONAL REQUIREMENTS
            - CHANGE AT LEAST 70% of the code structure and content
            - DO NOT preserve ANY generic descriptions from the template
            - COMPLETELY REWRITE every agent role, goal, and backstory
            - DELETE any methods or classes that aren't specifically needed
            - ADD domain-specific logic and functionality
            - REQUIRE deeper customization in EVERY aspect of the code
            - RENAME the main class to reflect the domain (e.g., "{domain_terminology[0] if domain_terminology else "Domain"}AgentFactory")
            - CREATE domain-specific agent roles like "{domain_terminology[0] if domain_terminology else "Domain"} Analyst"
            
            ## PREVIOUS UNSUCCESSFUL ATTEMPT
            ```python
            {transformed_code}
            ```
            
            ## WARNING
            Your response will be REJECTED if it's not substantially different from the template.
            """
            
            # Try again with a more forceful prompt
            completion = await ctx.deps.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.9,  # Even higher temperature for more divergent generation
                max_tokens=4000
            )
            
            transformed_code = completion.choices[0].message.content
            transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
            
            # Check the results of the second attempt
            difference_percentage = calculate_code_difference(agents_code, transformed_code)
            remaining_generic_count = sum(transformed_code.count(term) for term in generic_terms)
            domain_term_count = sum(1 for term in domain_terminology if term.lower() in transformed_code.lower())
            
            logger.info(f"Second attempt results: diff={difference_percentage:.2f}%, generic_terms={remaining_generic_count}, domain_terms={domain_term_count}")
            
            # If still not good enough, try one final time with explicit line-by-line requirements
            if difference_percentage < 50 or remaining_generic_count > 2 or domain_term_count < min(3, len(domain_terminology)):
                final_prompt = f"""
                CRITICAL FAILURE: Your customization is still too generic.
                
                You MUST create a COMPLETELY NEW VERSION of this agents.py file.
                
                REQUIREMENTS:
                - RENAME the class to "{domain_terminology[0] if domain_terminology else 'Domain'}AgentFactory"
                - DELETE all occurrences of "YouTube", "Transcript", "Expert", "Specialist"
                - CREATE new agent roles like "{domain_terminology[0]} Analyst", "{domain_terminology[1] if len(domain_terminology) > 1 else 'Specialized'} Researcher"
                - WRITE completely new backstories that mention: {', '.join(domain_terminology[:3])}
                - CHANGE all goals to focus specifically on {domain_terminology[0] if domain_terminology else 'the domain'}
                
                # USER REQUEST
                {user_query}
                
                # DOMAIN TERMS TO USE (MANDATORY)
                {', '.join(domain_terminology)}
                
                DO NOT PRESERVE ANY YOUTUBE-SPECIFIC TERMINOLOGY.
                START FROM SCRATCH if necessary.
                
                ## TOOL CLASSES TO USE
                {', '.join(tool_class_names)}
                """
                
                # Final attempt
                completion = await ctx.deps.openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=1.0,  # Maximum creativity
                    max_tokens=4000
                )
                
                transformed_code = completion.choices[0].message.content
                transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
        
        # Log the final results
        difference_percentage = calculate_code_difference(agents_code, transformed_code)
        remaining_generic_count = sum(transformed_code.count(term) for term in generic_terms)
        domain_term_count = sum(1 for term in domain_terminology if term.lower() in transformed_code.lower())
        
        logger.info(f"Final transformation results: diff={difference_percentage:.2f}%, generic_terms={remaining_generic_count}, domain_terms={domain_term_count}")
        
        return transformed_code
    
    except Exception as e:
        logger.error(f"Error adapting agents code: {e}")
        # Return original code on error
        return agents_code


async def analyze_agent_requirements(ctx: RunContext[MCPToolDeps], user_query: str) -> Dict[str, Any]:
    """
    Analyze user query to extract specific agent requirements.
    
    Args:
        ctx: RunContext with dependencies
        user_query: User's query or request
        
    Returns:
        Dictionary with agent requirements
    """
    try:
        prompt = f"""
        Analyze this user request for a CrewAI tool and extract specific requirements for agents.
        
        USER QUERY: {user_query}
        
        Please extract and provide:
        1. The number of agents explicitly or implicitly required (use a specific number, default to 0 if unclear)
        2. 5-10 domain-specific terms that should be used in agent descriptions
        3. Specific agent roles mentioned or implied in the request (e.g., "data analyst", "content writer")
        
        Format your response as a JSON object with these keys:
        - agent_count: number of agents (integer)
        - domain_terms: list of domain-specific terms (array of strings)
        - agent_roles: list of specific agent roles (array of strings)
        """
        
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        logger.info(f"Extracted agent requirements: {json.dumps(result)}")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing agent requirements: {e}")
        return {
            "agent_count": 0,
            "domain_terms": ["specialized", "customized", "domain-specific"],
            "agent_roles": []
        }


def calculate_code_difference(original: str, modified: str) -> float:
    """
    Calculate the percentage difference between two code strings.
    
    Args:
        original: Original code string
        modified: Modified code string
        
    Returns:
        Percentage difference (0-100)
    """
    import difflib
    
    # Normalize code by removing whitespace variations
    def normalize(code):
        return '\n'.join([line.strip() for line in code.split('\n') if line.strip()])
    
    original_norm = normalize(original)
    modified_norm = normalize(modified)
    
    # Use SequenceMatcher to calculate similarity
    matcher = difflib.SequenceMatcher(None, original_norm, modified_norm)
    similarity = matcher.ratio()
    
    # Convert to difference percentage
    difference = (1 - similarity) * 100
    
    return difference


@mcp_tool_agent.tool
async def adapt_tasks_to_requirements(
    ctx: RunContext[MCPToolDeps],
    tasks_code: str,
    agent_names: List[str],
    required_task_types: List[str],
    workflow_requirements: List[str],
    user_query: str,
    data_flow_requirements: List[str] = None
) -> str:
    """
    Adapt tasks code to meet specific requirements.
    
    Args:
        ctx: RunContext with dependencies
        tasks_code: Original tasks code
        agent_names: List of agent names
        required_task_types: List of required task types
        workflow_requirements: List of workflow requirements
        user_query: Original user query
        data_flow_requirements: Data flow requirements
        
    Returns:
        Adapted tasks code
    """
    try:
        # Analyze user query to determine task-specific needs
        task_needs = await analyze_task_requirements(ctx, user_query)
        domain_terminology = task_needs.get("domain_terms", [])
        specific_task_types = task_needs.get("task_types", [])
        workflow_patterns = task_needs.get("workflow_patterns", [])
        
        # Prepare a comprehensive, directive prompt
        prompt = f"""
        # TASK CODE TRANSFORMATION TASK

        You will COMPLETELY TRANSFORM this tasks.py template code into a HIGHLY SPECIALIZED version that 
        meets the exact requirements from the user query. You must make SUBSTANTIAL changes, not minor edits.
        
        ## USER QUERY
        {user_query}
        
        ## TASK REQUIREMENTS
        Required Task Types: {', '.join(required_task_types)}
        Specific Task Types from Analysis: {', '.join(specific_task_types)}
        Workflow Patterns: {', '.join(workflow_patterns)}
        Domain-Specific Terminology: {', '.join(domain_terminology)}
        
        ## WORKFLOW REQUIREMENTS
        {chr(10).join("- " + req for req in workflow_requirements)}
        
        ## AVAILABLE AGENT NAMES
        {', '.join(agent_names)}
        
        ## TRANSFORMATION REQUIREMENTS (MANDATORY)
        
        1. TASK DESCRIPTIONS AND PURPOSES:
           - COMPLETELY REWRITE all task descriptions with highly specific language
           - Each task must have DOMAIN-SPECIFIC instructions that reference real-world entities and concepts
           - Replace generic steps with DETAILED, INDUSTRY-SPECIFIC procedures
           - Expected outputs must be PRECISELY defined for the domain
        
        2. TASK STRUCTURE:
           - RUTHLESSLY DELETE any task types not needed for this specific user query
           - RENAME task classes and methods to directly reflect domain-specific operations
           - RESTRUCTURE the task hierarchy to better match the workflow requirements
        
        3. AGENT ASSIGNMENT:
           - ENSURE tasks are assigned to the correct specialized agents
           - All agent references must EXACTLY match the agent names provided
           - MODIFY task dependencies to create a logical workflow for this specific domain
        
        4. CODE CUSTOMIZATION:
           - CHANGE variable names to use domain terminology
           - REWRITE task factory methods to be domain-specific
           - ADD specialized validation and handling for domain-specific inputs
        
        5. DOMAIN SPECIFICITY:
           - All text in the file must use DOMAIN-SPECIFIC LANGUAGE related to {', '.join(domain_terminology[:3])}
           - Tasks must include references to SPECIFIC ENTITIES, TECHNOLOGIES, or PROCESSES from the domain
        
        ## ORIGINAL TASKS CODE
        ```python
        {tasks_code}
        ```
        
        ## IMPORTANT NOTES
        - Your adaptation must be DOMAIN-SPECIFIC, not generic
        - The final code should look SUBSTANTIALLY DIFFERENT from the template
        - Task descriptions should include SPECIFIC DOMAIN KNOWLEDGE, not generic instructions
        - Make STRUCTURAL changes, not just text substitutions
        - Return ONLY the Python code with NO explanations
        """
        
        # Call OpenAI to transform the code
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for more creative customization
            max_tokens=4000
        )
        
        # Extract the transformed code
        transformed_code = completion.choices[0].message.content
        
        # Clean up the code - remove any markdown formatting
        transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
        
        # Verify substantial changes were made
        if calculate_code_difference(tasks_code, transformed_code) < 40:
            # Less than 40% different - try again with a more forceful prompt
            logger.info("Initial tasks transformation not substantial enough, retrying with stronger prompt")
            
            enhanced_prompt = f"""
            Your previous adaptation was NOT TRANSFORMATIVE ENOUGH. You must make MUCH MORE EXTENSIVE CHANGES.
            
            {prompt}
            
            ## ADDITIONAL REQUIREMENTS
            - CHANGE AT LEAST 70% of the code structure and content
            - DO NOT preserve ANY generic descriptions from the template
            - COMPLETELY REWRITE every task description with domain-specific details
            - DELETE any methods or classes that aren't specifically needed
            - ADD domain-specific validation and processing logic
            - REQUIRE deeper customization in EVERY aspect of the code
            
            ## WARNING
            Your response will be REJECTED if it's not substantially different from the template.
            """
            
            # Try again with a more forceful prompt
            completion = await ctx.deps.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.9,  # Even higher temperature for more divergent generation
                max_tokens=4000
            )
            
            transformed_code = completion.choices[0].message.content
            transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
            
        return transformed_code
    
    except Exception as e:
        logger.error(f"Error adapting tasks code: {e}")
        # Return original code on error
        return tasks_code


async def analyze_task_requirements(ctx: RunContext[MCPToolDeps], user_query: str) -> Dict[str, Any]:
    """
    Analyze user query to extract specific task requirements.
    
    Args:
        ctx: RunContext with dependencies
        user_query: User's query or request
        
    Returns:
        Dictionary with task requirements
    """
    try:
        prompt = f"""
        Analyze this user request for a CrewAI tool and extract specific requirements for tasks.
        
        USER QUERY: {user_query}
        
        Please extract and provide:
        1. 5-10 domain-specific terms that should be used in task descriptions
        2. Specific task types mentioned or implied in the request (e.g., "data analysis", "content generation")
        3. Workflow patterns implied in the request (e.g., "sequential processing", "approval workflow")
        
        Format your response as a JSON object with these keys:
        - domain_terms: list of domain-specific terms (array of strings)
        - task_types: list of specific task types (array of strings)
        - workflow_patterns: list of workflow patterns (array of strings)
        """
        
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        logger.info(f"Extracted task requirements: {json.dumps(result)}")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing task requirements: {e}")
        return {
            "domain_terms": ["specialized", "customized", "domain-specific"],
            "task_types": [],
            "workflow_patterns": []
        }


@mcp_tool_agent.tool
async def adapt_crew_to_requirements(
    ctx: RunContext[MCPToolDeps],
    crew_code: str,
    agent_names: List[str],
    task_names: List[str],
    workflow_requirements: List[str],
    user_query: str,
    data_flow_requirements: List[str] = None
) -> str:
    """
    Adapt crew code to meet specific requirements.
    
    Args:
        ctx: RunContext with dependencies
        crew_code: Original crew code
        agent_names: List of agent names
        task_names: List of task names
        workflow_requirements: List of workflow requirements
        user_query: Original user query
        data_flow_requirements: Data flow requirements
        
    Returns:
        Adapted crew code
    """
    try:
        # Analyze user query to determine crew-specific needs
        crew_needs = await analyze_crew_requirements(ctx, user_query)
        domain_terminology = crew_needs.get("domain_terms", [])
        process_type = crew_needs.get("process_type", "sequential")
        workflow_complexity = crew_needs.get("workflow_complexity", "medium")
        
        # Prepare a comprehensive, directive prompt
        prompt = f"""
        # CREW CODE TRANSFORMATION TASK

        You will TRANSFORM this crew.py template code into a SPECIALIZED version that 
        meets the exact requirements from the user query. You must make significant changes, not minor edits.
        
        ## USER QUERY
        {user_query}
        
        ## CREW REQUIREMENTS
        Process Type: {process_type}
        Workflow Complexity: {workflow_complexity}
        Domain-Specific Terminology: {', '.join(domain_terminology)}
        
        ## WORKFLOW REQUIREMENTS
        {chr(10).join("- " + req for req in workflow_requirements)}
        
        ## AVAILABLE AGENT NAMES
        {', '.join(agent_names)}
        
        ## AVAILABLE TASK NAMES
        {', '.join(task_names)}
        
        ## TRANSFORMATION REQUIREMENTS (MANDATORY)
        
        1. CREW ORCHESTRATION:
           - ENSURE all agent and task imports match EXACTLY with the provided names
           - SET the appropriate process type ({process_type}) based on the workflow
           - RENAME the crew class to specifically describe its domain function
           - CUSTOMIZE the crew initialization with domain-specific parameters
        
        2. WORKFLOW STRUCTURE:
           - ADJUST the task assignment and dependencies to create a logical domain-specific workflow
           - MODIFY the process flow based on the workflow complexity ({workflow_complexity})
           - ENSURE proper agent-task pairing based on specializations
        
        3. CODE CUSTOMIZATION:
           - REPLACE generic variable names with domain-specific terminology
           - REWRITE logging statements to reference domain-specific events and data
           - ADD specialized error handling for domain-specific edge cases
        
        4. DOMAIN SPECIFICITY:
           - All text in the file must use DOMAIN-SPECIFIC LANGUAGE related to {', '.join(domain_terminology[:3])}
           - Configuration settings should be optimized for this specific domain
        
        ## ORIGINAL CREW CODE
        ```python
        {crew_code}
        ```
        
        ## IMPORTANT NOTES
        - The code MUST work with the agent_names and task_names provided
        - Your adaptation must be DOMAIN-SPECIFIC, not generic
        - Change the crew name and processes to be domain-specific
        - Return ONLY the Python code with NO explanations
        """
        
        # Call OpenAI to transform the code
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for more creative customization
            max_tokens=4000
        )
        
        # Extract the transformed code
        transformed_code = completion.choices[0].message.content
        
        # Clean up the code - remove any markdown formatting
        transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
        
        # Verify the agent_names and task_names are present in the transformed code
        missing_agents = [agent for agent in agent_names if agent not in transformed_code]
        missing_tasks = [task for task in task_names if task not in transformed_code]
        
        if missing_agents or missing_tasks:
            logger.warning(f"Transformed crew code is missing some agent or task references")
            logger.warning(f"Missing agents: {missing_agents}")
            logger.warning(f"Missing tasks: {missing_tasks}")
            
            # Try again with a more specific prompt
            fix_prompt = f"""
            Your adaptation is missing references to some required agents or tasks.
            
            {prompt}
            
            ## MISSING REFERENCES THAT MUST BE INCLUDED
            Missing Agent Names: {', '.join(missing_agents) if missing_agents else 'None'}
            Missing Task Names: {', '.join(missing_tasks) if missing_tasks else 'None'}
            
            ## ADDITIONAL REQUIREMENTS
            - You MUST include ALL the agent names and task names provided
            - Check imports, initializations, and usage of all agents and tasks
            - Make sure all names match EXACTLY as provided (case-sensitive)
            - Do not abbreviate or change any agent or task names
            
            ## WARNING
            Your code will not work if these references are missing.
            """
            
            # Try again with the fix prompt
            completion = await ctx.deps.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.5,  # Lower temperature for more precise corrections
                max_tokens=4000
            )
            
            transformed_code = completion.choices[0].message.content
            transformed_code = transformed_code.replace("```python", "").replace("```", "").strip()
        
        return transformed_code
    
    except Exception as e:
        logger.error(f"Error adapting crew code: {e}")
        # Return original code on error
        return crew_code


async def analyze_crew_requirements(ctx: RunContext[MCPToolDeps], user_query: str) -> Dict[str, Any]:
    """
    Analyze user query to extract specific crew requirements.
    
    Args:
        ctx: RunContext with dependencies
        user_query: User's query or request
        
    Returns:
        Dictionary with crew requirements
    """
    try:
        prompt = f"""
        Analyze this user request for a CrewAI tool and extract specific requirements for crew orchestration.
        
        USER QUERY: {user_query}
        
        Please extract and provide:
        1. 5-10 domain-specific terms that should be used in the crew implementation
        2. The appropriate process type (sequential or hierarchical) based on the request
        3. The appropriate workflow complexity (simple, medium, complex) based on the request
        
        Format your response as a JSON object with these keys:
        - domain_terms: list of domain-specific terms (array of strings)
        - process_type: either "sequential" or "hierarchical" (string)
        - workflow_complexity: one of "simple", "medium", or "complex" (string)
        """
        
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        logger.info(f"Extracted crew requirements: {json.dumps(result)}")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing crew requirements: {e}")
        return {
            "domain_terms": ["specialized", "customized", "domain-specific"],
            "process_type": "sequential",
            "workflow_complexity": "medium"
        }

@mcp_tool_agent.tool
async def generate_main_py(ctx: RunContext[MCPToolDeps],
                         tools_data: Dict[str, Any],
                         output_dir: str) -> str:
    """
    Generate a main.py file for the CrewAI project to demonstrate how to use the crew.
    
    Args:
        ctx: Run context with dependencies
        tools_data: Dictionary containing information about the tools
        output_dir: Directory to save the file
        
    Returns:
        Content of the generated main.py file
    """
    try:
        logger.info("Generating main.py file")
        
        # Extract the primary purpose for the name
        primary_purpose = ""
        if "tools" in tools_data and isinstance(tools_data["tools"], list) and len(tools_data["tools"]) > 0:
            primary_purpose = tools_data["tools"][0].get('purpose', '')
        elif "purpose" in tools_data:
            primary_purpose = tools_data.get("purpose", '')
        
        # Get a nice descriptive project name
        project_name = "CrewAI Project"
        if primary_purpose:
            words = primary_purpose.split()[:3]
            project_name = " ".join(words) + " Project"
        
        # Check if crew.py exists and read it to understand its structure
        crew_file_path = os.path.join(output_dir, "crew.py")
        crew_content = ""
        crew_import_pattern = ""
        
        if os.path.exists(crew_file_path):
            with open(crew_file_path, 'r') as f:
                crew_content = f.read()
                
            # Try to determine how to import from crew.py
            if "def get_crew" in crew_content:
                crew_import_pattern = "from crew import get_crew"
            elif "crew = Crew" in crew_content:
                crew_import_pattern = "from crew import crew"
            else:
                crew_import_pattern = "import crew"
        
        # Create the main.py prompt
        prompt = f"""
        Create a comprehensive main.py file for a CrewAI project based on the specific user requirements.
        
        PROJECT PURPOSE:
        {primary_purpose}
        
        HOW TO IMPORT FROM CREW.PY:
        {crew_import_pattern if crew_import_pattern else "Determine the best import based on standard CrewAI practices"}
        
        The main.py file should:
        1. Import the necessary modules from the project
        2. Create a simple command-line interface for running the crew
        3. Include proper error handling and logging setup
        4. Have a clean, well-documented structure
        5. Be ready to run as a standalone script
        
        FOLLOW THESE REQUIREMENTS CAREFULLY:
        - Set up proper logging to track execution
        - Create a main() function that runs the crew and handles results
        - Add command-line arguments parsing with argparse (e.g., --verbose flag, --output flag)
        - Protect the entry point with if __name__ == "__main__"
        - Add appropriate error handling for missing files or failed imports
        - Make sure the output is user-friendly and properly formatted
        - DO NOT add any functionality beyond running the crew - main.py should just be a runner
        
        Return ONLY the complete Python code for main.py without any explanations.
        """
        
        # Use model to generate main.py
        model_name = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
        
        # Use OpenAI API directly
        if ctx.deps.openai_client:
            response = await ctx.deps.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            main_py_content = response.choices[0].message.content
            
            # Remove any markdown code formatting
            main_py_content = main_py_content.replace("```python", "").replace("```", "").strip()
            
            # Save the file
            main_file_path = os.path.join(output_dir, "main.py")
            with open(main_file_path, "w") as f:
                f.write(main_py_content)
                
            logger.info(f"Generated and saved main.py to {main_file_path}")
            return main_py_content
        else:
            # Fallback to standard agent run
            result = await ctx.model.run(prompt)
            
            # Save the file
            main_file_path = os.path.join(output_dir, "main.py")
            with open(main_file_path, "w") as f:
                f.write(result.data)
                
            logger.info(f"Generated and saved main.py to {main_file_path}")
            return result.data
    
    except Exception as e:
        logger.error(f"Error generating main.py: {str(e)}", exc_info=True)
        
        # Create a minimal main.py as fallback
        fallback_content = """#!/usr/bin/env python3
# Main script for CrewAI project

import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the crew
try:
    # Try different import patterns
    try:
        from crew import get_crew
        
        def run_crew():
            crew = get_crew()
            return crew.kickoff()
    except (ImportError, AttributeError):
        try:
            from crew import crew
            
            def run_crew():
                return crew.kickoff()
        except (ImportError, AttributeError):
            import crew
            
            def run_crew():
                if hasattr(crew, 'get_crew'):
                    return crew.get_crew().kickoff()
                elif hasattr(crew, 'crew'):
                    return crew.crew.kickoff()
                else:
                    logger.error("Could not find crew object in crew.py")
                    return "Error: Could not find crew object"
except Exception as e:
    logger.error(f"Error importing crew: {str(e)}")
    
    def run_crew():
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Run CrewAI Project')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("Starting CrewAI project")
    
    try:
        result = run_crew()
        print("\\n===== RESULT =====")
        print(result)
        print("==================\\n")
        return result
    except Exception as e:
        logger.error(f"Error running crew: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()
"""
        
        # Save the fallback file
        main_file_path = os.path.join(output_dir, "main.py")
        with open(main_file_path, "w") as f:
            f.write(fallback_content)
            
        logger.info(f"Generated and saved fallback main.py to {main_file_path}")
        return fallback_content

# Common errors in tools.py file generation that need to be fixed
TOOLS_PY_COMMON_ERRORS = [
    {
        "error_type": "class_naming_inconsistency",
        "description": "Tool class names are inconsistent across files",
        "detection_pattern": r"YouTubeTranscriptMCPTool|MCPTool",
        "fix_pattern": r"class ([a-zA-Z0-9_]+)(?:MCPTool)\b",
        "fix_replacement": "class \\1MCPTool(\\1Tool)",
        "error_message": "Detected class naming inconsistency. Adding alias class for backward compatibility."
    },
    {
        "error_type": "missing_tool_suffix",
        "description": "Tool class doesn't have 'Tool' suffix",
        "detection_pattern": r"class\s+([a-zA-Z0-9_]+)\s*\(BaseTool\)",
        "fix_pattern": r"class\s+([a-zA-Z0-9_]+)\s*\(BaseTool\)",
        "fix_replacement": "class \\1Tool(BaseTool)",
        "error_message": "Detected tool class without 'Tool' suffix. Renaming for consistency."
    },
    {
        "error_type": "inconsistent_method_names",
        "description": "Tool methods called differently across files",
        "detection_pattern": r"def\s+(get_transcript|extract_transcript)",
        "fix_pattern": r"def\s+extract_transcript",
        "fix_replacement": "def get_transcript",
        "error_message": "Detected inconsistent method names. Standardizing method names."
    },
    {
        "error_type": "missing_alias_class",
        "description": "Missing alias class for backward compatibility",
        "detection_pattern": r"Tool\b(?!.*MCPTool)",
        "fix_pattern": r"class\s+([a-zA-Z0-9_]+)Tool\b(?!.*MCPTool)",
        "fix_replacement": "class \\1Tool\n\n# Alias class for backward compatibility\nclass \\1MCPTool(\\1Tool)",
        "error_message": "Adding alias class with MCPTool suffix for backward compatibility."
    }
]

def validate_tools_py_content(tools_py_content: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Validate and fix common errors in tools.py content.
    
    Args:
        tools_py_content: Content of the tools.py file
        
    Returns:
        Tuple of (fixed_content, list_of_applied_fixes)
    """
    import re
    applied_fixes = []
    fixed_content = tools_py_content
    
    logger.info("Validating tools.py content for common errors")
    
    for error in TOOLS_PY_COMMON_ERRORS:
        # Check if the error pattern exists in the content
        if re.search(error["detection_pattern"], fixed_content):
            logger.info(f"Detected error: {error['description']}")
            
            # Apply the fix
            old_content = fixed_content
            fixed_content = re.sub(error["fix_pattern"], error["fix_replacement"], fixed_content)
            
            # If content changed, we applied a fix
            if old_content != fixed_content:
                applied_fixes.append({
                    "error_type": error["error_type"],
                    "description": error["description"],
                    "message": error["error_message"]
                })
                logger.info(f"Applied fix: {error['error_message']}")
    
    # Special validation for YouTube Transcript Tool - ensure the correct class exists
    youtube_patterns = [
        (r"YouTubeTranscriptTool", r"YouTubeTranscriptMCPTool"),
        (r"YouTubeTranscriptMCPTool", r"YouTubeTranscriptTool")
    ]
    
    for base_pattern, alias_pattern in youtube_patterns:
        if re.search(base_pattern, fixed_content) and not re.search(alias_pattern, fixed_content):
            logger.info(f"Detected missing {alias_pattern} class in tools.py")
            
            # Extract the base class definition to create an alias
            base_class_match = re.search(f"class {base_pattern}\\([^)]*\\):(.*?)(?:class|$)", fixed_content, re.DOTALL)
            if base_class_match:
                # Add alias class after the base class
                base_class_end = base_class_match.end(1)
                prefix = fixed_content[:base_class_end]
                suffix = fixed_content[base_class_end:]
                
                # Create alias class definition
                alias_class = f"""

# Alias class for backward compatibility
class {alias_pattern}({base_pattern}):
    \"\"\"Alias for {base_pattern} for backward compatibility\"\"\"
    pass

"""
                fixed_content = prefix + alias_class + suffix
                
                applied_fixes.append({
                    "error_type": "missing_alias_class",
                    "description": f"Missing {alias_pattern} alias for {base_pattern}",
                    "message": f"Added alias class {alias_pattern} for backward compatibility"
                })
                logger.info(f"Added alias class {alias_pattern} for backward compatibility")
    
    # Return the fixed content and list of applied fixes
    return fixed_content, applied_fixes