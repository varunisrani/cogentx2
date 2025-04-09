from typing import Any, Dict, List, Optional
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


# Additional imports from original tools
from dotenv import load_dotenv
import datetime
import logging
import time
# CrewAI-compatible tools collection



# Helper functions
    def __init__(self):
        super().__init__()
        load_dotenv()
        
        self.logger, self.log_file_path = self._setup_logging("INFO")
        self.logger.info("Initializing Spotify MCP client")
        
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.smithery_api_key = os.getenv("SMITHERY_API_KEY")
        
        if not self.client_id or not self.client_secret:
            self.logger.error("Spotify API credentials not found")
            raise ValueError("Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        
        if not self.smithery_api_key:
            self.logger.warning("Smithery API key not provided. Some operations may fail.")
        
        self.auth_params = {
            "spotifyClientId": self.client_id,
            "spotifyClientSecret": self.client_secret
        }
        
        self.url = create_smithery_url(self.base_url, self.auth_params)
        if self.smithery_api_key:
            self.url += f"&api_key={self.smithery_api_key}"
        
        self.logger.info("Spotify MCP client URL configured")
        
        try:
            asyncio.run(self.preload_tools())
        except Exception as e:
            self.logger.warning(f"Could not preload tools: {str(e)}")
        
        self.logger.info("Spotify MCP client initialized successfully")
    
    def _setup_logging(self, log_level: str):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/spotify_operations_{timestamp}.log"
        
        level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        level = level_map.get(log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        return logging.getLogger("spotify_operations"), log_file
    
    async def preload_tools(self):
        try:
            async with websocket_client(self.url) as streams:
                async with mcp.ClientSession(*streams) as session:
                    self.available_tools = await self.get_available_tools_from_server(session)
                    self.logger.info(f"Preloaded {len(self.available_tools)} Spotify tools")
        except Exception as e:
            self.logger.error(f"Error preloading tools: {str(e)}")
            raise
    
    async def get_available_tools_from_server(self, session) -> Dict[str, Dict]:
        tools_result = await session.list_tools()
        
        tools_dict = {}
        if hasattr(tools_result, 'tools'):
            for tool in tools_result.tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    schema = {}
                    if hasattr(tool, 'inputSchema'):
                        schema = tool.inputSchema
                    
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
    
    async def _arun(self, operation: str, parameters: Dict[str, Any]) -> str:
        parameters["_timestamp"] = str(time.time())
        
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                if self.available_tools is None:
                    self.available_tools = await self.get_available_tools_from_server(session)
                
                if operation not in self.available_tools:
                    return f"Invalid operation: {operation}. Available operations include: {', '.join(list(self.available_tools.keys())[:5])}..."
                
                tool_details = self.available_tools[operation]
                required_params = tool_details.get("required", [])
                
                missing_params = [param for param in required_params if param not in parameters]
                if missing_params:
                    return f"Missing required parameters for {operation}: {', '.join(missing_params)}"
                
                self.logger.info(f"Calling Spotify API: {operation}")
                call_start_time = time.time()
                result = await session.call_tool(operation, parameters)
                call_duration = time.time() - call_start_time
                self.logger.info(f"API call completed in {call_duration:.2f} seconds")
                
                if hasattr(result, 'content') and result.content:
                    try:
                        response_data = json.loads(result.content[0].text)
                        return json.dumps(response_data, indent=2)
                    except json.JSONDecodeError:
                        return result.content[0].text
                else:
                    return f"Operation {operation} executed successfully but returned no content."
    
    def _run(self, input_data: str) -> str:
        self.logger.info(f"Processing input: {input_data}")
        
        try:
            operation_data = json.loads(input_data)
            if isinstance(operation_data, dict) and "operation" in operation_data:
                operation = operation_data["operation"]
                parameters = operation_data.get("parameters", {})
                return self._execute_operation(operation, parameters)
        except json.JSONDecodeError:
            pass
        
        operation = "search"
        parameters = {"query": input_data, "type": "track", "limit": 10}
        
        if "search for" in input_data.lower() or "find" in input_data.lower():
            search_type = "track"
            if "artist" in input_data.lower():
                search_type = "artist"
            elif "album" in input_data.lower():
                search_type = "album"
            elif "playlist" in input_data.lower():
                search_type = "playlist"
                
            parameters["type"] = search_type
            
        elif "recommendation" in input_data.lower() or "similar to" in input_data.lower():
            operation = "recommendations"
            parameters = {"seed_genres": "pop", "limit": 10}
            
        elif "get artist" in input_data.lower():
            operation = "get_artist"
            parameters = {"id": ""}
        
        self.logger.info(f"Interpreted as operation: {operation}")
        self.logger.info(f"With parameters: {json.dumps(parameters, indent=2)}")
        
        return self._execute_operation(operation, parameters)
    
    def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> str:
        self.logger.info(f"OPERATION: {operation}")
        safe_params = parameters.copy()
        self.logger.info(f"PARAMETERS: {json.dumps(safe_params, indent=2)}")
        
        try:
            result = asyncio.run(self._arun(operation, parameters))
            return result
        except Exception as e:
            self.logger.error(f"ERROR: {str(e)}")
            return f"Error executing Spotify operation: {str(e)}"
    
    async def list_available_operations(self) -> List[Dict[str, str]]:
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                tools_dict = await self.get_available_tools_from_server(session)
                
                operations = []
                for name, details in tools_dict.items():
                    operations.append({
                        "name": name,
                        "description": details["description"],
                        "required_params": details.get("required", [])
                    })
                
                return operations
    
    def get_available_operations(self) -> str:
        try:
            operations = asyncio.run(self.list_available_operations())
            
            result = "# Available Spotify Operations\n\n"
            for op in operations:
                result += f"## {op['name']}\n"
                result += f"Description: {op['description']}\n"
                if op['required_params']:
                    result += f"Required parameters: {', '.join(op['required_params'])}\n"
                result += "\n"
            
            return result
        except Exception as e:
            return f"Error retrieving available operations: {str(e)}"
    
    def _process_parameters_from_agent(self, params_dict: Dict) -> str:
        self.logger.info(f"Processing agent parameters: {params_dict}")
        
        if "description" in params_dict and "type" in params_dict:
            description = params_dict["description"]
            search_type = params_dict["type"]
            if search_type in ["track", "album", "artist", "playlist"]:
                return f"Find {search_type}s by {description}"
            else:
                return description
        elif "query" in params_dict:
            return params_dict["query"]
        else:
            return " ".join(str(v) for v in params_dict.values())
    
    def __call__(self, input_data: Union[str, Dict]) -> str:
        self.logger.info(f"Tool called with input_data type: {type(input_data)}")
        
        if isinstance(input_data, str):
            self.logger.info(f"Processing string input: {input_data}")
            return self._run(input_data)
        
        elif isinstance(input_data, dict):
            self.logger.info(f"Processing dictionary input: {str(input_data)[:100]}")
            
            if "operation" in input_data:
                operation = input_data["operation"]
                parameters = input_data.get("parameters", {})
                self.logger.info(f"Detected operation format: {operation}")
                return self._execute_operation(operation, parameters)
            
            elif "input_data" in input_data:
                inner_input = input_data["input_data"]
                self.logger.info(f"Detected input_data format: {str(inner_input)[:100]}")
                
                if isinstance(inner_input, str):
                    return self._run(inner_input)
                elif isinstance(inner_input, dict):
                    return self.__call__(inner_input)
                else:
                    return f"Invalid input_data type: {type(inner_input)}. Expected string or dict."
            
            elif "query" in input_data:
                query = input_data["query"]
                search_type = input_data.get("type", "track")
                limit = input_data.get("limit", 10)
                self.logger.info(f"Detected query format: query={query}, type={search_type}")
                
                return self._execute_operation("search", {
                    "query": query,
                    "type": search_type,
                    "limit": limit
                })
            
            elif "description" in input_data:
                description = input_data["description"]
                item_type = input_data.get("type", "track")
                self.logger.info(f"Detected description format: {description}, type={item_type}")
                
                query = f"Find {item_type}s by {description}" if item_type else f"Find {description}"
                return self._run(query)
            
            else:
                self.logger.warning(f"Unknown dictionary format: {input_data}")
                return "Please provide a search query or Spotify operation details."
        
        else:
            self.logger.error(f"Invalid input type: {type(input_data)}")
            return f"Invalid input type: {type(input_data)}. Please provide a string or dictionary."
class SpotifyMCPTool

# Alias class for backward compatibility
class SpotifyMCPMCPTool(SpotifyMCPTool)(SpotifyTool)

# Alias class for backward compatibility
class SpotifyMCPMCPTool(SpotifyMCPTool)(SpotifyMCPTool)(SpotifyTool)(BaseTool):
    """
    Tool for interacting with the Spotify API using MCP through Smithery.
    
    This tool allows users to perform various operations such as searching for music,
    retrieving recommendations, and managing playlists.
    
    Example usage:
    
    # Create client
    client = SpotifyMCPTool()
    
    # Get available operations
    print(client.get_available_operations())
    
    # Example search
    result = client({
        "operation": "search",
        "parameters": {
            "query": "The Beatles",
            "type": "artist",
            "limit": 5
        }
    })
    print(result)
    
    """
    
    name: str = "spotify_mcp_tool"
    description: str = "Interact with Spotify API to search music, get recommendations, and more"
    args_schema: type[BaseModel] = SpotifyToolInput
    
    client_id: Optional[str] = Field(default=None, description="Spotify API client ID")
    client_secret: Optional[str] = Field(default=None, description="Spotify API client secret")
    smithery_api_key: Optional[str] = Field(default=None, description="Smithery API key")
    
    logger: Optional[logging.Logger] = Field(default=None, description="Logger for the Spotify MCP client")
    log_file_path: Optional[str] = Field(default=None, description="Path to the log file")
    base_url: str = Field(default="wss://server.smithery.ai/@superseoworld/mcp-spotify/ws", description="Base URL for the Spotify MCP server")
    url: Optional[str] = Field(default=None, description="Full URL for the Spotify MCP server with authentication")
    auth_params: Dict[str, str] = Field(default_factory=dict, description="Authentication parameters for the Spotify MCP server")
    available_tools: Optional[Dict[str, Dict]] = Field(default=None, description="Cache of available Spotify tools")
    
    def __init__(self):
        super().__init__()
        load_dotenv()
        
        self.logger, self.log_file_path = self._setup_logging("INFO")
        self.logger.info("Initializing Spotify MCP client")
        
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.smithery_api_key = os.getenv("SMITHERY_API_KEY")
        
        if not self.client_id or not self.client_secret:
            self.logger.error("Spotify API credentials not found")
            raise ValueError("Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        
        if not self.smithery_api_key:
            self.logger.warning("Smithery API key not provided. Some operations may fail.")
        
        self.auth_params = {
            "spotifyClientId": self.client_id,
            "spotifyClientSecret": self.client_secret
        }
        
        self.url = create_smithery_url(self.base_url, self.auth_params)
        if self.smithery_api_key:
            self.url += f"&api_key={self.smithery_api_key}"
        
        self.logger.info("Spotify MCP client URL configured")
        
        try:
            asyncio.run(self.preload_tools())
        except Exception as e:
            self.logger.warning(f"Could not preload tools: {str(e)}")
        
        self.logger.info("Spotify MCP client initialized successfully")
    
    def _setup_logging(self, log_level: str):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/spotify_operations_{timestamp}.log"
        
        level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        level = level_map.get(log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        return logging.getLogger("spotify_operations"), log_file
    
    async def preload_tools(self):
        try:
            async with websocket_client(self.url) as streams:
                async with mcp.ClientSession(*streams) as session:
                    self.available_tools = await self.get_available_tools_from_server(session)
                    self.logger.info(f"Preloaded {len(self.available_tools)} Spotify tools")
        except Exception as e:
            self.logger.error(f"Error preloading tools: {str(e)}")
            raise
    
    async def get_available_tools_from_server(self, session) -> Dict[str, Dict]:
        tools_result = await session.list_tools()
        
        tools_dict = {}
        if hasattr(tools_result, 'tools'):
            for tool in tools_result.tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    schema = {}
                    if hasattr(tool, 'inputSchema'):
                        schema = tool.inputSchema
                    
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
    
    async def _arun(self, operation: str, parameters: Dict[str, Any]) -> str:
        parameters["_timestamp"] = str(time.time())
        
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                if self.available_tools is None:
                    self.available_tools = await self.get_available_tools_from_server(session)
                
                if operation not in self.available_tools:
                    return f"Invalid operation: {operation}. Available operations include: {', '.join(list(self.available_tools.keys())[:5])}..."
                
                tool_details = self.available_tools[operation]
                required_params = tool_details.get("required", [])
                
                missing_params = [param for param in required_params if param not in parameters]
                if missing_params:
                    return f"Missing required parameters for {operation}: {', '.join(missing_params)}"
                
                self.logger.info(f"Calling Spotify API: {operation}")
                call_start_time = time.time()
                result = await session.call_tool(operation, parameters)
                call_duration = time.time() - call_start_time
                self.logger.info(f"API call completed in {call_duration:.2f} seconds")
                
                if hasattr(result, 'content') and result.content:
                    try:
                        response_data = json.loads(result.content[0].text)
                        return json.dumps(response_data, indent=2)
                    except json.JSONDecodeError:
                        return result.content[0].text
                else:
                    return f"Operation {operation} executed successfully but returned no content."
    
    def _run(self, input_data: str) -> str:
        self.logger.info(f"Processing input: {input_data}")
        
        try:
            operation_data = json.loads(input_data)
            if isinstance(operation_data, dict) and "operation" in operation_data:
                operation = operation_data["operation"]
                parameters = operation_data.get("parameters", {})
                return self._execute_operation(operation, parameters)
        except json.JSONDecodeError:
            pass
        
        operation = "search"
        parameters = {"query": input_data, "type": "track", "limit": 10}
        
        if "search for" in input_data.lower() or "find" in input_data.lower():
            search_type = "track"
            if "artist" in input_data.lower():
                search_type = "artist"
            elif "album" in input_data.lower():
                search_type = "album"
            elif "playlist" in input_data.lower():
                search_type = "playlist"
                
            parameters["type"] = search_type
            
        elif "recommendation" in input_data.lower() or "similar to" in input_data.lower():
            operation = "recommendations"
            parameters = {"seed_genres": "pop", "limit": 10}
            
        elif "get artist" in input_data.lower():
            operation = "get_artist"
            parameters = {"id": ""}
        
        self.logger.info(f"Interpreted as operation: {operation}")
        self.logger.info(f"With parameters: {json.dumps(parameters, indent=2)}")
        
        return self._execute_operation(operation, parameters)
    
    def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> str:
        self.logger.info(f"OPERATION: {operation}")
        safe_params = parameters.copy()
        self.logger.info(f"PARAMETERS: {json.dumps(safe_params, indent=2)}")
        
        try:
            result = asyncio.run(self._arun(operation, parameters))
            return result
        except Exception as e:
            self.logger.error(f"ERROR: {str(e)}")
            return f"Error executing Spotify operation: {str(e)}"
    
    async def list_available_operations(self) -> List[Dict[str, str]]:
        async with websocket_client(self.url) as streams:
            async with mcp.ClientSession(*streams) as session:
                tools_dict = await self.get_available_tools_from_server(session)
                
                operations = []
                for name, details in tools_dict.items():
                    operations.append({
                        "name": name,
                        "description": details["description"],
                        "required_params": details.get("required", [])
                    })
                
                return operations
    
    def get_available_operations(self) -> str:
        try:
            operations = asyncio.run(self.list_available_operations())
            
            result = "# Available Spotify Operations\n\n"
            for op in operations:
                result += f"## {op['name']}\n"
                result += f"Description: {op['description']}\n"
                if op['required_params']:
                    result += f"Required parameters: {', '.join(op['required_params'])}\n"
                result += "\n"
            
            return result
        except Exception as e:
            return f"Error retrieving available operations: {str(e)}"
    
    def _process_parameters_from_agent(self, params_dict: Dict) -> str:
        self.logger.info(f"Processing agent parameters: {params_dict}")
        
        if "description" in params_dict and "type" in params_dict:
            description = params_dict["description"]
            search_type = params_dict["type"]
            if search_type in ["track", "album", "artist", "playlist"]:
                return f"Find {search_type}s by {description}"
            else:
                return description
        elif "query" in params_dict:
            return params_dict["query"]
        else:
            return " ".join(str(v) for v in params_dict.values())
    
    def __call__(self, input_data: Union[str, Dict]) -> str:
        self.logger.info(f"Tool called with input_data type: {type(input_data)}")
        
        if isinstance(input_data, str):
            self.logger.info(f"Processing string input: {input_data}")
            return self._run(input_data)
        
        elif isinstance(input_data, dict):
            self.logger.info(f"Processing dictionary input: {str(input_data)[:100]}")
            
            if "operation" in input_data:
                operation = input_data["operation"]
                parameters = input_data.get("parameters", {})
                self.logger.info(f"Detected operation format: {operation}")
                return self._execute_operation(operation, parameters)
            
            elif "input_data" in input_data:
                inner_input = input_data["input_data"]
                self.logger.info(f"Detected input_data format: {str(inner_input)[:100]}")
                
                if isinstance(inner_input, str):
                    return self._run(inner_input)
                elif isinstance(inner_input, dict):
                    return self.__call__(inner_input)
                else:
                    return f"Invalid input_data type: {type(inner_input)}. Expected string or dict."
            
            elif "query" in input_data:
                query = input_data["query"]
                search_type = input_data.get("type", "track")
                limit = input_data.get("limit", 10)
                self.logger.info(f"Detected query format: query={query}, type={search_type}")
                
                return self._execute_operation("search", {
                    "query": query,
                    "type": search_type,
                    "limit": limit
                })
            
            elif "description" in input_data:
                description = input_data["description"]
                item_type = input_data.get("type", "track")
                self.logger.info(f"Detected description format: {description}, type={item_type}")
                
                query = f"Find {item_type}s by {description}" if item_type else f"Find {description}"
                return self._run(query)
            
            else:
                self.logger.warning(f"Unknown dictionary format: {input_data}")
                return "Please provide a search query or Spotify operation details."
        
        else:
            self.logger.error(f"Invalid input type: {type(input_data)}")
            return f"Invalid input type: {type(input_data)}. Please provide a string or dictionary."

# Export all tools for use in CrewAI
__all__ = [
    "SpotifyMCPTool",
]
