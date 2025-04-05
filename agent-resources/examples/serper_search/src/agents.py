from pydantic_ai import Agent
import logging
import json
from .models import SearchResponse, SerperConfig
from .tools import create_serper_mcp_server

logger = logging.getLogger(__name__)

class SerperAgent:
    def __init__(self, config: SerperConfig):
        self.config = config
        logger.debug("Initializing Serper Agent...")
        
        try:
            # First create the MCP server to discover available tools
            logger.info("Setting up Serper MCP server...")
            self.mcp_server = create_serper_mcp_server(config.serper_api_key)
            
            # Get available MCP tools and log them in detail
            logger.info("\n" + "="*50)
            logger.info("DISCOVERING AVAILABLE MCP TOOLS")
            logger.info("="*50)
            
            # Get tools using listTools() method
            tools_result = self.mcp_server.listTools()
            logger.info("\nMCP Server Response:")
            logger.info("-"*50)
            logger.info(json.dumps(tools_result, indent=2))
            
            # Store available tools
            self.available_tools = tools_result.get('tools', [])
            
            # Log each available tool
            logger.info("\nAvailable Tools Summary:")
            logger.info("-"*50)
            for tool in self.available_tools:
                logger.info(f"\nTool: {tool.get('name')}")
                logger.info(f"Description: {tool.get('description')}")
                if tool.get('parameters'):
                    logger.info("Parameters:")
                    for param in tool['parameters']:
                        logger.info(f"  - {param.get('name')}: {param.get('type')} "
                                 f"(Required: {param.get('required', False)})")
                        if param.get('description'):
                            logger.info(f"    {param.get('description')}")
            
            logger.info("\n" + "="*50)
            logger.info(f"Total Tools Available: {len(self.available_tools)}")
            logger.info("="*50)
            
            self.agent = Agent(
                f'openai:{config.model_name}',
                system_prompt=(
                    f"""
                    You are a search assistant that provides accurate and relevant 
                    information from web searches. You have access to the following tools:
                    {json.dumps(self.available_tools, indent=2)}
                    
                    For each query, select the most appropriate tool based on its capabilities.
                    Summarize the results clearly and concisely.
                    """
                ),
                result_type=SearchResponse
            )
            logger.debug("Agent initialized with model: %s", config.model_name)
            logger.info("Serper agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SerperAgent: {str(e)}")
            raise
        
    async def search(self, query: str) -> SearchResponse:
        """Perform a search using Serper and process results"""
        try:
            logger.info(f"Executing search query: {query}")
            
            # Log which tools are available for this search
            tool_names = [t.get('name') for t in self.available_tools]
            logger.info(f"Available tools for search: {', '.join(tool_names)}")
            
            # Set up MCP servers for the agent
            self.agent.mcp_servers = [self.mcp_server]
            
            # Log the full request being sent
            request = (
                f"Search for: {query}\n"
                f"Use the most appropriate search tool from: {', '.join(tool_names)}.\n"
                f"Provide results in a structured format with title, snippet, and link for each result.\n"
                f"Include a brief summary of the key findings."
            )
            logger.debug(f"Sending request to agent:\n{request}")
            
            result = await self.agent.run(request)
            logger.info("Search completed successfully")
            return result.data
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise 