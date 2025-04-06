from pydantic_ai import Agent, RunContext
from models import SerperSpotifyAgentConfig
import logging
import asyncio
import json
import traceback
from contextlib import asynccontextmanager

class IntegratedAgent(Agent):
    """Agent that integrates Serper search and Spotify functionality"""
    
    def __init__(self, config: SerperSpotifyAgentConfig):
        super().__init__()
        self.config = config
        self.integrated_servers = []
        
    @asynccontextmanager
    async def run_mcp_servers(self):
        """Context manager to handle MCP server lifecycle"""
        try:
            # Start all servers
            for server in self.integrated_servers:
                if hasattr(server, 'session') and server.session:
                    if not server.session.is_connected():
                        await server.session.connect()
                elif hasattr(server, 'start'):
                    await server.start()
                elif hasattr(server, 'connect'):
                    await server.connect()
            
            yield
            
        finally:
            # Clean up servers
            for server in self.integrated_servers:
                try:
                    if hasattr(server, 'session') and server.session:
                        if server.session.is_connected():
                            await server.session.disconnect()
                    elif hasattr(server, 'stop'):
                        await server.stop()
                    elif hasattr(server, 'disconnect'):
                        await server.disconnect()
                except Exception as e:
                    logging.warning(f"Error cleaning up server: {str(e)}")
    
    async def execute_server_tool(self, server, tool_name: str, params: dict = None) -> dict:
        """Execute a tool on the specified server with proper error handling
        
        Args:
            server: The MCP server to use
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            dict: The tool execution result
        """
        try:
            # Try session invoke_tool first
            if hasattr(server, 'session') and server.session:
                return await server.session.invoke_tool(tool_name, params or {})
                
            # Try direct invoke_tool if available
            if hasattr(server, 'invoke_tool'):
                return await server.invoke_tool(tool_name, params or {})
                
            raise Exception(f"Server does not support tool execution")
            
        except Exception as e:
            logging.error(f"Error executing tool '{tool_name}': {str(e)}")
            logging.debug("Error details:", exc_info=True)
            raise
    
    async def run(self, query: str, context: RunContext = None) -> dict:
        """Run a query through the integrated agent
        
        Args:
            query: The user's query string
            context: Optional run context
            
        Returns:
            dict: The query result
        """
        try:
            # Process query based on content
            is_spotify_query = any(keyword in query.lower() for keyword in 
                                ['spotify', 'music', 'song', 'playlist', 'artist', 'album', 'play', 'track'])
            
            if is_spotify_query and len(self.integrated_servers) > 1:
                server = self.integrated_servers[1]  # Spotify server
                tool_name = 'searchTracks'  # Default Spotify search tool
                params = {'query': query, 'market': 'US'}
            else:
                server = self.integrated_servers[0]  # Serper server
                tool_name = 'search'  # Default Serper search tool
                params = {'query': query}
                
            # Execute query through appropriate server
            result = await self.execute_server_tool(server, tool_name, params)
            return result
            
        except Exception as e:
            logging.error(f"Error running query: {str(e)}")
            logging.error(f"Error details: {traceback.format_exc()}")
            raise

async def setup_agent(config: SerperSpotifyAgentConfig) -> IntegratedAgent:
    """Set up the integrated agent with configuration
    
    Args:
        config: The agent configuration
        
    Returns:
        IntegratedAgent: The configured agent instance
    """
    try:
        agent = IntegratedAgent(config)
        logging.info("Created integrated agent instance")
        
        # Initialize servers
        from tools import create_serper_server, create_spotify_server
        
        # Create and add Serper server
        serper_server = create_serper_server(config.SERPER_API_KEY)
        agent.integrated_servers.append(serper_server)
        logging.info("Added Serper server to agent")
        
        # Create and add Spotify server
        spotify_server = create_spotify_server(config.SPOTIFY_API_KEY)
        agent.integrated_servers.append(spotify_server)
        logging.info("Added Spotify server to agent")
        
        return agent
        
    except Exception as e:
        logging.error(f"Error setting up agent: {str(e)}")
        raise 