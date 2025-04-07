import asyncio
import logging
from pydantic_ai.mcp import MCPServerStdio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    # Create MCP server instance for YouTube Transcript
    server = MCPServerStdio(
        'npx',
        [
            '-y',
            '@smithery/cli@latest',
            'run',
            '@kimtaeyoon83/mcp-server-youtube-transcript',
            '--key',
            '040afa77-557a-4a7b-9169-3f1f2b9d685f'
        ]
    )
    
    try:
        # Try to start the server
        logging.info("Starting YouTube Transcript MCP server...")
        
        # Method 1: Use ensure_running if available
        if hasattr(server, 'ensure_running') and callable(server.ensure_running):
            try:
                await server.ensure_running()
                logging.info("Server started using ensure_running()")
            except Exception as e:
                logging.warning(f"Failed to start with ensure_running(): {str(e)}")
        
        # Method 2: Use start if available
        elif hasattr(server, 'start') and callable(server.start):
            try:
                await server.start()
                logging.info("Server started using start()")
            except Exception as e:
                logging.warning(f"Failed to start with start(): {str(e)}")
        
        # Try to get tools
        try:
            if hasattr(server, 'get_tools') and callable(server.get_tools):
                tools = await server.get_tools()
                logging.info(f"Retrieved {len(tools)} tools using get_tools()")
                for tool in tools:
                    logging.info(f"Tool: {tool.get('name')} - {tool.get('description')}")
            elif hasattr(server, 'list_tools') and callable(server.list_tools):
                tools = await server.list_tools()
                logging.info(f"Retrieved {len(tools)} tools using list_tools()")
                for tool in tools:
                    logging.info(f"Tool: {tool.get('name')} - {tool.get('description')}")
        except Exception as e:
            logging.error(f"Error getting tools: {str(e)}")
        
        # Wait for user input to keep the server running
        print("\nYouTube Transcript MCP Server is running.")
        print("Press Enter to exit...")
        await asyncio.sleep(10)  # Wait for 10 seconds
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        # Try to stop the server
        if hasattr(server, 'stop') and callable(server.stop):
            try:
                await server.stop()
                logging.info("Server stopped")
            except Exception as e:
                logging.warning(f"Error stopping server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
