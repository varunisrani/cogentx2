import asyncio
import logging
import sys  # Needed for sys.exit
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import traceback
import streamlit as st
from models import load_config
from agent import setup_agent
from tools import run_weather_query

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Weather MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='weather_agent.log', help='Log file path')
    parser.add_argument('--max-log-size', type=int, default=5 * 1024 * 1024, help='Maximum log file size in bytes')
    parser.add_argument('--log-backups', type=int, default=3, help='Number of log backups to keep')
    return parser.parse_args()

# Configure logging with colors and better formatting
def setup_logging(args):
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler for complete logs
    file_handler = RotatingFileHandler(
        args.log_file,
        maxBytes=args.max_log_size,
        backupCount=args.log_backups
    )
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Suppress verbose logging from libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    return root_logger

# Function to display tool usage in a user-friendly way
def display_tool_usage(tool_usage):
    if not tool_usage:
        st.info("📋 No specific tools were recorded for this query")
        return

    st.subheader("🔧 WEATHER TOOLS USED:")

    for i, tool in enumerate(tool_usage, 1):
        tool_name = tool.get('name', 'Unknown Tool')
        tool_params = tool.get('parameters', {})

        with st.expander(f"Tool {i}: {tool_name}"):
            if tool_params:
                st.write("Parameters:")
                for param, value in tool_params.items():
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:97] + "..."
                    st.text(f"- {param}: {value}")

def init_streamlit():
    st.set_page_config(
        page_title="Weather MCP Agent",
        page_icon="🌤️",
        layout="wide"
    )
    st.title("🌤️ Weather MCP Agent")

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "logger" not in st.session_state:
        args = parse_args()
        st.session_state.logger = setup_logging(args)

async def initialize_agent():
    if st.session_state.agent is None:
        try:
            # Load configuration
            config = load_config()

            # Setup agent
            st.session_state.agent = await setup_agent(config)
            return True
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.error("Please check the logs for more details.")
            return False
    return True

async def main():
    try:
        init_streamlit()

        # Initialize agent if not already done
        if not await initialize_agent():
            return

        # Sidebar with information
        with st.sidebar:
            st.header("ℹ️ Information")
            st.markdown("""
            ### Features:
            - Get current weather conditions
            - View temperature, humidity, wind speed
            - Get daily forecasts
            - Search weather by city name

            ### Examples:
            1. "What's the weather in London?"
            2. "Show me the forecast for Tokyo"
            3. "How's the weather in Paris?"
            """)

        # Main query input
        user_query = st.text_area("🌍 Enter your weather query:", height=100)

        if st.button("Get Weather", type="primary"):
            if not user_query:
                st.warning("Please enter a query first!")
                return

            try:
                with st.spinner("Getting weather information..."):
                    # Log the query
                    st.session_state.logger.info(f"Processing query: '{user_query}'")

                    # Run the query through the agent
                    async with st.session_state.agent.run_mcp_servers():
                        result, elapsed_time, tool_usage = await run_weather_query(st.session_state.agent, user_query)

                        # Log completion
                        st.session_state.logger.info(f"Query completed in {elapsed_time:.2f} seconds")

                        # Display results
                        st.success(f"Query completed in {elapsed_time:.2f} seconds")

                        # Display tool usage
                        display_tool_usage(tool_usage)

                        # Display results
                        st.header("Weather Results")
                        st.markdown("---")
                        st.write(result.data)

            except Exception as e:
                st.session_state.logger.error(f"Error processing query: {str(e)}")
                st.session_state.logger.error(f"Error details: {traceback.format_exc()}")
                st.error(f"Error: {str(e)}")
                st.error("Please try a different query or check the logs for details.")

                st.markdown("""
                ### Tips:
                1. Make sure your OpenWeather API key is valid
                2. Check if the city name is spelled correctly
                3. Try being more specific in your query
                """)
    finally:
        # Ensure we clean up properly when the function exits
        if 'agent' in st.session_state and st.session_state.agent:
            try:
                # Try to clean up MCP servers if they exist
                if hasattr(st.session_state.agent, 'mcp_servers'):
                    for server in st.session_state.agent.mcp_servers:
                        if hasattr(server, 'close') and callable(server.close):
                            await server.close()
                logging.info("MCP servers closed successfully")
            except Exception as cleanup_err:
                logging.error(f"Error closing MCP servers: {str(cleanup_err)}")

async def cleanup():
    """Properly clean up async resources"""
    try:
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            if task is not asyncio.current_task(loop=loop):
                task.cancel()

        # Wait for tasks to complete cancellation
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        logging.info("All tasks cleaned up successfully")
    except Exception as cleanup_err:
        logging.error(f"Error during cleanup: {str(cleanup_err)}")

def main_wrapper():
    """Wrapper function to handle asyncio.run and cleanup"""
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("User interrupted the process")
        sys.exit(0)  # Exit with success code for keyboard interrupt
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)  # Exit with error code
    finally:
        logging.info("Weather agent shutting down")
        logging.info("Exiting gracefully...")

if __name__ == '__main__':
    main_wrapper()
