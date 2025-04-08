import asyncio
import logging
import sys
import argparse
import colorlog
from logging.handlers import RotatingFileHandler
import os
import traceback

# Add the current directory to sys.path to make imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import the modules
from models import load_config
from agent import setup_agent
from tools import run_time_query

# Import streamlit conditionally to handle CLI mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Time MCP Agent')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='time_agent.log', help='Log file path')
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

def init_streamlit():
    """Initialize Streamlit UI and session state"""
    # Configure the page
    st.set_page_config(
        page_title="Time MCP Agent",
        page_icon="🕒",
        layout="wide"
    )
    st.title("🕒 Time MCP Agent")

    # Initialize session state variables
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "init_requested" not in st.session_state:
        st.session_state.init_requested = False
    if "logger" not in st.session_state:
        # Use default args for logging in Streamlit mode
        args = argparse.Namespace(
            verbose=False,
            log_file="time_agent_streamlit.log",
            max_log_size=5 * 1024 * 1024,  # 5MB
            log_backups=3
        )
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

async def process_query(agent, user_query: str):
    """Process a user query and handle any errors"""
    try:
        result, elapsed_time, tool_usage = await run_time_query(agent, user_query)
        return result, elapsed_time, tool_usage
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        raise

def streamlit_main():
    """Streamlit interface for the Time MCP Agent"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not available. Please install it with 'pip install streamlit'")
        return

    init_streamlit()

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        ### Time MCP Agent Features:
        - Get current time in any timezone
        - Convert between timezones
        - Calculate time differences
        - Parse and format dates/times

        ### Examples:
        1. "What time is it in Tokyo?"
        2. "Convert 3:30 PM EST to PST"
        3. "How many hours between 2pm and 8pm?"
        4. "Format 2025-04-08 14:30:00 as RFC3339"
        """)

    # Main query input
    user_query = st.text_area("⌨️ Enter your time query:", height=100)

    # Initialize agent button
    if st.session_state.agent is None:
        if st.button("Initialize Agent", type="primary"):
            st.session_state.init_requested = True

    # Handle initialization
    if st.session_state.get("init_requested", False) and st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            try:
                # Create a new event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Initialize the agent
                config = load_config()
                st.session_state.agent = loop.run_until_complete(setup_agent(config))
                st.success("Agent initialized successfully!")
                st.session_state.init_requested = False
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
                st.error("Please check the logs for more details.")
                st.session_state.init_requested = False

    # Process query button
    process_button = st.button("Process Query", type="primary", disabled=st.session_state.agent is None)

    if process_button:
        if not user_query:
            st.warning("Please enter a query first!")
        elif st.session_state.agent is None:
            st.warning("Please initialize the agent first!")
        else:
            try:
                with st.spinner("Processing your query..."):
                    # Log the query
                    st.session_state.logger.info(f"Processing query: '{user_query}'")

                    # Create a new event loop for async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Define an async function to run the query
                    async def run_query():
                        async with st.session_state.agent.run_mcp_servers():
                            return await process_query(st.session_state.agent, user_query)

                    # Run the query
                    result, elapsed_time, tool_usage = loop.run_until_complete(run_query())

                    # Log completion
                    st.session_state.logger.info(f"Query completed in {elapsed_time:.2f} seconds")

                    # Display results
                    st.success(f"Query completed in {elapsed_time:.2f} seconds")

                    # Display tool usage
                    if tool_usage:
                        with st.expander("🔧 Tools Used"):
                            for i, tool in enumerate(tool_usage, 1):
                                tool_name = tool.get('tool', 'Unknown Tool')
                                st.text(f"{i}. {tool_name}")
                                if 'params' in tool:
                                    st.json(tool['params'])

                    # Display results
                    st.header("Results")
                    st.markdown("---")
                    st.write(result.data)

            except Exception as e:
                st.session_state.logger.error(f"Error processing query: {str(e)}")
                st.session_state.logger.error(f"Error details: {traceback.format_exc()}")
                st.error(f"Error: {str(e)}")
                st.error("Please try a different query or check the logs for details.")

async def cli_main():
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args)

    try:
        # Load configuration
        logger.info("Starting Time MCP Agent")
        config = load_config()

        # Setup agent
        logger.info("Setting up agent...")
        agent = await setup_agent(config)
        logger.info("Time MCP Server started successfully")

        print("\n" + "="*50)
        print("Time MCP Agent Ready")
        print("Type 'exit' or press Ctrl+C to quit")
        print("="*50 + "\n")

        # Interactive loop
        while True:
            try:
                # Get user input
                user_query = input("🕒 > ")

                if user_query.lower() in ('exit', 'quit'):
                    break

                if not user_query.strip():
                    continue

                # Process the query
                logger.info(f"Processing query: '{user_query}'")

                async with agent.run_mcp_servers():
                    result, elapsed_time, tool_usage = await process_query(agent, user_query)

                    # Log completion and display results
                    logger.info(f"Query completed in {elapsed_time:.2f} seconds")
                    print(f"\nResult: {result.data}")
                    print(f"Time: {elapsed_time:.2f}s")

                    if tool_usage:
                        print("\nTools used:")
                        for i, tool in enumerate(tool_usage, 1):
                            tool_name = tool.get('name', 'Unknown Tool')
                            print(f"{i}. {tool_name}")
                    print()

            except KeyboardInterrupt:
                logger.info("User interrupted the process")
                break
            except EOFError:
                logger.info("EOF detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'exit' to quit")

    except KeyboardInterrupt:
        logger.info("User interrupted the process")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        logger.info("Time agent shutting down")
        logger.info("Exiting gracefully...")

def main():
    # Check if running under Streamlit
    try:
        # Check if running in Streamlit environment
        is_streamlit = STREAMLIT_AVAILABLE and hasattr(st, 'session_state')

        if is_streamlit:
            # When running with streamlit, we don't use asyncio.run directly
            # as Streamlit has its own event loop
            streamlit_main()
        else:
            try:
                asyncio.run(cli_main())
            except KeyboardInterrupt:
                sys.exit(0)
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
