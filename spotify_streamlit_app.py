import streamlit as st
import asyncio
import logging
import traceback
import subprocess
from datetime import datetime
import nest_asyncio

# Import Spotify agent components through our wrapper module
from spotify_wrapper import load_config, setup_agent, run_spotify_query

# Configure page
st.set_page_config(
    page_title="Spotify Agent",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and apply custom CSS
def load_css():
    with open("spotify_streamlit_style.css", "r") as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load custom CSS
try:
    load_css()
except Exception as e:
    st.warning(f"Could not load custom CSS: {e}")

# Apply nest_asyncio to allow running asyncio in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "config" not in st.session_state:
    st.session_state.config = None
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "tool_usage" not in st.session_state:
    st.session_state.tool_usage = None
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to add log entries
def add_log(level, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} [{level}] {message}"
    st.session_state.logs.append(log_entry)
    if level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)

# Function to check Node.js and npm
def check_node_npm():
    try:
        node_version = subprocess.check_output(['node', '--version']).decode().strip()
        npm_version = subprocess.check_output(['npm', '--version']).decode().strip()
        add_log("INFO", f"Node.js version: {node_version}, npm version: {npm_version}")
        return True
    except Exception as e:
        add_log("WARNING", f"Could not detect Node.js/npm: {str(e)}. Make sure these are installed.")
        return False

# Async function to initialize the agent
async def initialize_agent():
    try:
        add_log("INFO", "Starting Spotify Agent initialization")

        # Check Node.js and npm
        node_installed = check_node_npm()
        if not node_installed:
            st.error("Node.js and npm are required. Please install them and try again.")
            return False

        # Load configuration
        add_log("INFO", "Loading configuration...")
        st.session_state.config = load_config()

        # Setup agent
        add_log("INFO", "Setting up agent...")
        st.session_state.agent = await setup_agent(st.session_state.config)

        # Start MCP servers
        add_log("INFO", "Starting MCP servers...")
        st.session_state.mcp_context = st.session_state.agent.run_mcp_servers()
        await st.session_state.mcp_context.__aenter__()

        add_log("INFO", "Spotify Agent initialized successfully")
        st.session_state.is_initialized = True
        return True
    except Exception as e:
        add_log("ERROR", f"Error initializing Spotify Agent: {str(e)}")
        add_log("ERROR", traceback.format_exc())
        st.error(f"Error initializing Spotify Agent: {str(e)}")
        return False

# Async function to process a query
async def process_query(query):
    if not st.session_state.is_initialized:
        st.error("Spotify Agent is not initialized. Please initialize it first.")
        return

    try:
        add_log("INFO", f"Processing query: '{query}'")

        # Run the query through the agent
        result, elapsed_time, tool_usage = await run_spotify_query(st.session_state.agent, query)

        # Store the results in session state
        st.session_state.current_result = result
        st.session_state.tool_usage = tool_usage
        st.session_state.elapsed_time = elapsed_time

        # Add to query history
        st.session_state.query_history.append({
            "query": query,
            "result": result,
            "tool_usage": tool_usage,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        add_log("INFO", f"Query completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        add_log("ERROR", f"Error processing query: {str(e)}")
        add_log("ERROR", traceback.format_exc())
        st.error(f"Error processing query: {str(e)}")
        return False

# Function to run async functions in Streamlit
def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(func(*args, **kwargs))

# Sidebar
with st.sidebar:
    st.title("🎵 Spotify Agent")
    st.markdown("---")

    # Initialization section
    st.subheader("Initialization")
    if not st.session_state.is_initialized:
        if st.button("Initialize Spotify Agent", key="init_button"):
            with st.spinner("Initializing Spotify Agent..."):
                success = run_async(initialize_agent)
                if success:
                    st.success("Spotify Agent initialized successfully!")
                else:
                    st.error("Failed to initialize Spotify Agent. Check logs for details.")
    else:
        st.success("Spotify Agent is initialized and ready!")
        if st.button("Reinitialize", key="reinit_button"):
            st.session_state.is_initialized = False
            st.experimental_rerun()

    st.markdown("---")

    # Navigation
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Query History", "Logs", "About"])

    st.markdown("---")

    # System info
    st.subheader("System Info")
    node_installed = check_node_npm()
    if node_installed:
        st.info(f"Node.js and npm detected")
    else:
        st.warning("Node.js/npm not detected")

# Main content
if page == "Home":
    st.title("🎵 Spotify Agent")

    # Introduction
    st.markdown("""
    This is a Streamlit interface for the Spotify Agent. You can use it to query Spotify data and perform various operations.

    To get started:
    1. Initialize the Spotify Agent using the button in the sidebar
    2. Enter your query in the text area below
    3. Click "Run Query" to process your query
    4. View the results below
    """)

    # Query input
    st.subheader("Enter Your Query")
    query = st.text_area("What would you like to know about Spotify?",
                         height=100,
                         placeholder="e.g., 'Find the top 5 songs by Taylor Swift' or 'Create a playlist of relaxing jazz music'")

    # Process query button
    if st.button("Run Query", key="run_query_button", disabled=not st.session_state.is_initialized):
        if not query:
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Processing your query..."):
                success = run_async(process_query, query)
                if success:
                    st.success("Query processed successfully!")

    # Display current result
    if st.session_state.current_result:
        st.subheader("Results")

        # Display the result in a nice format
        result_data = st.session_state.current_result.data
        st.markdown(f"### Response\n{result_data}")

        # Display elapsed time
        st.info(f"Query completed in {st.session_state.elapsed_time:.2f} seconds")

        # Display tool usage in an expander
        if st.session_state.tool_usage:
            with st.expander("Tools Used"):
                for i, tool in enumerate(st.session_state.tool_usage, 1):
                    tool_name = tool.get('name', 'Unknown Tool')
                    tool_params = tool.get('parameters', {})

                    st.markdown(f"**{i}. Tool: {tool_name}**")
                    if tool_params:
                        st.markdown("Parameters:")
                        for param, value in tool_params.items():
                            # Truncate long values
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:97] + "..."
                            st.markdown(f"- {param}: {value}")
                    st.markdown("---")

elif page == "Query History":
    st.title("Query History")

    if not st.session_state.query_history:
        st.info("No queries have been run yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"{item['timestamp']} - {item['query']}"):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Result:** {item['result'].data}")
                st.markdown(f"**Elapsed Time:** {item['elapsed_time']:.2f} seconds")

                # Display tool usage
                if item['tool_usage']:
                    st.markdown("**Tools Used:**")
                    for j, tool in enumerate(item['tool_usage'], 1):
                        tool_name = tool.get('name', 'Unknown Tool')
                        st.markdown(f"{j}. {tool_name}")

elif page == "Logs":
    st.title("Logs")

    # Add log filter options
    log_filter = st.selectbox("Filter logs by level", ["All", "INFO", "WARNING", "ERROR"])

    # Clear logs button
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.experimental_rerun()

    # Display logs
    if not st.session_state.logs:
        st.info("No logs available.")
    else:
        log_text = ""
        for log in st.session_state.logs:
            if log_filter == "All" or log_filter in log:
                log_text += log + "\n"

        st.code(log_text, language="")

elif page == "About":
    st.title("About Spotify Agent")

    st.markdown("""
    ## Spotify Agent

    This is a Streamlit interface for the Spotify Agent, which uses the Model Context Protocol (MCP) to interact with the Spotify API.

    ### Features

    - Query Spotify data using natural language
    - Create and manage playlists
    - Search for songs, artists, and albums
    - Get recommendations based on your preferences
    - View detailed information about tracks, artists, and albums

    ### How It Works

    The Spotify Agent uses a large language model (LLM) to understand your queries and translate them into API calls to Spotify. It then processes the results and presents them in a human-readable format.

    ### Requirements

    - Node.js and npm
    - Spotify API credentials
    - Internet connection

    ### Credits

    This application was built using:
    - Streamlit
    - Pydantic AI
    - Model Context Protocol (MCP)
    - Spotify Web API
    """)

# Footer
st.markdown("---")
st.markdown("Spotify Agent Streamlit Interface | Built with ❤️ using Streamlit and Pydantic AI")
