from langgraph.types import Command
import streamlit as st
import uuid
import sys
import os
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_ui.log"),
        logging.StreamHandler()
    ]
)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.archon_graph import agentic_flow
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps
from utils.utils import get_clients

# Initialize clients
embedding_client, supabase = get_clients()

def get_thread_id():
    """Generate a thread ID based on the current timestamp for uniqueness."""
    if "thread_id" not in st.session_state:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.thread_id = f"{current_time}-{str(uuid.uuid4())[:8]}"
        st.info(f"Started new chat session with thread ID: {current_time}")
    return st.session_state.thread_id

async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text for the user_input prompt."""
    # Get the thread ID from the session state
    thread_id = get_thread_id()

    # Create the configuration with the thread ID
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Log the thread ID for debugging
    logger.info(f"Using thread ID: {thread_id} for conversation")

    # First message from user
    if len(st.session_state.messages) == 1:
        # For first message, use agentic_flow from archon_graph.py
        async for msg in agentic_flow.astream(
                {"latest_user_message": user_input}, config, stream_mode="custom"
            ):
                yield msg
    else:
        # For subsequent messages, continue the conversation
        async for msg in agentic_flow.astream(
            Command(resume=user_input), config, stream_mode="custom"
        ):
            yield msg

async def chat_tab():
    """Display the chat interface for talking to Archon."""
    st.write("Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.")
    st.write("Example: Build me an AI agent that can search the web with the Brave API.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if "thread_id" in st.session_state:
            del st.session_state.thread_id
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            
            # Add a spinner while loading
            with st.spinner("Archon is thinking..."):
                try:
                    # Run the agent with streaming
                    async for chunk in run_agent_with_streaming(user_input):
                        response_content += chunk
                        # Update the placeholder with the current response content
                        message_placeholder.markdown(response_content)
                except Exception as e:
                    error_msg = f"Error running agent: {str(e)}"
                    logger.error(error_msg)
                    message_placeholder.error(error_msg)
                    response_content = error_msg
        
        # Add the response to the conversation history
        st.session_state.messages.append({"type": "ai", "content": response_content})