import streamlit as st
import asyncio
from archon_graph import agentic_flow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Archon AI Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton > button {
        color: white;
        border: 2px solid #00CC99;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        border: 2px solid #FF69B4;
    }
    
    .stChatMessage {
        border-left: 4px solid #00CC99;
    }
    
    .stChatInput > div {
        border: 2px solid #00CC99 !important;
    }
    
    .stChatInput > div:focus-within {
        box-shadow: none !important;
        border: 2px solid #FF69B4 !important;
        outline: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("Archon AI Chat")
st.markdown("""
This interface allows you to interact with the Archon AI agent system.
Enter your request below and the system will process it through multiple specialized agents.
""")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent_state = {
        'messages': [],
        'scope': '',
        'agent_context': {}
    }

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like the AI agent to help you with?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update agent state
    st.session_state.agent_state['latest_user_message'] = prompt
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Stream handler for agent output
        def stream_handler(content: str):
            if 'current_response' not in st.session_state:
                st.session_state.current_response = ""
            st.session_state.current_response += content
            message_placeholder.markdown(st.session_state.current_response)
        
        try:
            # Run the agent workflow
            result = asyncio.run(
                agentic_flow.ainvoke(
                    st.session_state.agent_state,
                    config={"configurable": {"stream_handler": stream_handler}}
                )
            )
            
            # Update session state with result
            st.session_state.agent_state = result
            
            # Add assistant's response to chat history
            if 'current_response' in st.session_state:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": st.session_state.current_response
                })
                del st.session_state.current_response
                
        except Exception as e:
            logger.error(f"Error running agent workflow: {e}", exc_info=True)
            message_placeholder.error(f"An error occurred: {str(e)}")

# Minimal sidebar with essential information and reset button
with st.sidebar:
    st.image("public/ArchonLightGrey.png", width=300)
    
    # Context information (if available)
    if st.session_state.agent_state.get('agent_context'):
        with st.expander("Current Context", expanded=False):
            context = st.session_state.agent_state['agent_context']
            if 'project_spec' in context:
                st.markdown("**Project Specification:**")
                st.code(context['project_spec'])
            if 'architecture' in context:
                st.markdown("**Architecture:**")
                st.code(context['architecture'])
            if 'implementation_plan' in context:
                st.markdown("**Implementation Plan:**")
                st.code(context['implementation_plan'])
    
    # Reset conversation button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_state = {
            'messages': [],
            'scope': '',
            'agent_context': {}
        }
        st.rerun() 