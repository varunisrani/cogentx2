import streamlit as st
import uuid
import logging
from crew_stream import (
    agentic_flow, mcp_tools_available, process_mcp_request
)

logger = logging.getLogger(__name__)

async def crew_tab():
    """Display the Crew interface with both standard and MCP modes."""
    st.markdown("""
    <h1>Cogentx Crew</h1>
    <p style="font-size: 1.2rem; margin-bottom: 2rem;">Create intelligent agents that help with various tasks using natural language instructions.</p>
    """, unsafe_allow_html=True)

    # Create a container at the top for mode selection
    with st.container(border=True):
        st.markdown("### Crew Mode")
        st.markdown("Select how you want to use the crew:")

        # Initialize mode selection in session state if not present
        if "crew_mode" not in st.session_state:
            st.session_state.crew_mode = "standard"

        # Create two columns for mode selection cards
        mode_col1, mode_col2 = st.columns(2)

        with mode_col1:
            standard_card_style = "background-color: " + ("#E3F2FD" if st.session_state.crew_mode == "standard" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.crew_mode == "standard" else "#e0e0e0")

            st.markdown(f"""
            <div style="{standard_card_style}" onclick="document.getElementById('standard_mode_btn').click()">
                <div style="font-size: 32px; margin-bottom: 10px;">🧠</div>
                <h3 style="margin: 0; margin-bottom: 5px;">Standard Mode</h3>
                <p style="margin: 0; color: #666;">Create agents using templates and standard approach</p>
            </div>
            """, unsafe_allow_html=True)

            # Hidden button for the click handler
            standard_mode = st.button("🧠 Standard Mode", key="standard_mode_btn", use_container_width=True)
            if standard_mode:
                st.session_state.crew_mode = "standard"
                st.rerun()

        with mode_col2:
            mcp_card_style = "background-color: " + ("#E3F2FD" if st.session_state.crew_mode == "mcp" else "white") + "; padding: 20px; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; border: 2px solid " + ("#1E88E5" if st.session_state.crew_mode == "mcp" else "#e0e0e0")

            st.markdown(f"""
            <div style="{mcp_card_style}" onclick="document.getElementById('mcp_mode_btn').click()">
                <div style="font-size: 32px; margin-bottom: 10px;">🔌</div>
                <h3 style="margin: 0; margin-bottom: 5px;">MCP Mode</h3>
                <p style="margin: 0; color: #666;">Create agents with advanced MCP tool integrations</p>
            </div>
            """, unsafe_allow_html=True)

            # Hidden button for the click handler
            mcp_mode = st.button("🔌 MCP Mode", key="mcp_mode_btn", use_container_width=True)
            if mcp_mode:
                st.session_state.crew_mode = "mcp"
                st.rerun()

    # Display current mode info
    st.markdown(f"""
    <div style="background-color: #E3F2FD; border-left: 4px solid #1E88E5; padding: 12px; border-radius: 4px; margin: 16px 0;">
        <p style="margin: 0;"><strong>Currently in {st.session_state.crew_mode.title()} Mode</strong>: {{
            "Using traditional template-based approach for agent creation." if st.session_state.crew_mode == "standard"
            else "Using MCP tools for advanced API integrations and agent creation."
        }}</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history in session state if not present
    if "crew_messages" not in st.session_state:
        st.session_state.crew_messages = []

    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.crew_messages = []
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.crew_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user - adjust placeholder based on the selected mode
    placeholder_text = "What would you like the crew to do?" if st.session_state.crew_mode == "standard" else "Describe what you want to build with MCP tools..."
    user_input = st.chat_input(placeholder_text)

    if user_input:
        # Add user message to chat history
        st.session_state.crew_messages.append({"role": "user", "content": user_input})

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message

            # Add a spinner with appropriate message based on mode
            spinner_message = "Crew is thinking..." if st.session_state.crew_mode == "standard" else "Searching for relevant MCP tools..."
            with st.spinner(spinner_message):
                try:
                    # Check which mode to use
                    if st.session_state.crew_mode == "mcp":
                        # Process with MCP mode
                        if not mcp_tools_available:
                            error_msg = "MCP tools functionality is not available. Please check if the required modules are installed."
                            message_placeholder.error(error_msg)
                            response_content = error_msg
                        else:
                            # Use MCP processing
                            response_content = await process_mcp_request(user_input, "single")
                            message_placeholder.markdown(response_content)
                    else:
                        # Standard mode processing
                        config = {
                            "configurable": {
                                "thread_id": str(uuid.uuid4())
                            }
                        }

                        # Run the agent with streaming
                        async for chunk in agentic_flow.astream(
                                {"latest_user_message": user_input}, config, stream_mode="custom"
                            ):
                            response_content += chunk
                            # Update the placeholder with the current response content
                            message_placeholder.markdown(response_content)
                except Exception as e:
                    error_msg = f"Error running crew: {str(e)}"
                    logger.error(error_msg)
                    message_placeholder.error(error_msg)
                    response_content = error_msg

        # Add the response to the conversation history
        st.session_state.crew_messages.append({"role": "assistant", "content": response_content})