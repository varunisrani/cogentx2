import streamlit as st
import os
import sys

# Try to import streamlit-terminal
try:
    from streamlit_terminal import st_terminal
    TERMINAL_AVAILABLE = True
except ImportError:
    TERMINAL_AVAILABLE = False
    st.error("streamlit-terminal is not installed. Install it with: pip install streamlit-terminal")
    st.stop()

# Page configuration
st.set_page_config(page_title="Terminal Comparison", layout="wide")

st.title("Terminal Implementation Comparison")
st.write("This page compares the streamlit-terminal component with the custom Archon terminal implementation.")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Streamlit-Terminal Component")
    st.write("""
    A lightweight terminal component that can be easily integrated into any Streamlit app.
    Features:
    - Simple integration
    - Component-based approach
    - Easy to use as a dependency
    """)
    
    # Configure the terminal
    show_welcome = st.checkbox("Show Welcome Message", value=True, key="st_welcome")
    term_height = st.slider("Terminal Height", min_value=200, max_value=600, value=400, step=50, key="st_height")
    custom_cmd = st.text_input("Custom Command (optional)", key="st_cmd",
                            placeholder="Leave empty for interactive terminal or enter a command to run")
    
    # Create the terminal
    st.subheader("Terminal")
    full_outputs, updated_outputs = st_terminal(
        key="streamlit_terminal",
        command=custom_cmd if custom_cmd else "",
        height=term_height,
        show_welcome_message=show_welcome
    )
    
    # Display outputs
    with st.expander("Terminal Outputs"):
        st.subheader("Full Outputs")
        st.write(full_outputs)
        st.subheader("Updated Outputs")
        st.write(updated_outputs)

with col2:
    st.header("Use Archon Terminal")
    st.write("""
    The full-featured Archon terminal implementation with advanced features.
    Features:
    - Command history
    - Directory tracking
    - Process management
    - Colorful output
    - Detailed logging
    """)
    
    st.markdown(f"""
    ### Open Archon Terminal
    
    Click the button below to open the full Archon terminal implementation in a new tab.
    
    <a href="terminal_app.py" target="_blank">
        <button style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        ">
            Open Archon Terminal
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    # Sample of Archon terminal features
    st.subheader("Archon Terminal Features Preview")
    
    # Show example of command history
    st.write("**Command History**")
    st.code("""
    $ ls -la
    $ python --version
    $ echo "Hello, World!"
    """)
    
    # Show example of directory tracking
    st.write("**Directory Tracking**")
    st.code("""
    $ pwd
    /home/user/projects
    $ cd documents
    Changed directory to /home/user/projects/documents
    $ pwd
    /home/user/projects/documents
    """)
    
    # Show example of process management
    st.write("**Process Management**")
    st.code("""
    $ python -u test/clock.py
    Current time: 10:15:30
    Current time: 10:15:31
    Current time: 10:15:32
    ...
    Execution time: 10.345 seconds
    """)

# Add comparison table
st.header("Feature Comparison")
feature_data = {
    "Feature": [
        "Interactive Terminal", 
        "Command Execution",
        "Real-time Output",
        "Command History",
        "Directory Tracking (cd command)",
        "Process Management",
        "Multi-line Commands",
        "Support for stdin",
        "Color Output",
        "Code Editor Integration",
        "Downloadable Logs",
        "Installation Complexity"
    ],
    "Streamlit-Terminal": [
        "✅", 
        "✅", 
        "✅", 
        "❌", 
        "❌", 
        "Limited", 
        "❌", 
        "Limited", 
        "✅", 
        "❌", 
        "❌",
        "Simple (pip install)"
    ],
    "Archon Terminal": [
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅", 
        "✅",
        "Moderate (part of Archon)"
    ]
}

st.table(feature_data)

# Add recommendations section
st.header("When to Use Which Implementation")

st.subheader("Use Streamlit-Terminal When:")
st.markdown("""
- You need a simple terminal interface in your Streamlit app
- You want a clean, component-based approach
- You prefer a lightweight dependency
- You don't need advanced features like directory tracking
""")

st.subheader("Use Archon Terminal When:")
st.markdown("""
- You need a full-featured terminal experience
- You require command history and directory tracking
- You want detailed process information and management
- You need support for multi-line commands and script execution
""")

# Add setup instructions
st.header("Setup Instructions")

st.subheader("Streamlit-Terminal Setup")
st.code("""
# Install the package
pip install streamlit-terminal

# Import and use in your Streamlit app
import streamlit as st
from streamlit_terminal import st_terminal

# Basic usage
st_terminal(key="terminal")

# With custom command
st_terminal(key="terminal", command="echo 'Hello, World!'")
""")

st.subheader("Archon Terminal Setup")
st.write("The Archon terminal is part of the Archon project and can be used by running the terminal_app.py script.")
st.code("""
# Clone the repository
git clone https://github.com/your-username/Archon.git

# Install requirements
pip install -r requirements.txt

# Run the terminal app
streamlit run terminal_app.py
""")

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ for terminal enthusiasts") 