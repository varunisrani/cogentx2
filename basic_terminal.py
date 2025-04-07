import streamlit as st
from streamlit_terminal import st_terminal

st.set_page_config(page_title="Basic Terminal", layout="wide")

st.title("Basic Terminal Implementation")
st.write("A simple terminal interface using the streamlit-terminal component.")

# Terminal configuration
st.subheader("Terminal Settings")
show_welcome = st.checkbox("Show Welcome Message", value=True)
term_height = st.slider("Terminal Height", min_value=200, max_value=800, value=400, step=50)
custom_cmd = st.text_input("Custom Command (optional)", 
                          placeholder="Leave empty for interactive terminal or enter a command to run")

# Create the terminal
st.subheader("Terminal")
full_outputs, updated_outputs = st_terminal(
    key="terminal",
    command=custom_cmd if custom_cmd else "",
    height=term_height,
    show_welcome_message=show_welcome
)

# Show outputs
with st.expander("Terminal Outputs"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Full Outputs")
        st.write(full_outputs)
    with col2:
        st.subheader("Updated Outputs")
        st.write(updated_outputs)

# Add sidebar with useful information
st.sidebar.title("Terminal Information")
st.sidebar.markdown("""
## What is streamlit-terminal?

streamlit-terminal is a Streamlit component that provides an interactive terminal interface within your Streamlit app.

## Basic Features
- Run shell commands in a web interface
- Real-time command output streaming
- Support for ANSI color codes
- Custom command execution

## How to Use
1. Type your command in the box above or use the terminal directly
2. Press Enter to execute the command
3. View the real-time output in the terminal
4. Expand "Terminal Outputs" to see structured command output data

## Example Commands
```bash
# List files
ls -la

# Print current directory
pwd

# Run a Python script
python -c "print('Hello from Python!')"

# Display system information
uname -a
```
""")

# Make the terminal look better with custom CSS
st.markdown("""
<style>
    /* VS Code-like styling */
    .stApp {
        background-color: #1e1e1e;
    }
    
    h1, h2, h3, h4, h5, h6, .stMarkdown, p {
        color: #d4d4d4 !important;
    }
    
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background-color: #3c3c3c !important;
        color: #cccccc !important;
        border: 1px solid #555 !important;
    }
    
    .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: #9cdcfe !important;
    }
    
    .stSlider div[data-baseweb="slider"] {
        background-color: #3c3c3c !important;
    }
    
    .stSlider div[data-baseweb="thumb"] {
        background-color: #569cd6 !important;
    }
    
    .stButton button {
        background-color: #0e639c !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #1177bb !important;
    }
    
    .stExpander {
        border: 1px solid #3c3c3c !important;
        background-color: #252526 !important;
    }
    
    .stExpander summary {
        color: #9cdcfe !important;
    }
</style>
""", unsafe_allow_html=True) 