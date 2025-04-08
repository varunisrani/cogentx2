import streamlit as st
import asyncio
import os
import sys
import logging
from openai import AsyncOpenAI
from supabase import Client
import logfire

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import MCP tool specific functions
from archon.mcp_tools.mcp_tool_graph import mcp_tool_flow, combined_adaptive_flow
from archon.mcp_tools.mcp_tool_coder import MCPToolDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_streamlit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mcp_streamlit')

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

# Initialize OpenAI client
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
is_ollama = "localhost" in base_url.lower()

if is_ollama:
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
elif os.getenv("OPENAI_API_KEY"):
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai_client = None

# Initialize Supabase client
if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
    supabase = Client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
else:
    supabase = None

# Set page configuration
st.set_page_config(
    page_title="MCP2 Tool Builder",
    page_icon="🛠️",
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
    
    .tool-mode-selector {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("MCP2 Tool Builder")
st.markdown("""
This interface allows you to create CrewAI agents with integrated MCP tools for external API services.
Describe what you want to build, and the system will find and integrate the appropriate MCP tools.
""")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.generated_files = []
    st.session_state.tool_info = {}

if 'tool_mode' not in st.session_state:
    st.session_state.tool_mode = "single"

# Tool mode selection
tool_mode_col1, tool_mode_col2 = st.columns(2)
with tool_mode_col1:
    single_mode = st.button("🔨 Single Tool Mode", 
                           use_container_width=True,
                           help="Create a single MCP tool")
    if single_mode:
        st.session_state.tool_mode = "single"

with tool_mode_col2:
    multi_mode = st.button("🔧 Multiple Tools Mode", 
                          use_container_width=True,
                          help="Create and integrate multiple MCP tools that work together")
    if multi_mode:
        st.session_state.tool_mode = "multiple"

# Show current mode
st.info(f"Currently in {'Single Tool' if st.session_state.tool_mode == 'single' else 'Multiple Tools'} Mode")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process user request through MCP flow
async def process_mcp_request(user_request, tool_mode="single"):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # For multiple tools mode, enhance the request
    if tool_mode == "multiple":
        # Create a more specific directory
        timestamp = asyncio.get_event_loop().time()
        output_dir = os.path.join(output_dir, f"multi_tools_{int(timestamp)}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhance the user request to ensure multiple tools are detected
        enhanced_request = f"""Create multiple integrated MCP tools that work together: {user_request}
        
This should generate MULTIPLE tools that can be used together in a CrewAI setup, ensuring they're properly integrated.
"""
        user_request = enhanced_request
    
    try:
        # Process the request through MCP tool flow
        result = await combined_adaptive_flow(
            user_request=user_request,
            openai_client=openai_client,
            supabase_client=supabase,
            base_dir=output_dir,
            use_adaptive_synthesis=True
        )
        
        # Store the generated files and tool info in session state
        st.session_state.generated_files = result.get("files", [])
        st.session_state.tool_info = result.get("tool_info", {})
        
        # Create a formatted response
        if result.get("success", False):
            if tool_mode == "single":
                if "tool_info" in result and result["tool_info"].get("purpose", ""):
                    response = f"""✅ **Successfully integrated MCP tool: {result['tool_info']['purpose']}**

Generated the following files:
"""
                else:
                    response = f"""✅ **Successfully generated code**

Generated the following files:
"""
            else:
                # Multiple tools mode
                tool_count = result.get("tool_info", {}).get("tool_count", 0)
                tool_names = result.get("tool_info", {}).get("tool_names", [])
                if tool_names:
                    tool_names_str = ", ".join(tool_names[:3])
                    if len(tool_names) > 3:
                        tool_names_str += f" and {len(tool_names) - 3} more"
                    response = f"""✅ **Successfully integrated {tool_count} MCP tools: {tool_names_str}**

Generated the following files:
"""
                else:
                    response = f"""✅ **Successfully generated {tool_count} MCP tools**

Generated the following files:
"""
            
            for file_path in result.get("files", []):
                response += f"- `{os.path.basename(file_path)}`\n"
            
            output_directory = result.get("tool_info", {}).get("directory", output_dir)
            response += f"\nAll files are saved in: `{output_directory}`"
        else:
            response = f"""⚠️ **Error in processing request**

{result.get('message', 'An unknown error occurred.')}
"""
            
        return response
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}", exc_info=True)
        return f"❌ **Error processing your request**\n\n{str(e)}"

# Chat input
if prompt := st.chat_input(f"What would you like to build with {'a single' if st.session_state.tool_mode == 'single' else 'multiple'} MCP tool{'s' if st.session_state.tool_mode == 'multiple' else ''}?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching for relevant MCP tools...")
        
        try:
            # Process the request based on the current mode
            response = asyncio.run(process_mcp_request(prompt, st.session_state.tool_mode))
            
            # Display the response
            message_placeholder.markdown(response)
            
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            message_placeholder.error(f"An error occurred: {str(e)}")

# Sidebar with additional information and controls
with st.sidebar:
    st.title("MCP2 Tools")
    
    # Information about MCP tools
    st.markdown("""
    **Model Context Protocol (MCP) Tools** enable AI agents to integrate with external services like:
    
    - GitHub
    - Spotify
    - YouTube
    - Twitter
    - Google Sheets
    - And many more!
    
    Simply describe what you want to build, and the system will find and integrate the appropriate tools.
    """)
    
    # Tool mode information
    st.markdown("### Tool Modes")
    st.markdown("""
    - **Single Tool Mode**: Create one tool for a specific service
    - **Multiple Tools Mode**: Create several integrated tools that work together
    """)
    
    # Generated files section
    if st.session_state.generated_files:
        st.markdown("### Generated Files")
        for file_path in st.session_state.generated_files:
            file_name = os.path.basename(file_path)
            st.code(f"{file_name}")
    
    # Tool info section
    if st.session_state.tool_info:
        st.markdown("### MCP Tool Info")
        st.markdown(f"**Purpose:** {st.session_state.tool_info.get('purpose', '')}")
        
        # Display multiple tool names if available
        if 'tool_names' in st.session_state.tool_info:
            tool_names = st.session_state.tool_info.get('tool_names', [])
            st.markdown(f"**Tools:** {', '.join(tool_names)}")
            
        if 'tool_count' in st.session_state.tool_info:
            st.markdown(f"**Tool Count:** {st.session_state.tool_info.get('tool_count', 1)}")
            
        if 'directory' in st.session_state.tool_info:
            st.markdown(f"**Output Directory:** {st.session_state.tool_info.get('directory', '')}")
    
    # Reset conversation button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generated_files = []
        st.session_state.tool_info = {}
        st.rerun()

# Function to initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'tool_info' not in st.session_state:
        st.session_state.tool_info = {}
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
    if 'code_files' not in st.session_state:
        st.session_state.code_files = {}

# Define default output directory
default_output_dir = os.path.join(os.getcwd(), "output")

# Define the main function before it's called
def main():
    # Set up the app state
    init_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="MCP Tool Creator",
        page_icon="🔧",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("# MCP Tool Creator")
        st.markdown("---")
        st.markdown("### Settings")
        
        # Add output directory setting
        global output_dir
        output_dir = st.text_input("Output Directory", value=default_output_dir)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Add a reset button
        if st.button("Reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.experimental_rerun()
        
        # Add app info to sidebar
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool creates MCP-compatible CrewAI tools 
        based on natural language descriptions.
        
        It can generate single tools or multiple integrated tools
        that work together in a CrewAI workflow.
        """)
        
        st.markdown("### Examples")
        st.markdown("""
        - "Create a tool to search for GitHub repositories"
        - "Generate a Spotify tool that can create playlists"
        - "Make tools for Twitter and Gmail that can work together"
        """)
        
        # Show API status
        st.markdown("---")
        st.markdown("### API Status")
        if openai_client:
            st.success("OpenAI: Connected")
        else:
            st.error("OpenAI: Not connected")
            
        if supabase:
            st.success("Supabase: Connected")
        else:
            st.error("Supabase: Not connected")
    
    # Main app layout
    st.title("MCP Tool Creator")
    st.markdown("Create MCP-compatible tools for CrewAI with natural language descriptions")
    
    # Use tabs for main UI sections
    tab1, tab2 = st.tabs(["Create Tool", "View Code"])
    
    with tab1:
        # Show the input form using the new function
        show_input_form()
        
        # Show tool info if available
        if st.session_state.tool_info:
            st.markdown("---")
            show_tool_info()
    
    with tab2:
        # Show code viewer if enabled
        if st.session_state.get("show_code", False) and st.session_state.get("code_files"):
            st.subheader("Generated Code")
            
            # Create a selectbox for file selection
            file_names = list(st.session_state.code_files.keys())
            selected_file = st.selectbox("Select a file to view:", file_names)
            
            # Show the selected file content
            if selected_file:
                st.code(st.session_state.code_files[selected_file], language="python")
                
                # Add download button
                file_content = st.session_state.code_files[selected_file]
                st.download_button(
                    label=f"Download {selected_file}",
                    data=file_content,
                    file_name=selected_file,
                    mime="text/plain"
                )
        else:
            st.info("Create a tool first, then click 'View Code' to see the generated files.")

# Main script execution
if __name__ == "__main__":
    # Just call our new main() function 
    main()

# UI for the main input form
def show_input_form():
    with st.form(key="mcp_tool_form"):
        st.markdown("### 🔧 MCP Tool Generator")
        
        # Add description / request
        user_request = st.text_area(
            "Describe what you need:", 
            height=100,
            help="Describe what your tool should do in natural language. Include any specific services like 'Spotify', 'GitHub', etc."
        )
        
        # Enable multi-tool mode option
        multi_tool_mode = st.checkbox(
            "Enable multi-tool mode", 
            value=False, 
            help="Generate multiple tools that work together in a CrewAI framework"
        )
        
        # Submit button and processing
        submitted = st.form_submit_button("Create MCP Tool")
        
        if submitted:
            if not user_request.strip():
                st.error("Please enter a description of the tool you need.")
                return
            
            # If multi-tool mode is enabled, enhance the request if needed
            if multi_tool_mode and "multiple" not in user_request.lower() and " and " not in user_request.lower():
                # Add multi-tool indicator to request if not already present
                user_request = f"Create multiple integrated MCP tools that work together: {user_request}"
                st.info("Enhanced request to create multiple integrated tools.")
            
            # Show processing message
            with st.spinner("Processing your request..."):
                try:
                    result = process_tool_request(user_request)
                    st.session_state.tool_info = result.get("tool_info", {})
                    st.session_state.last_request = user_request
                    
                    # Display the result
                    if result.get("success", False):
                        st.success("Tool created successfully!")
                        
                        # Check tool info for better display
                        is_multi_tool = st.session_state.tool_info.get("is_multi_tool", False)
                        tool_count = st.session_state.tool_info.get("tool_count", 1)
                        tool_names = st.session_state.tool_info.get("tool_names", [])
                        
                        if is_multi_tool and tool_count > 1:
                            if tool_names:
                                tools_str = ", ".join(tool_names)
                                response = f"""✅ **Successfully integrated {tool_count} MCP tools: {tools_str}**
                                
                                The tools have been integrated into a complete CrewAI setup with:
                                - tools.py (tool implementations)
                                - agents.py (agent definitions)
                                - tasks.py (task definitions)
                                - crew.py (workflow orchestration)
                                
                                You can use these files directly in your project.
                                """
                            else:
                                response = f"""✅ **Successfully created {tool_count} integrated MCP tools**
                                
                                The tools have been integrated into a complete CrewAI setup.
                                """
                        else:
                            # Single tool mode
                            if "tool_info" in result and result["tool_info"].get("purpose", ""):
                                response = f"""✅ **Successfully integrated MCP tool: {result['tool_info']['purpose']}**
                                
                                The tool has been integrated with CrewAI and is ready to use.
                                """
                            else:
                                response = "✅ **Successfully created MCP tool**"
                            
                        st.markdown(response)
                        
                        # Display created files
                        st.subheader("Generated Files")
                        
                        # Format and display file list
                        files = result.get("files", [])
                        if files:
                            if is_multi_tool and tool_count > 1:
                                # Group files by type for multi-tool mode
                                file_groups = {
                                    "Tool Implementations": [],
                                    "CrewAI Setup": [],
                                    "Examples & Others": []
                                }
                                
                                for file in files:
                                    basename = os.path.basename(file)
                                    if basename == "tools.py":
                                        file_groups["Tool Implementations"].append((basename, file, "Contains the MCP tool implementations"))
                                    elif basename in ["agents.py", "tasks.py", "crew.py"]:
                                        file_groups["CrewAI Setup"].append((basename, file, 
                                            "Agent definitions" if basename == "agents.py" else
                                            "Task definitions" if basename == "tasks.py" else
                                            "Workflow orchestration"))
                                    else:
                                        file_groups["Examples & Others"].append((basename, file, "Additional file"))
                                
                                # Display each group
                                for group_name, group_files in file_groups.items():
                                    if group_files:
                                        st.markdown(f"**{group_name}:**")
                                        for basename, file, description in group_files:
                                            st.markdown(f"- **{basename}**: {description}")
                                            
                                # Special instructions for multi-tool setup
                                st.markdown("### How to Use")
                                st.info("""
                                1. Copy these files to your project
                                2. Install required dependencies: `pip install crewai dotenv`
                                3. Set up any required API keys in a .env file
                                4. Run the workflow: `python crew.py`
                                """)
                            else:
                                # Simple list for single tool mode
                                for file in files:
                                    basename = os.path.basename(file)
                                    st.markdown(f"- **{basename}**")
                        else:
                            st.warning("No files were generated.")
                        
                        # Calculate and show output directory
                        output_directory = result.get("tool_info", {}).get("directory", output_dir)
                        st.markdown(f"**Output Directory:** `{output_directory}`")
                        
                        # Button to view code
                        if st.button("View Code"):
                            st.session_state.show_code = True
                            st.session_state.code_files = {}
                            
                            # Read file contents
                            for file_path in files:
                                try:
                                    with open(file_path, 'r') as f:
                                        st.session_state.code_files[os.path.basename(file_path)] = f.read()
                                except Exception as e:
                                    st.error(f"Error reading file {file_path}: {str(e)}")
                            
                            # Trigger rerun to show code
                            st.experimental_rerun()
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error in Streamlit app: {str(e)}", exc_info=True)

# UI for displaying tool information
def show_tool_info():
    st.subheader("🔍 Tool Information")
    
    if st.session_state.tool_info:
        # Show purpose
        st.markdown(f"**Purpose:** {st.session_state.tool_info.get('purpose', '')}")
        
        # Show tool names if available
        if 'tool_names' in st.session_state.tool_info:
            tool_names = st.session_state.tool_info.get('tool_names', [])
            if tool_names:
                st.markdown(f"**Tools:** {', '.join(tool_names)}")
        
        # Show tool count if available
        if 'tool_count' in st.session_state.tool_info:
            st.markdown(f"**Tool Count:** {st.session_state.tool_info.get('tool_count', 1)}")
        
        # Show directory if available
        if 'directory' in st.session_state.tool_info:
            st.markdown(f"**Output Directory:** {st.session_state.tool_info.get('directory', '')}")
            
        # Show if it's a multi-tool setup
        is_multi_tool = st.session_state.tool_info.get('is_multi_tool', False)
        if is_multi_tool:
            st.markdown("**Type:** Multi-tool CrewAI setup")
        else:
            st.markdown("**Type:** Single tool")
    else:
        st.info("No tool has been created yet.")
        st.session_state.tool_info = {} 