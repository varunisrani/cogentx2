import streamlit as st

def home_tab():
    """Home tab content."""
    
    st.markdown("""
    # Welcome to Archon
    
    Archon is a platform for building, managing, and running multi-agent AI systems using the MCP (Multi-Agent Coordination Protocol) framework.
    
    ## Features
    
    - **Agent Runner**: Run and test MCP agents interactively
    - **Template Browser**: Browse and use pre-built agent templates
    - **Workbench**: Manage your agent projects and files
    - **Code Editor**: Edit agent code with a full-featured code editor
    - **No-Code Builder**: Build agents visually with drag-and-drop components
    
    ## Getting Started
    
    1. Browse the **Template Browser** to find agent templates
    2. Use the **Workbench** to manage your agent projects
    3. Edit code in the **Code Editor** or build visually in the **No-Code Builder**
    4. Run your agents in the **Agent Runner**
    
    ## Documentation
    
    For more information about Archon and the MCP framework, check out the documentation at [github.com/sjuu/Archon](https://github.com/sjuu/Archon).
    """)
    
    # Display cards for each main feature
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("🚀 Agent Runner")
            st.write("Run and test MCP agents interactively.")
            if st.button("Go to Agent Runner", key="home_agent_runner_btn"):
                st.session_state.selected_tab = "Agent Runner"
                st.rerun()
        
        with st.container(border=True):
            st.subheader("🧰 Workbench")
            st.write("Manage your agent projects and files.")
            if st.button("Go to Workbench", key="home_workbench_btn"):
                st.session_state.selected_tab = "Workbench"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.subheader("📚 Template Browser")
            st.write("Browse and use pre-built agent templates.")
            if st.button("Go to Template Browser", key="home_template_browser_btn"):
                st.session_state.selected_tab = "Template Browser"
                st.rerun()
        
        with st.container(border=True):
            st.subheader("🔧 Code Editor")
            st.write("Edit agent code with a full-featured code editor.")
            if st.button("Go to Code Editor", key="home_code_editor_btn"):
                st.session_state.selected_tab = "Code Editor"
                st.rerun()
    
    # Feature highlight
    st.subheader("✨ Featured: Visual Builder")
    with st.container(border=True):
        st.write("Build agents visually with our new drag-and-drop interface - no coding required!")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            The **No-Code Builder** allows you to:
            - Visualize code structure and components
            - Drag and drop elements to build your agent
            - Use pre-built templates and components
            - Generate clean code automatically
            """)
        with col2:
            if st.button("Try It Now!", key="home_visual_builder_btn", use_container_width=True):
                st.session_state.selected_tab = "No-Code Builder"
                st.rerun() 