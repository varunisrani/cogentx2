import streamlit as st
import os

def workbench_tab():
    """Workbench tab content."""
    
    st.info("This is a placeholder for the Workbench tab. The actual implementation would allow managing agent projects and files.")
    
    # Placeholder UI
    st.write("## Projects")
    
    # Get actual workbench files if they exist
    workbench_dir = os.path.join(os.getcwd(), "workbench")
    has_files = os.path.exists(workbench_dir) and len(os.listdir(workbench_dir)) > 0
    
    if has_files:
        files = os.listdir(workbench_dir)
        python_files = [f for f in files if f.endswith('.py')]
        json_files = [f for f in files if f.endswith('.json')]
        other_files = [f for f in files if not f.endswith(('.py', '.json'))]
        
        if python_files:
            st.write("### Python Files")
            for file in python_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    st.button("Edit", key=f"edit_{file}")
        
        if json_files:
            st.write("### Configuration Files")
            for file in json_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    st.button("Edit", key=f"edit_{file}")
        
        if other_files:
            st.write("### Other Files")
            for file in other_files:
                st.write(file)
    else:
        st.write("No files found in the workbench directory.")
        if st.button("Create New Project"):
            st.session_state.create_new_project = True
            st.rerun()
    
    # Project actions
    st.write("## Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("New File", use_container_width=True)
    
    with col2:
        st.button("Import Template", use_container_width=True)
    
    with col3:
        st.button("Export Project", use_container_width=True) 