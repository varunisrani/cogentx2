import os
import sys
import streamlit as st
from code_editor import code_editor
import json
import subprocess
import re
import time
from typing import Dict, List, Optional, Tuple, Any
import tempfile

def get_workbench_files() -> Dict[str, List[str]]:
    """Get all Python files in the workbench directory, organized by type."""
    workbench_dir = os.path.join(os.getcwd(), "workbench")
    
    if not os.path.exists(workbench_dir):
        return {}
    
    files = {
        "Main Files": [],
        "Model Files": [],
        "Agent Files": [],
        "Tool Files": [],
        "Other Files": []
    }
    
    for file in os.listdir(workbench_dir):
        file_path = os.path.join(workbench_dir, file)
        if not os.path.isfile(file_path):
            continue
            
        if file.endswith('.py'):
            if file == 'main.py':
                files["Main Files"].append(file)
            elif 'model' in file.lower():
                files["Model Files"].append(file)
            elif 'agent' in file.lower():
                files["Agent Files"].append(file)
            elif 'tool' in file.lower():
                files["Tool Files"].append(file)
            else:
                files["Other Files"].append(file)
    
    # Remove empty categories
    return {k: v for k, v in files.items() if v}

def get_file_content(file_path: str) -> str:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

def save_file_content(file_path: str, content: str) -> bool:
    """Save content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

def execute_python_code(code: str) -> Tuple[bool, str]:
    """Execute Python code and return the result."""
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        process = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10  # Limit execution time
        )
        
        if process.returncode == 0:
            return True, process.stdout
        else:
            return False, process.stderr
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (10 seconds)"
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

def nocode_editor_tab():
    """Main function for the Code Editor tab."""
    
    # Get all Python files in the workbench directory
    files_by_category = get_workbench_files()
    
    if not files_by_category:
        st.warning("⚠️ No Python files found in the workbench directory. Run the code generator first to create some files.")
        return
    
    # Create a better UI layout with tabs and sidebar
    st.sidebar.markdown("## 📂 File Browser")
    
    # Create a dictionary to store expanded states for each category
    if "expanded_categories" not in st.session_state:
        st.session_state.expanded_categories = {
            "Main Files": True,
            "Model Files": True,
            "Agent Files": True,
            "Tool Files": True,
            "Other Files": True
        }
    
    # Create a search box for files
    search_query = st.sidebar.text_input("🔍 Search files", key="file_search")
    
    # Display files by category with a modern look
    file_count = 0
    selected_file = None
    
    for category, files in files_by_category.items():
        # Filter files based on search query
        if search_query:
            filtered_files = [f for f in files if search_query.lower() in f.lower()]
        else:
            filtered_files = files
            
        if not filtered_files:
            continue
            
        # Make the category header look nice with icons
        category_icon = {
            "Main Files": "🚀",
            "Model Files": "📊",
            "Agent Files": "🤖",
            "Tool Files": "🔧",
            "Other Files": "📄"
        }.get(category, "📁")
        
        # Create expandable sections that look nicer
        with st.sidebar.expander(f"{category_icon} {category} ({len(filtered_files)})", expanded=st.session_state.expanded_categories.get(category, True)):
            for file in filtered_files:
                file_count += 1
                
                # Create a clickable file name
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(f"{file}", key=f"file_{file}", use_container_width=True):
                        selected_file = file
                        # Store selection in session state
                        st.session_state.selected_file = file
                        # Reset file content view
                        if "file_content" in st.session_state:
                            st.session_state.pop("file_content", None)
                
                with col2:
                    # Add a small info button to show file details
                    if st.button("ℹ️", key=f"info_{file}", help=f"Show info about {file}"):
                        workbench_dir = os.path.join(os.getcwd(), "workbench")
                        file_path = os.path.join(workbench_dir, file)
                        st.session_state.show_file_info = file
    
    # If there's a selected file in session state, use that
    if "selected_file" in st.session_state:
        selected_file = st.session_state.selected_file
    
    # Show file info if requested
    if "show_file_info" in st.session_state:
        file_to_show = st.session_state.show_file_info
        workbench_dir = os.path.join(os.getcwd(), "workbench")
        file_path = os.path.join(workbench_dir, file_to_show)
        
        try:
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            file_modified = time.ctime(file_stats.st_mtime)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"### File: {file_to_show}")
            st.sidebar.markdown(f"**Size:** {file_size} bytes")
            st.sidebar.markdown(f"**Last Modified:** {file_modified}")
            
            if st.sidebar.button("Close Info", key="close_info"):
                st.session_state.pop("show_file_info", None)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error getting file info: {str(e)}")
    
    # Main content area - show welcome or file editor
    if not selected_file:
        # Show welcome screen
        st.markdown("## 👋 Welcome to the Code Editor")
        
        # Create a clean, modern welcome screen
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Edit your agent code directly in the browser
            
            This editor lets you modify and test the Python files in your workbench 
            directory without needing to use a separate code editor.
            
            **Key features:**
            - Syntax highlighting for Python
            - Auto-completion for code
            - Execute code directly in the browser
            - Modern editor with theme support
            
            **To get started:**
            1. Select a file from the sidebar
            2. Edit the code as needed
            3. Click 'Save Changes' when done
            4. Use 'Run Code' to test your changes
            """)
        
        with col2:
            # Show a visually appealing illustration or stats
            st.markdown("### Project Statistics")
            
            # Calculate some stats
            total_files = sum(len(files) for files in files_by_category.values())
            
            # Create a metrics display
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Files", total_files)
            
            # Try to count total lines of code
            try:
                total_lines = 0
                workbench_dir = os.path.join(os.getcwd(), "workbench")
                for category, files in files_by_category.items():
                    for file in files:
                        file_path = os.path.join(workbench_dir, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            total_lines += len(f.readlines())
                
                col_b.metric("Lines of Code", total_lines)
            except:
                col_b.metric("Lines of Code", "N/A")
            
            # Show when the workbench was last modified
            try:
                workbench_dir = os.path.join(os.getcwd(), "workbench")
                last_modified = max(os.path.getmtime(os.path.join(workbench_dir, file)) 
                                   for category, files in files_by_category.items() 
                                   for file in files)
                col_c.metric("Last Modified", time.strftime("%d %b", time.localtime(last_modified)))
            except:
                col_c.metric("Last Modified", "N/A")
            
            # Add a "Quick Start" section
            st.markdown("### Quick Start")
            
            if st.button("📄 Edit main.py", use_container_width=True):
                # Try to find and open main.py
                if "Main Files" in files_by_category and files_by_category["Main Files"]:
                    st.session_state.selected_file = "main.py"
                    st.rerun()
    else:
        # Load file content if not already in session state
        if "file_content" not in st.session_state:
            workbench_dir = os.path.join(os.getcwd(), "workbench")
            file_path = os.path.join(workbench_dir, selected_file)
            
            try:
                st.session_state.file_content = get_file_content(file_path)
                # Also store original content for comparison
                st.session_state.original_file_content = st.session_state.file_content
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Create a nice header
        file_icon = "📄"
        if selected_file.endswith('.py'):
            file_icon = "🐍"
        
        st.markdown(f"## {file_icon} Editing: {selected_file}")
        
        # Add a toolbar with helpful buttons
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        
        with col1:
            if st.button("🔄 Reset Changes", use_container_width=True):
                if "original_file_content" in st.session_state:
                    st.session_state.file_content = st.session_state.original_file_content
                    st.success("Changes reset to original file content")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            editor_theme = st.selectbox(
                "🎨 Theme", 
                ["default", "monokai", "github", "tomorrow", "kuroir", "twilight", "solarized_dark", "solarized_light"],
                key="editor_theme"
            )
        
        with col3:
            editor_keybinding = st.selectbox(
                "⌨️ Keybindings", 
                ["vscode", "default", "emacs", "vim"],
                key="editor_keybinding"
            )
        
        with col4:
            editor_font_size = st.selectbox(
                "📏 Font Size",
                [12, 14, 16, 18, 20],
                index=1,
                key="editor_font_size"
            )
            
        with col5:
            # Autorun toggle
            st.session_state.autorun = st.toggle("🚀 Auto-Run on Save", value=False, key="autorun_toggle")
        
        # Create the code editor with the chosen theme
        edited_code = code_editor(
            code=st.session_state.file_content,
            lang="python",
            theme=editor_theme,
            shortcuts=editor_keybinding,
            height=[500, 700],
            buttons=[
                {
                    "name": "Save Changes",
                    "feather": "Save",
                    "primary": True,
                    "hasText": True,
                    "showWithIcon": True,
                    "commands": ["submit"]
                }
            ],
            options={
                "wrap": True,
                "fontSize": editor_font_size,
                "showLineNumbers": True,
                "showGutter": True,
                "showPrintMargin": False,
                "enableBasicAutocompletion": True,
                "enableLiveAutocompletion": True,
                "highlightActiveLine": True
            },
            response_mode=["blur", "debounce"],
            key=f"code_editor_{selected_file}"
        )
        
        # Handle code saving
        if edited_code["type"] == "submit":
            workbench_dir = os.path.join(os.getcwd(), "workbench")
            file_path = os.path.join(workbench_dir, selected_file)
            
            if save_file_content(file_path, edited_code["text"]):
                st.success(f"✅ {selected_file} saved successfully!")
                
                # Update the file content in session state
                st.session_state.file_content = edited_code["text"]
                
                # Auto-run if enabled
                if st.session_state.autorun:
                    st.session_state.run_after_save = True
                
                # Give a visual feedback
                with st.spinner("Updating..."):
                    time.sleep(0.5)  # Brief pause for visual feedback
            else:
                st.error(f"Failed to save {selected_file}")
        
        # Add Execute Code button
        if st.button("🚀 Run Code", use_container_width=True) or st.session_state.get('run_after_save', False):
            st.session_state.pop('run_after_save', None)  # Clear flag
            
            st.markdown("### 📟 Execution Output")
            
            with st.spinner("Running code..."):
                success, output = execute_python_code(st.session_state.file_content)
                
                # Show output in a nice container
                output_container = st.container(border=True)
                
                with output_container:
                    if success:
                        st.markdown("#### ✅ Code executed successfully")
                        if output.strip():
                            st.code(output, language="bash")
                        else:
                            st.info("No output produced")
                    else:
                        st.markdown("#### ❌ Code execution failed")
                        st.error(output)
        
        # Add a file details section
        with st.expander("📊 File Details", expanded=False):
            try:
                file_path = os.path.join(os.path.join(os.getcwd(), "workbench"), selected_file)
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                file_modified = time.ctime(file_stats.st_mtime)
                
                # Count lines of code and other metrics
                code_lines = st.session_state.file_content.split('\n')
                total_lines = len(code_lines)
                comment_lines = sum(1 for line in code_lines if line.strip().startswith('#'))
                blank_lines = sum(1 for line in code_lines if not line.strip())
                
                # Create a nice grid of metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Lines", total_lines)
                col2.metric("Code Lines", total_lines - comment_lines - blank_lines)
                col3.metric("Comments", comment_lines)
                col4.metric("Blank Lines", blank_lines)
                
                # File information
                st.markdown(f"**File Size:** {file_size} bytes")
                st.markdown(f"**Last Modified:** {file_modified}")
                st.markdown(f"**Full Path:** {file_path}")
                
            except Exception as e:
                st.error(f"Error reading file details: {str(e)}")
                
        # Add a help section
        with st.expander("❓ Keyboard Shortcuts", expanded=False):
            st.markdown("""
            | Action | VSCode Shortcut | Vim Shortcut | Emacs Shortcut |
            | ------ | --------------- | ------------ | -------------- |
            | Save | Ctrl+S | :w | C-x C-s |
            | Find | Ctrl+F | / | C-s |
            | Replace | Ctrl+H | :%s/find/replace/g | M-% |
            | Undo | Ctrl+Z | u | C-/ |
            | Redo | Ctrl+Y | Ctrl+R | C-? |
            | Comment | Ctrl+/ | gc | M-; |
            """)

    # Add a footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>Archon Code Editor • Built with ❤️ using Streamlit</div>", 
        unsafe_allow_html=True
    ) 