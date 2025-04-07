import os
import sys
import json
import streamlit as st
from streamlit_elements import elements, dashboard, mui, html, nivo
import tempfile
import subprocess
import time
from typing import Dict, List, Any, Optional

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
        "Config Files": [],
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
        elif file.endswith('.json'):
            files["Config Files"].append(file)
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

def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """Analyze a Python file to extract functions, classes, imports."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
            analysis_script = f"""
import ast
import json
import sys

def analyze_python_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        result = {{
            "imports": [],
            "functions": [],
            "classes": [],
            "variables": []
        }}
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    result["imports"].append({{
                        "name": name.name,
                        "line": node.lineno
                    }})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    result["imports"].append({{
                        "name": f"{{module}}.{{name.name}}",
                        "line": node.lineno
                    }})
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                result["functions"].append({{
                    "name": node.name,
                    "args": args,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, 'end_lineno', node.lineno + 10),
                    "docstring": ast.get_docstring(node) or ""
                }})
            elif isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.append({{
                            "name": child.name,
                            "args": [arg.arg for arg in child.args.args],
                            "lineno": child.lineno,
                            "end_lineno": getattr(child, 'end_lineno', child.lineno + 5)
                        }})
                result["classes"].append({{
                    "name": node.name,
                    "methods": methods,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, 'end_lineno', node.lineno + 20),
                    "docstring": ast.get_docstring(node) or ""
                }})
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result["variables"].append({{
                            "name": target.id,
                            "lineno": node.lineno
                        }})
        
        return result
    except Exception as e:
        return {{"error": str(e)}}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = analyze_python_file(file_path)
        print(json.dumps(result))
    else:
        print(json.dumps({{"error": "No file path provided"}}))
"""
            tmp.write(analysis_script)
            tmp_path = tmp.name
        
        # Run the script
        result = subprocess.run(
            [sys.executable, tmp_path, file_path],
            capture_output=True,
            text=True
        )
        
        os.unlink(tmp_path)
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"error": "Failed to parse analysis results"}
        else:
            return {"error": f"Analysis failed: {result.stderr}"}
    except Exception as e:
        return {"error": f"Failed to analyze code: {str(e)}"}

def analyze_json_file(file_path: str) -> Dict[str, Any]:
    """Analyze a JSON file to extract structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {
            "keys": [],
            "structure": data
        }
        
        # Extract top-level keys
        if isinstance(data, dict):
            for key in data.keys():
                result["keys"].append(key)
        
        return result
    except Exception as e:
        return {"error": f"Failed to analyze JSON: {str(e)}"}

def generate_function_node(function_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a dashboard node for a function."""
    return {
        "id": f"function_{function_info['name']}",
        "type": "function",
        "data": {
            "name": function_info["name"],
            "args": function_info["args"],
            "docstring": function_info.get("docstring", ""),
            "lineno": function_info["lineno"]
        },
        "position": {
            "x": 0,
            "y": function_info["lineno"] * 10
        }
    }

def generate_class_node(class_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a dashboard node for a class."""
    return {
        "id": f"class_{class_info['name']}",
        "type": "class",
        "data": {
            "name": class_info["name"],
            "methods": class_info["methods"],
            "docstring": class_info.get("docstring", ""),
            "lineno": class_info["lineno"]
        },
        "position": {
            "x": 300,
            "y": class_info["lineno"] * 10
        }
    }

def get_mcp_templates() -> List[Dict[str, Any]]:
    """Get available MCP templates with descriptions."""
    # In a real implementation, this would query your template database or API
    # For demonstration, we'll hardcode some examples
    return [
        {
            "id": "github_agent",
            "name": "GitHub Agent",
            "description": "Agent for interacting with GitHub repositories",
            "components": ["main.py", "models.py", "agent.py", "tools.py", "mcp.json"]
        },
        {
            "id": "spotify_agent",
            "name": "Spotify Agent",
            "description": "Agent for interacting with Spotify API",
            "components": ["main.py", "models.py", "agent.py", "tools.py", "mcp.json"]
        },
        {
            "id": "youtube_agent",
            "name": "YouTube Agent",
            "description": "Agent for fetching and analyzing YouTube content",
            "components": ["main.py", "models.py", "agent.py", "tools.py", "mcp.json"]
        },
        {
            "id": "file_processing",
            "name": "File Processing",
            "description": "Framework for processing and analyzing files",
            "components": ["main.py", "models.py", "processor.py", "tools.py"]
        }
    ]

def get_code_components() -> List[Dict[str, Any]]:
    """Get reusable code components."""
    return [
        {
            "id": "http_request",
            "name": "HTTP Request Component",
            "type": "function",
            "category": "Networking",
            "code": '''
async def make_http_request(url: str, method: str = "GET", headers: dict = None, data: dict = None) -> dict:
    """Make an HTTP request and return the response.
    
    Args:
        url: URL to make the request to
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Request headers
        data: Request data for POST/PUT methods
        
    Returns:
        Dictionary containing response data
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = await client.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        return {
            "status_code": response.status_code,
            "text": response.text,
            "json": response.json() if response.headers.get("content-type") == "application/json" else None,
            "headers": dict(response.headers)
        }
'''
        },
        {
            "id": "file_loader",
            "name": "File Loader Component",
            "type": "function",
            "category": "File I/O",
            "code": '''
def load_file(file_path: str, encoding: str = "utf-8") -> str:
    """Load a file and return its contents.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        File contents as a string
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
'''
        },
        {
            "id": "json_parser",
            "name": "JSON Parser Component",
            "type": "function",
            "category": "Data Processing",
            "code": '''
def parse_json(json_str: str) -> dict:
    """Parse JSON string into a Python dictionary.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as a dictionary
    """
    import json
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
'''
        },
        {
            "id": "mcp_server",
            "name": "MCP Server Component",
            "type": "code_block",
            "category": "MCP",
            "code": '''
async def run_mcp_servers():
    """Run MCP servers defined in mcp.json."""
    import subprocess
    import json
    import os
    
    # Load MCP configuration
    with open("mcp.json", "r") as f:
        mcp_config = json.load(f)
    
    # Start each MCP server
    servers = []
    for server_name, server_config in mcp_config.get("mcpServers", {}).items():
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        # Merge environment variables
        full_env = os.environ.copy()
        full_env.update(env)
        
        # Start the server
        server_process = subprocess.Popen(
            [command, *args],
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        servers.append({"name": server_name, "process": server_process})
    
    # Return servers for later cleanup
    return servers
'''
        }
    ]

def nocode_builder_tab():
    """Main function for the No-Code Builder tab."""
    st.title("Visual Code Builder")
    
    # Using a wider layout
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 95%;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Dashboard styling */
    .dashboard-item {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .dashboard-item-header {
        font-weight: bold;
        margin-bottom: 8px;
        color: #1565c0;
    }
    
    .dashboard-item-content {
        font-size: 14px;
    }
    
    /* Function node */
    .function-node {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    
    /* Class node */
    .class-node {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    
    /* Template node */
    .template-node {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    
    /* Component node */
    .component-node {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for the selected file
    if "nocode_selected_file" not in st.session_state:
        st.session_state.nocode_selected_file = None
    
    if "nocode_file_content" not in st.session_state:
        st.session_state.nocode_file_content = None
    
    if "nocode_file_analysis" not in st.session_state:
        st.session_state.nocode_file_analysis = None
    
    if "nocode_dashboard_items" not in st.session_state:
        st.session_state.nocode_dashboard_items = []
    
    if "nocode_dropped_components" not in st.session_state:
        st.session_state.nocode_dropped_components = []
    
    # Tabs for different views
    builder_tab, templates_tab, components_tab = st.tabs(["Visual Builder", "MCP Templates", "Code Components"])
    
    with templates_tab:
        st.subheader("MCP Agent Templates")
        st.markdown("Drag and drop these templates to build your agent quickly.")
        
        # Get MCP templates
        templates = get_mcp_templates()
        
        # Display templates in a grid
        cols = st.columns(2)
        for i, template in enumerate(templates):
            with cols[i % 2]:
                with st.container(border=True):
                    st.subheader(template["name"])
                    st.write(template["description"])
                    st.write(f"Components: {', '.join(template['components'])}")
                    
                    if st.button(f"Use Template", key=f"use_template_{template['id']}"):
                        st.session_state.nocode_selected_template = template
                        st.rerun()
    
    with components_tab:
        st.subheader("Reusable Code Components")
        st.markdown("Drag and drop these components to add functionality to your code.")
        
        # Get code components
        components = get_code_components()
        
        # Filter components by category
        categories = sorted(set(component["category"] for component in components))
        selected_category = st.selectbox("Filter by category", ["All"] + categories)
        
        filtered_components = components
        if selected_category != "All":
            filtered_components = [c for c in components if c["category"] == selected_category]
        
        # Display components in a grid
        for component in filtered_components:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(component["name"])
                    st.write(f"Type: {component['type']}")
                    st.write(f"Category: {component['category']}")
                
                with col2:
                    if st.button("Preview", key=f"preview_{component['id']}"):
                        st.session_state.nocode_preview_component = component
                    
                    if st.button("Add to Project", key=f"add_{component['id']}"):
                        if "nocode_dropped_components" not in st.session_state:
                            st.session_state.nocode_dropped_components = []
                        
                        st.session_state.nocode_dropped_components.append(component)
                        st.success(f"Added {component['name']} to your project!")
        
        # Preview component if selected
        if "nocode_preview_component" in st.session_state:
            with st.expander("Component Preview", expanded=True):
                st.code(st.session_state.nocode_preview_component["code"], language="python")
    
    with builder_tab:
        # Create columns for sidebar and main content
        file_col, content_col = st.columns([1, 3])
        
        # Sidebar for file selection
        with file_col:
            st.subheader("Files")
            
            # Get workbench files
            files_by_category = get_workbench_files()
            
            if not files_by_category:
                st.warning("No files found in workbench directory. Generate some files first.")
            else:
                # Display files by category
                for category, files in files_by_category.items():
                    with st.expander(category, expanded=True):
                        for file in files:
                            if st.button(file, key=f"select_file_{file}", use_container_width=True):
                                workbench_dir = os.path.join(os.getcwd(), "workbench")
                                file_path = os.path.join(workbench_dir, file)
                                
                                # Read file content
                                content = get_file_content(file_path)
                                
                                # Analyze file
                                if file.endswith('.py'):
                                    analysis = analyze_python_file(file_path)
                                elif file.endswith('.json'):
                                    analysis = analyze_json_file(file_path)
                                else:
                                    analysis = {"error": "Unsupported file type"}
                                
                                # Update session state
                                st.session_state.nocode_selected_file = file
                                st.session_state.nocode_file_content = content
                                st.session_state.nocode_file_analysis = analysis
                                
                                # Generate dashboard items
                                dashboard_items = []
                                
                                if "functions" in analysis:
                                    for func in analysis["functions"]:
                                        dashboard_items.append(generate_function_node(func))
                                
                                if "classes" in analysis:
                                    for cls in analysis["classes"]:
                                        dashboard_items.append(generate_class_node(cls))
                                
                                st.session_state.nocode_dashboard_items = dashboard_items
                                
                                st.rerun()
        
        # Main area - display the builder interface
        with content_col:
            if st.session_state.nocode_selected_file:
                file = st.session_state.nocode_selected_file
                
                # File information
                st.header(f"Visual Builder: {file}")
                
                # Show a warning if file analysis failed
                if "error" in st.session_state.nocode_file_analysis:
                    st.error(f"Error analyzing file: {st.session_state.nocode_file_analysis['error']}")
                
                # Display dashboard with draggable components
                dashboard_height = 600
                
                # Create dashboard with Streamlit Elements
                with elements("nocode_builder_dashboard"):
                    # Init dashboard with the draggable items
                    with dashboard.Grid(st.session_state.nocode_dashboard_items, draggableHandle=".item-drag-handle"):
                        # Create dashboard items for each node
                        for item in st.session_state.nocode_dashboard_items:
                            # Function node
                            if item["type"] == "function":
                                with mui.Card(key=item["id"], sx={"maxWidth": 300, "margin": "10px", "backgroundColor": "#e3f2fd"}):
                                    mui.CardHeader(
                                        title=mui.Typography(
                                            item["data"]["name"], className="item-drag-handle", variant="h6"
                                        ),
                                        subheader=f"Line: {item['data']['lineno']}"
                                    )
                                    mui.CardContent(
                                        html.div(
                                            [
                                                html.p(f"Args: {', '.join(item['data']['args'])}" if item['data']['args'] else "No arguments"),
                                                html.p(item['data']['docstring'] if item['data']['docstring'] else "No docstring")
                                            ]
                                        )
                                    )
                                    with mui.CardActions:
                                        mui.Button("Edit", variant="outlined", size="small")
                                        mui.Button("Delete", variant="outlined", size="small", color="error")
                            
                            # Class node
                            elif item["type"] == "class":
                                with mui.Card(key=item["id"], sx={"maxWidth": 300, "margin": "10px", "backgroundColor": "#e8f5e9"}):
                                    mui.CardHeader(
                                        title=mui.Typography(
                                            item["data"]["name"], className="item-drag-handle", variant="h6"
                                        ),
                                        subheader=f"Line: {item['data']['lineno']}"
                                    )
                                    with mui.CardContent:
                                        html.div(html.p(item['data']['docstring'] if item['data']['docstring'] else "No docstring"))
                                        
                                        if item["data"]["methods"]:
                                            mui.Typography("Methods:", variant="subtitle2")
                                            with mui.List(dense=True):
                                                for method in item["data"]["methods"]:
                                                    mui.ListItem(
                                                        mui.ListItemText(
                                                            method["name"],
                                                            f"Args: {', '.join(method['args'])}" if method['args'] else "No arguments"
                                                        )
                                                    )
                                    
                                    with mui.CardActions:
                                        mui.Button("Edit", variant="outlined", size="small")
                                        mui.Button("Delete", variant="outlined", size="small", color="error")
                        
                        # Add dropped components to the dashboard
                        for i, component in enumerate(st.session_state.nocode_dropped_components):
                            with mui.Card(key=f"component_{i}", sx={"maxWidth": 300, "margin": "10px", "backgroundColor": "#f3e5f5"}):
                                mui.CardHeader(
                                    title=mui.Typography(
                                        component["name"], className="item-drag-handle", variant="h6"
                                    ),
                                    subheader=f"Type: {component['type']}"
                                )
                                with mui.CardContent:
                                    html.div(html.p(f"Category: {component['category']}"))
                                
                                with mui.CardActions:
                                    mui.Button("Edit", variant="outlined", size="small")
                                    mui.Button("Delete", variant="outlined", size="small", color="error")
                
                # Buttons to update and save the file
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("👀 View Code", use_container_width=True):
                        workbench_dir = os.path.join(os.getcwd(), "workbench")
                        file_path = os.path.join(workbench_dir, file)
                        content = get_file_content(file_path)
                        st.session_state.nocode_view_code = content
                
                with col2:
                    if st.button("💾 Generate Code", use_container_width=True):
                        # This would generate code from the visual components
                        # For now, we'll just show a success message
                        st.success("Code generation not implemented in this demo")
                
                with col3:
                    if st.button("🚀 Run Code", use_container_width=True):
                        # This would run the code
                        # For now, we'll just show a success message
                        st.success("Code execution not implemented in this demo")
                
                # Display code if requested
                if "nocode_view_code" in st.session_state:
                    with st.expander("Code Preview", expanded=True):
                        st.code(st.session_state.nocode_view_code, language="python")
            else:
                # No file selected, show instructions
                st.info("👈 Select a file from the list to start building visually")
                
                st.markdown("""
                ### Visual Code Builder
    
                This tool lets you:
                
                1. **Visualize Code Structure** - See functions, classes, and their relationships
                2. **Drag and Drop Components** - Rearrange and modify code elements visually
                3. **Add Pre-built Components** - Use templates and code snippets for common tasks
                4. **Generate Clean Code** - Convert your visual design to clean, working code
                
                Select a file from the list to get started, or explore the MCP Templates and Code Components tabs.
                """)
                
                # Sample dashboard for demonstration
                with st.container(border=True):
                    st.subheader("Sample Visual Builder")
                    st.image("https://github.com/okld/streamlit-elements/raw/main/github_banner.png", use_container_width=True)
            
    # Add footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>Archon Visual Code Builder • No-code solution for MCP agents</div>", 
        unsafe_allow_html=True
    ) 