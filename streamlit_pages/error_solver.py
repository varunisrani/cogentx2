from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import ast
import re
import time
import hashlib
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from pydantic import BaseModel, Field
from functools import cached_property
import networkx as nx
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeContext(BaseModel):
    """Rich code context model"""
    file_path: str
    content: str
    imports: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    ast_tree: Optional[ast.AST] = None

class ErrorAnalysis(BaseModel):
    """Comprehensive error analysis model"""
    error_type: str
    error_message: str
    stack_trace: List[Dict[str, Any]]
    affected_files: List[str]
    root_cause: Optional[str] = None
    suggested_fixes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    
class CodeFix(BaseModel):
    """Detailed code fix model"""
    file_path: str
    original_code: str
    fixed_code: str
    changes: List[Dict[str, Any]]
    explanation: str
    impact_analysis: Dict[str, Any]
    confidence_score: float

class ErrorSolver:
    def __init__(self):
        self.openai_client = OpenAI()
        self.project_root = self._find_project_root()
        self.dependency_graph = nx.DiGraph()
        self.file_contexts: Dict[str, CodeContext] = {}
        
    def _find_project_root(self) -> Path:
        """Find project root by looking for common markers"""
        current = Path.cwd()
        while current != current.parent:
            if any((current / marker).exists() for marker in ['.git', 'pyproject.toml', 'setup.py']):
                return current
            current = current.parent
        return Path.cwd()

    def analyze_error(self, error_text: str) -> ErrorAnalysis:
        """Perform deep error analysis"""
        # Parse error information
        error_info = self._parse_error_info(error_text)
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Collect relevant files
        affected_files = self._find_affected_files(error_info)
        
        # Build rich context
        for file_path in affected_files:
            self._analyze_file(file_path)
            
        # Get AI analysis
        analysis = self._get_ai_analysis(error_info, affected_files)
        
        return analysis

    def _parse_error_info(self, error_text: str) -> Dict[str, Any]:
        """Extract detailed error information"""
        lines = error_text.split('\n')
        stack_trace = []
        
        for line in lines:
            if "File " in line:
                match = re.search(r'File "([^"]+)", line (\d+), in (.+)', line)
                if match:
                    stack_trace.append({
                        'file': match.group(1),
                        'line': int(match.group(2)),
                        'context': match.group(3)
                    })
                    
        error_type = None
        error_message = None
        for line in reversed(lines):
            if ': ' in line:
                parts = line.split(': ', 1)
                if any(err in parts[0] for err in ['Error', 'Exception']):
                    error_type = parts[0]
                    error_message = parts[1]
                    break
                    
        return {
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace
        }

    def _build_dependency_graph(self):
        """Build project dependency graph"""
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    with open(file_path) as f:
                        content = f.read()
                    
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for name in node.names:
                                    self.dependency_graph.add_edge(str(file_path), name.name)
                            elif isinstance(node, ast.ImportFrom):
                                self.dependency_graph.add_edge(str(file_path), node.module)
                    except:
                        continue

    def _find_affected_files(self, error_info: Dict[str, Any]) -> List[str]:
        """Find all potentially affected files"""
        affected = set()
        
        # Add files from stack trace
        for entry in error_info['stack_trace']:
            affected.add(entry['file'])
            
        # Add dependent files
        for file in list(affected):
            if file in self.dependency_graph:
                affected.update(nx.descendants(self.dependency_graph, file))
                affected.update(nx.ancestors(self.dependency_graph, file))
                
        return list(affected)

    def _analyze_file(self, file_path: str):
        """Build rich context for a file"""
        if file_path in self.file_contexts:
            return
            
        try:
            with open(file_path) as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            context = CodeContext(
                file_path=file_path,
                content=content,
                ast_tree=tree
            )
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    context.imports.extend(name.name for name in node.names)
                elif isinstance(node, ast.ImportFrom):
                    context.imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    context.classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    context.functions.append(node.name)
                    
            self.file_contexts[file_path] = context
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

    def _get_ai_analysis(self, error_info: Dict[str, Any], affected_files: List[str]) -> ErrorAnalysis:
        """Get AI analysis of the error"""
        # Build context for AI
        context = []
        for file in affected_files:
            if file in self.file_contexts:
                file_ctx = self.file_contexts[file]
                context.append(f"File: {file}\n{file_ctx.content}\n")
                
        # Get AI analysis
        prompt = f"""
Error Type: {error_info['error_type']}
Error Message: {error_info['error_message']}

Stack Trace:
{self._format_stack_trace(error_info['stack_trace'])}

Relevant Code:
{chr(10).join(context)}

Analyze this error and provide:
1. Root cause
2. Suggested fixes with confidence scores
3. Potential impact of fixes
"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Python developer analyzing code errors."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse AI response
        analysis = self._parse_ai_response(response.choices[0].message.content)
        
        return ErrorAnalysis(
            error_type=error_info['error_type'],
            error_message=error_info['error_message'],
            stack_trace=error_info['stack_trace'],
            affected_files=affected_files,
            root_cause=analysis['root_cause'],
            suggested_fixes=analysis['fixes'],
            confidence_score=analysis['confidence']
        )

    def generate_fix(self, analysis: ErrorAnalysis) -> List[CodeFix]:
        """Generate fixes based on analysis"""
        fixes = []
        
        for suggestion in analysis.suggested_fixes:
            file_path = suggestion['file']
            if file_path not in self.file_contexts:
                continue
                
            original = self.file_contexts[file_path].content
            
            # Get AI to generate specific fix
            prompt = f"""
Original code:
{original}

Error analysis:
{suggestion['description']}

Generate a precise fix that:
1. Addresses the root cause
2. Maintains code style
3. Has minimal impact
4. Includes detailed explanation
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer generating code fixes."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            fix_info = self._parse_fix_response(response.choices[0].message.content)
            
            fixes.append(CodeFix(
                file_path=file_path,
                original_code=original,
                fixed_code=fix_info['code'],
                changes=fix_info['changes'],
                explanation=fix_info['explanation'],
                impact_analysis=fix_info['impact'],
                confidence_score=suggestion['confidence']
            ))
            
        return fixes

    def apply_fixes(self, fixes: List[CodeFix], approved_files: List[str]) -> bool:
        """Apply approved fixes"""
        try:
            # Only apply to approved files
            for fix in fixes:
                if fix.file_path in approved_files:
                    with open(fix.file_path, 'w') as f:
                        f.write(fix.fixed_code)
            return True
        except Exception as e:
            logger.error(f"Error applying fixes: {e}")
            return False

    def _format_stack_trace(self, stack_trace: List[Dict[str, Any]]) -> str:
        """Format stack trace for AI prompt"""
        return '\n'.join(
            f"  File {entry['file']}, line {entry['line']}, in {entry['context']}"
            for entry in stack_trace
        )

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        # Implementation to parse structured response
        # Returns dict with root_cause, fixes, confidence
        pass

    def _parse_fix_response(self, response: str) -> Dict[str, Any]:
        """Parse AI fix response"""
        # Implementation to parse structured fix
        # Returns dict with code, changes, explanation, impact
        pass

def error_solver_tab():
    """Streamlit UI for error solver"""
    st.title("🔧 Advanced Error Solver")
    
    # Initialize solver
    if 'error_solver' not in st.session_state:
        st.session_state.error_solver = ErrorSolver()
        
    # Error input
    error_text = st.text_area(
        "Error Message",
        height=200,
        placeholder="Paste the complete error message including traceback..."
    )
    
    if error_text:
        with st.spinner("🔍 Analyzing error..."):
            # Analyze error
            analysis = st.session_state.error_solver.analyze_error(error_text)
            
            # Display analysis
            st.subheader("Error Analysis")
            st.write(f"**Error Type:** {analysis.error_type}")
            st.write(f"**Root Cause:** {analysis.root_cause}")
            
            # Show affected files
            st.subheader("Affected Files")
            for file in analysis.affected_files:
                st.write(f"- `{file}`")
            
            # Generate fixes
            fixes = st.session_state.error_solver.generate_fixes(analysis)
            
            # Display fixes
            st.subheader("Suggested Fixes")
            approved_files = []
            
            for fix in fixes:
                with st.expander(f"Fix for {fix.file_path} (Confidence: {fix.confidence_score:.2f})"):
                    st.write("**Changes:**")
                    for change in fix.changes:
                        st.write(f"- {change}")
                        
                    st.write("**Explanation:**")
                    st.write(fix.explanation)
                    
                    st.write("**Impact Analysis:**")
                    for area, impact in fix.impact_analysis.items():
                        st.write(f"- {area}: {impact}")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(fix.fixed_code, language="python")
                    with col2:
                        if st.checkbox("Approve this fix", key=fix.file_path):
                            approved_files.append(fix.file_path)
            
            # Apply fixes button
            if approved_files and st.button("Apply Approved Fixes"):
                with st.spinner("Applying fixes..."):
                    success = st.session_state.error_solver.apply_fixes(fixes, approved_files)
                    if success:
                        st.success("✅ Fixes applied successfully!")
                    else:
                        st.error("❌ Error applying fixes")

if __name__ == "__main__":
    error_solver_tab()
              File "app.py", line 25, in <module>
                result = process_data(data)
              File "app.py", line 10, in process_data
                formatted = print_data(data)
            NameError: name 'print_data' is not defined
            ```
            
            **Example 2: ImportError**
            ```
            Traceback (most recent call last):
              File "ml_model.py", line 3, in <module>
                import tensorflow as tf
            ModuleNotFoundError: No module named 'tensorflow'
            ```
            
            **Example 3: SyntaxError**
            ```
            File "utils.py", line 15
                if x == 5
                        ^
            SyntaxError: invalid syntax
            ```
            
            **Example 4: Multi-file Error**
            ```
            Traceback (most recent call last):
              File "main.py", line 5, in <module>
                from data_processor import process
              File "/Users/dev/project/data_processor.py", line 12, in <module>
                from utils import format_data
              File "/Users/dev/project/utils.py", line 8, in <module>
                class DataFormatter
            SyntaxError: expected ':'
            ```
            """)
        else:
            st.session_state.show_examples = False
        
        # Error text area with better placeholder
        st.text_area(
            "Error Message", 
            key="error_text", 
            height=200, 
            placeholder="Paste the full error message including traceback here for best results...",
            help="For best results, include the complete traceback which shows the chain of function calls that led to the error."
        )
        
        # Processing status indicator
        if st.session_state.processing_step:
            step_text = {
                "analyzing": "🔍 Analyzing error...",
                "generating_fixes": "🔧 Generating fixes...",
                "applying_fixes": "✅ Applying fixes..."
            }.get(st.session_state.processing_step, "Processing...")
            
            st.info(step_text)
        
        # Analyze button with better UX
        analyze_button = st.button(
            "🔍 Analyze Error", 
            on_click=analyze_error, 
            disabled=not st.session_state.error_text.strip() or st.session_state.processing_step is not None,
            help="Analyze the error to identify the cause and potential fixes"
        )
        
        # Analysis Results with tabbed interface for better navigation
        if st.session_state.error_analysis:
            st.subheader("📊 Error Analysis Results")
            
            error_type = st.session_state.error_analysis.error_type or "Unknown Error"
            error_msg = st.session_state.error_analysis.error_message or ""
            file_name = st.session_state.error_analysis.file_name or "Unknown file"
            line_number = st.session_state.error_analysis.line_number or "?"
            
            # Show error details with better formatting
            st.markdown(f"**Error Type:** `{error_type}`")
            st.markdown(f"**Message:** `{error_msg}`")
            st.markdown(f"**Main File:** `{file_name}`")
            st.markdown(f"**Line:** `{line_number if line_number else 'Unknown'}`")
            
            # Display AI analysis if available
            if st.session_state.error_analysis.ai_analysis:
                with st.expander("🤖 AI Analysis", expanded=True):
                    st.markdown(st.session_state.error_analysis.ai_analysis)
            
            # Display traceback info with better visualization
            if st.session_state.error_analysis.traceback:
                with st.expander("🔍 Traceback Information"):
                    for idx, entry in enumerate(st.session_state.error_analysis.traceback):
                        file_info = f"`{entry.file_name}`"
                        line_info = f"line `{entry.line_number}`"
                        content_info = f"```python\n{entry.line_content}\n```" if entry.line_content else ""
                        
                        st.markdown(f"**{idx+1}.** {file_info}, {line_info}")
                        if content_info:
                            st.markdown(content_info)
            
            # Display file contents with improved tabbed interface
            if st.session_state.error_analysis.file_contents:
                st.subheader("📄 File Contents and Proposed Fixes")
                
                # Create tabs for each file
                file_names = list(st.session_state.error_analysis.file_contents.keys())
                if file_names:
                    # Ensure main file is first
                    if file_name in file_names:
                        file_names.remove(file_name)
                        file_names.insert(0, file_name)
                    
                    tabs = st.tabs(file_names)
                    
                    for i, (tab_file_name, tab) in enumerate(zip(file_names, tabs)):
                        with tab:
                            content = st.session_state.error_analysis.file_contents[tab_file_name]
                            is_main_file = tab_file_name == file_name
                            
                            # Display original content with better formatting
                            st.markdown(f"**{tab_file_name}**{' (Main Error File)' if is_main_file else ''}")
                            
                            # Highlight affected line with advanced context
                            if is_main_file and line_number and content and isinstance(content, str):
                                lines = content.split('\n')
                                
                                if 0 < line_number <= len(lines):
                                    # Show context (5 lines before and after) with better highlighting
                                    start_line = max(0, line_number - 6)
                                    end_line = min(len(lines), line_number + 5)
                                    
                                    context_code = []
                                    for idx, line in enumerate(lines[start_line:end_line], start=start_line + 1):
                                        if idx == line_number:
                                            context_code.append(f"➡️ {idx}: {line}")
                                        else:
                                            context_code.append(f"   {idx}: {line}")
                                    
                                    st.code("\n".join(context_code), language="python")
                                    
                                    # Option to view full file
                                    with st.expander("View Full File"):
                                        st.code(content, language="python")
                                else:
                                    st.code(content, language="python")
                            else:
                                # Use an expander for long files
                                if content and isinstance(content, str) and len(content.splitlines()) > 50:
                                    with st.expander("View File Content"):
                                        st.code(content, language="python")
                                else:
                                    st.code(content, language="python")
                            
                            # Show fix proposals with enhanced diff view
                            if tab_file_name in st.session_state.proposed_changes:
                                fix_info = st.session_state.proposed_changes[tab_file_name]
                                
                                st.markdown(f"**Proposed Fix:** {fix_info.get('message', 'Unknown fix')}")
                                
                                # Display diff between original and fixed content with improved visualization
                                with st.expander("View Changes", expanded=True):
                                    original_lines = content.split('\n') if content and isinstance(content, str) else []
                                    fixed_lines = fix_info.get('fixed_content', '').split('\n') if fix_info.get('fixed_content') and isinstance(fix_info.get('fixed_content'), str) else []
                                    
                                    # Use difflib for better diff display
                                    diff_output = []
                                    d = difflib.Differ()
                                    diff = list(d.compare(original_lines, fixed_lines))
                                    
                                    for line in diff:
                                        if line.startswith('- '):
                                            diff_output.append(f"🔴 {line[2:]}")
                                        elif line.startswith('+ '):
                                            diff_output.append(f"🟢 {line[2:]}")
                                        elif line.startswith('  '):
                                            diff_output.append(f"   {line[2:]}")
                                    
                                    st.code("\n".join(diff_output), language="diff")
                                
                                # Checkbox to apply this fix with better confirmation
                                st.session_state.proposed_changes[tab_file_name]["apply"] = st.checkbox(
                                    f"✅ Apply fix to {tab_file_name}", 
                                    value=False,
                                    key=f"apply_{tab_file_name}",
                                    help="Select to apply this fix"
                                )
            
            # Apply Fix Button with improved confirmation
            if st.session_state.proposed_changes:
                has_selected_changes = any(fix.get("apply", False) for fix in st.session_state.proposed_changes.values())
                
                if has_selected_changes:
                    st.warning("⚠️ Applying fixes will modify your files. Backups will be created, but please review changes carefully.")
                
                fix_button = st.button(
                    "✅ Apply Selected Fixes", 
                    on_click=apply_fix,
                    disabled=not has_selected_changes or st.session_state.processing_step is not None,
                    help="Apply the selected fixes to your code files"
                )
    
    with right_col:
        # Help and guidance section first for better UX
        with st.expander("ℹ️ How to Use the Error Solver", expanded=not st.session_state.error_analysis):
            st.markdown("""
            ### How to Fix Errors:
            
            1. **Paste your error**: Include the complete traceback for best results
            2. **Analyze the error**: The AI will identify the problem and search for related files
            3. **Review the AI analysis**: See what the AI believes is causing the issue
            4. **Review suggested fixes**: Carefully examine the proposed changes in each file
            5. **Select and apply fixes**: Choose which fixes to apply by checking the boxes
            6. **Confirm changes**: Click "Apply Selected Fixes" to implement the changes
            
            ### Tips for Best Results:
            
            - Include the **full traceback** from your error, not just the last line
            - The solver can detect issues across **multiple files**
            - Always **review proposed changes** before applying them
            - For complex errors, you may need to fix **multiple files**
            - **Backups** are automatically created when applying fixes
            """)
        
        # Recent fixes list for better user tracking
        st.subheader("📋 Fix History")
        
        if st.session_state.error_fix_results:
            with st.expander("Recently Applied Fixes", expanded=True):
                for fix in st.session_state.error_fix_results:
                    st.success(f"✅ Fixed {fix['file_name']}: {fix['message']}")
                    st.markdown(f"**Path:** `{fix['file_path']}`")
                    st.markdown(f"**Applied at:** {fix['timestamp']}")
                    if fix.get('backup_path'):
                        st.markdown(f"**Backup:** `{fix['backup_path']}`")
