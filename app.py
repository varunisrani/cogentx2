import streamlit as st
from streamlit_terminal import st_terminal

# Set wide mode
st.set_page_config(layout="wide")

st.markdown("🚧 **Work in Progress** 🚧")

st.markdown("# Streamlit Terminal")

# Example 1: Basic Usage
st.markdown("## Example 1: Basic Usage")
st.write("Use this terminal to execute commands.")
st.write("For example, type `ls` after the `>` prompt and press `Enter` to run the command.")
st_terminal(key="terminal1", show_welcome_message=True)

st.markdown("""
            ```python
            # Source code of the above example
            from streamlit_terminal import st_terminal
            st_terminal(key="terminal1", show_welcome_message=True)
            ```""")

# Example 2: Custom Terminal
st.markdown("## Example 2: Custom Terminal")

st.markdown("### Set a Command from Input")
st.write("In this terminal, you can't type commands directly, but you can set a custom command using the `command` parameter.")
st.write("Click the |> button at the top-right of the terminal to run the command.")
cmd = st.text_input("Command", "python -u test/clock.py")
st_terminal(key="terminal4", command=cmd, height=-1, max_height=250, disable_input=True)

# Example 3: Get Output from Terminal
st.markdown("## Example 3: Get Output from Terminal")

full_outputs, updated_outputs = st_terminal(key="terminal5", command="python -u test/clock.py", height=200)

with st.container(height=500):
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Full Outputs")
        st.write(full_outputs)

    with cols[1]:
        st.markdown("### Updated Outputs")
        st.write(updated_outputs)


# Example 4: Colorful Output
st.markdown("## Example 4: Colorful Output")
st.write("This terminal displays colorful output.")
colorful_command = st.text_input("Colorful command", r'''echo -e "\x1b[31mError: Something went wrong\x1b[0m" &&
echo -e "\x1b[33mWarning: Check your configuration\x1b[0m" &&
echo -e "\x1b[32mSuccess: Operation completed successfully\x1b[0m"''')
st_terminal(key="terminal2", command=colorful_command, height=-1)

# Example 5: npm and nvm Support
st.markdown("## Example 5: npm and nvm Support")
st.write("This terminal supports npm and nvm commands.")

# npm version tab
st.markdown("### npm Version")
st.write("Run npm --version to check the installed npm version.")
st_terminal(key="terminal_npm", command="npm --version", height=150)

# Node.js version tab
st.markdown("### Node.js Version")
st.write("Run node --version to check the installed Node.js version.")
st_terminal(key="terminal_node", command="node --version", height=150)

# nvm list tab
st.markdown("### NVM Installed Versions")
st.write("Run nvm list to see all installed Node.js versions (if nvm is installed).")
st_terminal(key="terminal_nvm", command="nvm list", height=200)

# npm packages tab
st.markdown("### npm Packages")
st.write("Run npm list to see installed npm packages in the current directory.")
st_terminal(key="terminal_npm_list", command="npm list --depth=0", height=250)

# Interactive npm terminal
st.markdown("### Interactive npm Terminal")
st.write("Use this terminal to run any npm or nvm command.")
st_terminal(key="terminal_npm_interactive", height=400, show_welcome_message=True, welcome_message="Welcome to the npm/nvm terminal! Try commands like 'npm --version', 'node --version', or 'nvm list'.")

# Example 6: Interactive MCP Server Terminal
st.markdown("## Example 6: Interactive MCP Server Terminal")
st.write("This terminal is designed for running MCP servers and capturing their output in a VS Code-like experience.")
st.write("Try running commands like `python spotify_agent/streamlit_main.py --streamlit` or `python github_agent/main.py`")

st.markdown("### Spotify MCP Server Terminal")
st.write("This terminal is optimized for running the Spotify MCP server and interacting with it.")
st.write("1. Run the command `python spotify_agent/streamlit_main.py --streamlit`")
st.write("2. Wait for the server to start and show the prompt '🎵 Enter your Spotify query:'")
st.write("3. Type your query in the input field and press Enter or click Send")
spotify_terminal = st_terminal(key="terminal_spotify", height=500, show_welcome_message=True,
                         welcome_message="Spotify MCP Server Terminal - Run the Spotify agent and interact with it.")

st.markdown("### GitHub MCP Server Terminal")
st.write("This terminal is optimized for running the GitHub MCP server and interacting with it.")
github_terminal = st_terminal(key="terminal_github", height=500, show_welcome_message=True,
                         welcome_message="GitHub MCP Server Terminal - Run the GitHub agent and interact with it.")
