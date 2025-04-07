# Archon Terminal Application

The Archon Terminal Application is an interactive terminal interface built with Streamlit that provides a powerful command-line experience directly in your web browser.

## Features

- **Interactive Terminal**: Execute shell commands directly from your browser
- **Command History**: Track and recall previously executed commands
- **Full Logs**: View complete command history with timestamps
- **Multi-line Commands**: Support for complex, multi-line scripts
- **Process Management**: Start, monitor, and terminate processes
- **ANSI Color Support**: Display colorful terminal output
- **Streamlit Terminal Integration**: Includes the streamlit-terminal component for a simpler terminal experience

## Terminal Tabs

The application provides multiple views through tabs:

1. **Terminal**: The main interactive terminal with command input and real-time output
2. **Full Logs**: Complete history of all commands and outputs
3. **Streamlit Terminal**: A simpler terminal interface using the streamlit-terminal component

## Streamlit Terminal Integration

The Archon Terminal includes integration with the streamlit-terminal component, providing a simpler alternative to the full-featured Archon Terminal.

### Comparison

**Streamlit-Terminal:**
- Good for simple integration where you need a basic terminal
- Useful when you want a clean component-based approach
- Easier to use as a dependency in other Streamlit apps

**Archon Terminal:**
- More feature-rich with advanced styling and command history
- Better for complex terminal operations and scripting
- Includes full logs and detailed output processing

### Streamlit Terminal Examples

The Streamlit Terminal tab includes three examples:

1. **Basic Usage**: A simple terminal for executing commands
2. **Custom Command**: A terminal that runs a specific command you provide
3. **Colorful Output**: A terminal that demonstrates ANSI color support

## Requirements

- Streamlit
- Optional: streamlit-ace for enhanced code editing
- Optional: streamlit-terminal for the Streamlit Terminal component
- Optional: rich for enhanced ANSI color handling

## Installation

```bash
pip install streamlit
pip install streamlit-ace  # Optional, for enhanced code editing
pip install streamlit-terminal  # Optional, for the streamlit-terminal component
pip install rich  # Optional, for enhanced ANSI color handling
```

## Usage

```bash
streamlit run terminal_app.py
```

## Contributing

Contributions to improve the terminal application are welcome! Feel free to submit pull requests with enhancements or bug fixes.

## License

MIT License
