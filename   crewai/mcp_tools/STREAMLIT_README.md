# MCP2 Tool Builder - Streamlit UI

This Streamlit application provides a user-friendly interface for creating CrewAI agents with integrated MCP (Model Context Protocol) tools. It allows users to describe what they want to build in natural language, and the system will automatically find and integrate the appropriate MCP tools.

## Features

- **Conversational Interface**: Chat-based UI for describing your agent requirements
- **Automatic MCP Tool Discovery**: Finds relevant MCP tools based on your description
- **Code Generation**: Generates complete CrewAI agent code with MCP tool integration
- **File Management**: Creates organized directory structure for generated code
- **Tool Documentation**: Provides information about the integrated MCP tools

## Getting Started

### Prerequisites

- Python 3.9+
- Streamlit
- OpenAI API key (for embeddings and completions)
- Supabase connection (for tool storage and retrieval)

### Environment Variables

Make sure the following environment variables are set:

```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
PRIMARY_MODEL=gpt-4o-mini  # or your preferred model
```

### Running the Application

1. Navigate to the project directory:
   ```bash
   cd /path/to/archon
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run archon/mcp_tools/streamlit_app.py
   ```

3. Open your browser at http://localhost:8501

## Usage Examples

Here are some examples of requests you can make:

1. "Build a Spotify playlist generator that creates playlists based on mood"
2. "Create a Twitter bot that posts daily weather updates"
3. "Develop a GitHub issue tracker that summarizes open issues daily"
4. "Make a YouTube video analyzer that extracts key topics from videos"

## Output

For each request, the application:

1. Searches for relevant MCP tools
2. Generates agent code with integrated MCP tools
3. Creates directory structure with all necessary files
4. Provides a README with setup instructions

The generated code is saved to the `output/crew_project` directory by default.

## Troubleshooting

If you encounter issues:

1. Check that your environment variables are set correctly
2. Ensure you have the required permissions for the Supabase database
3. Verify that the OpenAI API key has sufficient quota

## Extending

To add new MCP tools:

1. Add tool definitions to the Supabase database
2. Include example code showing how to use the tool
3. Provide connection scripts for authentication if needed

## Related Documentation

- [MCP Tools Module](README.md) - Main documentation for the MCP tools module
- [CrewAI Documentation](https://docs.crewai.com/) - Documentation for CrewAI 