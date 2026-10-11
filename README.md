# Open Notebook MCP Server

<!-- mcp-name: io.github.Epochal-dev/open-notebook -->

An MCP (Model Context Protocol) server that provides tools to interact with the [Open Notebook](https://github.com/lfnovo/open-notebook) API. This server enables AI assistants like Claude to manage notebooks, sources, notes, search content, and interact with AI models through Open Notebook.

## Features

- **Notebooks Management**: Create, read, update, and delete notebooks
- **Sources Management**: Add and manage content sources (links, uploads, text)
- **Notes Management**: Create and organize notes within notebooks
- **Search & AI**: Search content using vector/text search and ask questions
- **Models Management**: Configure and manage AI models
- **Chat Sessions**: Create and manage chat conversations
- **Settings**: Access and update application settings
- **Progressive Disclosure**: Efficient tool discovery with `search_capabilities`

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/PiotrAleksander/open-notebook-mcp.git
cd open-notebook-mcp

# Install with uv
uv sync
```

### Using pip

```bash
pip install -e .
```

## Configuration

The server requires configuration to connect to your Open Notebook instance:

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required: URL of your Open Notebook instance
OPEN_NOTEBOOK_URL=http://localhost:5055

# Optional: Authentication password (if APP_PASSWORD is set in Open Notebook)
OPEN_NOTEBOOK_PASSWORD=your_password_here

# Optional: Transport configuration (default: stdio)
MCP_TRANSPORT=stdio  # or streamable-http for remote deployment
```

### Example Configuration

For local development with default Open Notebook settings:

```bash
# .env
OPEN_NOTEBOOK_URL=http://localhost:5055
```

If you've configured authentication in Open Notebook:

```bash
# .env
OPEN_NOTEBOOK_URL=http://localhost:5055
OPEN_NOTEBOOK_PASSWORD=my_secure_password
```

## Usage

### Running the Server

#### Development Mode (STDIO)

For local use with AI assistants:

```bash
uv run open-notebook-mcp
```

Or using the MCP CLI:

```bash
mcp dev src/open_notebook_mcp/server.py
```

#### Production Mode (Streamable HTTP)

For remote deployment:

```bash
MCP_TRANSPORT=streamable-http HOST=0.0.0.0 PORT=8000 uv run open-notebook-mcp
```

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "open-notebook": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/open-notebook-mcp",
        "open-notebook-mcp"
      ],
      "env": {
        "OPEN_NOTEBOOK_URL": "http://localhost:5055",
        "OPEN_NOTEBOOK_PASSWORD": "your_password_if_needed"
      }
    }
  }
}
```

### Discovering Available Tools

The server implements progressive disclosure. Use the `search_capabilities` tool to discover available functionality:

```python
# Get a summary of all tools
search_capabilities(query="", detail="summary", limit=50)

# Search for specific functionality
search_capabilities(query="notebook", detail="summary", limit=10)

# Get full details for a specific tool
search_capabilities(query="create_notebook", detail="full", limit=1)
```

### Example Workflows

#### Creating and Managing Notebooks

```python
# Create a new notebook
result = create_notebook(
    name="AI Research",
    description="Research on AI applications"
)
notebook_id = result["notebook"]["id"]

# List all notebooks
notebooks = list_notebooks(archived=False, limit=20)

# Update a notebook
update_notebook(
    notebook_id=notebook_id,
    name="AI Research (Updated)"
)

# Get a specific notebook
notebook = get_notebook(notebook_id=notebook_id)
```

#### Adding Sources

```python
# Add a web source
source = create_source(
    notebook_id=notebook_id,
    type="link",
    url="https://example.com/ai-article",
    title="AI Research Article",
    embed=True  # Generate embeddings
)

# List sources in a notebook
sources = list_sources(notebook_id=notebook_id, limit=20)
```

#### Creating Notes

```python
# Create a note
note = create_note(
    notebook_id=notebook_id,
    title="Key Findings",
    content="Important insights about AI applications...",
    topics=["AI", "Research"]
)

# Update a note
update_note(
    note_id=note["note"]["id"],
    content="Updated insights..."
)
```

#### Searching and Asking Questions

```python
# Search content
results = search(
    query="artificial intelligence",
    type="vector",
    notebook_id=notebook_id,
    limit=10
)

# List available models first
models = list_models(limit=50)
model_id = models["models"][0]["id"]

# Ask a question
answer = ask_simple(
    question="What are the main AI applications mentioned?",
    strategy_model=model_id,
    answer_model=model_id,
    final_answer_model=model_id,
    notebook_id=notebook_id
)
```

#### Chat Sessions

```python
# Create a chat session
session = create_chat_session(
    notebook_id=notebook_id,
    title="Research Discussion"
)
session_id = session["session"]["id"]

# Build context
context = get_chat_context(notebook_id=notebook_id)

# Send a message
response = execute_chat(
    session_id=session_id,
    message="What are the key insights from my research?",
    context=context["context"]
)

# Get session history
history = get_chat_session(session_id=session_id)
```

## Available Tools

The server provides 39 tools across multiple categories:

### Meta Tools

- `search_capabilities` - Progressive tool discovery

### Notebooks (5 tools)

- `list_notebooks`, `get_notebook`, `create_notebook`, `update_notebook`, `delete_notebook`

### Sources (5 tools)

- `list_sources`, `get_source`, `create_source`, `update_source`, `delete_source`

### Notes (5 tools)

- `list_notes`, `get_note`, `create_note`, `update_note`, `delete_note`

### Search (3 tools)

- `search`, `ask_question`, `ask_simple`

### Models (5 tools)

- `list_models`, `get_model`, `create_model`, `delete_model`, `get_default_models`

### Chat (7 tools)

- `list_chat_sessions`, `create_chat_session`, `get_chat_session`, `update_chat_session`, `delete_chat_session`, `execute_chat`, `get_chat_context`

### Settings (2 tools)

- `get_settings`, `update_settings`

## Architecture

This server follows MCP best practices:

- **Progressive Disclosure**: Use `search_capabilities` to minimize context usage
- **Context Efficiency**: Small outputs by default, with limit parameters
- **Dual Transport**: Supports both STDIO (local) and Streamable HTTP (remote)
- **Error Handling**: Structured error messages with actionable hints
- **Timeouts**: 30-second default timeout for all API requests
- **Authentication**: Optional Bearer token authentication

## Development

### Project Structure

```
open-notebook-mcp/
├── src/
│   └── open_notebook_mcp/
│       ├── __init__.py
│       └── server.py          # Main MCP server implementation
├── tests/                      # (to be added)
├── pyproject.toml
├── README.md
└── .env.example
```

### Testing

Test the server using the MCP Inspector:

```bash
mcp dev src/open_notebook_mcp/server.py
```

or

```bash
npx @modelcontextprotocol/inspector uv --directory ./src/open_notebook_mcp "run" "server.py"
```

This opens an interactive inspector where you can:

1. Browse available tools
2. Test tool calls
3. Inspect responses
4. Debug errors

### Adding New Tools

To add new tools:

1. Add a `Capability` entry to the `CAPABILITIES` tuple
2. Implement the tool function with `@mcp.tool()` decorator
3. Follow naming conventions: `verb_noun` (e.g., `list_notebooks`)
4. Include proper docstrings and type hints
5. Return structured responses with `request_id`

## Requirements

- Python 3.12+
- Open Notebook instance (local or remote)
- Dependencies: `mcp[cli]>=2,<3`, `httpx>=0.28.1`

## Contributing

Contributions are welcome! Please ensure:

- Follow the existing code structure and patterns
- Add tools to the `CAPABILITIES` index
- Include proper type hints and docstrings
- Test with MCP Inspector before submitting

## License

See LICENSE file for details.

## Links

- [Open Notebook](https://github.com/lfnovo/open-notebook)
- [Open Notebook API Reference](https://github.com/lfnovo/open-notebook/blob/main/docs/development/api-reference.md)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://gofastmcp.com/)

## Support

For issues related to:

- **This MCP server**: Open an issue in this repository
- **Open Notebook itself**: Visit the [Open Notebook repository](https://github.com/lfnovo/open-notebook)
- **MCP protocol**: Check the [MCP documentation](https://modelcontextprotocol.io/)
