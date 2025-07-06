# Memvid MCP Server 🎥

A Model Context Protocol (MCP) server that exposes Memvid video memory functionalities to AI clients. This server allows you to encode text, PDFs, and other content into video memory format for efficient semantic search and chat interactions.

## 🌟 Features

- **Text Encoding**: Add text chunks or full text documents to video memory
- **PDF Processing**: Extract and encode content from PDF files
- **Video Memory Building**: Generate compressed video representations of your data
- **Semantic Search**: Query your encoded data using natural language
- **Chat Interface**: Have conversations with your encoded knowledge base
- **Multi-Connection Support**: Handle multiple concurrent client connections
- **Comprehensive Logging**: Detailed logging to stderr for debugging
- **Graceful Shutdown**: Proper resource cleanup and signal handling

## 📋 Requirements

- Python 3.10 or higher
- uv package manager
- memvid package
- MCP-compatible client (e.g., Claude Desktop)

## 🚀 Installation

### 1. Set up the environment
```bash
cd /memvid_mcp_server
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 2. Install dependencies
```bash
uv add -e .
```
### H.265 Encoding with Docker

To enable H.265 video compression, you need to build the `memvid-h265` Docker container. This container provides the necessary FFmpeg environment for H.265
encoding.

1.  **Navigate to the `memvid` repository root:**
    ```bash
    cd /memvid
    ```
2.  **Build the Docker image:**
    ```bash
    docker build -f docker/Dockerfile -t memvid-h265 docker/
    ```
    This command builds the Docker image named `memvid-h265` using the `Dockerfile` located in the `docker/` directory.

Once the Docker image is built, `memvid` will automatically detect and use it when `video_codec='h265'` is specified in `build_video`.

### 3. Test the server (optional)
```bash
uv run python memvid_mcp_server/main.py
```

## ⚙️ Configuration

### Claude Desktop Setup

1. Copy the example configuration:
```bash
cp example_mcp_config.json ~/.config/claude-desktop/config.json
```

2. Or manually add to your Claude Desktop config:
```json
{
  "mcpServers": {
    "memvid-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/memvid_mcp_server",
        "run",
        "python",
        "memvid_mcp_server/main.py"
      ],
      "env": {
        "PYTHONPATH": "/home/ty/Repositories/memvid_mcp_server",
        "PYTHONWARNINGS": "ignore"
      }
    }
  }
}
```

3. Restart Claude Desktop to load the server.

## 🛠️ Available Tools

### `get_server_status`
Check the current status of the memvid server including version information.

### `add_chunks`
Add a list of text chunks to the encoder.
- **chunks**: List of text strings to add

### `add_text`
Add a single text document to the encoder.
- **text**: Text content to add
- **metadata**: Optional metadata dictionary

### `add_pdf`
Process and add a PDF file to the encoder.
- **pdf_path**: Path to the PDF file

### `build_video`
Build the video memory from all added content.
- **video_path**: Output path for the video file
- **index_path**: Output path for the index file
- **codec**: Video codec to use ('h265' or 'h264', default: 'h265')
- **show_progress**: Whether to show progress during build (default: True)
- **auto_build_docker**: Whether to auto-build docker if needed (default: True)
- **allow_fallback**: Whether to allow fallback options (default: True)

### `search_memory`
Perform semantic search on the built video memory.
- **query**: Natural language search query
- **top_k**: Number of results to return (default: 5)

### `chat_with_memvid`
Have a conversation with your encoded knowledge base.
- **message**: Message to send to the chat system

## 📖 Usage Workflow

1. **Add Content**: Use `add_text`, `add_chunks`, or `add_pdf` to add your data
2. **Build Video**: Use `build_video` to create the video memory representation
3. **Search or Chat**: Use `search_memory` for queries or `chat_with_memvid` for conversations

## 🔧 Development

### Testing
```bash
# Install development dependencies
uv add --dev pytest pytest-asyncio black ruff mypy

# Run tests
uv run pytest

# Format code
uv run black memvid_mcp_server/
uv run ruff check memvid_mcp_server/
```

### Debugging
- Check logs in Claude Desktop: `~/Library/Logs/Claude/mcp*.log` (macOS) or equivalent
- Enable debug logging by setting `LOG_LEVEL=DEBUG` in environment
- Use `get_server_status` tool to check server state

## 🔧 Troubleshooting

### Common Issues

1. **JSON Parsing Errors**: All output is properly redirected to stderr to prevent protocol interference
2. **Import Errors**: The server gracefully handles missing memvid package with clear error messages
3. **Connection Issues**: Check Claude Desktop logs and use `get_server_status` to diagnose issues
4. **Video Build Failures**: Ensure sufficient disk space and valid paths

### Logging Configuration

The server implements comprehensive stdout redirection to prevent any library output from interfering with the MCP JSON-RPC protocol:

- All memvid operations are wrapped with stdout redirection
- Progress bars, warnings, and model loading messages are captured
- Only structured JSON responses are sent to Claude Desktop
- All diagnostic information is logged to stderr

### Error Messages

- **"Memvid not available"**: Install the memvid package: `uv add memvid`
- **"Video memory not built"**: Run `build_video` before searching or chatting
- **"LLM not available"**: Expected warning - memvid will work without external LLM providers

## 📄 License

MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Related Projects

- [Memvid](https://github.com/tomayac/memvid) - The underlying video memory technology
- [Model Context Protocol](https://modelcontextprotocol.io/) - The protocol specification
- [Claude Desktop](https://claude.ai/download) - MCP-compatible AI client

---

Generated with improvements for production reliability and MCP best practices.
