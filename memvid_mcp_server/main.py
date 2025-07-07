"""
Memvid MCP Server - An MCP server to expose Memvid functionalities to AI clients.

A FastMCP server that provides video memory encoding, searching, and chat capabilities.
Includes Docker lifecycle management for automatic container setup and teardown.
"""

import asyncio
import atexit
import glob
import logging
import os
import signal
import sys
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stdout

# from io import StringIO
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP

# Import Docker lifecycle manager
try:
    from .docker_lifecycle import DockerLifecycleManager
except ImportError:
    # Fallback for direct script execution
    from docker_lifecycle import DockerLifecycleManager

# Suppress ALL warnings to prevent JSON parsing errors
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Redirect stdout for the entire process to prevent any library output
class StderrRedirect:
    def write(self, s):
        sys.stderr.write(s)
    def flush(self):
        sys.stderr.flush()

# Store original stdout
_original_stdout = sys.stdout

# Import memvid components with stdout redirected
try:
    with redirect_stdout(sys.stderr):
        from memvid import MemvidChat, MemvidEncoder, MemvidRetriever
except ImportError as e:
    # Log import error to stderr and continue
    print(f"Warning: Could not import memvid: {e}", file=sys.stderr)
    MemvidEncoder = None
    MemvidRetriever = None
    MemvidChat = None

# Configure logging to stderr for MCP compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class ServerState:
    """Centralized state management for the memvid MCP server."""

    def __init__(self):
        self.connections: dict[str, Any] = {}
        self.initialized = False
        self.encoder: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.chat: Optional[Any] = None
        self.docker_manager = DockerLifecycleManager()
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize server resources."""
        if self.initialized:
            return

        try:
            logger.info("Initializing memvid MCP server with Docker lifecycle management")

            # Initialize Docker environment first
            docker_success = await self.docker_manager.initialize()
            if docker_success:
                logger.info("✅ Docker environment ready - memvid package will handle containers")
            else:
                logger.warning("⚠️ Docker initialization failed, memvid will use native mode")

            # Check if memvid is available
            if MemvidEncoder is None:
                logger.warning("Memvid not available - tools will return error messages")
            else:
                # Initialize encoder only when needed to avoid startup errors
                logger.info("Memvid available - ready to create encoder on demand")

            self.initialized = True
            logger.info("Server initialization completed")

        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        if not self.initialized:
            return

        logger.info("Cleaning up memvid MCP server")

        try:
            # Clean up Docker resources first
            await self.docker_manager.cleanup()

            # Close connections
            for conn_id, conn in self.connections.items():
                try:
                    if hasattr(conn, 'close'):
                        if asyncio.iscoroutinefunction(conn.close):
                            await conn.close()
                        else:
                            conn.close()
                except Exception as e:
                    logger.warning(f"Connection cleanup failed for {conn_id}: {e}")

            # Clear state
            self.connections.clear()
            self.encoder = None
            self.retriever = None
            self.chat = None
            self.initialized = False

            logger.info("Server cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global state instance
_server_state = ServerState()


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerState]:
    """Manage server lifecycle."""
    try:
        await _server_state.initialize()
        yield _server_state
    except Exception as e:
        logger.error(f"Lifespan error: {e}")
        raise
    finally:
        await _server_state.cleanup()


# Initialize FastMCP with proper lifespan
mcp = FastMCP("memvid-mcp-server", lifespan=lifespan)


def _check_memvid_available() -> bool:
    """Check if memvid is available."""
    return MemvidEncoder is not None


def _ensure_encoder() -> Any:
    """Ensure encoder is initialized."""
    if not _check_memvid_available():
        raise RuntimeError("Memvid not available - please install memvid package")

    if _server_state.encoder is None:
        if MemvidEncoder is None:
            raise RuntimeError("MemvidEncoder is not available. Please install the memvid package.")
        # Create encoder with stdout redirected to prevent progress bars
        with redirect_stdout(sys.stderr):
            _server_state.encoder = MemvidEncoder()
        logger.info("Created new MemvidEncoder instance")

    return _server_state.encoder


@mcp.tool()
async def add_chunks(ctx: Context, chunks: list[str]) -> dict[str, Any]:
    """Add text chunks to the Memvid encoder.

    WORKFLOW REQUIREMENT: After adding all content, you MUST call build_video()
    before search_memory() or chat_with_memvid() will work.

    Args:
        chunks: A list of text chunks to add to the encoder.

    Returns:
        Status dictionary with success/error information.

    Note: This only stages content. Call build_video() to complete the workflow.
    """
    try:
        encoder = _ensure_encoder()
        # Redirect stdout during add_chunks operation
        with redirect_stdout(sys.stderr):
            encoder.add_chunks(chunks)
        logger.info(f"Successfully added {len(chunks)} chunks to encoder")
        return {"status": "success", "chunks_added": len(chunks)}
    except Exception as e:
        logger.error(f"Failed to add chunks: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def add_text(ctx: Context, text: str, metadata: Optional[dict] = None) -> dict[str, Any]:
    """Add text to the Memvid encoder.

    WORKFLOW REQUIREMENT: After adding all content, you MUST call build_video()
    before search_memory() or chat_with_memvid() will work.

    Args:
        text: The text content to add.
        metadata: Optional metadata for the text.

    Returns:
        Status dictionary with success/error information.

    Note: This only stages content. Call build_video() to complete the workflow.
    """
    try:
        encoder = _ensure_encoder()
        # Redirect stdout during add_text operation
        with redirect_stdout(sys.stderr):
            encoder.add_text(text)
        logger.info("Successfully added text to encoder")
        return {"status": "success", "text_length": len(text)}
    except Exception as e:
        logger.error(f"Failed to add text: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def add_pdf(ctx: Context, pdf_path: str) -> dict[str, Any]:
    """Add a PDF file to the Memvid encoder.

    WORKFLOW REQUIREMENT: After adding all content, you MUST call build_video()
    before search_memory() or chat_with_memvid() will work.

    Args:
        pdf_path: The path to the PDF file to add.

    Returns:
        Status dictionary with success/error information.

    Note: This only stages content. Call build_video() to complete the workflow.
    """
    try:
        encoder = _ensure_encoder()
        # Redirect stdout during add_pdf operation
        with redirect_stdout(sys.stderr):
            encoder.add_pdf(pdf_path)
        logger.info(f"Successfully added PDF: {pdf_path}")
        return {"status": "success", "pdf_path": pdf_path}
    except Exception as e:
        logger.error(f"Failed to add PDF {pdf_path}: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def build_video(
    ctx: Context,
    video_path: str,
    index_path: str,
    codec: str = 'h265',
    show_progress: bool = True,
    auto_build_docker: bool = True,
    allow_fallback: bool = True
) -> dict[str, Any]:
    """Build the video memory from the added chunks.

    Args:
        video_path: The path to save the video file.
        index_path: The path to save the index file.
        codec: Video codec to use ('h265' or 'h264', default: 'h265').
        show_progress: Whether to show progress during build (default: True).
        auto_build_docker: Whether to auto-build docker if needed (default: True).
        allow_fallback: Whether to allow fallback options (default: True).

    Returns:
        Status dictionary with success/error information.
    """
    try:
        encoder = _ensure_encoder()

        # Call build_video with stdout redirected to prevent progress bars
        with redirect_stdout(sys.stderr):
            result = encoder.build_video(
                output_file=video_path,
                index_file=index_path,
                codec=codec,
                show_progress=show_progress,
                auto_build_docker=auto_build_docker,
                allow_fallback=allow_fallback
            )

        # Initialize retriever and chat with stdout redirected
        if MemvidRetriever and MemvidChat:
            with redirect_stdout(sys.stderr):
                _server_state.retriever = MemvidRetriever(video_path, index_path)
                # Custom config with higher max_tokens for longer responses
                chat_config = {
                    "llm": {
                        "max_tokens": 8192,
                        "temperature": 0.1,
                        "context_window": 32000,
                    },
                    "chat": {
                        "max_history": 10,
                        "context_chunks": 8,  # More context chunks
                    }
                }
                _server_state.chat = MemvidChat(
                    video_path,
                    index_path,
                    llm_provider='google',  # Back to Google as requested
                    llm_model='gemini-2.0-flash-exp',
                    config=chat_config
                )

        logger.info(f"Successfully built video: {video_path}")
        return {
            "status": "success",
            "video_path": video_path,
            "index_path": index_path,
            "codec": codec,
            "build_result": result
        }
    except Exception as e:
        logger.error(f"Failed to build video: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def search_memory(ctx: Context, query: str, top_k: int = 5) -> dict[str, Any]:
    """Perform a semantic search on the video memory.

    Args:
        query: The natural language query to search for.
        top_k: The number of top results to retrieve.

    Returns:
        Status dictionary with search results or error information.
    """
    try:
        if not _server_state.retriever:
            return {
                "status": "error",
                "message": "Video memory not built. Call build_video first."
            }

        # Perform search with stdout redirected to prevent embedding model output
        with redirect_stdout(sys.stderr):
            results = _server_state.retriever.search(query, top_k=top_k)
        logger.info(f"Search completed for query: '{query}' with {len(results)} results")
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def chat_with_memvid(ctx: Context, message: str) -> dict[str, Any]:
    """Chat with the Memvid memory.

    Args:
        message: The message to send to the chat system.

    Returns:
        Status dictionary with chat response or error information.
    """
    try:
        if not _server_state.chat:
            return {
                "status": "error",
                "message": "Video memory not built. Call build_video first."
            }

        # Perform chat with stdout redirected to prevent LLM library output
        with redirect_stdout(sys.stderr):
            # Check if LLM is available, if not, provide full context without truncation
            if not _server_state.chat.llm_client:
                # Get full context without LLM truncation
                context_chunks = _server_state.chat.search_context(message, top_k=8)
                if context_chunks:
                    response = "Based on the knowledge base, here's what I found:\n\n"
                    for i, chunk in enumerate(context_chunks):
                        response += f"{i+1}. {chunk}\n\n"
                    response = response.strip()
                else:
                    response = "I couldn't find any relevant information in the knowledge base."
            else:
                response = _server_state.chat.chat(
                    message,
                    max_context_tokens=400000  # Increase context for better responses
                )
        logger.info(f"Chat completed for message: '{message[:8164]}...'")
        return {"status": "success", "message": message, "response": response}
    except Exception as e:
        logger.error(f"Chat failed for message '{message}': {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def list_video_memories(ctx: Context, directory: str = ".") -> dict[str, Any]:
    """List available video memory files in a directory.

    Searches for .mp4/.json pairs that represent complete video memories.

    Args:
        directory: Directory to search for video memory files (default: current directory)

    Returns:
        Dictionary with list of discovered video memory files and their info.
    """
    try:
        import glob
        import os

        # Find all .mp4 files in directory
        mp4_pattern = os.path.join(directory, "*.mp4")
        mp4_files = glob.glob(mp4_pattern)

        video_memories = []
        for mp4_file in mp4_files:
            base_name = os.path.splitext(mp4_file)[0]
            json_file = f"{base_name}.json"

            if os.path.exists(json_file):
                try:
                    stat_mp4 = os.stat(mp4_file)
                    stat_json = os.stat(json_file)

                    video_memories.append({
                        "name": os.path.basename(base_name),
                        "video_path": mp4_file,
                        "index_path": json_file,
                        "video_size_mb": round(stat_mp4.st_size / (1024*1024), 2),
                        "index_size_kb": round(stat_json.st_size / 1024, 2),
                        "created": stat_mp4.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Could not stat files for {base_name}: {e}")

        video_memories.sort(key=lambda x: x["created"], reverse=True)

        return {
            "status": "success",
            "directory": directory,
            "video_memories": video_memories,
            "count": len(video_memories)
        }

    except Exception as e:
        logger.error(f"Failed to list video memories in {directory}: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def load_video_memory(ctx: Context, video_path: str, index_path: str) -> dict[str, Any]:
    """Load an existing video memory for search and chat.

    This allows switching to a different video memory without rebuilding.

    Args:
        video_path: Path to existing .mp4 video memory file
        index_path: Path to existing .json index file

    Returns:
        Status dictionary indicating success/failure of loading operation.
    """
    try:
        import os

        # Verify files exist
        if not os.path.exists(video_path):
            return {"status": "error", "message": f"Video file not found: {video_path}"}
        if not os.path.exists(index_path):
            return {"status": "error", "message": f"Index file not found: {index_path}"}

        if not _check_memvid_available():
            return {"status": "error", "message": "Memvid not available - please install memvid package"}

        # Check if MemvidRetriever and MemvidChat are available
        if MemvidRetriever is None or MemvidChat is None:
            return {"status": "error", "message": "MemvidRetriever or MemvidChat is not available. Please ensure the memvid package is installed correctly."}

        # Load retriever and chat with existing video memory
        with redirect_stdout(sys.stderr):
            _server_state.retriever = MemvidRetriever(video_path, index_path)
            # Custom config with higher max_tokens for longer responses
            chat_config = {
                "llm": {
                    "max_tokens": 8192,
                    "temperature": 0.1,
                    "context_window": 32000,
                },
                "chat": {
                    "max_history": 10,
                    "context_chunks": 8,  # More context chunks
                }
            }
            _server_state.chat = MemvidChat(
                video_path,
                index_path,
                llm_provider='google',  # Back to Google as requested
                llm_model='gemini-2.0-flash-exp',
                config=chat_config
            )

        logger.info(f"Successfully loaded video memory: {video_path}")
        return {
            "status": "success",
            "video_path": video_path,
            "index_path": index_path,
            "message": f"Loaded video memory: {os.path.basename(video_path)}"
        }

    except Exception as e:
        logger.error(f"Failed to load video memory {video_path}: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_current_video_info(ctx: Context) -> dict[str, Any]:
    """Get information about the currently loaded video memory.

    Returns details about the active video memory including paths and stats.

    Returns:
        Dictionary with current video memory information or error if none loaded.
    """
    try:
        if not _server_state.retriever or not _server_state.chat:
            return {
                "status": "info",
                "message": "No video memory currently loaded. Use build_video or load_video_memory first."
            }

        # Try to get basic info from the retriever/chat objects
        info = {
            "status": "success",
            "retriever_ready": _server_state.retriever is not None,
            "chat_ready": _server_state.chat is not None,
        }

        # Try to extract file paths if available in the objects
        try:
            if hasattr(_server_state.retriever, 'video_path'):
                info["video_path"] = _server_state.retriever.video_path
            if hasattr(_server_state.retriever, 'index_path'):
                info["index_path"] = _server_state.retriever.index_path

            # Get file sizes if paths are available
            if "video_path" in info and "index_path" in info:
                import os
                if os.path.exists(info["video_path"]):
                    stat_mp4 = os.stat(info["video_path"])
                    info["video_size_mb"] = round(stat_mp4.st_size / (1024*1024), 2)
                if os.path.exists(info["index_path"]):
                    stat_json = os.stat(info["index_path"])
                    info["index_size_kb"] = round(stat_json.st_size / 1024, 2)

        except Exception as e:
            logger.warning(f"Could not extract detailed info: {e}")
            info["message"] = "Video memory loaded but could not extract detailed file info"

        return info

    except Exception as e:
        logger.error(f"Failed to get current video info: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_server_status(ctx: Context) -> dict[str, Any]:
    """Get the current status of the memvid server.

    Returns:
        Server status information including version details and Docker status.
    """
    try:
        # Check memvid version if available
        memvid_version = "unknown"
        if _check_memvid_available():
            try:
                import memvid
                memvid_version = getattr(memvid, '__version__', 'no version info')
            except Exception:
                memvid_version = "version check failed"

        # Get Docker status
        docker_status = _server_state.docker_manager.get_status()

        status = {
            "status": "success",
            "server_initialized": _server_state.initialized,
            "memvid_available": _check_memvid_available(),
            "memvid_version": memvid_version,
            "encoder_ready": _server_state.encoder is not None,
            "retriever_ready": _server_state.retriever is not None,
            "chat_ready": _server_state.chat is not None,
            "active_connections": len(_server_state.connections),
            "docker_status": docker_status
        }
        return status
    except Exception as e:
        logger.error(f"Failed to get server status: {e}")
        return {"status": "error", "message": str(e)}


# Signal handling for graceful shutdown
def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, initiating shutdown")
        _server_state._shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def cleanup_handler() -> None:
    """Cleanup handler for atexit."""
    if _server_state.initialized:
        logger.info("Running cleanup handler")
        try:
            # Run cleanup in a new event loop if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_server_state.cleanup())
            loop.close()
        except Exception as e:
            logger.error(f"Cleanup handler error: {e}")


# Register cleanup handler
atexit.register(cleanup_handler)


if __name__ == "__main__":
    # Set up signal handlers
    setup_signal_handlers()

    try:
        logger.info("Starting memvid MCP server")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)
