"""
Memvid MCP Server - An MCP server to expose Memvid functionalities to AI clients.

A FastMCP server that provides video memory encoding, searching, and chat capabilities.
Includes Docker lifecycle management for automatic container setup and teardown.
"""

import asyncio
import atexit
import glob
import json
import logging
import os
import signal
import sys
import warnings
from contextlib import redirect_stdout

# from io import StringIO
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP

# Suppress ALL warnings to prevent JSON parsing errors BEFORE importing memvid
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Redirect stdout for the entire process to prevent any library output
class StderrRedirect:
    def __init__(self):
        self.buffer = sys.stderr.buffer if hasattr(sys.stderr, 'buffer') else None

    def write(self, s):
        sys.stderr.write(s)

    def flush(self):
        sys.stderr.flush()

    def __getattr__(self, name):
        # Delegate any other attributes to stderr
        return getattr(sys.stderr, name)

# Store original stdout for MCP communication
_original_stdout = sys.stdout

# DON'T redirect stdout globally - this breaks MCP communication!
# Instead, we'll redirect only during specific memvid operations

# Import Docker lifecycle manager
try:
    from .docker_lifecycle import DockerLifecycleManager
except ImportError:
    # Fallback for direct script execution
    from docker_lifecycle import DockerLifecycleManager

# Configure logging to stderr for MCP compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Import memvid - using redirect_stdout only when needed to preserve MCP communication
try:
    from memvid import MemvidChat, MemvidEncoder, MemvidRetriever  # noqa: E402
except ImportError as e:
    logger.warning(f"Failed to import memvid: {e}")
    MemvidChat = MemvidEncoder = MemvidRetriever = None


def _detect_and_configure_faiss() -> str:
    """Detect and configure the best available FAISS implementation."""
    faiss_type = "none"

    try:
        # Try GPU FAISS first
        import faiss
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            # Test if GPU is actually available and working
            try:
                # Create a small test index to verify GPU functionality
                test_index = faiss.IndexFlatL2(128)
                gpu_res = faiss.StandardGpuResources()
                faiss.index_cpu_to_gpu(gpu_res, 0, test_index)
                faiss_type = "gpu"
                logger.info(f"✅ GPU FAISS detected and working - {faiss.get_num_gpus()} GPU(s) available")
            except Exception as e:
                logger.warning(f"GPU FAISS detected but not functional: {e}, falling back to CPU")
                faiss_type = "cpu"
        else:
            faiss_type = "cpu"
            logger.info("🖥️ Using CPU FAISS (no GPU detected or GPU FAISS not available)")

    except ImportError:
        logger.warning("⚠️ FAISS not available - vector search may not work")
        faiss_type = "none"

    return faiss_type


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
        # Track currently loaded memory
        self.current_video_path: Optional[str] = None
        self.current_index_path: Optional[str] = None
        # Memory library state
        self.available_memories: dict[str, dict] = {}
        # FAISS configuration
        self.faiss_type: str = "none"
        # Library directory path (where the server is installed)
        self.library_dir = self._detect_library_directory()

    def _detect_library_directory(self) -> str:
        """Detect the library directory relative to the server installation."""
        try:
            # Get the directory containing this script
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            library_dir = os.path.join(script_dir, "library")
            
            # Create library directory if it doesn't exist
            os.makedirs(library_dir, exist_ok=True)
            
            logger.info(f"Using library directory: {library_dir}")
            return library_dir
        except Exception as e:
            logger.error(f"Failed to detect library directory: {e}")
            # Fallback to current directory
            fallback_dir = os.path.join(os.getcwd(), "library")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir

    def resolve_file_path(self, path: str, file_type: str = "video") -> str:
        """Resolve file path, using library directory for relative paths."""
        if os.path.isabs(path):
            # Absolute path - use as is
            return path
        else:
            # Relative path - resolve to library directory
            resolved_path = os.path.join(self.library_dir, path)
            logger.info(f"Resolved relative path '{path}' to '{resolved_path}'")
            return resolved_path

    async def initialize(self) -> None:
        """Initialize server resources."""
        if self.initialized:
            return

        try:
            logger.info("Initializing memvid MCP server with Docker lifecycle management")

            # Detect and configure FAISS
            faiss_type = _detect_and_configure_faiss()
            self.faiss_type = faiss_type

            # Initialize Docker environment first
            with redirect_stdout(sys.stderr):
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

    def update_memory_library(self, video_path: str, index_path: str) -> None:
        """Update the memory library with a new or loaded memory."""
        try:
            name = os.path.basename(os.path.splitext(video_path)[0])
            self.available_memories[name] = {
                "video_path": video_path,
                "index_path": index_path,
                "loaded": True
            }
            logger.info(f"Updated memory library with: {name}")
        except Exception as e:
            logger.warning(f"Failed to update memory library: {e}")

    def set_active_memory(self, video_path: str, index_path: str) -> None:
        """Set the currently active memory and update library."""
        self.current_video_path = video_path
        self.current_index_path = index_path
        self.update_memory_library(video_path, index_path)

    async def cleanup(self) -> None:
        """Clean up server resources."""
        if not self.initialized:
            return

        logger.info("Cleaning up memvid MCP server")

        try:
            # Clean up Docker resources first
            try:
                with redirect_stdout(sys.stderr):
                    await self.docker_manager.cleanup()
            except Exception as e:
                logger.warning(f"Docker cleanup failed: {e}")

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
            self.current_video_path = None
            self.current_index_path = None
            self.available_memories.clear()
            self.initialized = False

            logger.info("Server cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global state instance
_server_state = ServerState()


# Initialize FastMCP without lifespan - simpler approach
mcp = FastMCP("memvid-mcp-server")


def _check_memvid_available() -> bool:
    """Check if memvid is available."""
    return MemvidEncoder is not None


async def _ensure_server_initialized() -> None:
    """Ensure server is initialized before processing tools."""
    if not _server_state.initialized:
        try:
            await _server_state.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise RuntimeError(f"Server initialization failed: {e}")


async def _ensure_encoder() -> Any:
    """Ensure encoder is initialized."""
    await _ensure_server_initialized()

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


async def _rebuild_memory_from_json(video_path: str, index_path: str) -> bool:
    """Rebuild a memory from its JSON metadata when the video is corrupted/incompatible."""
    try:
        # Resolve paths (in case relative paths were passed)
        resolved_video_path = _server_state.resolve_file_path(video_path, "video")
        resolved_index_path = _server_state.resolve_file_path(index_path, "index")
        
        logger.info(f"Attempting to rebuild memory from {resolved_index_path}")

        # Read the JSON metadata
        with open(resolved_index_path, 'r') as f:
            metadata = json.load(f)

        if 'metadata' not in metadata:
            logger.error("Invalid JSON format - missing metadata")
            return False

        # Extract all text chunks
        chunks = []
        for item in metadata['metadata']:
            if 'text' in item:
                chunks.append(item['text'])

        if not chunks:
            logger.error("No text chunks found in metadata")
            return False

        logger.info(f"Found {len(chunks)} chunks to rebuild")

        # Initialize encoder and add chunks
        try:
            encoder = await _ensure_encoder()
        except RuntimeError as e:
            logger.error(f"Cannot rebuild - encoder not available: {e}")
            return False

        with redirect_stdout(sys.stderr):
            encoder.add_chunks(chunks)

        # Rebuild the video
        with redirect_stdout(sys.stderr):
            encoder.build_video(
                output_file=resolved_video_path,
                index_file=resolved_index_path,
                codec='h265',
                show_progress=False,  # Silent rebuild
                auto_build_docker=True,
                allow_fallback=True
            )

        logger.info(f"Successfully rebuilt memory: {resolved_video_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to rebuild memory: {e}")
        return False


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
        encoder = await _ensure_encoder()
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
        encoder = await _ensure_encoder()
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
        encoder = await _ensure_encoder()
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
        video_path: The path to save the video file (relative paths use library directory).
        index_path: The path to save the index file (relative paths use library directory).
        codec: Video codec to use ('h265' or 'h264', default: 'h265').
        show_progress: Whether to show progress during build (default: True).
        auto_build_docker: Whether to auto-build docker if needed (default: True).
        allow_fallback: Whether to allow fallback options (default: True).

    Returns:
        Status dictionary with success/error information.
    """
    try:
        encoder = await _ensure_encoder()

        # Resolve paths using library directory for relative paths
        resolved_video_path = _server_state.resolve_file_path(video_path, "video")
        resolved_index_path = _server_state.resolve_file_path(index_path, "index")

        # Call build_video with stdout redirected to prevent progress bars
        with redirect_stdout(sys.stderr):
            build_result = encoder.build_video(
                output_file=resolved_video_path,
                index_file=resolved_index_path,
                codec=codec,
                show_progress=show_progress,
                auto_build_docker=auto_build_docker,
                allow_fallback=allow_fallback
            )

        # Initialize retriever and chat with stdout redirected
        if MemvidRetriever and MemvidChat:
            try:
                # Check if MemvidRetriever and MemvidChat are available
                if MemvidRetriever is None or MemvidChat is None:
                    logger.error("MemvidRetriever or MemvidChat is not available")
                    return {
                        "status": "partial_success",
                        "video_path": resolved_video_path,
                        "index_path": resolved_index_path,
                        "codec": codec,
                        "build_result": build_result,
                        "warning": "Video built but MemvidRetriever/Chat not available for initialization"
                    }

                with redirect_stdout(sys.stderr):
                    _server_state.retriever = MemvidRetriever(resolved_video_path, resolved_index_path)
                    _server_state.chat = MemvidChat(resolved_video_path, resolved_index_path)

                # Update state tracking
                _server_state.set_active_memory(resolved_video_path, resolved_index_path)

                logger.info(f"Successfully built and loaded video memory: {resolved_video_path}")
            except Exception as e:
                logger.error(f"Failed to initialize retriever/chat after build: {e}")
                # Build succeeded but initialization failed
                return {
                    "status": "partial_success",
                    "video_path": resolved_video_path,
                    "index_path": resolved_index_path,
                    "codec": codec,
                    "build_result": build_result,
                    "warning": f"Video built but failed to initialize retriever/chat: {str(e)}"
                }

        return {
            "status": "success",
            "video_path": resolved_video_path,
            "index_path": resolved_index_path,
            "codec": codec,
            "build_result": build_result,
            "message": "Video memory built and loaded successfully"
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
                "message": "No video memory loaded. Call build_video or load_video_memory first."
            }

        # Perform search with stdout redirected to prevent embedding model output
        with redirect_stdout(sys.stderr):
            results = _server_state.retriever.search(query, top_k=top_k)

        if results is None:
            return {
                "status": "error",
                "message": "Search returned None - retriever may be corrupted. Try loading or rebuilding the memory."
            }

        logger.info(f"Search completed for query: '{query}' with {len(results)} results")

        # Include current memory info in response
        current_memory = "Unknown"
        if _server_state.current_video_path:
            current_memory = os.path.basename(os.path.splitext(_server_state.current_video_path)[0])

        return {
            "status": "success",
            "query": query,
            "results": results,
            "current_memory": current_memory,
            "result_count": len(results)
        }
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
                "message": "No video memory loaded. Call build_video or load_video_memory first."
            }

        # Perform chat with stdout redirected to prevent LLM library output
        with redirect_stdout(sys.stderr):
            response = _server_state.chat.chat(message)

        if response is None:
            return {
                "status": "error",
                "message": "Chat returned None - chat system may be corrupted. Try loading or rebuilding the memory."
            }

        logger.info(f"Chat completed for message: '{message[:8164]}...'")

        # Include current memory info in response
        current_memory = "Unknown"
        if _server_state.current_video_path:
            current_memory = os.path.basename(os.path.splitext(_server_state.current_video_path)[0])

        return {
            "status": "success",
            "message": message,
            "response": response,
            "current_memory": current_memory
        }
    except Exception as e:
        logger.error(f"Chat failed for message '{message}': {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def list_video_memories(ctx: Context, directory: str = None) -> dict[str, Any]:
    """List available video memory files in a directory.

    Searches for .mp4/.json pairs that represent complete video memories.
    Also updates the server's memory library state.

    Args:
        directory: Directory to search for video memory files (defaults to server library directory)

    Returns:
        Dictionary with list of discovered video memory files and their info.
    """
    try:
        # Use library directory as default
        if directory is None:
            directory = _server_state.library_dir
        
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

                    memory_info = {
                        "name": os.path.basename(base_name),
                        "video_path": mp4_file,
                        "index_path": json_file,
                        "video_size_mb": round(stat_mp4.st_size / (1024*1024), 2),
                        "index_size_kb": round(stat_json.st_size / 1024, 2),
                        "created": stat_mp4.st_mtime,
                        "is_current": mp4_file == _server_state.current_video_path
                    }

                    video_memories.append(memory_info)

                    # Update memory library (but don't mark as loaded unless it's current)
                    name = memory_info["name"]
                    if name not in _server_state.available_memories:
                        _server_state.available_memories[name] = {
                            "video_path": mp4_file,
                            "index_path": json_file,
                            "loaded": memory_info["is_current"]
                        }

                except Exception as e:
                    logger.warning(f"Could not stat files for {base_name}: {e}")

        video_memories.sort(key=lambda x: x["created"], reverse=True)

        return {
            "status": "success",
            "directory": directory,
            "video_memories": video_memories,
            "count": len(video_memories),
            "current_memory": _server_state.current_video_path
        }

    except Exception as e:
        logger.error(f"Failed to list video memories in {directory}: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def load_video_memory(ctx: Context, video_path: str, index_path: str) -> dict[str, Any]:
    """Load an existing video memory for search and chat.

    This allows switching to a different video memory without rebuilding.
    If the memory fails to load, it will attempt to auto-rebuild from the JSON metadata.

    Args:
        video_path: Path to existing .mp4 video memory file (relative paths use library directory)
        index_path: Path to existing .json index file (relative paths use library directory)

    Returns:
        Status dictionary indicating success/failure of loading operation.
    """
    try:
        # Resolve paths using library directory for relative paths
        resolved_video_path = _server_state.resolve_file_path(video_path, "video")
        resolved_index_path = _server_state.resolve_file_path(index_path, "index")
        
        # Verify files exist
        if not os.path.exists(resolved_video_path):
            return {"status": "error", "message": f"Video file not found: {resolved_video_path}"}
        if not os.path.exists(resolved_index_path):
            return {"status": "error", "message": f"Index file not found: {resolved_index_path}"}

        if not _check_memvid_available():
            return {"status": "error", "message": "Memvid not available - please install memvid package"}

        # Attempt to load the memory directly first
        def try_load_memory():
            """Try to load the memory and return success status."""
            try:
                logger.info(f"Attempting to load retriever from {resolved_video_path}, {resolved_index_path}")
                logger.info(f"File sizes - Video: {os.path.getsize(resolved_video_path)} bytes, Index: {os.path.getsize(resolved_index_path)} bytes")

                # Check if MemvidRetriever and MemvidChat are available
                if MemvidRetriever is None or MemvidChat is None:
                    raise RuntimeError("MemvidRetriever or MemvidChat is not available. Please ensure the memvid package is installed correctly.")

                # Try initializing retriever first
                with redirect_stdout(sys.stderr):
                    retriever = MemvidRetriever(resolved_video_path, resolved_index_path)
                    chat = MemvidChat(resolved_video_path, resolved_index_path)

                # Test if retriever actually works
                logger.info("Testing retriever functionality...")
                with redirect_stdout(sys.stderr):
                    test_results = retriever.search("test", top_k=1)

                if test_results is None:
                    raise Exception("Retriever returned None for test search")

                logger.info(f"Test search returned: {len(test_results)} results")

                # If we get here, everything works
                _server_state.retriever = retriever
                _server_state.chat = chat
                _server_state.set_active_memory(resolved_video_path, resolved_index_path)

                return True, "Memory loaded successfully"

            except Exception as e:
                logger.error(f"Direct load failed: {e}")
                return False, str(e)

        # First attempt: try to load directly
        success, message = try_load_memory()

        if success:
            logger.info(f"Successfully loaded video memory: {resolved_video_path}")
            return {
                "status": "success",
                "video_path": resolved_video_path,
                "index_path": resolved_index_path,
                "message": f"Loaded video memory: {os.path.basename(resolved_video_path)}"
            }

        # If direct load failed, attempt auto-rebuild
        logger.info(f"Direct load failed ({message}), attempting auto-rebuild...")

        rebuild_success = await _rebuild_memory_from_json(resolved_video_path, resolved_index_path)
        if not rebuild_success:
            return {
                "status": "error",
                "message": f"Failed to load memory and auto-rebuild failed. Original error: {message}"
            }

        # Try loading again after rebuild
        success, message = try_load_memory()

        if success:
            logger.info(f"Successfully loaded video memory after rebuild: {resolved_video_path}")
            return {
                "status": "success",
                "video_path": resolved_video_path,
                "index_path": resolved_index_path,
                "message": f"Auto-rebuilt and loaded video memory: {os.path.basename(resolved_video_path)}",
                "rebuilt": True
            }
        else:
            return {
                "status": "error",
                "message": f"Memory rebuilt but still failed to load: {message}"
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
                "message": "No video memory currently loaded. Use build_video or load_video_memory first.",
                "retriever_ready": _server_state.retriever is not None,
                "chat_ready": _server_state.chat is not None,
                "available_memories": list(_server_state.available_memories.keys())
            }

        # Get basic info from state tracking
        info = {
            "status": "success",
            "retriever_ready": _server_state.retriever is not None,
            "chat_ready": _server_state.chat is not None,
            "video_path": _server_state.current_video_path,
            "index_path": _server_state.current_index_path,
        }

        # Add memory name
        if _server_state.current_video_path:
            info["memory_name"] = os.path.basename(os.path.splitext(_server_state.current_video_path)[0])

        # Get file sizes if paths are available
        if info["video_path"] and info["index_path"]:
            try:
                if os.path.exists(info["video_path"]):
                    stat_mp4 = os.stat(info["video_path"])
                    info["video_size_mb"] = round(stat_mp4.st_size / (1024*1024), 2)
                    info["video_created"] = stat_mp4.st_mtime
                if os.path.exists(info["index_path"]):
                    stat_json = os.stat(info["index_path"])
                    info["index_size_kb"] = round(stat_json.st_size / 1024, 2)
            except Exception as e:
                logger.warning(f"Could not get file stats: {e}")
                info["warning"] = "Could not get file statistics"

        # Include memory library info
        info["available_memories"] = list(_server_state.available_memories.keys())
        info["total_memories"] = len(_server_state.available_memories)

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
        await _ensure_server_initialized()

        # Check memvid version if available
        memvid_version = "unknown"
        if _check_memvid_available():
            try:
                with redirect_stdout(sys.stderr):
                    import memvid
                    memvid_version = getattr(memvid, '__version__', 'no version info')
            except Exception:
                memvid_version = "version check failed"

        # Get Docker status
        docker_status = _server_state.docker_manager.get_status()

        # Current memory info
        current_memory_name = "None"
        if _server_state.current_video_path:
            current_memory_name = os.path.basename(os.path.splitext(_server_state.current_video_path)[0])

        status = {
            "status": "success",
            "server_initialized": _server_state.initialized,
            "memvid_available": _check_memvid_available(),
            "memvid_version": memvid_version,
            "faiss_type": _server_state.faiss_type,
            "encoder_ready": _server_state.encoder is not None,
            "retriever_ready": _server_state.retriever is not None,
            "chat_ready": _server_state.chat is not None,
            "current_memory": current_memory_name,
            "current_video_path": _server_state.current_video_path,
            "current_index_path": _server_state.current_index_path,
            "available_memories": len(_server_state.available_memories),
            "memory_library": list(_server_state.available_memories.keys()),
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
            # Run cleanup in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_server_state.cleanup())
            finally:
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
