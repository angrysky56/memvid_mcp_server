"""Basic tests for memvid MCP server."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memvid_mcp_server.main import ServerState, _check_memvid_available


class TestServerState:
    """Test ServerState functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test server state initialization."""
        state = ServerState()
        assert not state.initialized
        assert state.encoder is None
        assert state.retriever is None
        assert state.chat is None

        await state.initialize()
        assert state.initialized

        await state.cleanup()
        assert not state.initialized


class TestMemvidAvailability:
    """Test memvid availability checking."""

    def test_memvid_availability(self):
        """Test memvid availability check."""
        # This will depend on whether memvid is actually installed
        result = _check_memvid_available()
        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_server_startup():
    """Test that the server can start up without errors."""
    from memvid_mcp_server.main import mcp
    
    # Basic smoke test - ensure the server object exists and has tools
    assert mcp is not None
    assert hasattr(mcp, '_tools')
    assert len(mcp._tools) > 0


if __name__ == "__main__":
    pytest.main([__file__])
