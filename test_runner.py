#!/usr/bin/env python3
"""
Test runner for memvid MCP server.
This script can be used to test the server functionality.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memvid_mcp_server.main import _check_memvid_available, _server_state


async def test_server_functionality():
    """Test basic server functionality."""
    print("Testing memvid MCP server...")
    
    # Test 1: Check memvid availability
    print(f"Memvid available: {_check_memvid_available()}")
    
    # Test 2: Initialize server state
    try:
        await _server_state.initialize()
        print("✓ Server state initialized successfully")
    except Exception as e:
        print(f"✗ Server state initialization failed: {e}")
        return False
    
    # Test 3: Check server status
    try:
        status = {
            "initialized": _server_state.initialized,
            "memvid_available": _check_memvid_available(),
            "encoder_ready": _server_state.encoder is not None,
        }
        print(f"✓ Server status: {status}")
    except Exception as e:
        print(f"✗ Server status check failed: {e}")
        return False
    
    # Test 4: Cleanup
    try:
        await _server_state.cleanup()
        print("✓ Server cleanup completed successfully")
    except Exception as e:
        print(f"✗ Server cleanup failed: {e}")
        return False
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_server_functionality())
    sys.exit(0 if success else 1)
