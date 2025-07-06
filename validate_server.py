#!/usr/bin/env python3
"""
Quick validation script for memvid MCP server.
This script checks if the server can be imported and started without errors.
"""

import sys
import os
import asyncio
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_import():
    """Test that the server can be imported without errors."""
    try:
        from memvid_mcp_server.main import mcp, _check_memvid_available
        print("✓ Server imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_server_startup():
    """Test that the server can start up briefly."""
    try:
        # Create a temporary script to test server startup
        test_script = """
import sys
import os
import asyncio
sys.path.insert(0, r'{}')

async def test_startup():
    try:
        from memvid_mcp_server.main import _server_state
        await _server_state.initialize()
        await _server_state.cleanup()
        print("SERVER_STARTUP_SUCCESS")
        return True
    except Exception as e:
        print(f"SERVER_STARTUP_ERROR: {{e}}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_startup())
    sys.exit(0 if success else 1)
""".format(project_root)

        # Write to temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_script = f.name

        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            if "SERVER_STARTUP_SUCCESS" in result.stdout:
                print("✓ Server startup test passed")
                return True
            else:
                print(f"✗ Server startup failed: {result.stdout} {result.stderr}")
                return False
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_script)
            except:
                pass
                
    except Exception as e:
        print(f"✗ Server startup test error: {e}")
        return False


def test_mcp_tools():
    """Test that MCP tools are properly registered."""
    try:
        from memvid_mcp_server.main import mcp
        
        # Check that tools are registered
        tools = getattr(mcp, '_tools', {})
        expected_tools = [
            'add_chunks', 'add_text', 'add_pdf', 'build_video', 
            'search_memory', 'chat_with_memvid', 'get_server_status'
        ]
        
        missing_tools = [tool for tool in expected_tools if tool not in tools]
        
        if missing_tools:
            print(f"✗ Missing tools: {missing_tools}")
            return False
        else:
            print(f"✓ All {len(expected_tools)} tools registered correctly")
            return True
            
    except Exception as e:
        print(f"✗ Tool registration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Validating memvid MCP server...")
    print(f"Project root: {project_root}")
    
    tests = [
        ("Import Test", test_import),
        ("Server Startup Test", test_server_startup),
        ("MCP Tools Test", test_mcp_tools),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")
    
    print(f"\n--- Results ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All validation tests passed! Server is ready to use.")
        return True
    else:
        print("✗ Some validation tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
