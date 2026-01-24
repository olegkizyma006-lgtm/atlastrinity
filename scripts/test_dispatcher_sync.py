import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Prep path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock logger and config before imports


async def test_tool_selection():
    from src.brain.tool_dispatcher import ToolDispatcher

    # Mock MCPManager
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value={"success": True, "result": "mocked_output"})

    dispatcher = ToolDispatcher(mock_mcp)

    print("\n--- Starting Tool Selection Tests ---")

    # Test Case 1: Search Routing (Memory Server)
    print("\n1. Testing 'search' (Memory Server) routing...")
    await dispatcher.resolve_and_dispatch("search", {"query": "weather in Kyiv"})
    mock_mcp.call_tool.assert_awaited_with("memory", "search", {"query": "weather in Kyiv"})
    print("Success: 'search' routed to memory.search")

    # Test Case 2: Terminal Routing (macos-use)
    print("\n2. Testing 'bash' (macos-use) routing...")
    mock_mcp.call_tool.reset_mock()
    await dispatcher.resolve_and_dispatch("bash", {"command": "ls -la"})
    mock_mcp.call_tool.assert_awaited_with("macos-use", "execute_command", {"command": "ls -la"})
    print("Success: 'bash' routed to macos-use.execute_command")

    # Test Case 3: Discovery (Discovery First Policy)
    print("\n3. Testing 'discovery' (macos-use) routing...")
    mock_mcp.call_tool.reset_mock()
    await dispatcher.resolve_and_dispatch("discovery", {})
    mock_mcp.call_tool.assert_awaited_with("macos-use", "macos-use_list_tools_dynamic", {})
    print("Success: 'discovery' routed to macos-use.macos-use_list_tools_dynamic")

    # Test Case 4: Heuristic Keyword Priority
    print("\n4. Testing keyword priority (Git -> macos-use)...")
    mock_mcp.call_tool.reset_mock()
    await dispatcher.resolve_and_dispatch("git_status", {"porcelain": True})
    mock_mcp.call_tool.assert_awaited_with(
        "macos-use",
        "execute_command",
        {"command": "git status --porcelain"},
    )
    print("Success: 'git_status' routed to macos-use.execute_command through legacy handler")

    # Test Case 5: Direct Fetch (macos-use)
    print("\n5. Testing 'fetch' (macos-use) routing...")
    mock_mcp.call_tool.reset_mock()
    await dispatcher.resolve_and_dispatch("fetch", {"url": "https://google.com"})
    mock_mcp.call_tool.assert_awaited_with(
        "macos-use",
        "macos-use_fetch_url",
        {"url": "https://google.com"},
    )
    print("Success: 'fetch' routed to macos-use.macos-use_fetch_url")

    # Test Case 6: Verify search never goes to puppeteer (critical safeguard)
    print("\n6. Testing search routing safeguard...")
    try:
        # This should raise an exception, not route to puppeteer
        from src.brain.tool_dispatcher import ToolDispatcher

        _server, _tool, _args = dispatcher._handle_browser("search", {"query": "test"})
        print("ERROR: Search was incorrectly routed to browser tools!")
        raise AssertionError("Search should never be handled by _handle_browser")
    except ValueError as e:
        if "search" in str(e) and "memory server" in str(e):
            print("Success: Search routing safeguard working correctly")
        else:
            print(f"ERROR: Unexpected error message: {e}")
            raise AssertionError("Wrong error message for search routing")

    print("\n--- All Dispatcher Tests Passed Successfully! ---")


if __name__ == "__main__":
    asyncio.run(test_tool_selection())
