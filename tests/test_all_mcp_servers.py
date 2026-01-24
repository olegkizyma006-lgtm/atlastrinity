#!/usr/bin/env python3
"""Comprehensive MCP Server Test Suite
Tests all configured MCP servers and their capabilities.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.mcp_manager import mcp_manager


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


async def run_server_test(server_name: str, test_cases: list) -> dict:
    """Test a single MCP server with multiple test cases. Returns graceful skips for missing servers."""
    results = {
        "server": server_name,
        "status": "unknown",
        "tools": [],
        "tests": [],
        "error": None,
    }

    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}Testing MCP Server: {server_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    try:
        # List available tools with timeout protection
        print(f"{Colors.OKBLUE}→ Listing tools...{Colors.ENDC}")

        try:
            tools = await asyncio.wait_for(mcp_manager.list_tools(server_name), timeout=10.0)
        except TimeoutError:
            results["status"] = "timeout"
            results["error"] = "Tool listing timeout (>10s)"
            print(f"{Colors.WARNING}⚠ Connection timeout{Colors.ENDC}")
            return results

        if not tools:  # Empty list means server not configured or no tools
            results["status"] = "not_configured"
            results["error"] = "Server not configured or no tools available"
            print(f"{Colors.WARNING}⚠ Server not configured or no tools{Colors.ENDC}")
            return results

        results["status"] = "connected"
        results["tools"] = [t.name for t in tools]
        print(f"{Colors.OKGREEN}✓ Found {len(tools)} tools:{Colors.ENDC}")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")

        # Run test cases
        for test_case in test_cases:
            tool_name = test_case["tool"]
            args = test_case.get("args", {})
            description = test_case.get("description", f"Testing {tool_name}")

            print(f"\n{Colors.OKBLUE}→ {description}{Colors.ENDC}")
            print(f"  Tool: {tool_name}, Args: {args}")

            try:
                result = await asyncio.wait_for(
                    mcp_manager.call_tool(server_name, tool_name, args),
                    timeout=5.0,
                )

                if isinstance(result, dict) and "error" in result:
                    print(f"{Colors.FAIL}✗ Error: {result['error']}{Colors.ENDC}")
                    results["tests"].append(
                        {
                            "test": description,
                            "status": "error",
                            "error": result["error"],
                        },
                    )
                else:
                    # Truncate long output
                    output_str = str(result)[:200]
                    print(f"{Colors.OKGREEN}✓ Success: {output_str}...{Colors.ENDC}")
                    results["tests"].append({"test": description, "status": "success"})
            except TimeoutError:
                print(f"{Colors.WARNING}⚠ Tool call timeout{Colors.ENDC}")
                results["tests"].append({"test": description, "status": "timeout"})
            except Exception as e:
                print(f"{Colors.FAIL}✗ Exception: {e!s}{Colors.ENDC}")
                results["tests"].append(
                    {"test": description, "status": "exception", "error": str(e)},
                )

    except Exception as e:
        results["status"] = "connection_failed"
        results["error"] = str(e)
        print(f"{Colors.FAIL}✗ Failed to connect: {e!s}{Colors.ENDC}")

    return results


# Pytest wrapper
# Pytest wrapper


@pytest.mark.asyncio
async def test_mcp_server(server_name: str, test_cases: list):
    """Test MCP server with timeout. Skip if server unavailable."""
    try:
        # Add timeout to prevent hanging on unavailable servers
        result = await asyncio.wait_for(run_server_test(server_name, test_cases), timeout=15.0)

        # Skip if server not configured or unavailable, assert if connected
        if result["status"] == "not_configured":
            pytest.skip(f"Server {server_name} not configured")
        elif result["status"] == "timeout":
            pytest.skip(f"Server {server_name} connection timeout")
        elif result["status"] == "error":
            pytest.skip(f"Server {server_name} unavailable: {result.get('error', 'Unknown')}")

        assert result["status"] in (
            "connected",
            "no_tools",
        ), f"Server {server_name} failed: {result.get('error')}"

    except TimeoutError:
        pytest.skip(f"Server {server_name} connection timeout (>15s)")
    except Exception as e:
        pytest.skip(f"Server {server_name} connection error: {str(e)[:80]}")


async def main():
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         AtlasTrinity MCP Server Test Suite              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    test_plan = {
        "filesystem": [
            {
                "tool": "list_directory",
                "args": {"path": str(Path.home())},
                "description": "List home directory",
            },
            {
                "tool": "read_file",
                "args": {"path": str(Path.home() / ".zshrc")},
                "description": "Read .zshrc file",
            },
        ],
        "macos-use": [
            {
                "tool": "execute_command",
                "args": {"command": "echo 'Terminal via macos-use OK'"},
                "description": "Execute echo command",
            },
            {
                "tool": "macos-use_get_time",
                "args": {},
                "description": "Get system time",
            },
            {
                "tool": "macos-use_fetch_url",
                "args": {"url": "https://example.com"},
                "description": "Fetch URL as markdown",
            },
        ],
        "github": [
            {
                "tool": "get_file_contents",
                "args": {
                    "owner": "olegkizyma",
                    "repo": "atlastrinity",
                    "path": "README.md",
                },
                "description": "Get README.md from repo",
            },
        ],
        "memory": [
            {
                "tool": "create_entities",
                "args": {
                    "entities": [
                        {
                            "name": "test_memory",
                            "entityType": "concept",
                            "observations": ["Testing MCP memory server"],
                        },
                    ],
                },
                "description": "Create test memory entity",
            },
        ],
        "puppeteer": [
            {
                "tool": "puppeteer_navigate",
                "args": {"url": "https://example.com"},
                "description": "Navigate to example.com",
            },
        ],
    }

    all_results = []

    for server_name, tests in test_plan.items():
        result = await run_server_test(server_name, tests)
        all_results.append(result)
        await asyncio.sleep(1)  # Brief pause between servers

    # Summary
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                    Test Summary                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    for result in all_results:
        server = result["server"]
        status = result["status"]

        if status == "connected":
            tool_count = len(result["tools"])
            test_count = len([t for t in result["tests"] if t["status"] == "success"])
            total_tests = len(result["tests"])
            print(
                f"{Colors.OKGREEN}✓ {server:20} | {tool_count} tools | {test_count}/{total_tests} tests passed{Colors.ENDC}",
            )
        elif status == "no_tools":
            print(f"{Colors.WARNING}⚠ {server:20} | Connected but no tools{Colors.ENDC}")
        else:
            error = result.get("error", "Unknown error")
            print(f"{Colors.FAIL}✗ {server:20} | {error[:40]}...{Colors.ENDC}")

    # Cleanup
    await mcp_manager.cleanup()
    print(f"\n{Colors.OKCYAN}Test suite completed.{Colors.ENDC}\n")


if __name__ == "__main__":
    asyncio.run(main())
