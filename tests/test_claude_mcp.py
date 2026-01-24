import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
from src.brain.mcp_manager import mcp_manager


async def verify_official_mcp():
    servers_to_test = ["filesystem", "sequential-thinking", "fetch"]

    print("=== Official MCP Verification ===")

    for server in servers_to_test:
        print(f"\n--- Testing Server: {server} ---")
        try:
            # 1. Connect
            session = await mcp_manager.get_session(server)
            if not session:
                print(f"❌ '{server}' failed to connect (check logs/npx/bunx).")
                continue
            print(f"✅ Connected to '{server}'.")

            # 2. List tools
            tools = await mcp_manager.list_tools(server)
            print(
                f"Found {len(tools)} tools: {[t.name for t in tools[:5]]}{'...' if len(tools) > 5 else ''}",
            )

            # 3. Simple execution test
            if server == "filesystem":
                # List home dir
                res = await mcp_manager.call_tool(
                    server,
                    "list_directory",
                    {"path": os.path.expanduser("~")},
                )
                print(
                    f"Execution test (list_directory): {'SUCCESS' if not hasattr(res, 'isError') or res.isError is False else 'FAILURE'}",
                )

            elif server == "sequential-thinking":
                # Start a thought
                res = await mcp_manager.call_tool(
                    server,
                    "sequentialthinking",
                    {
                        "thought": "Testing Claude's thinking mcp server",
                        "thoughtNumber": 1,
                        "totalThoughts": 1,
                        "nextThoughtNeeded": False,
                    },
                )
                print(
                    f"Execution test (thinking): {'SUCCESS' if not hasattr(res, 'isError') or res.isError is False else 'FAILURE'}",
                )

            elif server == "fetch":
                # Fetch a simple URL (Google)
                res = await mcp_manager.call_tool(
                    server,
                    "fetch_url",
                    {"url": "https://www.google.com"},
                )
                print(
                    f"Execution test (fetch): {'SUCCESS' if not hasattr(res, 'isError') or res.isError is False else 'FAILURE'}",
                )

        except Exception as e:
            print(f"❌ Error testing '{server}': {e}")

    print("\n--- Cleanup ---")
    await mcp_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(verify_official_mcp())
