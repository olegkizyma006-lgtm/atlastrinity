import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.brain.logger import logger
from src.brain.mcp_manager import mcp_manager


async def test_filesystem_server():
    logger.info("🧪 Starting Smoke Test: Filesystem MCP Server")

    SERVER = "filesystem"

    try:
        # 1. List Tools
        tools = await mcp_manager.list_tools(SERVER)
        tool_names = [t.name for t in tools]
        logger.info(f"✅ Filesystem Tools available: {tool_names}")

        # 2. List Directory (Home or tmp)
        # We know from config it has access to HOME and /tmp
        # On macOS /tmp is a symlink to /private/tmp, and the node server requires exact match or contained path
        target_dir = "/private/tmp"
        logger.info(f"Testing list_directory on {target_dir}...")

        res_list = await mcp_manager.call_tool(SERVER, "list_directory", {"path": target_dir})
        logger.info(f"List Result: {str(res_list)[:200]}...")  # Truncate log

        # 3. Write and Read File
        test_file = os.path.join(target_dir, "trinity_fs_test.txt")
        test_content = "Hello from Trinity Filesystem Check"

        logger.info(f"Testing write_file to {test_file}...")
        await mcp_manager.call_tool(
            SERVER,
            "write_file",
            {"path": test_file, "content": test_content},
        )

        logger.info("Testing read_file...")
        res_read = await mcp_manager.call_tool(SERVER, "read_file", {"path": test_file})

        content = ""
        if hasattr(res_read, "content") and res_read.content:
            content = res_read.content[0].text

        if content == test_content:
            logger.info("✅ Read/Write verification PASSED.")
        else:
            logger.error(f"❌ Content mismatch. Expected '{test_content}', got '{content}'")

        # Cleanup (optional, but good practice)
        # Note: filesystem might not have delete_file depending on version, usually it does not explicit delete easily
        # via standard MCP filesystem often doesn't expose delete to safety?
        # Actually standard filesystem server usually has no delete. We skip delete.

    except Exception as e:
        logger.error(f"💥 Filesystem test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await mcp_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_filesystem_server())
