import asyncio
import os
import stat
import sys
from pathlib import Path

# Determine CONFIG_ROOT to check creation
CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
WORKSPACE_DIR = CONFIG_ROOT / "workspace"

# Add src to path for MCP Manager
sys.path.append(os.path.abspath(os.getcwd()))
from src.brain.config import ensure_dirs  # Trigger creation logic
from src.brain.mcp_manager import mcp_manager


async def verify_workspace():
    print("--- 1. Testing Creation & Permissions ---")

    # 1. Trigger ensure_dirs (usually happens on import of config.py, but calling explicitly to be sure)
    ensure_dirs()

    if not WORKSPACE_DIR.exists():
        print(f"❌ FAILURE: Workspace directory not created at {WORKSPACE_DIR}")
        return
    else:
        print(f"✅ Directory exists: {WORKSPACE_DIR}")

    # Check permissions
    st = os.stat(WORKSPACE_DIR)
    mode = stat.S_IMODE(st.st_mode)
    print(f"Permissions: {oct(mode)} (Expected 777 or rwxrwxrwx)")

    if mode == 0o777:
        print("✅ Permissions match 777.")
    else:
        # Note: on some systems umask might affect this, but we explicitly set os.chmod(..., 0o777)
        # 0o777 is 511 decimal.
        print(
            f"⚠️ Permissions are {oct(mode)}. This might be due to umask or OS restrictions, but check if writable.",
        )

    print("\n--- 2. Testing MCP Write Access ---")
    test_file = str(WORKSPACE_DIR / "mcp_test_file.txt")

    try:
        # We need to restart MCP server or ensure it picks up the new config
        # Since this script imports mcp_manager fresh, it should load the new config.json
        print("Connecting to filesystem...")
        # Force reload config in manager? mcp_manager loads config in __init__
        # We might need to re-init it or just trust it loads the file we just wrote.
        mcp_manager.config = mcp_manager._load_config()

        content = "Hello from Global Workspace!"
        result = await mcp_manager.call_tool(
            "filesystem",
            "write_file",
            {"path": test_file, "content": content},
        )
        print(f"Write Result: {result}")

        if "error" not in str(result).lower() and os.path.exists(test_file):
            print(f"✅ SUCCESS: MCP successfully wrote to {test_file}")
            # Cleanup
            os.remove(test_file)
        else:
            print("❌ FAILURE: MCP write failed or file not found.")

    except Exception as e:
        print(f"❌ FAILURE: MCP interaction error: {e}")
        import traceback

        traceback.print_exc()

    await mcp_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(verify_workspace())
