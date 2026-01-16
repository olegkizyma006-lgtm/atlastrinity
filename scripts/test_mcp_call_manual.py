
import asyncio
import sys
import os
from pathlib import Path

# Fix path to include src
sys.path.append(os.getcwd())

from src.brain.mcp_manager import MCPManager
from src.brain.logger import logger

async def test_tool_call():
    manager = MCPManager()
    server_name = "filesystem"
    tool_name = "list_directory"
    args = {"path": "/Users/olegkizyma/Documents/GitHub/atlastrinity"}
    
    print(f"--- Testing tool call: {server_name}.{tool_name} ---")
    
    try:
        # Ensure server is connected
        success = await manager.ensure_servers_connected([server_name])
        if not success.get(server_name):
            print(f"Failed to connect to {server_name}")
            return

        # Call tool
        print(f"Calling {tool_name} with {args}...")
        result = await manager.call_tool(server_name, tool_name, args)
        
        print("\nResult:")
        print(result)
        
        if result and not getattr(result, 'is_error', False):
            print("\n✅ Success! Tool call worked correctly.")
        else:
            print("\n❌ Failure. Tool call returned error or was empty.")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_tool_call())
