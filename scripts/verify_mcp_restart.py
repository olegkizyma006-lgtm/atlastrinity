
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from brain.mcp_manager import MCPManager
from brain.config_loader import config
from brain.logger import logger

async def test_restart():
    """
    Test the restart_mcp_server functionality.
    We will target a less critical server like 'filesystem' or 'memory'.
    """
    print("Initializing MCP Manager...")
    target_server = "filesystem"
    manager = MCPManager()
    # Ensure server is running
    await manager.get_session(target_server)
    
    print(f"Targeting server: {target_server}")
    
    # 1. Check if it's running
    status = await manager.health_check(target_server)
    print(f"Initial status (healthy): {status}")
    
    if target_server not in manager.sessions and target_server not in manager._connection_tasks:
         # Note: sessions might be empty if connected but not used, but connection task should be there
         pass
        
    # We grab the session object if possible
    original_client = await manager.get_session(target_server)
    
    print(f"Original Client ID: {id(original_client)}")
    
    # 2. Restart
    print("Triggering restart...")
    success = await manager.restart_server(target_server)
    
    if success:
        print("Restart reported SUCCESS.")
    else:
        print("Restart reported FAILURE.")
        await manager.cleanup()
        return

    # 3. Check new state
    new_client = await manager.get_session(target_server)
    print(f"New Client ID: {id(new_client)}")
    
    if id(new_client) != id(original_client):
        print("✅ Client object replaced. Restart confirmed.")
    else:
        print("❌ Client object ID UNCHANGED. Restart failed or mocked?")
        
    # 4. Cleanup
    await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_restart())
