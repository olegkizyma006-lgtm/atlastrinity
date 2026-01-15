
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

async def check_mcp():
    from brain.mcp_manager import mcp_manager
    from brain.config import ensure_dirs
    
    ensure_dirs()
    # No .initialize() method exists on MCPManager
    
    servers = mcp_manager.config.get("mcpServers", {})
    results = {}
    
    for server_name in servers:
        if server_name.startswith("_") or servers[server_name].get("disabled"):
            continue
        
        try:
            # list_tools will automatically call get_session and connect if needed
            tools = await mcp_manager.list_tools(server_name)
            if tools:
                results[server_name] = {
                    "status": "ONLINE",
                    "tools_count": len(tools)
                }
            else:
                # check if it's connected
                if server_name in mcp_manager.sessions:
                    results[server_name] = {
                        "status": "ONLINE (but no tools?)",
                        "tools_count": 0
                    }
                else:
                    results[server_name] = {
                        "status": "OFFLINE",
                        "error": "Failed to get session or tools"
                    }
        except Exception as e:
            results[server_name] = {
                "status": "OFFLINE",
                "error": str(e)
            }
    
    print("MCP SERVER STATUS REPORT:")
    for name, res in results.items():
        print(f"[{name}] {res['status']} | {res.get('tools_count', '')} {res.get('error', '')}")

if __name__ == "__main__":
    asyncio.run(check_mcp())
