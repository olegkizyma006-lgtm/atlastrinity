
import json
import os
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
CONFIG_PATH = Path.home() / ".config/atlastrinity/mcp/config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tools to test
TOOLS_TO_TEST = [
    # Core
    ("vibe_which", {}),
    ("vibe_prompt", {"prompt": "Hello, are you working?", "timeout_s": 30}), 
    ("vibe_list_sessions", {"limit": 1}),
    
    # Config/System
    ("vibe_get_config", {}),
    ("vibe_get_system_context", {}),
    
    # Analysis/Coding (Mock/Safe)
    ("vibe_ask", {"question": "What is 2+2?"}),
    
    # Error Handling checks
    ("vibe_session_details", {"session_id_or_file": "non_existent_session_id"}),
    ("vibe_execute_subcommand", {"subcommand": "invalid-subcommand"}),
]

async def run_vibe_tool(server_config: Dict[str, Any], tool_name: str, tool_args: Dict[str, Any]) -> bool:
    print(f"\n--- Testing Vibe -> {tool_name} ---")
    
    cmd = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env", {})
    
    # Resolve
    if cmd == "python3":
        cmd = sys.executable
    if "${PROJECT_ROOT}" in cmd:
        cmd = cmd.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    
    full_cmd = [cmd] + [arg.replace("${HOME}", str(Path.home())).replace("${PROJECT_ROOT}", str(PROJECT_ROOT)) for arg in args]
    
    run_env = os.environ.copy()
    run_env.update(env)
    
    try:
        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=run_env
        )
    except Exception as e:
        print(f"❌ Failed to start process: {e}")
        return False

    success = False
    
    try:
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "vibe-tester"}}
        }
        
        if process.stdin:
            process.stdin.write(json.dumps(init_request).encode() + b"\n")
            await process.stdin.drain()
            
            # Read Init Response
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                if not line: break
                try:
                    msg = json.loads(line.decode())
                    if msg.get("id") == 1:
                        # Initialized
                        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode() + b"\n")
                        await process.stdin.drain()
                        break
                except:
                    pass
            
            # Call Tool
            call_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_args}
            }
            
            print(f"Calling {tool_name} with {tool_args}...")
            process.stdin.write(json.dumps(call_request).encode() + b"\n")
            await process.stdin.drain()
            
            # Read Tool Response
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=30.0) # Longer timeout for AI
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for tool execution")
                    break
                    
                if not line: break
                
                try:
                    msg = json.loads(line.decode())
                    if msg.get("id") == 2:
                        # print(f"Raw response: {msg}")
                        
                        if "error" in msg:
                             # Check if error is expected (e.g. invalid subcommand)
                            err_msg = msg['error'].get('message', '')
                            if tool_name == "vibe_execute_subcommand" and "Unknown subcommand" in str(msg['error']):
                                print("✅ Correctly rejected invalid subcommand")
                                success = True
                            elif tool_name == "vibe_session_details" and "not found" in str(msg['result'] if 'result' in msg else msg.get('error')): # Vibe often returns error inside result dict or as error
                                # Check result for error logic
                                pass # Let's handle result below if it's there
                            else:
                                print(f"❌ Error: {msg['error']}")
                                # Special case for session details returning error in tool logic
                                
                        result = msg.get("result", {})
                        
                        # Handle tool-specific success checks
                        if tool_name == "vibe_session_details":
                             content = result.get("content", [])
                             text_res = "".join([c["text"] for c in content if c["type"] == "text"])
                             if '"success": false' in text_res or "not found" in text_res:
                                 print("✅ Correctly handled missing session")
                                 success = True
                        
                        elif tool_name == "vibe_execute_subcommand":
                             if msg.get("result", {}).get("error"):
                                 # Some tools return error inside result
                                 print("✅ Correctly rejected invalid subcommand (in result)")
                                 success = True
                             elif "error" in msg and "Unknown subcommand" in str(msg['error']):
                                 print("✅ Correctly rejected invalid subcommand (in error)")
                                 success = True

                        elif tool_name == "vibe_which":
                            # Check content for binary path
                            content = result.get("content", [])
                            text_res = "".join([c["text"] for c in content if c["type"] == "text"])
                            if "binary" in text_res:
                                print(f"✅ Found binary info: {text_res[:100]}...")
                                success = True
                            else:
                                print(f"❌ Unexpected response: {text_res}")
                                
                        else:
                            # General success check
                            print("✅ Tool executed successfully")
                            content = result.get("content", [])
                            text_res = "".join([c["text"] for c in content if c["type"] == "text"])
                            print(f"Output preview: {text_res[:200]}...")
                            success = True
                            
                        break
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        try:
            process.terminate()
            await process.wait()
        except:
            pass
            
    return success

async def main():
    if not CONFIG_PATH.exists():
        print("Config not found")
        return

    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    
    server_config = data.get("mcpServers", {}).get("vibe")
    if not server_config:
        print("Vibe server config not found")
        return
        
    results = {}
    for tool, args in TOOLS_TO_TEST:
        results[tool] = await run_vibe_tool(server_config, tool, args)
        
    print("\n=== VIBE DEEP TEST SUMMARY ===")
    all_pass = True
    for tool, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{tool}: {status}")
        if not passed: all_pass = False

    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
