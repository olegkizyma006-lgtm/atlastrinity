import asyncio
import json
import os
import sys
from pathlib import Path

# Constants
CONFIG_PATH = Path.home() / ".config/atlastrinity/mcp/config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

async def run_debug():
    if not CONFIG_PATH.exists():
        print("Config not found")
        return

    with open(CONFIG_PATH) as f:
        data = json.load(f)

    server_config = data.get("mcpServers", {}).get("vibe")
    if not server_config:
        print("Vibe server config not found")
        return

    cmd = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env", {})

    if cmd == "python3":
        cmd = sys.executable
    
    full_cmd = [cmd] + [arg.replace("${HOME}", str(Path.home())) for arg in args]
    
    print(f"Starting server: {full_cmd}")
    process = await asyncio.create_subprocess_exec(
        *full_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env},
    )

    # Init
    init_req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "debug"}}
    }
    process.stdin.write(json.dumps(init_req).encode() + b"\n")
    await process.stdin.drain()

    # Read Init
    while True:
        line = await process.stdout.readline()
        if not line: break
        msg = json.loads(line.decode())
        if msg.get("id") == 1:
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode() + b"\n")
            await process.stdin.drain()
            break

    # Call Subcommand
    req = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {
            "name": "vibe_execute_subcommand",
            "arguments": {"subcommand": "invalid-subcommand"}
        }
    }
    print("Sending request...")
    process.stdin.write(json.dumps(req).encode() + b"\n")
    await process.stdin.drain()

    # Read Response
    while True:
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            print(f"RAW RECEIVED: {line.decode().strip()}")
            if not line: break
            msg = json.loads(line.decode())
            if msg.get("id") == 2:
                print(f"Decoded Response: {json.dumps(msg, indent=2)}")
                break
        except Exception as e:
            print(f"Error awaiting response: {e}")
            break

    process.terminate()

if __name__ == "__main__":
    asyncio.run(run_debug())
