import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Constants
CONFIG_PATH = Path.home() / ".config/atlastrinity/mcp/config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_mcp_server(name: str, config: dict[str, Any]) -> bool:
    print(f"\n--- Inspecting {name} ---")

    cmd = config.get("command")
    args = config.get("args", [])
    env = config.get("env", {})
    passed = False

    # Resolve placeholders
    if cmd is None:
        print(f"❌ Missing command for {name}")
        return False

    if cmd == "python3":
        cmd = sys.executable
    if "${PROJECT_ROOT}" in cmd:
        cmd = cmd.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))

    # Check if disabled
    if config.get("disabled", False):
        print(f"Skipping {name} (disabled)")
        return True

    full_cmd: list[str] = [cmd] + [
        (arg or "")
        .replace("${HOME}", str(Path.home()))
        .replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
        .replace("${GITHUB_TOKEN}", os.environ.get("GITHUB_TOKEN", ""))
        for arg in args
    ]

    # Prepare env
    run_env = os.environ.copy()
    run_env.update(env)

    print(f"Command: {' '.join(full_cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=run_env,
        )
    except Exception as e:
        print(f"❌ Failed to start process: {e}")
        return False

    async def read_stream(stream, label):
        while True:
            line = await stream.readline()
            if not line:
                break
            # print(f"[{label}] {line.decode().strip()}")

    # We need to send initialization
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",  # Try latest
            "capabilities": {},
            "clientInfo": {"name": "atlas-inspector", "version": "1.0"},
        },
    }

    json_line = json.dumps(init_request) + "\n"

    try:
        if process.stdin and process.stdout:
            process.stdin.write(json_line.encode())
            await process.stdin.drain()

            # Read response
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                except TimeoutError:
                    print("❌ Timeout waiting for initialization response")
                    break

                if not line:
                    break

                try:
                    msg = json.loads(line.decode())
                    if msg.get("id") == 1:
                        print("✅ Initialization successful")
                        # Send initialized notification
                        process.stdin.write(
                            json.dumps(
                                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                            ).encode()
                            + b"\n",
                        )
                        await process.stdin.drain()

                        # List tools
                        process.stdin.write(
                            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode()
                            + b"\n",
                        )
                        await process.stdin.drain()

                    elif msg.get("id") == 2:
                        tools = msg.get("result", {}).get("tools", [])
                        tool_names = [t["name"] for t in tools]
                        print(f"✅ Found {len(tools)} tools: {', '.join(tool_names[:5])}...")
                        passed = True
                        break

                    # Some servers act as clients too or send notifications, ignore those
                except json.JSONDecodeError:
                    print(f"Received non-json: {line.decode().strip()}")

    except Exception as e:
        print(f"❌ Error during interaction: {e}")
    finally:
        try:
            process.terminate()
            await process.wait()
        except:
            pass

    return passed


async def main():
    if not CONFIG_PATH.exists():
        print(f"Config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH) as f:
        data = json.load(f)

    servers = data.get("mcpServers", {})
    results = {}

    for name, config in servers.items():
        if name.startswith("_"):
            continue
        results[name] = await run_mcp_server(name, config)

    print("\n\n=== SUMMARY ===")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
