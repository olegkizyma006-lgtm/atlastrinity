import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

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


async def run_vibe_tool(
    server_config: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> bool:
    print(f"\n--- Testing Vibe -> {tool_name} ---")

    cmd = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env", {})

    # Resolve
    if cmd is None:
        return False
    if cmd == "python3":
        cmd = sys.executable
    if "${PROJECT_ROOT}" in cmd:
        cmd = cmd.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))

    full_cmd: list[str] = [cmd] + [
        arg.replace("${HOME}", str(Path.home())).replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
        for arg in args
    ]

    run_env = os.environ.copy()
    run_env.update(env)

    try:
        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=run_env,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
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
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "vibe-tester"},
            },
        }

        if process.stdin:
            process.stdin.write(json.dumps(init_request).encode() + b"\n")
            await process.stdin.drain()

            # Read Init Response
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                    if msg.get("id") == 1:
                        # Initialized
                        process.stdin.write(
                            json.dumps(
                                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                            ).encode()
                            + b"\n",
                        )
                        await process.stdin.drain()
                        break
                except:
                    pass

            # Call Tool
            call_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_args},
            }

            print(f"Calling {tool_name} with {tool_args}...")
            process.stdin.write(json.dumps(call_request).encode() + b"\n")
            await process.stdin.drain()

            # Read Tool Response
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=30.0,
                    )  # Longer timeout for AI
                except TimeoutError:
                    print("❌ Timeout waiting for tool execution")
                    break

                if not line:
                    break

                try:
                    msg = json.loads(line.decode())
                    if msg.get("id") == 2:
                        if "error" in msg:
                            if (
                                tool_name == "vibe_execute_subcommand"
                                and "Unknown subcommand" in str(msg["error"])
                            ):
                                print("✅ Correctly rejected invalid subcommand")
                                success = True
                            else:
                                print(f"❌ Error: {msg['error']}")
                        else:
                            result = msg.get("result", {})

                            # Handle tool-specific success checks
                            if tool_name == "vibe_session_details":
                                content = result.get("content", [])
                                text_res = "".join(
                                    [c["text"] for c in content if c["type"] == "text"],
                                )
                                if '"success": false' in text_res or "not found" in text_res:
                                    print("✅ Correctly handled missing session")
                                    success = True

                            elif tool_name == "vibe_execute_subcommand":
                                # Check structuredContent or result error
                                struct = result.get("structuredContent", {})
                                if (
                                    result.get("error")
                                    or struct.get("error")
                                    or "Unknown subcommand" in str(struct.get("error", ""))
                                ):
                                    print("✅ Correctly rejected invalid subcommand (in result)")
                                    success = True
                                else:
                                    # Fallback to checking text content
                                    content = result.get("content", [])
                                    text_res = "".join(
                                        [c["text"] for c in content if c["type"] == "text"],
                                    )
                                    if (
                                        "Unknown subcommand" in text_res
                                        or '"success": false' in text_res
                                    ):
                                        print("✅ Correctly rejected invalid subcommand (in text)")
                                        success = True

                            elif tool_name == "vibe_which":
                                # Check content for binary path
                                content = result.get("content", [])
                                text_res = "".join(
                                    [c["text"] for c in content if c["type"] == "text"],
                                )
                                if "binary" in text_res:
                                    print(f"✅ Found binary info: {text_res[:100]}...")
                                    success = True
                            else:
                                # General success check
                                print("✅ Tool executed successfully")
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

    with open(CONFIG_PATH) as f:
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
        if not passed:
            all_pass = False

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
