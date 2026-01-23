import asyncio
import json
import os

BINARY_PATH = "vendor/mcp-server-macos-use/.build/release/mcp-server-macos-use"


async def main():
    if not os.path.exists(BINARY_PATH):
        print(f"Binary not found at {BINARY_PATH}")
        return

    print(f"Spawning {BINARY_PATH}...")
    process = await asyncio.create_subprocess_exec(
        BINARY_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    request_id = 0

    async def send_request(method, params=None):
        nonlocal request_id
        request_id += 1
        msg = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params:
            msg["params"] = params

        from typing import cast

        stdin = cast("asyncio.subprocess.Process", process).stdin  # type: ignore
        assert stdin is not None
        stdin.write(json.dumps(msg).encode() + b"\n")
        await stdin.drain()
        return request_id

    try:
        # Initialize
        await send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )

        while process.stdout is not None:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == 1:
                    print("<- Connected and Initialized")
                    break
            except:
                pass

        # Send initialized
        if process.stdin:
            process.stdin.write(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
                + b"\n",
            )
            await process.stdin.drain()

        # 1. Test Dynamic Help
        print("\n[Test 1] Dynamic Help...")
        rid = await send_request(
            "tools/call", {"name": "macos-use_list_tools_dynamic", "arguments": {}},
        )
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                        print(f"✗ Help failed: {resp['error']}")
                    else:
                        content = resp["result"]["content"][0]["text"]
                        print(f"✓ Help received ({len(content)} chars)")
                        # print(content[:200])
                    break
            except:
                pass

        # 2. Test Notes List Folders
        print("\n[Test 2] Notes: List Folders...")
        rid = await send_request(
            "tools/call", {"name": "macos-use_notes_list_folders", "arguments": {}},
        )
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                        print(f"✗ Notes failed: {resp['error']}")
                    else:
                        content = resp["result"]["content"][0]["text"]
                        print(f"✓ Notes folders: {content}")
                    break
            except:
                pass

        # 3. Test Mail Inbox Read
        print("\n[Test 3] Mail: Read Inbox...")
        rid = await send_request(
            "tools/call", {"name": "macos-use_mail_read_inbox", "arguments": {"limit": 3}},
        )
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                        print(f"✗ Mail failed: {resp['error']}")
                    else:
                        content = resp["result"]["content"][0]["text"]
                        print(f"✓ Mail Inbox (top 3): {content[:100]}...")
                    break
            except:
                pass

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()


if __name__ == "__main__":
    asyncio.run(main())
