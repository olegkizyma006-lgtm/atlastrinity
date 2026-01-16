import asyncio
import json
import os
import subprocess
import sys

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
        stderr=asyncio.subprocess.PIPE
    )

    request_id = 0

    async def send_request(method, params=None):
        nonlocal request_id
        request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            msg["params"] = params
        
        # print(f"-> Sending {method}...")
        process.stdin.write(json.dumps(msg).encode() + b"\n")
        await process.stdin.drain()
        return request_id

    try:
        # Initialize
        await send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        })

        # Read initialize response
        while True:
            line = await process.stdout.readline()
            if not line: break
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == 1:
                    print("<- Connected and Initialized")
                    break
            except: pass
        
        # Send initialized
        process.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }).encode() + b"\n")
        await process.stdin.drain()

        # 1. Test Spotlight
        print("\n[Test 1] Spotlight Search (checking for this file)...")
        rid = await send_request("tools/call", {
            "name": "macos-use_spotlight_search",
            "arguments": {"query": "test_macos_use_universal.py"}
        })
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                        print(f"✗ Spotlight failed: {resp['error']}")
                    else:
                        content = resp["result"]["content"][0]["text"]
                        print(f"✓ Spotlight result count: {len(content.splitlines())}")
                        if "test_macos_use_universal.py" in content:
                             print("  Found self!")
                    break
            except: pass

        # 2. Test Notification
        print("\n[Test 2] Send Notification...")
        rid = await send_request("tools/call", {
            "name": "macos-use_send_notification",
            "arguments": {"title": "Test from Script", "message": "Universal MCP is working!"}
        })
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                     if "error" in resp: print(f"✗ Notification failed: {resp['error']}")
                     else: print("✓ Notification sent")
                     break
            except: pass

        # 3. Test Reminders (List unavailable without auth, but tool should run)
        print("\n[Test 3] Fetch Reminders...")
        rid = await send_request("tools/call", {
            "name": "macos-use_reminders",
            "arguments": {}
        })
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp: print(f"✗ Reminders failed: {resp['error']}")
                    else: 
                        content = resp['result']['content'][0]['text']
                        print(f"✓ Reminders returned: {content[:50]}...")
                    break
            except: pass

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    asyncio.run(main())
