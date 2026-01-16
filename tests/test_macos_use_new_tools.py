import asyncio
import json
import os
import subprocess
import sys

BINARY_PATH = "vendor/mcp-server-macos-use/.build/release/mcp-server-macos-use"

async def read_stream(stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        yield line

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
        
        print(f"-> Sending {method}...")
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
                    print("<- Received initialize response")
                    break
            except:
                pass

        # Send initialized notification
        process.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }).encode() + b"\n")
        await process.stdin.drain()

        # Test Fetch
        print("\n[Test 1] Fetching URL...")
        rid = await send_request("tools/call", {
            "name": "macos-use_fetch_url",
            "arguments": {"url": "https://example.com"}
        })
        
        while True:
            line = await process.stdout.readline()
            if not line: break
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                         print(f"✗ Fetch failed: {resp['error']}")
                    else:
                         content = resp["result"]["content"][0]["text"]
                         if "Example Domain" in content:
                             print("✓ Fetch successful")
                         else:
                             print(f"✗ Fetch content unexpected: {content[:100]}...")
                    break
            except: 
                pass

        # Test Time
        print("\n[Test 2] Getting Time...")
        rid = await send_request("tools/call", {
            "name": "macos-use_get_time",
            "arguments": {}
        })
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    if "error" in resp:
                        print(f"✗ Time failed: {resp['error']}")
                    else:
                        print(f"✓ Time: {resp['result']['content'][0]['text']}")
                    break
            except: pass

        # Test AppleScript
        print("\n[Test 3] AppleScript...")
        rid = await send_request("tools/call", {
            "name": "macos-use_run_applescript",
            "arguments": {"script": 'return "Hello Native"'}
        })
        while True:
            line = await process.stdout.readline()
            try:
                resp = json.loads(line.decode())
                if resp.get("id") == rid:
                    content = resp["result"]["content"][0]["text"]
                    if "Hello Native" in content:
                        print("✓ AppleScript successful")
                    else:
                        print(f"✗ AppleScript result: {content}")
                    break
            except: pass

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    asyncio.run(main())
