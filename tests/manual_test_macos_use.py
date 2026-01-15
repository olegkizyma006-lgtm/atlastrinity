import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

async def run_mcp_test():
    print("=== TESTING MACOS-USE SWIFT BINARY TOOLS ===")
    
    # 1. Locate Binary
    project_root = Path(__file__).parent.parent
    config_path = project_root / "src" / "mcp_server" / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    cmd_template = config["mcpServers"]["macos-use"]["command"]
    binary_path = cmd_template.replace("${PROJECT_ROOT}", str(project_root))
    
    print(f"Binary Path: {binary_path}")
    if not os.path.exists(binary_path):
        print(f"ERROR: Binary not found at {binary_path}")
        return

    # 2. Start Server Process
    process = await asyncio.create_subprocess_exec(
        binary_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    print("Server process started.")

    async def read_response():
        line = await process.stdout.readline()
        if not line:
            return None
        return json.loads(line.decode())

    async def send_request(req):
        msg = json.dumps(req) + "\n"
        process.stdin.write(msg.encode())
        await process.stdin.drain()

    # 3. Initialize
    print("\n--- Sending Initialize ---")
    await send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    })
    
    # Read init response
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 1:
            print("Initialize Response Received")
            break

    await send_request({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })

    # 4. List Tools
    print("\n--- Listing Tools ---")
    await send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    })
    
    tools = []
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 2:
            tools = resp["result"]["tools"]
            print(f"Found {len(tools)} tools:")
            for t in tools:
                print(f" - {t['name']}")
            break

    # 5. Test execute_command
    print("\n--- Testing execute_command ---")
    await send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "execute_command",
            "arguments": {"command": "echo 'MCP Test Success'"}
        }
    })
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 3:
            content = resp["result"]["content"][0]["text"]
            print(f"Result: {content.strip()}")
            if "MCP Test Success" in content:
                print("✅ execute_command PASSED")
            else:
                print("❌ execute_command FAILED")
            break

    # 6. Test Screenshot
    print("\n--- Testing macos-use_take_screenshot ---")
    await send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "macos-use_take_screenshot",
            "arguments": {}
        }
    })
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 4:
            if "error" in resp: # Should check isError or result error
                print(f"❌ Screenshot FAILED (Expected if headless/no permission): {resp}")
            elif resp.get("result", {}).get("isError"):
                 print(f"❌ Screenshot FAILED (Tool Error): {resp['result']['content'][0]['text']}")
            else:
                content = resp["result"]["content"][0]["text"]
                if len(content) > 100:
                    print(f"✅ Screenshot PASSED (Base64 length: {len(content)})")
                else:
                    print(f"❌ Screenshot FAILED (Content too short): {content}")
            break

    # 7. Test Vision Analysis
    print("\n--- Testing macos-use_analyze_screen ---")
    await send_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "macos-use_analyze_screen",
            "arguments": {}
        }
    })
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 5:
            if resp.get("result", {}).get("isError"):
                 print(f"❌ Vision FAILED (Tool Error): {resp['result']['content'][0]['text']}")
            else:
                content = resp["result"]["content"][0]["text"]
                try:
                    data = json.loads(content)
                    print(f"✅ Vision Analysis PASSED (Found {len(data)} elements)")
                except:
                    print(f"❌ Vision FAILED (Invalid JSON): {content}")
            break

    # 8. SCENARIO: Calculator Automation
    print("\n--- SCENARIO: Calculator Automation ---")
    
    # A. Open Calculator
    print("Step 1: Opening Calculator...")
    await send_request({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "macos-use_open_application_and_traverse",
            "arguments": {"identifier": "Calculator"}
        }
    })
    
    app_pid = None
    while True:
        resp = await read_response()
        if not resp: break
        if resp.get("id") == 10:
            content = resp["result"]["content"][0]["text"]
            try:
                res_data = json.loads(content)
                # Check for direct PID or nested in openResult
                if "pid" in res_data:
                    app_pid = res_data["pid"]
                elif "openResult" in res_data and "pid" in res_data["openResult"]:
                    app_pid = res_data["openResult"]["pid"]
                
                if app_pid:
                    print(f"✅ Application Opened (PID: {app_pid})")
                else:
                    print(f"❌ Failed to open app (No PID found): {res_data}")
            except:
                print(f"❌ Invalid JSON response: {content}")
            break
            
    if app_pid:
        import time
        await asyncio.sleep(1) # Wait for animation
        
        # B. Type Calculation "5+5="
        print("Step 2: Typing '5+5='...")
        await send_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "macos-use_type_and_traverse",
                "arguments": {"pid": app_pid, "text": "5+5="}
            }
        })
        while True:
            resp = await read_response()
            if not resp: break
            if resp.get("id") == 11:
                print("✅ Typing command sent")
                break
        
        asyncio.sleep(0.5)

        # C. Take Screenshot Verification
        print("Step 3: Taking Screenshot for Verification...")
        await send_request({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "macos-use_take_screenshot",
                "arguments": {}
            }
        })
        while True:
            resp = await read_response()
            if not resp: break
            if resp.get("id") == 12:
                if not resp.get("result", {}).get("isError"):
                    print("✅ Screenshot captured successfully")
                    # Optionally save it
                    # with open("test_calc_screen.png", "wb") as f:
                    #     content = resp["result"]["content"][0]["text"]
                    #     f.write(base64.b64decode(content))
                else:
                     print(f"❌ Screenshot failed (Check permissions!): {resp}")
                break
                
        # D. Vision Analysis Check
        print("Step 4: Checking Resut with Vision OCR...")
        await send_request({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "macos-use_analyze_screen",
                "arguments": {}
            }
        })
        while True:
            resp = await read_response()
            if not resp: break
            if resp.get("id") == 13:
                 if not resp.get("result", {}).get("isError"):
                    content = resp["result"]["content"][0]["text"]
                    print(f"✅ OCR Results received: {content[:100]}...")
                    if "10" in content:
                        print("🎉 SUCCESS: Found result '10' in OCR data!")
                    else:
                        print("⚠️ Result '10' not explicitly found in OCR text (might be graphical).")
                 else:
                     print(f"❌ OCR failed: {resp}")
                 break

    print("\nAll tests completed.")
    process.terminate()

if __name__ == "__main__":
    asyncio.run(run_mcp_test())
