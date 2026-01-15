import json
import os
import subprocess
import sys
import time


def run_mcp_command(binary_path, commands):
    if not os.path.exists(binary_path):
        return [f"Error: Binary not found at {binary_path}"]

    process = subprocess.Popen(
        [binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "verify-client", "version": "1.0"},
        },
        "id": 0,
    }
    process.stdin.write(json.dumps(init_req) + "\n")
    process.stdin.flush()
    init_resp = process.stdout.readline()

    # 2. List tools to see all capabilities
    list_req = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}
    process.stdin.write(json.dumps(list_req) + "\n")
    process.stdin.flush()
    list_resp = process.stdout.readline()

    results = [list_resp]

    # 3. Call a specific tool (Open Calculator)
    for i, (method, params) in enumerate(commands):
        tool_req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": method, "arguments": params},
            "id": i + 2,
        }
        process.stdin.write(json.dumps(tool_req) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        results.append(line)
        time.sleep(1)

    process.terminate()
    return results


if __name__ == "__main__":
    project_root = "/Users/olegkizyma/Documents/GitHub/atlastrinity"
    binary = f"{project_root}/vendor/mcp-server-macos-use/.build/arm64-apple-macosx/release/mcp-server-macos-use"

    # Try to open Calculator as a test
    commands = [
        ("macos-use_open_application_and_traverse", {"identifier": "Calculator"}),
    ]

    print("Running verification...")
    outputs = run_mcp_command(binary, commands)

    for i, out in enumerate(outputs):
        try:
            data = json.loads(out)
            if i == 0:
                print("\n--- Available Tools ---")
                tools = data.get("result", {}).get("tools", [])
                for t in tools:
                    print(f"- {t['name']}: {t['description']}")
            else:
                print(f"\n--- Tool execution {i} result ---")
                if "error" in data:
                    print(f"Error: {data['error']}")
                else:
                    text_content = data["result"]["content"][0]["text"]
                    result_data = json.loads(text_content)
                    print(f"Success! App opened. PID: {result_data.get('pid')}")
                    if result_data.get("traversalAfter"):
                        elements = result_data["traversalAfter"].get("elements") or []
                        print(f"Elements found: {len(elements)}")
                        print("Sample elements:")
                        for el in elements[:5]:
                            print(
                                f"  - {el.get('role')}: {el.get('text') or el.get('value') or 'N/A'}"
                            )
        except Exception as e:
            print(f"Raw output {i}: {out[:200]}...")
            print(f"Error parsing: {e}")
