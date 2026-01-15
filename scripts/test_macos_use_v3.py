import json
import subprocess
import sys
import time


def run_mcp_command(binary_path, commands):
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
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
        "id": 0,
    }
    process.stdin.write(json.dumps(init_req) + "\n")
    process.stdin.flush()
    process.stdout.readline()  # Consume init response

    results = []
    for i, (method, params) in enumerate(commands):
        tool_req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": method, "arguments": params},
            "id": i + 1,
        }
        process.stdin.write(json.dumps(tool_req) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        results.append(line)
        time.sleep(2)  # Give it time for traversal

    # Read any remaining stderr
    stderr_output = ""
    try:
        # We can't easily read non-blocking without more complex code,
        # but let's just close stdin and read all.
        process.stdin.close()
        stderr_output = process.stderr.read()
    except Exception:
        pass

    process.terminate()
    return results, stderr_output


if __name__ == "__main__":
    project_root = "/Users/olegkizyma/Documents/GitHub/atlastrinity"
    binary = f"{project_root}/vendor/mcp-server-macos-use/.build/arm64-apple-macosx/release/mcp-server-macos-use"

    # Task: Open TextEdit with explicit traversal
    commands = [
        (
            "macos-use_open_application_and_traverse",
            {"identifier": "TextEdit", "traverseAfter": True},
        ),
    ]

    results, stderr = run_mcp_command(binary, commands)
    print("--- STDERR ---")
    print(stderr)
    print("--- RESULTS ---")
    for res in results:
        try:
            parsed = json.loads(res)
            tool_result_text = parsed["result"]["content"][0]["text"]
            tool_result = json.loads(tool_result_text)
            print(f"Tool Result Summary: {tool_result.keys()}")
            if "traversalAfter" in tool_result:
                elements = tool_result["traversalAfter"].get("elements") or []
                print(f"Traversal items count: {len(elements)}")
                for item in elements[:10]:
                    print(
                        f" - {item.get('role', 'N/A')}: {item.get('text', 'N/A')} ({item.get('identifier', 'N/A')})"
                    )
            else:
                print("No traversalAfter found.")
        except Exception as e:
            print(f"Error parsing result: {e}")
            print(res)
