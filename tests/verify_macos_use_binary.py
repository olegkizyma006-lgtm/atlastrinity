import asyncio
import json
import os
import subprocess


async def run_verification():
    binary_path = "/Users/olegkizyma/Documents/GitHub/atlastrinity/vendor/mcp-server-macos-use/.build/release/mcp-server-macos-use"

    if not os.path.exists(binary_path):
        print(f"Error: Binary not found at {binary_path}")
        return

    print(f"Starting verification of binary at: {binary_path}")

    process = subprocess.Popen(
        [binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    request_id = 0

    def send_request(method, params=None, is_notification=False):
        nonlocal request_id
        req = {"jsonrpc": "2.0", "method": method}
        curr_id = None
        if not is_notification:
            request_id += 1
            req["id"] = request_id
            curr_id = request_id

        if params is not None:
            req["params"] = params

        json_str = json.dumps(req)
        print(
            f"-> Sending {method} (Notification)"
            if is_notification
            else f"-> Sending {method} (ID: {curr_id})",
        )
        from typing import cast

        stdin = cast("subprocess.Popen", process).stdin
        assert stdin is not None
        stdin.write(json_str + "\n")
        stdin.flush()
        return curr_id

    def truncate_obj(obj, max_len=100):
        if isinstance(obj, dict):
            return {k: truncate_obj(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_obj(i, max_len) for i in obj]
        elif isinstance(obj, str):
            if len(obj) > max_len:
                return obj[:max_len] + f"... ({len(obj)} chars)"
            return obj
        return obj

    def read_response():
        from typing import cast

        stdout = cast("subprocess.Popen", process).stdout
        assert stdout is not None
        line = stdout.readline()
        if not line:
            print("DEBUG: Read empty line from stdout (EOF?)")
            return None
        # Don't print raw line if it's huge
        if len(line) > 500:
            print(f"DEBUG: Read from stdout: {line[:200]}... ({len(line)} chars)")
        else:
            print(f"DEBUG: Read from stdout: {line.strip()}")

        try:
            data = json.loads(line)
            # Print truncated version for debugging
            print(f"DEBUG: Parsed JSON: {json.dumps(truncate_obj(data), indent=2)}")
            return data
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to decode JSON: {e}")
            return None

    # 1. Initialize
    send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    )
    init_resp = read_response()
    print(f"<- Init Response: {json.dumps(truncate_obj(init_resp), indent=2)}")

    send_request("notifications/initialized", is_notification=True)

    # 2. List Tools
    send_request("tools/list")
    list_resp = read_response()
    if not list_resp or "result" not in list_resp:
        print("Failed to list tools.")
        process.terminate()
        return

    tools = list_resp["result"]["tools"]
    print(f"\nFound {len(tools)} tools.")

    # 3. Test Each Tool
    test_params = {
        "macos-use_open_application_and_traverse": {"identifier": "com.apple.finder"},
        "macos-use_click_and_traverse": {"x": 100, "y": 100},
        "macos-use_right_click_and_traverse": {"x": 100, "y": 100},
        "macos-use_double_click_and_traverse": {"x": 100, "y": 100},
        "macos-use_drag_and_drop_and_traverse": {
            "startX": 100,
            "startY": 100,
            "endX": 110,
            "endY": 110,
        },
        "macos-use_type_and_traverse": {"text": "verifying", "pid": 0},
        "macos-use_press_key_and_traverse": {"keyName": "Return", "modifierFlags": []},
        "macos-use_scroll_and_traverse": {"direction": "down", "amount": 1},
        "macos-use_refresh_traversal": {"pid": 0},
        "macos-use_window_management": {"action": "make_front", "pid": 0},
        "execute_command": {"command": "echo 'verified'"},
        "terminal": {"command": "echo 'verified'"},
        "macos-use_take_screenshot": {},
        "screenshot": {},
        "macos-use_analyze_screen": {},
        "ocr": {},
        "analyze": {},
        "macos-use_set_clipboard": {"text": "mcp_verified"},
        "macos-use_get_clipboard": {},
        "macos-use_system_control": {"action": "mute"},
        "macos-use_fetch_url": {"url": "https://example.com"},
        "macos-use_get_time": {},
        "macos-use_run_applescript": {"script": 'return "running"'},
        "macos-use_calendar_events": {
            "start": "2024-01-01T00:00:00.000Z",
            "end": "2024-01-02T00:00:00.000Z",
        },
        "macos-use_create_event": {"title": "Test Event", "date": "2024-01-01T00:00:00.000Z"},
        "macos-use_reminders": {},
        "macos-use_create_reminder": {"title": "Test Reminder"},
        "macos-use_spotlight_search": {"query": "README.md"},
        "macos-use_send_notification": {"title": "Test", "message": "Verification running"},
        "macos-use_notes_list_folders": {},
        "macos-use_notes_create_note": {"body": "Test note from MCP"},
        "macos-use_notes_get_content": {"name": "Test note from MCP"},
        "macos-use_mail_send": {"to": "test@example.com", "subject": "Test", "body": "Hello"},
        "macos-use_mail_read_inbox": {"limit": 1},
        "macos-use_list_tools_dynamic": {},
    }

    summary = []

    for tool in tools:
        name = tool["name"]
        args = test_params.get(name, {})
        print(f"\n--- Testing tool: {name} ---")

        try:
            send_request("tools/call", {"name": name, "arguments": args})

            await asyncio.sleep(0.5)
            resp = read_response()

            if resp and "result" in resp and not resp.get("error"):
                is_error = resp["result"].get("isError", False)
                if is_error:
                    status = "FAILED (Tool reported error)"
                    detail = json.dumps(truncate_obj(resp["result"]))
                else:
                    status = "PASSED"
                    detail = "Success"
            else:
                status = "FAILED (RPC/System error)"
                detail = json.dumps(truncate_obj(resp))

            summary.append((name, status, detail))
        except Exception as e:
            print(f"Exception during {name}: {e}")
            summary.append((name, "FAILED (Exception)", str(e)))
            if "Broken pipe" in str(e):
                print("Server process seems to have died. Exiting test loop.")
                break

    print("\n\n=== VERIFICATION SUMMARY ===")
    for name, status, detail in summary:
        print(f"{name:40} | {status:25} | {detail}")

    process.terminate()
    process.wait()

    stderr_output = process.stderr.read()
    if stderr_output:
        print("\n=== STDERR LOGS ===")
        print(stderr_output)


if __name__ == "__main__":
    asyncio.run(run_verification())
