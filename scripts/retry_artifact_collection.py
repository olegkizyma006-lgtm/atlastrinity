#!/usr/bin/env python3
import asyncio
import re
import sys

sys.path.insert(0, "src")
from brain.agents.grisha import Grisha
from brain.agents.tetyana import Tetyana
from brain.mcp_manager import mcp_manager


async def run():
    tet = Tetyana(model_name="grok-code-fast-1")
    step = {
        "id": 424,
        "action": "Launch a browser and navigate to https://accounts.google.com/signup.",
        "tool": "browser",
        "args": {"action": "navigate", "url": "https://accounts.google.com/signup"},
    }

    print("Executing step via Tetyana...")
    res = await tet.execute_step(step, attempt=1)
    print("Tetyana result:", res.success, res.result[:200])

    # Find the latest note for step_424
    notes_search = await mcp_manager.call_tool(
        "notes",
        "search_notes",
        {"tags": ["step_424"], "limit": 5},
    )
    structured = getattr(notes_search, "structuredContent", None) or getattr(
        notes_search,
        "content",
        None,
    )
    print("Notes search:", structured if structured else notes_search)

    # Read note content if available and extract file path (first file path)
    note_id = None
    if isinstance(notes_search, dict):
        notes = notes_search.get("notes", [])
        if notes:
            note_id = notes[0]["id"]
    elif hasattr(notes_search, "structuredContent"):
        sc = notes_search.structuredContent.get("result", {})
        notes = sc.get("notes", [])
        if notes:
            note_id = notes[0]["id"]

    screenshot_path = None
    if note_id:
        note = await mcp_manager.call_tool("notes", "read_note", {"note_id": note_id})
        if isinstance(note, dict) and note.get("success"):
            content = note.get("content", "")
        elif (
            hasattr(note, "content") and len(note.content) > 0 and hasattr(note.content[0], "text")
        ):
            import json

            try:
                content = json.loads(note.content[0].text).get("content", "")
            except Exception:
                content = note.content[0].text
        else:
            content = ""
        m = re.search(r"(/[^\s]+\.png)", content)
        if m:
            screenshot_path = m.group(1)
            print("Found screenshot path:", screenshot_path)

    # Ask Grisha to verify using the saved screenshot if found
    gr = Grisha()
    result = {"success": True, "output": "Navigated to accounts.google.com/signup"}
    verification = await gr.verify_step(step, result, screenshot_path=screenshot_path)
    print("Grisha verification:", verification.verified, verification.confidence)

    await mcp_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(run())
