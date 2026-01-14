import pytest
import asyncio

from brain.agents.tetyana import Tetyana
from brain.mcp_manager import mcp_manager


@pytest.mark.asyncio
async def test_tetyana_reads_grisha_note(tmp_path):
    # Create a note as Grisha would
    title = "Test Grisha Note for Tetyana"
    content = "This is a test rejection note from Grisha for step 777."
    res = await mcp_manager.call_tool("notes", "create_note", {
        "title": title,
        "content": content,
        "category": "verification_report",
        "tags": ["grisha", "step_777"],
    })
    assert res is not None

    tet = Tetyana(model_name="grok-code-fast-1")
    feedback = await tet.get_grisha_feedback(777)

    assert feedback is not None and "test rejection note" in feedback.lower()
