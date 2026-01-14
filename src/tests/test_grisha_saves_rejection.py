import pytest

from brain.agents.grisha import Grisha, VerificationResult
from brain.mcp_manager import mcp_manager


@pytest.mark.asyncio
async def test_grisha_saves_rejection_report():
    gr = Grisha()
    step = {"id": 999, "action": "Fake action", "expected_result": "Expect"}
    verification = VerificationResult(
        step_id=999,
        verified=False,
        confidence=0.1,
        description="Fake description",
        issues=["issue1"],
        voice_message="Rejected",
    )

    # Call internal save method
    await gr._save_rejection_report(999, step, verification)

    # Check notes
    notes_search = await mcp_manager.call_tool("notes", "search_notes", {"tags": ["step_999"], "limit": 5})
    assert notes_search is not None

    # Check memory entity
    mem = await mcp_manager.call_tool("memory", "get_entity", {"name": "grisha_rejection_step_999"})
    assert mem is not None
    # ensure memory entity contains observations
    if hasattr(mem, 'content'):
        # CallToolResult case: parse structuredContent
        sc = getattr(mem, 'structuredContent', None)
        res = sc.get('result') if sc else None
        assert res and 'name' in res
    else:
        assert mem.get('success') is True
