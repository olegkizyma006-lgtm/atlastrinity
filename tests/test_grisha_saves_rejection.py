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

    # Check filesystem for report
    import os
    import glob
    
    reports_dir = os.path.expanduser("~/.config/atlastrinity/reports")
    assert os.path.exists(reports_dir)
    
    # Find latest report
    files = glob.glob(os.path.join(reports_dir, "rejection_step_999_*.md"))
    assert len(files) > 0, "No rejection report found in filesystem"
    
    latest_report = files[-1]
    content = open(latest_report, "r").read()
    assert "Fake description" in content
    assert "issue1" in content

    # Check memory entity
    mem = await mcp_manager.call_tool("memory", "get_entity", {"name": "grisha_rejection_step_999"})
    assert mem is not None
    # ensure memory entity contains observations
    if hasattr(mem, "content"):
        # CallToolResult case: parse structuredContent
        sc = getattr(mem, "structuredContent", None)
        res = sc.get("result") if sc else None
        assert res and "name" in res
    else:
        assert mem.get("success") is True
