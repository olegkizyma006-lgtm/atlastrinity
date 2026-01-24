import pytest

from src.brain.agents.grisha import VerificationResult
from src.brain.agents.tetyana import StepResult
from src.brain.orchestrator import Trinity


@pytest.mark.asyncio
async def test_recovery_uses_grisha_and_announces_grisha_message(monkeypatch):
    t = Trinity()

    # Fake Tetyana to always return a failed step
    async def fake_execute_step(step_copy, attempt=1):
        return StepResult(
            step_id=step_copy.get("id", 1),
            success=False,
            result="failed",
            error="simulated",
        )

    monkeypatch.setattr(t.tetyana, "execute_step", fake_execute_step)

    # Fake Grisha to return a rejection with a voice message
    async def fake_verify_step(step, result, screenshot_path=None, goal_context="", task_id=None):
        return VerificationResult(
            step_id=step.get("id"),
            verified=False,
            confidence=0.1,
            description="Bad",
            issues=["issue1"],
            voice_message="GRISHA_MESSAGE",
        )

    monkeypatch.setattr(t.grisha, "verify_step", fake_verify_step)

    # Capture spoken messages
    spoken = []

    async def fake_speak(agent, message):
        spoken.append((agent, message))

    monkeypatch.setattr(t, "_speak", fake_speak)

    # Run the executor on a single failing step
    steps = [{"action": "do something", "expected_result": ""}]

    res = await t._execute_steps_recursive(steps, parent_prefix="", depth=0)

    # Expect overall result to be False (failed task)
    assert res is False

    # Expect Grisha to have announced rejection (first spoken message by grisha during validation)
    assert any(s[0] == "grisha" and "GRISHA_MESSAGE" in s[1] for s in spoken)
