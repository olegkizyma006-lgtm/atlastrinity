import pytest

from brain.message_bus import AgentMsg, MessageType, message_bus


@pytest.mark.asyncio
async def test_tetyana_hearing_bus_messages():
    """Test that Tetyana correctly receives and processes messages from the bus."""
    # 1. Inject a message into the bus for Tetyana
    msg = AgentMsg(
        from_agent="grisha",
        to_agent="tetyana",
        message_type=MessageType.REJECTION,
        payload={
            "feedback": "Testing bus: change your strategy to use terminal instead of filesystem.",
        },
        step_id="999",
    )
    await message_bus.send(msg)

    # 2. Mock a step that Tetyana would execute
    step = {
        "id": "999",
        "action": "create a file",
        "tool": "filesystem",
        "args": {"path": "test.txt", "content": "hello"},
    }

    # 3. Simulate orchestrator's behavior: retrieve bus messages and inject into step
    bus_messages = await message_bus.receive("tetyana", mark_read=True)
    assert len(bus_messages) > 0
    step["bus_messages"] = [m.to_dict() for m in bus_messages]  # type: ignore

    # 4. Initialize Tetyana and verify she sees it (we check if the reasoning prompt call would include it)
    # Since we can't easily mock the LLM response here without complex fixtures,
    # we verify that the prompt generation logic (which we updated) includes it.
    from typing import Any

    from brain.prompts import AgentPrompts

    step_typed: Any = step
    bus_messages_typed: Any = step["bus_messages"]
    prompt = AgentPrompts.tetyana_reasoning_prompt(
        step=str(step_typed),
        context={},
        bus_messages=bus_messages_typed,
    )

    assert "Testing bus: change your strategy" in prompt
    assert "REAL-TIME MESSAGES FROM OTHER AGENTS (Bus):" in prompt


@pytest.mark.asyncio
async def test_grisha_step_centric_verification():
    """Verify that Grisha's prompt now includes step-centric instructions."""
    from brain.prompts import AgentPrompts

    prompt = AgentPrompts.grisha_verification_prompt(
        strategy_context="Use terminal",
        step_id=1,
        step_action="Create file",
        expected="File exists",
        actual="File found",
        context_info={},
        history=[],
    )

    assert "CRITICAL VERIFICATION RULE" in prompt
    assert "You are verifying STEP 1" in prompt
    assert "Do NOT reject the result because the overall task/goal is not yet finished" in prompt
