from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.behavior_engine import BehaviorEngine, WorkflowEngine


@pytest.fixture
def mock_behavior_engine():
    be = MagicMock(spec=BehaviorEngine)
    be.config = {
        "workflows": {
            "test_workflow": {
                "stages": [
                    {
                        "name": "stage1",
                        "steps": [
                            {"action": "internal.test_action", "params": {"msg": "hello ${user}"}},
                        ],
                    },
                    {
                        "name": "stage2",
                        "steps": [
                            {
                                "if": "${condition}",
                                "then": {
                                    "action": "internal.test_action",
                                    "params": {"msg": "conditional"},
                                },
                                "else": {
                                    "action": "internal.test_action",
                                    "params": {"msg": "fallback"},
                                },
                            },
                        ],
                    },
                ],
            },
        },
    }
    return be


@pytest.mark.asyncio
async def test_workflow_execution(mock_behavior_engine):
    # Mocking dependencies that might trigger import errors or side effects
    import sys

    sys.modules["src.brain.db.manager"] = MagicMock()
    sys.modules["src.brain.services_manager"] = MagicMock()
    sys.modules["src.brain.state_manager"] = MagicMock()

    # Mock internal action
    mock_action = AsyncMock()

    # We need to monkeypatch the internal action registry for this test
    # Re-import to ensure mocks are used
    if "src.brain.internal_actions" in sys.modules:
        del sys.modules["src.brain.internal_actions"]

    from src.brain import internal_actions

    internal_actions._INTERNAL_ACTIONS["internal.test_action"] = mock_action

    engine = WorkflowEngine(mock_behavior_engine)

    # Case 1: True Condition
    context = {"user": "Oleg", "condition": True}
    success = await engine.execute_workflow("test_workflow", context)

    assert success is True
    assert mock_action.call_count == 2

    # Check calls
    calls = [c.kwargs for c in mock_action.call_args_list]
    assert {"msg": "hello Oleg"} in calls
    assert {"msg": "conditional"} in calls

    # Case 2: False Condition
    mock_action.reset_mock()
    context = {"user": "World", "condition": False}
    success = await engine.execute_workflow("test_workflow", context)

    assert success is True
    assert mock_action.call_count == 2

    calls = [c.kwargs for c in mock_action.call_args_list]
    assert {"msg": "hello World"} in calls
    assert {"msg": "fallback"} in calls
