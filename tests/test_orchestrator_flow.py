import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.orchestrator import Trinity


async def test_orchestrator_planning():
    print("Initializing Trinity for dry-run test...")
    orch = Trinity()

    # Mock atlas and other dependencies to avoid external calls
    orch.atlas = AsyncMock()
    orch.atlas.analyze_request = AsyncMock(return_value={"intent": "task"})
    orch.atlas.create_plan = AsyncMock(return_value=MagicMock(steps=[{"action": "test"}]))
    orch.atlas.get_voice_message = MagicMock(return_value="Plan created")

    orch.voice = AsyncMock()
    orch.voice.speak = AsyncMock()

    # Mock DB/State manager to avoid real connections
    orch._verify_db_ids = AsyncMock()

    print("Testing orchestrator.run() flow...")
    # We don't want to run the whole thing as it has many dependencies,
    # but we want to verify the logic flow of planning

    user_request = "Зроби щось корисне"

    # We'll try to run the planning part of run() specifically or just check if it's reachable.
    # Since we fixed the indentation, it should now call analyze_request and create_plan.

    # Let's mock the whole run method's internal parts if needed,
    # but the goal is to see if we can trigger planning.

    # Actually, let's just use a simple check: if we call run, does it reach planning?
    # We can mock state_manager to be available and see if it passes the planning phase.

    from src.brain.state_manager import state_manager

    state_manager.available = True
    state_manager.publish_event = AsyncMock()

    # Run a subset of logic or the whole run with deep mocks
    try:
        # We might hit more errors due to unmocked stuff, but we catch them
        await orch.run(user_request)
    except Exception as e:
        print(f"Caught expected/unexpected error during run: {e}")

    # Verification: Check if analyze_request was called (it was inside the once-broken indentation)
    if orch.atlas.analyze_request.called:
        print("SUCCESS: Orchestrator reached planning phase (analyze_request called).")
    else:
        print("FAIL: Orchestrator did NOT reach planning phase.")


if __name__ == "__main__":
    asyncio.run(test_orchestrator_planning())
