
import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, AsyncMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brain.orchestrator import Trinity
from brain.state_manager import state_manager
from brain.agents.atlas import TaskPlan

async def test_resumption():
    print("--- STARTING RESUMPTION VERIFICATION ---")
    
    # 1. Setup Mock State
    orch = Trinity()
    session_id = "test_resumption_session"
    orch.current_session_id = session_id
    
    test_plan = {
        "steps": [
            {"id": "1", "action": "Step 1 (Already Done)", "success": True},
            {"id": "2", "action": "Step 2 (Pending)", "success": False}
        ]
    }
    
    orch.state = {
        "messages": [{"type": "human", "content": "Proof of concept task"}],
        "current_plan": test_plan,
        "step_results": [
            {"step_id": "1", "action": "Step 1", "success": True, "result": "Done"}
        ],
        "system_state": "idle"
    }
    
    # Save to Redis
    if state_manager.available:
        state_manager.save_session(session_id, orch.state)
        
        # Set restart flag
        restart_key = state_manager._key("restart_pending")
        restart_metadata = {
            "reason": "Verification Test",
            "session_id": session_id,
            "timestamp": "2026-01-19T20:00:00"
        }
        state_manager.redis.set(restart_key, json.dumps(restart_metadata))
        print(f"Set restart_pending flag and saved session {session_id}")
    else:
        print("Redis unavailable. Test skipped.")
        return

    # 2. Simulate System Restart (Initialize new Orchestrator)
    print("\n--- SIMULATING RESTART ---")
    new_orch = Trinity()
    
    # Mock initialize dependencies
    new_orch.atlas.analyze_request = AsyncMock()
    new_orch.tetyana.execute_step = AsyncMock(return_value=MagicMock(success=True, result="Success Step 2", step_id="2"))
    new_orch.grisha.verify_step = AsyncMock()
    
    await new_orch.initialize()
    
    print(f"Resumption pending: {getattr(new_orch, '_resumption_pending', False)}")
    print(f"Current session: {new_orch.current_session_id}")
    
    if getattr(new_orch, "_resumption_pending", False):
        print("SUCCESS: Orchestrator detected restart and restored state.")
    else:
        print("FAILED: Orchestrator did NOT detect restart flag.")
        return

    # 3. Verify step skipping in _execute_steps_recursive
    print("\n--- VERIFYING STEP SKIPPING ---")
    # We'll call run directly to see if it skips Step 1
    # Note: run() will call _execute_steps_recursive
    
    # Mock run behavior to avoid infinite loops or broad execution
    await new_orch.run("Proof of concept task")
    
    # Check if execute_step was called FOR STEP 1
    # Tetyana.execute_step was called with step 2?
    call_args_list = new_orch.tetyana.execute_step.call_args_list
    steps_called = [args[0][0].get("id") for args in call_args_list]
    print(f"Tetyana execute_step called for IDs: {steps_called}")
    
    if "1" not in steps_called and "2" in steps_called:
        print("SUCCESS: Step 1 was skipped, Step 2 was executed.")
    else:
        print(f"FAILED: Step skipping logic incorrect. Called: {steps_called}")

    # Cleanup
    state_manager.redis.delete(restart_key)

if __name__ == "__main__":
    asyncio.run(test_resumption())
