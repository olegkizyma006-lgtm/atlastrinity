
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

# Mock LLM response
class MockLLM:
    def __init__(self, *args, **kwargs):
        self.model_name = "mock"
    async def ainvoke(self, messages):
        content = str(messages)
        # 1. Atlas Evaluation Mock
        if "Tetyana wants to DEVIATE" in content:
             return MagicMock(content='''{
                "approved": true,
                "reason": "The proposed deviation is more efficient.",
                "decision_factors": {"efficiency_gain": "high", "risk": "low"},
                "new_instructions": "Proceed with the optimized step.",
                "voice_message": "DEVIATION_APPROVED"
            }''')
        return MagicMock(content="{}")

# Patch CopilotLLM globally
mock_copilot = MagicMock()
mock_copilot.CopilotLLM = MockLLM
sys.modules["providers.copilot"] = mock_copilot

from src.brain.agents import Atlas
from src.brain.prompts import AgentPrompts

async def test_atlas_evaluation():
    print("\n--- Testing Atlas Deviation Evaluation ---")
    atlas = Atlas()
    
    mock_step = {"id": 1, "action": "Old slow action"}
    proposed = "Skip this, file exists."
    plan = ["1. Old"]
    
    print("Evaluating deviation...")
    result = await atlas.evaluate_deviation(mock_step, proposed, plan)
    
    print(f"Evaluation Result: {result}")
    
    if result.get("approved") is True and result.get("reason") and "decision_factors" in result:
        print("✅ Atlas successfully approved deviation with decision factors.")
    else:
        print("❌ Atlas failed to approve deviation or missing factors.")

async def main():
    await test_atlas_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
