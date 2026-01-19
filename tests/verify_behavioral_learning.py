
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

# Patch CopilotLLM globally
mock_copilot = MagicMock()
sys.modules["providers.copilot"] = mock_copilot

from src.brain.memory import long_term_memory

async def test_memory_functionality():
    print("\n--- Testing Behavioral Learning Memory ---")
    
    if not long_term_memory.available:
        print("Skipping: ChromaDB not available.")
        return

    # 1. Store a deviation
    print("Saving test deviation...")
    original = "Create folder at /tmp/test_with_factors"
    deviation = "Check if exists first"
    reason = "Efficiency"
    factors = {"time_saved": "500ms", "risk": "low"}
    res = long_term_memory.remember_behavioral_change(
        original, deviation, reason, "Saved", {"step_id": 999}, decision_factors=factors
    )
    
    if res:
        print("✅ Deviation saved successfully.")
    else:
        print("❌ Failed to save deviation.")
        return

    # 2. Recall it
    print("Recalling behavior...")
    # Allow some time for ChromaDB to index if needed (usually instant for small data)
    await asyncio.sleep(1)
    
    lessons = long_term_memory.recall_behavioral_logic(original, n_results=1)
    
    if lessons and "Check if exists first" in str(lessons[0]['document']):
        print("✅ Deviation recalled successfully.")
        print(f"Lesson: {lessons[0]['document']}")
        if "Decision Factors" in str(lessons[0]['document']):
             print("✅ Decision factors found in documentation.")
        else:
             print("❌ Decision factors MISSING in documentation.")
    else:
        print("❌ Failed to recall deviation.")
        print(f"Got: {lessons}")

async def main():
    await test_memory_functionality()

if __name__ == "__main__":
    asyncio.run(main())
