import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brain.agents.grisha import Grisha
from brain.agents.atlas import Atlas
from brain.context import shared_context

async def test_grisha_audit():
    print("Testing Grisha Agent Audit...")
    grisha = Grisha()
    error = "Permission denied: /root/secret.txt"
    vibe_report = "I tried to read /root/secret.txt but it failed. I should use sudo or check permissions."
    context = {"goal": "Read a secret file"}
    
    result = await grisha.audit_vibe_fix(error, vibe_report, context)
    print(f"Audit Result: {result.get('audit_verdict')}")
    print(f"Reasoning: {result.get('reasoning')}")
    assert "audit_verdict" in result
    print("Grisha Audit Test Passed!")

async def test_atlas_review():
    print("\nTesting Atlas Agent Healing Review...")
    atlas = Atlas()
    error = "Permission denied: /root/secret.txt"
    vibe_report = "I suggest using sudo to read the file."
    grisha_audit = {
        "audit_verdict": "APPROVE",
        "reasoning": "Sudo is appropriate for this path if user authorized it."
    }
    context = {"goal": "Read a secret file"}
    
    result = await atlas.evaluate_healing_strategy(error, vibe_report, grisha_audit, context)
    print(f"Decision: {result.get('decision')}")
    print(f"Reason: {result.get('reason')}")
    assert "decision" in result
    print("Atlas Review Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_grisha_audit())
    asyncio.run(test_atlas_review())
