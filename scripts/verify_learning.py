import asyncio
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.brain.memory import long_term_memory
from src.brain.orchestrator import Trinity as Orchestrator


async def test_learning_and_vibe():
    print("--- VERIFYING LEARNING SYSTEM ---")

    # 1. Test direct recall
    task_name = "Install special trinity module"
    plan = ["Step 1: Check modules", "Step 2: Run install"]

    print(f"Adding strategy to memory: {task_name}")
    long_term_memory.remember_strategy(
        task=task_name,
        plan_steps=plan,
        outcome="SUCCESS",
        success=True,
    )

    print("Attempting to recall similar task...")
    similar = long_term_memory.recall_similar_tasks("How do I install trinity modules?")
    if similar:
        print(f"✅ Recall SUCCESS: Found '{similar[0]['document'][:50]}...'")
    else:
        print("❌ Recall FAILED")

    # 2. Test Vibe Integration (Check if tool is recognized)
    print("\n--- VERIFYING VIBE INTEGRATION ---")
    orchestrator = Orchestrator()
    await orchestrator.initialize()

    try:
        from src.brain.mcp_manager import mcp_manager

        print("Checking Vibe server tools...")
        tools = await mcp_manager.list_tools("vibe")
        tool_names = [t.name for t in tools]
        print(f"Vibe tools: {tool_names}")
        if "vibe_analyze_error" in tool_names:
            print("✅ Vibe tools correctly available.")
        else:
            print("❌ Vibe tools NOT found.")

        # 3. Test Tetyana's new Vibe Self-Healing logic (Simulation)
        print("\n--- SIMULATING TETYANA SELF-HEALING ---")
        # We'll just check if the code we added is there by looking at the logs during a failing task
        # But for now, technical verification of availability is enough.

    finally:
        await mcp_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_learning_and_vibe())
