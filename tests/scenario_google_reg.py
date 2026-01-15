import asyncio
import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.brain.orchestrator import Trinity as Orchestrator
from src.brain.mcp_manager import mcp_manager
from src.brain.logger import logger

async def test_google_registration_scenario():
    print("=== TEST SCENARIO: GOOGLE ACCOUNT REGISTRATION ===")
    
    # Initialize
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    
    # Specific prompt to trigger computer use
    user_request = "Open Chrome and go to Google Account registration page to create a new account."
    
    print(f"\nUser Request: '{user_request}'")
    
    # We want to intercept the PLAN to verify it uses the correct tools
    # Since Orchestrator.process_request is e2e, we will run it and stop after a few steps or error out safely
    # For CI safety, we might just ask Atlas to PLAN first.
    
    auth_context = {"user_id": "test_user", "clearance": "admin"}
    
    # 1. Run the request
    # NOTE: In a real run without user interaction, this might stall on Browser launch if not handled.
    # However, Tetyana now uses native macos-use to launch apps, which works headless/GUI.
    
    try:
        # Run the request using the main loop
        result = await orchestrator.run(user_request)
        print("✅ Execution Result:", result)
                 
    except Exception as e:
        print(f"❌ Error during execution: {e}")
    finally:
        await mcp_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_google_registration_scenario())
