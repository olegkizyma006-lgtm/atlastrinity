
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

async def run_autonomous_google_reg():
    print("🚀 STARTING AUTONOMOUS GOOGLE ACCOUNT REGISTRATION DEMO")
    print("-------------------------------------------------------")
    
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    
    # Detailed instruction for the agents
    data_prompt = (
        "Open Google Chrome, navigate to the Google Account Creation page, and start registering a new account with these details:\n"
        "- First Name: Master\n"
        "- Last Name: Atlas\n"
        "- Desired Username: master.atlas.2026\n"
        "- Password: TrinityPassword2026!\n"
        "- Birthday: October 10, 1990\n"
        "- Gender: Rather not say\n"
        "Proceed through the steps (Name -> Birthday -> Username -> Password). "
        "Stop only if it asks for a phone number for SMS verification."
    )
    
    print(f"Instruction sent to Atlas:\n{data_prompt}\n")
    
    try:
        # Run the full pipeline
        # Using a timeout to prevent infinite loops in demo
        result = await asyncio.wait_for(orchestrator.run(data_prompt), timeout=600)
        print("\n🏁 Demo result status:", result)
                 
    except asyncio.TimeoutError:
        print("\n⏳ Demo timed out after 10 minutes.")
    except Exception as e:
        print(f"\n❌ Unexpected error in demo: {e}")
    finally:
        await mcp_manager.cleanup()
        print("\n✨ Demo cleanup complete.")

if __name__ == "__main__":
    asyncio.run(run_autonomous_google_reg())
