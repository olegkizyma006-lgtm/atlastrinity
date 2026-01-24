import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from brain.agents.grisha import Grisha


async def test_grisha_real():
    print("Testing Grisha with Real Implementation...")

    # Check if API key is present
    api_key = os.getenv("COPILOT_API_KEY") or os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("SKIP: COPILOT_API_KEY not set")
        return

    grisha = Grisha()

    # 1. Test Screenshot
    print("Step 1: Taking screenshot...")
    screenshot_path = await grisha.take_screenshot()
    if screenshot_path and os.path.exists(screenshot_path):
        print(f"Screenshot saved to: {screenshot_path}")
    else:
        print("Screenshot FAILED")
        return

    # 2. Test Verification
    print("Step 2: Verifying step...")
    mock_step = {
        "id": 1,
        "action": "Check if a file exists",
        "expected_result": "Information about successful file check on screen",
        "requires_verification": True,
    }
    mock_result = {
        "step_id": 1,
        "success": True,
        "result": "File checked successfully",
        "output": "File 'test.txt' found.",
    }

    try:
        verification = await grisha.verify_step(
            step=mock_step,
            result=mock_result,
            screenshot_path=screenshot_path,
        )

        print("\nVerification Results:")
        print(f"Verified: {verification.verified}")
        print(f"Confidence: {verification.confidence}")
        print(f"Description: {verification.description}")
        print(f"Issues: {verification.issues}")

    except Exception as e:
        import traceback

        print(f"ERROR during verification: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if screenshot_path and os.path.exists(screenshot_path):
            os.remove(screenshot_path)


if __name__ == "__main__":
    asyncio.run(test_grisha_real())
