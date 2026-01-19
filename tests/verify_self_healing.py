
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

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
        # Return a mock JSON response for Grisha
        return MagicMock(content='''{
            "root_cause": "Permission denied error indicates lack of privilege.",
            "technical_advice": "Use sudo for protected directories.",
            "voice_message": "FEEDBACK_GENERATED"
        }''')

# 1. Mock `providers.copilot` BEFORE importing agents
mock_copilot = MagicMock()
mock_copilot.CopilotLLM = MockLLM
sys.modules["providers.copilot"] = mock_copilot

# 2. NOW import agents
from src.brain.agents import Grisha, Tetyana
from src.brain.prompts import AgentPrompts

async def test_grisha_analysis():
    print("\n--- Testing Grisha Analysis ---")
    grisha = Grisha()
    
    mock_step = {
        "id": 1,
        "action": "Create a directory at /root/protected",
        "tool": "execute_command",
        "args": {"command": "mkdir /root/protected"},
        "full_plan": "1. Create dir. 2. Done."
    }
    mock_error = "Permission denied: /root/protected"
    
    print("Analyzing mocked failure...")
    result = await grisha.analyze_failure(mock_step, mock_error)
    
    print(f"Analysis Result: {result.get('analysis', {})}")
    print(f"Feedback Text: {result.get('feedback_text')}")
    
    if result.get("feedback_text") and "FEEDBACK" in result.get("feedback_text"):
        print("✅ Grisha successfully generated feedback.")
    else:
        print("❌ Grisha failed to generate feedback.")

async def test_tetyana_feedback_usage():
    print("\n--- Testing Tetyana Feedback Usage ---")
    tetyana = Tetyana()
    
    # Mock step with injected feedback
    mock_step = {
        "id": 2,
        "action": "Fix directory creation",
        "grisha_feedback": "Use sudo or change path to home directory.",
        "tool": "terminal",
        "args": {"command": "mkdir /root/protected"}
    }
    
    # We can't easily execute Tetyana without a full LLM response mock, 
    # but we can check if the prompt generation includes the feedback.
    
    prompt = AgentPrompts.tetyana_reasoning_prompt(
        str(mock_step),
        {"mock": "context"},
        feedback=mock_step["grisha_feedback"]
    )
    
    if "PREVIOUS REJECTION FEEDBACK" in prompt and "Use sudo" in prompt:
        print("✅ Tetyana prompt correctly includes Grisha's feedback.")
    else:
        print("❌ Tetyana prompt missing feedback.")
        print(prompt[:500])

async def main():
    await test_grisha_analysis()
    await test_tetyana_feedback_usage()

if __name__ == "__main__":
    asyncio.run(main())
