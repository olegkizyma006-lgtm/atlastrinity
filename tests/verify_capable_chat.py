import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.agents.atlas import Atlas
from src.brain.orchestrator import Trinity
from langchain_core.messages import HumanMessage

async def test_chat_intelligence():
    print("=== Testing Capable Chat Intelligence ===")
    atlas = Atlas()
    
    # Scenario 1: Greeting (Simple Chat)
    print("\nScenario 1: Greeting")
    analysis = await atlas.analyze_request("Привіт, як справи?")
    print(f"Intent detected: {analysis.get('intent')} (Reason: {analysis.get('reason')})")
    
    # Scenario 2: Information Seeking (News/Weather - should be 'chat' now)
    print("\nScenario 2: Information Seeking (News)")
    analysis = await atlas.analyze_request("Яка зараз погода в Києві та останні новини?")
    print(f"Intent detected: {analysis.get('intent')} (Reason: {analysis.get('reason')})")
    
    # Scenario 3: Code Explanation (Information Seeking)
    print("\nScenario 3: Code Explanation")
    analysis = await atlas.analyze_request("Поясни, як працює скрипт setup_dev.py?")
    print(f"Intent detected: {analysis.get('intent')} (Reason: {analysis.get('reason')})")
    
    # Scenario 4: Task intent (State change - should still be 'task')
    print("\nScenario 4: State Change (Task)")
    analysis = await atlas.analyze_request("Створи файл test.txt на робочому столі")
    print(f"Intent detected: {analysis.get('intent')} (Reason: {analysis.get('reason')})")

async def test_chat_brevity():
    print("\n=== Testing Chat Brevity for Greetings ===")
    atlas = Atlas()
    
    # Test Greeting
    print("Sending: 'Привіт Atlas'")
    try:
        response = await atlas.chat("Привіт Atlas")
        print(f"Atlas Response: {response}")
        word_count = len(response.split())
        print(f"Response length: {word_count} words.")
        if word_count < 25:
             print("✓ Brevity test PASSED (Concise response)")
        else:
             print("⚠ Brevity test FAILED (Response too long)")
    except Exception as e:
        print(f"Chat brevity test skipped or failed (likely missing live LLM/Redis context): {e}")

if __name__ == "__main__":
    asyncio.run(test_chat_intelligence())
    asyncio.run(test_chat_brevity())
