"""
Test script for Vibe MCP Server tools.
Verifies that tools are correctly wrapped and functional.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import vibe server functions directly
from src.mcp_server.vibe_server import (
    vibe_which,
    vibe_prompt,
    vibe_analyze_error,
    handle_long_prompt,
    VIBE_WORKSPACE,
    INSTRUCTIONS_DIR,
)

class MockContext:
    """Mock context for testing."""
    def __init__(self):
        self.output = []
    async def info(self, msg):
        print(f"[INFO] {msg}")
        self.output.append(msg)
    async def error(self, msg):
        print(f"[ERROR] {msg}")
        self.output.append(msg)
    async def log(self, level, message, logger_name=None):
        print(f"[{level.upper()}] {message}")
        self.output.append(message)


async def test_vibe_which():
    """Test that vibe_which locates the binary."""
    print("\n=== TEST: vibe_which ===")
    ctx = MockContext()
    result = await vibe_which(ctx)
    print(f"Result: {result}")
    
    if result.get("success"):
        print(f"✅ Vibe binary found at: {result.get('binary')}")
        print(f"   Version: {result.get('version')}")
        return True
    else:
        print(f"❌ Failed: {result.get('error')}")
        return False


async def test_prepare_prompt_small():
    """Test that small prompts don't create files."""
    print("\n=== TEST: handle_long_prompt (small) ===")
    small_prompt = "Create a hello world Python script."
    result, file_path = handle_long_prompt(small_prompt)
    
    if file_path is None:
        print(f"✅ Small prompt returned directly (no file created)")
        print(f"   Prompt: {result[:50]}...")
        return True
    else:
        print(f"❌ Unexpected file created: {file_path}")
        return False


async def test_prepare_prompt_large():
    """Test that large prompts create files in INSTRUCTIONS_DIR."""
    print("\n=== TEST: handle_long_prompt (large) ===")
    large_prompt = "A" * 3000
    result, file_path = handle_long_prompt(large_prompt)
    
    if file_path is not None and INSTRUCTIONS_DIR in file_path:
        print(f"✅ Large prompt offloaded to INSTRUCTIONS_DIR")
        print(f"   File: {file_path}")
        print(f"   Result prompt: {result[:80]}...")
        
        # Cleanup test file
        Path(file_path).unlink(missing_ok=True)
        return True
    else:
        print(f"❌ File not in correct location: {file_path}")
        return False


async def test_vibe_prompt_small_task():
    """Test vibe_prompt with a simple task."""
    print("\n=== TEST: vibe_prompt (create hello.py) ===")
    
    ctx = MockContext()
    
    # Use short prompt
    prompt = "Create a file called 'hello_vibe_test.py' with a simple hello world script."
    
    print(f"Sending prompt to Vibe CLI...")
    
    result = await vibe_prompt(
        ctx=ctx,
        prompt=prompt,
        timeout_s=600,
        max_turns=5,
    )
    
    print(f"\n--- Result ---")
    print(f"Success: {result.get('success')}")
    
    if not result.get('success'):
        print(f"❌ Vibe failed: {result.get('error')}")
        if result.get('stderr'):
            print(f"Stderr: {result.get('stderr')[:500]}")
        return False

    # Check if file was created in workspace
    test_file = Path(VIBE_WORKSPACE) / "hello_vibe_test.py"
    if test_file.exists():
        print(f"\n✅ File created successfully in workspace!")
        print(f"   Content: {test_file.read_text()[:200]}")
        # Cleanup
        test_file.unlink()
        return True
    else:
        print(f"⚠️ Vibe ran but file not found at {test_file}")
        return False


async def test_vibe_arg_filtering():
    """Test that forbidden arguments like --no-tui are filtered out."""
    print("\n=== TEST: vibe_prompt (argument filtering) ===")
    ctx = MockContext()
    
    # Send prompt with forbidden argument
    result = await vibe_prompt(
        ctx=ctx,
        prompt="version",
        args=["--no-tui"],
        timeout_s=30,
        max_turns=1,
    )
    
    # Check if command in result contains --no-tui
    command = result.get("command", [])
    if "--no-tui" not in command:
        print(f"✅ Argument --no-tui was correctly filtered out")
        print(f"   Command run: {' '.join(command)}")
        return True
    else:
        print(f"❌ Failed: --no-tui was NOT filtered out")
        print(f"   Command run: {' '.join(command)}")
        return False


async def test_vibe_analyze_error():
    """Test vibe_analyze_error to ensure prompt variable is fixed."""
    print("\n=== TEST: vibe_analyze_error ===")
    ctx = MockContext()
    
    result = await vibe_analyze_error(
        ctx=ctx,
        error_message="Test error message for analysis",
        auto_fix=False,
        timeout_s=60,
    )
    
    if result.get("success") or result.get("returncode") is not None:
        print(f"✅ vibe_analyze_error executed without name errors")
        return True
    else:
        print(f"❌ Failed: {result.get('error')}")
        return False


async def main():
    print("=" * 60)
    print("VIBE MCP SERVER - INTEGRATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Check binary
    results.append(("vibe_which", await test_vibe_which()))
    
    # Test 2: Small prompt (no file)
    results.append(("_prepare_prompt_arg (small)", await test_prepare_prompt_small()))
    
    # Test 3: Large prompt (file in INSTRUCTIONS_DIR)
    results.append(("_prepare_prompt_arg (large)", await test_prepare_prompt_large()))
    
    # Test 4: Argument filtering
    results.append(("vibe_arg_filtering", await test_vibe_arg_filtering()))
    
    # Test 5: Analyze error (fix check)
    results.append(("vibe_analyze_error", await test_vibe_analyze_error()))
    
    # Test 6: Actually run Vibe to create a file
    results.append(("vibe_prompt (create file)", await test_vibe_prompt_small_task()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
