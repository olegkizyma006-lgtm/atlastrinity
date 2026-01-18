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
    _prepare_prompt_arg,
    VIBE_WORKSPACE,
    INSTRUCTIONS_DIR,
)

class MockContext:
    """Mock context for testing."""
    async def info(self, msg):
        print(f"[INFO] {msg}")
    async def error(self, msg):
        print(f"[ERROR] {msg}")


async def test_vibe_which():
    """Test that vibe_which locates the binary."""
    print("\n=== TEST: vibe_which ===")
    result = await vibe_which()
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
    print("\n=== TEST: _prepare_prompt_arg (small) ===")
    small_prompt = "Create a hello world Python script."
    result, file_path = _prepare_prompt_arg(small_prompt)
    
    if file_path is None:
        print(f"✅ Small prompt returned directly (no file created)")
        print(f"   Prompt: {result[:50]}...")
        return True
    else:
        print(f"❌ Unexpected file created: {file_path}")
        return False


async def test_prepare_prompt_large():
    """Test that large prompts create files in INSTRUCTIONS_DIR."""
    print("\n=== TEST: _prepare_prompt_arg (large) ===")
    large_prompt = "A" * 3000
    result, file_path = _prepare_prompt_arg(large_prompt, cwd="/some/random/path")
    
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
    
    # Use short prompt that fits in CLI args
    prompt = "Create a file called '/tmp/hello_vibe_test.py' with a simple hello world script that prints 'Hello from Vibe MCP!'."
    
    print(f"Sending prompt to Vibe CLI...")
    print(f"Prompt: {prompt}")
    print(f"VIBE_WORKSPACE: {VIBE_WORKSPACE}")
    print(f"INSTRUCTIONS_DIR: {INSTRUCTIONS_DIR}")
    
    result = await vibe_prompt(
        ctx=ctx,
        prompt=prompt,
        timeout_s=600,
        max_turns=5,
    )
    
    print(f"\n--- Result ---")
    print(f"Success: {result.get('success')}")
    print(f"Return code: {result.get('returncode')}")
    
    if result.get('stderr'):
        print(f"Stderr (first 500 chars): {result.get('stderr', '')[:500]}")
    
    if result.get('success'):
        # Check if file was created
        test_file = Path("/tmp/hello_vibe_test.py")
        if test_file.exists():
            print(f"\n✅ File created successfully!")
            print(f"   Content: {test_file.read_text()[:200]}")
            return True
        else:
            print(f"⚠️ Vibe ran but file not found at expected location")
            return False
    else:
        print(f"❌ Vibe failed: {result.get('error')}")
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
    
    # Test 4: Actually run Vibe to create a file
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
