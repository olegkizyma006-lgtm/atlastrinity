#!/usr/bin/env python3
"""
Example of how to correctly use the vibe_prompt tool.

This demonstrates the proper way to call vibe_prompt with all required parameters.
"""

import asyncio
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def example_vibe_prompt_usage():
    """Example of correct vibe_prompt usage."""
    
    # Import the vibe server module
    from src.mcp_server.vibe_server import vibe_prompt
    
    # Create a mock context (in real usage, this comes from MCP framework)
    class MockContext:
        async def log(self, level, message, logger_name="vibe_mcp"):
            print(f"[{level.upper()}] {message}")
    
    ctx = MockContext()
    
    # CORRECT USAGE: Always include the required 'prompt' parameter
    result = await vibe_prompt(
        ctx=ctx,
        prompt="Analyze the current system status and provide recommendations",  # REQUIRED
        cwd="/Users/dev/Documents/GitHub/atlastrinity",  # Optional
        timeout_s=300,  # Optional
        auto_approve=True,  # Optional (default: True)
        max_turns=5,  # Optional (default: 10)
    )
    
    print("Vibe prompt result:", result)
    
    # INCORRECT USAGE (would cause the error you saw):
    # result = await vibe_prompt(
    #     ctx=ctx,
    #     step_id="1",  # Wrong parameter name
    #     timeout=600,  # Wrong parameter name (should be timeout_s)
    #     workspace="/some/path",  # Wrong parameter name (should be cwd)
    #     # Missing required 'prompt' parameter!
    # )
    
    # CORRECTED VERSION of the above:
    # result = await vibe_prompt(
    #     ctx=ctx,
    #     prompt="Your prompt text here",  # REQUIRED parameter
    #     cwd="/some/path",  # Correct parameter name
    #     timeout_s=600,  # Correct parameter name
    # )


if __name__ == "__main__":
    print("Running vibe_prompt usage example...")
    asyncio.run(example_vibe_prompt_usage())