
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_routing():
    print("🔍 Testing Tetyana Workspace Re-routing Logic...")
    
    from src.brain.agents.tetyana import Tetyana
    from src.brain.config_loader import SystemConfig
    
    # Mock dependencies
    tetyana = Tetyana()
    
    # Mock mcp_manager
    tetyana.mcp_manager = MagicMock()
    tetyana.mcp_manager.call_tool = AsyncMock(return_value={"content": [{"text": "success"}]})
    
    # 1. Test Vibe re-routing when CWD is repo root
    repo_root = str(Path.cwd().absolute())
    args = {"prompt": "hi", "cwd": repo_root}
    
    print(f"Input CWD: {repo_root} (Repo Root)")
    
    # We need to mock the internal _call_mcp_direct or just call it
    # Since we want to test the logic inside _call_mcp_direct, let's call it
    # Note: we need to handle the fact that it's a private method
    
    try:
        # Mocking the validation to skip macos-use specific stuff
        tetyana._validate_macos_use_args = lambda tool, args: args
        
        await tetyana._call_mcp_direct("vibe", "vibe_prompt", args)
        
        expected_workspace = str(Path.home() / "AtlasProjects")
        actual_cwd = args["cwd"]
        
        print(f"Final CWD: {actual_cwd}")
        
        if actual_cwd == expected_workspace:
            print("✅ SUCCESS: Vibe re-routed to ~/AtlasProjects")
        else:
            print(f"❌ FAILURE: Vibe stayed in {actual_cwd}")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_routing())
