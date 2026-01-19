
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.brain.mcp_registry import SERVER_CATALOG, TOOL_SCHEMAS, VIBE_DOCUMENTATION, VOICE_PROTOCOL
    
    print(f"Catalog size: {len(SERVER_CATALOG)}")
    print(f"Schemas size: {len(TOOL_SCHEMAS)}")
    print(f"Vibe Docs length: {len(VIBE_DOCUMENTATION)}")
    print(f"Voice Protocol length: {len(VOICE_PROTOCOL)}")
    
    assert "macos-use" in SERVER_CATALOG, "macos-use missing from Catalog"
    assert "vibe_prompt" in TOOL_SCHEMAS, "vibe_prompt missing from Schemas"
    assert len(VIBE_DOCUMENTATION) > 100, "Vibe Documentation seems too short"
    assert len(VOICE_PROTOCOL) > 50, "Voice Protocol seems too short"
    
    print("\n✅ Registry verification passed!")
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)
except AssertionError as e:
    print(f"\n❌ Assertion Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    sys.exit(1)
