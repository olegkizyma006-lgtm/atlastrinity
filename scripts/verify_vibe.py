
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.brain.mcp_manager import mcp_manager
from src.brain.logger import logger

async def test_vibe_server():
    logger.info("🧪 Starting Smoke Test: Vibe MCP Server")
    
    SERVER = "vibe"
    
    try:
        # 1. List functionalities (Tools)
        tools = await mcp_manager.list_tools(SERVER)
        tool_names = [t.name for t in tools]
        logger.info(f"✅ Vibe Tools available: {tool_names}")
        
        if "vibe_prompt" not in tool_names:
            logger.error("❌ vibe_prompt tool missing!")
            return

        # 2. Run simple prompt
        logger.info("Testing vibe_prompt...")
        # Vibe CLI might be slow or require internet (Mistral API), so we use a small timeout if possible, 
        # but the tool definition has timeout_s.
        
        res = await mcp_manager.call_tool(
            SERVER, 
            "vibe_prompt", 
            {
                "prompt": "Say 'Vibe is working' and nothing else.", 
                "args": ["--quiet"], 
                "timeout_s": 300
            }
        )
        
        logger.info(f"Result: {res}")
        
        # Check content
        content = ""
        if hasattr(res, 'content') and res.content:
            content = res.content[0].text
        elif isinstance(res, dict):
            content = str(res)
            
        if "Vibe is working" in content or "working" in content:
             logger.info("✅ Vibe Prompt received expected response.")
        else:
             logger.warning(f"⚠️ Vibe response might be unexpected (check logs): {content}")

    except Exception as e:
        logger.error(f"💥 Vibe test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await mcp_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_vibe_server())
