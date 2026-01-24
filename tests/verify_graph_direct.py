#!/usr/bin/env python3
"""Verify Graph Server Functionality"""

import asyncio
import os
import sys

# Setup path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.brain.db.manager import db_manager


async def test_graph_generation():
    print("=== Testing Graph Server Integration ===")

    # Initialize DB
    await db_manager.initialize()

    # Initialize MCP (needs config)
    # We will use the graph server directly via its class or invoke python module
    # But since it's an MCP server, we should try to "connect" or at least verify the tool function

    from src.mcp_server.graph_server import generate_mermaid

    print("Generating Mermaid Diagram...")
    result = await generate_mermaid()

    print("\n[RESULT PREVIEW]")
    print(result[:200] + "..." if len(result) > 200 else result)

    if "graph TD" in result:
        print("\n✅ Graph Generation Successful")
        return True
    else:
        print("\n❌ Graph Generation Failed (Invalid format)")
        return False


if __name__ == "__main__":
    try:
        asyncio.run(test_graph_generation())
    except KeyboardInterrupt:
        pass
