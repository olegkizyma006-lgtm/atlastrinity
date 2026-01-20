import asyncio
import json
from src.mcp_server.duckduckgo_search_server import duckduckgo_search

async def test_generic():
    print("Testing generic search for 'Україна'...")
    result = duckduckgo_search("Україна")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_generic())
