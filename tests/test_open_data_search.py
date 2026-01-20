import asyncio
import json
from src.mcp_server.duckduckgo_search_server import open_data_search

async def test_open_data():
    print("Testing specialized open data search for 'бюджет'...")
    result = open_data_search("бюджет")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_open_data())
