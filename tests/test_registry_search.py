import asyncio
import json
from src.mcp_server.duckduckgo_search_server import business_registry_search

async def test_search():
    print("Testing specialized business registry search for 'Кардинал-Клінінг'...")
    result = business_registry_search("Кардинал-Клінінг")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_search())
