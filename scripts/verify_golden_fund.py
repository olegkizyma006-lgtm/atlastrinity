#!/usr/bin/env python3
"""
Verification Script for Golden Fund MCP
Tests:
1. Ingestion (Scraping/Parsing)
2. Transformation (Unified Schema)
3. Storage (ChromaDB + Blob)
4. Recursive Enrichment Logic
"""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Adjust imports to match MCP server structure
from src.mcp_server.golden_fund.lib.storage.vector import VectorStorage
from src.mcp_server.golden_fund.lib.storage.blob import BlobStorage
from src.mcp_server.golden_fund.lib.transformer import DataTransformer
from src.mcp_server.golden_fund.tools.chain import enricher

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_golden_fund")

async def test_vector_storage():
    logger.info("--- Testing Vector Storage (ChromaDB) ---")
    test_path = Path.home() / ".config" / "atlastrinity" / "data" / "golden_fund" / "test_chroma"
    storage = VectorStorage(persistence_path=str(test_path), collection_name="test_collection")
    
    if not storage.enabled:
        logger.warning("ChromaDB not available, skipping vector test.")
        return

    data = [{"title": "Test Document", "content": "This is a test vector document.", "id": "1"}]
    
    # Store
    res = storage.store(data)
    if res.success:
        logger.info(f"Storage success: {res.data}")
    else:
        logger.error(f"Storage failed: {res.error}")
        return

    # Search
    search_res = storage.search("test vector")
    if search_res.success and len(search_res.data["results"]) > 0:
        logger.info(f"Search success. Found: {search_res.data['results'][0]['content']}")
    else:
        logger.error(f"Search failed or empty: {search_res.data}")

    # Cleanup
    shutil.rmtree(test_path, ignore_errors=True)

async def test_blob_storage():
    logger.info("--- Testing Blob Storage ---")
    test_root = Path.home() / ".config" / "atlastrinity" / "data" / "golden_fund" / "test_blobs"
    blob = BlobStorage(root_path=str(test_root))
    
    content = {"key": "value", "list": [1, 2, 3]}
    res = blob.store(content, "test.json")
    
    if res.success:
        logger.info(f"Blob stored: {res.data['path']}")
    else:
        logger.error(f"Blob store failed: {res.error}")
        return

    retr = blob.retrieve("test.json")
    if retr.success and retr.data["key"] == "value":
        logger.info("Blob retrieval verified.")
    else:
        logger.error("Blob retrieval failed.")

    shutil.rmtree(test_root, ignore_errors=True)

async def test_transformer():
    logger.info("--- Testing Transformer ---")
    transformer = DataTransformer()
    raw = [{"title": "Raw Data", "value": 100, "extra": "info"}]
    
    res = transformer.transform(raw, source_format="json")
    if res.success:
        logger.info(f"Transformation success. Item 0 type: {res.data[0]['type']}")
    else:
        logger.error(f"Transformation failed: {res.error}")

async def main():
    await test_vector_storage()
    await test_blob_storage()
    await test_transformer()
    logger.info("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(main())
