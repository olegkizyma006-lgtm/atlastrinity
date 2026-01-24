import asyncio
import logging
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP

from .lib.storage import SearchStorage, VectorStorage
from .lib.storage.blob import BlobStorage
from .tools.chain import recursive_enrichment
from .tools.ingest import ingest_dataset as ingest_impl

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("golden_fund")

# Initialize storage
vector_store = VectorStorage()
search_store = SearchStorage()

# Create FastMCP server
mcp = FastMCP("golden_fund")


@mcp.tool()
async def search_golden_fund(query: str, mode: str = "semantic") -> str:
    """
    Search the Golden Fund knowledge base.

    Args:
        query: The search query.
        mode: Search mode - 'semantic', 'keyword', 'hybrid', or 'recursive' (chains external data).
    """
    logger.info(f"Searching Golden Fund: {query} (mode={mode})")

    if mode == "semantic":
        result = vector_store.search(query)
    elif mode == "keyword":
        result = search_store.search(query)
    elif mode == "hybrid":
        vec_res = vector_store.search(query)
        txt_res = search_store.search(query)
        return f"Hybrid Results:\nVector: {vec_res}\nText: {txt_res}"
    elif mode == "recursive":
        return await recursive_enrichment(query)
    else:
        result = vector_store.search(query)

    return str(result)


blob_store = BlobStorage()


@mcp.tool()
async def store_blob(content: str, filename: str | None = None) -> str:
    """Store raw data as a blob (mock MinIO)."""
    return str(blob_store.store(content, filename))


@mcp.tool()
async def retrieve_blob(filename: str) -> str:
    """Retrieve raw data blob."""
    return str(blob_store.retrieve(filename))


@mcp.tool()
async def ingest_dataset(url: str, type: str, process_pipeline: list[str] | None = None) -> str:
    """
    Ingest a dataset into the Golden Fund.

    Args:
        url: URL of the dataset or API endpoint.
        type: Type of data source (e.g., 'api', 'web_page', 'csv_url').
        process_pipeline: List of processing steps (e.g., ['extract_entities', 'vectorize']).
    """
    if process_pipeline is None:
        process_pipeline = []
    return await ingest_impl(url, type, process_pipeline)


@mcp.tool()
async def probe_entity(entity_id: str, depth: int = 1) -> str:
    """
    Probe the knowledge graph for an entity to explore relationships.

    Args:
        entity_id: ID or name of the entity to probe.
        depth: How deep to traverse the graph relationships.
    """
    logger.info(f"Probing entity: {entity_id} (depth={depth})")
    return f"Probing results for {entity_id} (Placeholder)"


@mcp.tool()
async def add_knowledge_node(
    content: str, metadata: dict[str, Any], links: list[dict[str, str]] | None = None
) -> str:
    """
    Manually add a confirmed knowledge node to the Golden Fund.

    Args:
        content: The core information/text of the node.
        metadata: Key-value metadata pairs.
        links: List of links to other nodes [{'relation': 'related_to', 'target_id': '...'}]
    """
    if links is None:
        links = []
    logger.info(f"Adding knowledge node: {content[:50]}...")
    return "Node added successfully (Placeholder)"


if __name__ == "__main__":
    mcp.run()
