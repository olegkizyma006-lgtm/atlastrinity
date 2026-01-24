"""
Recursive Enrichment Chain for Golden Fund
Implements "Search -> Miss -> External Fetch -> Ingest -> Link -> Search" loop.
"""

import asyncio
import logging
from typing import Any, Optional

from ..lib.connectors.ckan_connector import CKANConnector
from ..lib.storage import VectorStorage
from .ingest import ingest_dataset

logger = logging.getLogger("golden_fund.tools.chain")


class RecursiveEnricher:
    def __init__(self):
        self.ckan = CKANConnector()
        self.vector_store = VectorStorage()
        self.max_depth = 2

    async def enrich_and_search(self, query: str, depth: int = 0) -> str:
        """
        Recursively search and enrich the knowledge base.
        """
        if depth > self.max_depth:
            return f"Max recursion depth reached for query: {query}"

        # 1. Local Search
        local_results = self.vector_store.search(query, limit=3)
        if (
            local_results.success
            and local_results.data
            and len(local_results.data.get("results", [])) > 0
        ):
            # Check confidence score
            first_score = 0
            if local_results.data["results"]:
                first_score = local_results.data["results"][0].get("score", 0)
            if first_score > 0.8:
                return f"Found locally (Score {first_score}): {local_results.data}"

        logger.info(
            f"Local confidence low/miss for '{query}'. Initiating external enrichment (Depth {depth})"
        )

        # 2. External Search (CKAN)
        packages = self.ckan.search_packages(query, rows=3)
        if not packages:
            return f"No local or external data found for '{query}'"

        enrichment_summary = []

        # 3. Ingest and Chain
        for pkg in packages:
            resources = self.ckan.find_resources_by_format(pkg, ["CSV", "JSON"])
            if resources:
                target_resource = resources[0]  # Take first valid resource
                url = self.ckan.get_resource_url(target_resource)

                # Ingest
                logger.info(f"Enriching with: {pkg.get('title')} ({url})")
                ingest_result = await ingest_dataset(
                    url, type="api", process_pipeline=["parse"]
                )  # Assume API/Direct link
                enrichment_summary.append(f"Ingested '{pkg.get('title')}': {ingest_result}")

        # 4. Re-Search Local
        retry_results = self.vector_store.search(query, limit=5)

        return (
            "Enrichment Complete:\n"
            + "\n".join(enrichment_summary)
            + f"\n\nFinal Search Results:\n{retry_results.data}"
        )


# Singleton instance
enricher = RecursiveEnricher()


async def recursive_enrichment(query: str, depth: int = 2) -> str:
    """Public entry point for the tool."""
    return await enricher.enrich_and_search(query, depth=0)  # Reset depth for fresh call
