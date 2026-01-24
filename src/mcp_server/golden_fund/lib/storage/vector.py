"""
Vector Storage Adapter for Golden Fund (ChromaDB)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# Try importing chromadb, handle failure for CI/Bootstrap
try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from .types import StorageResult

logger = logging.getLogger("golden_fund.storage.vector")


class VectorStorage:
    """
    Vector storage adapter using ChromaDB for persistence.
    """

    def __init__(
        self, persistence_path: str | None = None, collection_name: str = "golden_fund_vectors"
    ):
        if persistence_path is None:
            self.persistence_path = (
                Path.home() / ".config" / "atlastrinity" / "data" / "golden_fund" / "chroma_db"
            )
        else:
            self.persistence_path = Path(persistence_path)

        self.collection_name = collection_name
        self.enabled = CHROMA_AVAILABLE
        self.client = None
        self.collection = None

        if self.enabled:
            try:
                self.persistence_path.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(self.persistence_path))
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                logger.info(f"VectorStorage (ChromaDB) initialized at {self.persistence_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.enabled = False
        else:
            logger.warning(
                "ChromaDB not available. VectorStorage disabled or running in simulation mode."
            )

    def store(self, data: dict[str, Any] | list[dict[str, Any]]) -> StorageResult:
        """Store data with generated embeddings."""
        if not self.enabled:
            return StorageResult(
                False, "vector", error="VectorStorage is disabled (ChromaDB missing)"
            )

        try:
            if not data:
                return StorageResult(False, "vector", error="No data provided")

            if isinstance(data, dict):
                data = [data]

            ids = [str(uuid.uuid4()) for _ in data]
            documents = []
            metadatas = []

            for record in data:
                # Prepare document content (stringify)
                doc_content = json.dumps(record, default=str)
                documents.append(doc_content)

                # Prepare metadata (flatten or select subset)
                # Chroma metadata must be int, float, str, bool
                meta = {}
                for k, v in record.items():
                    if isinstance(v, str | int | float | bool):
                        meta[k] = v
                    elif isinstance(v, list):
                        meta[k] = str(v)  # Flatten lists
                    elif isinstance(v, dict):
                        meta[k] = json.dumps(v)  # Flatten dicts
                meta["timestamp"] = datetime.now().isoformat()
                metadatas.append(meta)

            # Upsert into Chroma
            if self.collection:
                self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

            record_count = len(data)
            logger.info(
                f"Persisted {record_count} records to ChromaDB collection '{self.collection_name}'"
            )

            return StorageResult(
                True,
                "vector",
                data={
                    "collection": self.collection_name,
                    "records_inserted": record_count,
                    "path": str(self.persistence_path),
                },
            )

        except Exception as e:
            msg = f"Vector storage failed: {e}"
            logger.error(msg)
            return StorageResult(False, "vector", error=msg)

    def search(self, query: str, limit: int = 5) -> StorageResult:
        """Search for similar records."""
        if not self.enabled:
            return StorageResult(False, "vector", error="VectorStorage is disabled")

        try:
            logger.info(f"Searching ChromaDB '{self.collection_name}' for query: {query}")

            if self.collection:
                results = self.collection.query(query_texts=[query], n_results=limit)
            else:
                return StorageResult(False, "vector", error="Chroma collection not initialized")

            # Format results
            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "score": 1.0 - results["distances"][0][i]
                            if results["distances"]
                            else 0,  # Chroma returns distance
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                        }
                    )

            return StorageResult(True, "vector", data={"results": formatted_results})

        except Exception as e:
            return StorageResult(False, "vector", error=f"Vector search failed: {e}")
