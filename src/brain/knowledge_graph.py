"""
Knowledge Graph (GraphChain)
Bridges Structured Data (SQL/SQLite) and Semantic Data (ChromaDB)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .db.manager import db_manager
from .db.schema import KGEdge, KGNode
from .memory import long_term_memory

logger = logging.getLogger("brain.knowledge_graph")


class KnowledgeGraph:
    """
    Manages the Knowledge Graph.
    - Stores nodes/edges in PostgreSQL (Structured)
    - Syncs text content to ChromaDB (Semantic)
    """

    def __init__(self):
        self.chroma_collection_name = "knowledge_graph_nodes"

    async def add_node(
        self,
        node_type: str,
        node_id: str,
        attributes: Dict[str, Any] = None,
        sync_to_vector: bool = True,
    ) -> bool:
        """
        Add or update a node in the graph.

        Args:
            node_type: FILE, TASK, TOOL, CONCEPT
            node_id: Unique URI (e.g. file:///path, task:123)
            attributes: Metadata dict
            sync_to_vector: If True, content is embedded in ChromaDB
        """
        if not db_manager.available:
            return False

        attributes = attributes or {}

        try:
            async with await db_manager.get_session() as session:
                # Attempt insert; if it conflicts (existing id), update fields
                try:
                    new_node = KGNode(id=node_id, type=node_type, attributes=attributes)
                    session.add(new_node)
                    await session.commit()
                except IntegrityError:
                    # Existing node - update in place
                    await session.rollback()
                    existing = await session.get(KGNode, node_id)
                    if existing:
                        existing.type = node_type
                        existing.attributes = attributes
                        existing.last_updated = datetime.now()
                        session.add(existing)
                        await session.commit()

            # Semantic Sync
            if sync_to_vector and long_term_memory.available:
                # Create a text representation for embedding
                # e.g. "FILE: src/main.py. Description: Main entry point..."
                description = attributes.get("description", "")
                content = attributes.get("content", "")

                if description or content:
                    desc = attributes.get("description", "No description")
                    content = attributes.get("content", "")

                    text_repr = f"[{node_type}] ID: {node_id}\n"
                    text_repr += f"SUMMARY: {desc}\n"
                    if content:
                        text_repr += f"CONTENT:\n{content}\n"

                    # Sanitize metadata for ChromaDB (only allows str, int, float, bool)
                    sanitized_metadata = {
                        "type": node_type,
                        "last_updated": datetime.now().isoformat(),
                    }
                    for k, v in attributes.items():
                        if isinstance(v, (list, dict)):
                            sanitized_metadata[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            sanitized_metadata[k] = v

                    long_term_memory.add_knowledge_node(
                        node_id=node_id,
                        text=text_repr,
                        metadata=sanitized_metadata,
                    )

            logger.info(f"[GRAPH] Node stored: {node_id}")
            return True

        except Exception as e:
            logger.error(f"[GRAPH] Failed to add node {node_id}: {e}")
            return False

    async def add_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        """Create a relationship between two nodes."""
        if not db_manager.available:
            return False

        try:
            async with await db_manager.get_session() as session:
                # CRITICAL: Verify node existence to avoid FK violations
                # kg_edges_source_id_fkey or kg_edges_target_id_fkey
                check_src = await session.execute(select(KGNode).where(KGNode.id == source_id))
                if not check_src.scalar():
                    logger.warning(f"[GRAPH] Source node {source_id} not found. Cannot add edge.")
                    return False

                check_tg = await session.execute(select(KGNode).where(KGNode.id == target_id))
                if not check_tg.scalar():
                    logger.warning(f"[GRAPH] Target node {target_id} not found. Cannot add edge.")
                    return False

                entry = KGEdge(source_id=source_id, target_id=target_id, relation=relation)
                session.add(entry)
                await session.commit()
            logger.info(f"[GRAPH] Edge: {source_id} -[{relation}]-> {target_id}")
            return True
        except Exception as e:
            logger.error(f"[GRAPH] Failed to add edge: {e}")
            return False

    async def get_graph_data(self) -> Dict[str, Any]:
        """Fetch all nodes and edges for visualization."""
        if not db_manager.available:
            return {"nodes": [], "edges": []}

        try:
            async with await db_manager.get_session() as session:
                # Fetch all nodes
                nodes_result = await session.execute(select(KGNode))
                nodes = [
                    {"id": n.id, "type": n.type, "attributes": n.attributes}
                    for n in nodes_result.scalars()
                ]

                # Fetch all edges
                edges_result = await session.execute(select(KGEdge))
                edges = [
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation,
                    }
                    for e in edges_result.scalars()
                ]

                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"[GRAPH] Failed to fetch graph data: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}


knowledge_graph = KnowledgeGraph()
