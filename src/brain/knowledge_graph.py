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
    - Stores nodes/edges in SQLite (Structured)
    - Syncs text content to ChromaDB (Semantic)
    """

    def __init__(self):
        self.chroma_collection_name = "knowledge_graph_nodes"

    async def add_node(
        self,
        node_type: str,
        node_id: str,
        attributes: Dict[str, Any] = {},
        sync_to_vector: bool = True,
        namespace: str = "global",
        task_id: Optional[str] = None
    ) -> bool:
        """
        Add or update a node in the graph.

        Args:
            node_type: FILE, TASK, TOOL, CONCEPT, DATASET
            node_id: Unique URI (e.g. file:///path, task:123)
            attributes: Metadata dict
            sync_to_vector: If True, content is embedded in ChromaDB
            namespace: Isolation bucket (default: global)
            task_id: Associated Task UUID string
        """
        if not db_manager.available:
            return False

        attributes = attributes or {}

        try:
            async with await db_manager.get_session() as session:
                # Attempt insert; if it conflicts (existing id), update fields
                try:
                    new_node = KGNode(id=node_id, type=node_type, namespace=namespace, task_id=task_id, attributes=attributes)
                    session.add(new_node)
                    await session.commit()
                except IntegrityError:
                    # Existing node - update in place
                    await session.rollback()
                    existing = await session.get(KGNode, node_id)
                    if existing:
                        existing.type = node_type
                        existing.namespace = namespace
                        existing.task_id = task_id
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
                        "namespace": namespace,
                        "task_id": task_id or "",
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
                        namespace=namespace,
                        task_id=task_id or ""
                    )

            logger.info(f"[GRAPH] Node stored: {node_id} (Namespace: {namespace})")
            return True

        except Exception as e:
            logger.error(f"[GRAPH] Failed to add node {node_id}: {e}")
            return False

    async def add_edge(self, source_id: str, target_id: str, relation: str, namespace: str = "global") -> bool:
        """Create a relationship between two nodes."""
        if not db_manager.available:
            return False

        try:
            async with await db_manager.get_session() as session:
                # CRITICAL: Verify node existence to avoid FK violations
                check_src = await session.execute(select(KGNode).where(KGNode.id == source_id))
                if not check_src.scalar():
                    logger.warning(f"[GRAPH] Source node {source_id} not found. Cannot add edge.")
                    return False

                check_tg = await session.execute(select(KGNode).where(KGNode.id == target_id))
                if not check_tg.scalar():
                    logger.warning(f"[GRAPH] Target node {target_id} not found. Cannot add edge.")
                    return False

                entry = KGEdge(source_id=source_id, target_id=target_id, relation=relation, namespace=namespace)
                session.add(entry)
                await session.commit()
            logger.info(f"[GRAPH] Edge: {source_id} -[{relation}]-> {target_id} (Namespace: {namespace})")
            return True
        except Exception as e:
            logger.error(f"[GRAPH] Failed to add edge: {e}")
            return False

    async def batch_add_nodes(self, nodes: List[Dict[str, Any]], namespace: str = "global") -> Dict[str, Any]:
        """
        Optimized batch insertion of nodes.
        Used for bulk data ingestion.
        """
        if not db_manager.available or not nodes:
            return {"success": False, "count": 0}

        try:
            from sqlalchemy import insert
            
            # Prepare rows for SQLite
            rows = []
            for n in nodes:
                rows.append({
                    "id": n["node_id"],
                    "type": n.get("node_type", "ENTITY"),
                    "namespace": namespace,
                    "task_id": n.get("task_id"),
                    "attributes": n.get("attributes", {}),
                    "last_updated": datetime.now()
                })

            async with await db_manager.get_session() as session:
                # Use bulk upsert/insert logic
                # For SQLite, we can't easily do 'on conflict', but for new batches we use insert
                # To be safe and simple, we do it in a loop if overhead is low, 
                # or use core insert if we know they are new.
                stmt = insert(KGNode).values(rows)
                # Note: insert().values() doesn't handle conflicts. 
                # For big background tasks, we usually assume new data.
                try:
                    await session.execute(stmt)
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    # Fallback to individual add for mixed state
                    for row in rows:
                        await self.add_node(
                            node_type=row["type"],
                            node_id=row["id"],
                            attributes=row["attributes"],
                            namespace=row["namespace"],
                            task_id=row["task_id"],
                            sync_to_vector=True # Batch vectorization is harder
                        )

            return {"success": True, "count": len(nodes)}
        except Exception as e:
            logger.error(f"[GRAPH] Batch insert failed: {e}")
            return {"success": False, "error": str(e)}

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
