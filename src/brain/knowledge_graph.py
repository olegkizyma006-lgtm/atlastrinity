"""
Knowledge Graph (GraphChain)
Bridges Structured Data (SQL/SQLite) and Semantic Data (ChromaDB)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, cast

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
        attributes: dict[str, Any] | None = None,
        sync_to_vector: bool = True,
        namespace: str = "global",
        task_id: str | None = None,
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
        if attributes is None:
            attributes = {}
        if not db_manager.available:
            return False

        attributes = attributes or {}

        try:
            async with await db_manager.get_session() as session:
                # Attempt insert; if it conflicts (existing id), update fields
                try:
                    new_node = KGNode(
                        id=node_id,
                        type=node_type,
                        namespace=namespace,
                        task_id=task_id,
                        attributes=attributes,
                    )
                    session.add(new_node)
                    await session.commit()
                except IntegrityError:
                    # Existing node - update in place
                    await session.rollback()
                    existing = await session.get(KGNode, node_id)
                    if existing:
                        existing.type = node_type
                        existing.namespace = namespace
                        if namespace == "global":
                            existing.task_id = None
                        elif task_id:
                            existing.task_id = (
                                uuid.UUID(task_id)
                                if isinstance(task_id, str)
                                else cast(Any, task_id)
                            )

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
                        task_id=task_id or "",
                    )

            logger.info(f"[GRAPH] Node stored: {node_id} (Namespace: {namespace})")
            return True

        except Exception as e:
            logger.error(f"[GRAPH] Failed to add node {node_id}: {e}")
            return False

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        attributes: dict[str, Any] | None = None,
        namespace: str = "global",
    ) -> bool:
        """Create a relationship between two nodes."""
        if attributes is None:
            attributes = {}
        if not db_manager.available:
            return False

        try:
            async with await db_manager.get_session() as session:
                new_edge = KGEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    namespace=namespace,
                    attributes=attributes,
                )
                session.add(new_edge)
                await session.commit()

            logger.info(
                f"[GRAPH] Edge: {source_id} -[{relation}]-> {target_id} (Namespace: {namespace})"
            )
            return True
        except IntegrityError:
            # Maybe nodes don't exist yet, or duplicate edge
            return False
        except Exception as e:
            logger.error(f"[GRAPH] Failed to add edge: {e}")
            return False

    async def batch_add_nodes(
        self, nodes: list[dict[str, Any]], namespace: str = "global"
    ) -> dict[str, Any]:
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
                rows.append(
                    {
                        "id": n["node_id"],
                        "type": n.get("node_type", "ENTITY"),
                        "namespace": namespace,
                        "task_id": n.get("task_id"),
                        "attributes": n.get("attributes", {}),
                        "last_updated": datetime.now(),
                    }
                )

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
                            node_type=str(row["type"]),
                            node_id=str(row["id"]),
                            attributes=cast(dict[str, Any], row["attributes"]),
                            namespace=str(row["namespace"]),
                            task_id=cast(str | None, row["task_id"]),
                            sync_to_vector=True,  # Batch vectorization is harder
                        )

            return {"success": True, "count": len(nodes)}
        except Exception as e:
            logger.error(f"[GRAPH] Batch insert failed: {e}")
            return {"success": False, "error": str(e)}

    async def promote_node(
        self, node_id: str, target_namespace: str = "global", agent_name: str = "atlas"
    ) -> bool:
        """
        Elevate a node and its immediate relationships to a new namespace.
        Part of the 'Golden Fund' architecture.
        """
        if not db_manager.available:
            return False

        try:
            async with await db_manager.get_session() as session:
                # 1. Update Node in SQL
                existing = await session.get(KGNode, node_id)
                if not existing:
                    return False

                old_namespace = existing.namespace
                existing.namespace = target_namespace
                # Reset task_id if promoting to global
                if target_namespace == "global":
                    existing.task_id = None

                session.add(existing)

                # 2. Update Edges in SQL
                from sqlalchemy import update

                stmt_edges = (
                    update(KGEdge)
                    .where((KGEdge.source_id == node_id) | (KGEdge.target_id == node_id))
                    .values(namespace=target_namespace)
                )
                await session.execute(stmt_edges)

                # 3. Log Promotion
                from src.brain.db.schema import KnowledgePromotion

                promotion_log = KnowledgePromotion(
                    node_id=node_id,
                    old_namespace=old_namespace,
                    target_namespace=target_namespace,
                    promoted_by=agent_name,
                    reason="Promoted to Golden Fund for long-term retention",
                )
                session.add(promotion_log)

                await session.commit()

            # 3. Update Vector Store
            if long_term_memory.available:
                # ChromaDB metadata update
                try:
                    long_term_memory.knowledge.update(
                        ids=[node_id],
                        metadatas=cast(
                            Any,
                            [
                                {
                                    "namespace": target_namespace,
                                    "task_id": ""
                                    if target_namespace == "global"
                                    else existing.task_id or "",
                                }
                            ],
                        ),
                    )
                except Exception as ve:
                    logger.warning(f"[GRAPH] Vector metadata update failed during promotion: {ve}")

            logger.info(
                f"[GRAPH] Node {node_id} promoted from {old_namespace} to {target_namespace}"
            )
            return True
        except Exception as e:
            logger.error(f"[GRAPH] Promotion failed for {node_id}: {e}")
            return False

    async def get_graph_data(self, namespace: str | None = None) -> dict[str, Any]:
        """Fetch nodes and edges for visualization, optionally filtered by namespace."""
        if not db_manager.available:
            return {"nodes": [], "edges": []}

        try:
            async with await db_manager.get_session() as session:
                # 1. Fetch nodes
                stmt_nodes = select(KGNode)
                if namespace:
                    stmt_nodes = stmt_nodes.where(KGNode.namespace == namespace)

                nodes_result = await session.execute(stmt_nodes)
                nodes = [
                    {
                        "id": n.id,
                        "type": n.type,
                        "attributes": n.attributes,
                        "namespace": n.namespace,
                    }
                    for n in nodes_result.scalars()
                ]

                # 2. Fetch edges
                stmt_edges = select(KGEdge)
                if namespace:
                    stmt_edges = stmt_edges.where(KGEdge.namespace == namespace)

                edges_result = await session.execute(stmt_edges)
                edges = [
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation,
                        "namespace": e.namespace,
                        "attributes": e.attributes,
                    }
                    for e in edges_result.scalars()
                ]

                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"[GRAPH] Failed to fetch graph data: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}


knowledge_graph = KnowledgeGraph()
