import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

# Setup paths for internal imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, "..", "..")
sys.path.insert(0, os.path.abspath(root))

from brain.db.manager import db_manager  # noqa: E402
from brain.knowledge_graph import knowledge_graph  # noqa: E402
from brain.memory import long_term_memory  # noqa: E402

server = FastMCP("memory")


def _get_id(name: str) -> str:
    """Standardize entity ID format"""
    name = str(name).strip()
    if name.startswith("entity:"):
        return name
    return f"entity:{name}"


def _normalize_entity(ent: Dict[str, Any]) -> Dict[str, Any]:
    name = str(ent.get("name", "")).strip()
    entity_type = str(ent.get("entityType", "concept")).strip() or "concept"
    observations = ent.get("observations") or []
    if not isinstance(observations, list):
        observations = [str(observations)]
    observations = [str(o) for o in observations if str(o).strip()]
    return {"name": name, "entityType": entity_type, "observations": observations}


@server.tool()
async def create_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create or update multiple entities in the knowledge graph (SQLite + ChromaDB).

    Args:
        entities: List of entity dictionaries. Each must have a 'name' field.
                 Optional fields: 'entityType' (default: 'ENTITY'), 'observations' (list of strings).
    """
    if not isinstance(entities, list) or not entities:
        return {"error": "entities must be a non-empty list"}

    await db_manager.initialize()

    created: List[str] = []
    updated: List[str] = []

    for ent in entities:
        n = _normalize_entity(ent)
        name = n["name"]
        if not name:
            continue

        node_id = _get_id(name)
        
        # Attributes for SQLite
        attributes = {
            "entity_type": n["entityType"],
            "observations": n["observations"],
            "description": f"Entity of type {n['entityType']} with {len(n['observations'])} observations.",
            "content": "\n".join(n["observations"])
        }

        success = await knowledge_graph.add_node(
            node_type="ENTITY",
            node_id=node_id,
            attributes=attributes,
            sync_to_vector=True
        )
        
        if success:
            created.append(name) # Simplification: SQLite upsert doesn't differentiate easily here
        
    return {"success": True, "created": created, "backend": "sqlite+chromadb"}


@server.tool()
async def batch_add_nodes(nodes: List[Dict[str, Any]], namespace: str = "global") -> Dict[str, Any]:
    """
    Optimized batch insertion of multiple nodes into the Knowledge Graph.
    
    Args:
        nodes: List of dicts, each with 'node_id', 'node_type', and 'attributes'.
        namespace: Isolation bucket for these nodes.
    """
    await db_manager.initialize()
    return await knowledge_graph.batch_add_nodes(nodes, namespace=namespace)


@server.tool()
async def bulk_ingest_table(file_path: str, table_name: str, namespace: str = "global", task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest a large table (CSV/JSON/XLSX) into the Knowledge Graph as a DATASET node.
    This creates a summary node and indexes the content for semantic recall.
    
    Args:
        file_path: Path to the data file.
        table_name: Name of the dataset for the KG.
        namespace: Isolation bucket.
        task_id: Optional association with a task.
    """
    import pandas as pd
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
        
    try:
        # Load data
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        elif path.suffix.lower() in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        else:
            return {"error": f"Unsupported format: {path.suffix}"}
            
        # Create a summary node in the KG
        row_count = len(df)
        cols = list(df.columns)
        summary = f"Dataset '{table_name}' with {row_count} rows and columns: {', '.join(cols)}."
        
        attributes = {
            "description": summary,
            "row_count": row_count,
            "columns": cols,
            "file_path": str(path.absolute()),
            "content": df.head(5).to_string() # Store preview 
        }
        
        node_id = f"dataset:{table_name.lower().replace(' ', '_')}"
        await knowledge_graph.add_node(
            node_type="DATASET",
            node_id=node_id,
            attributes=attributes,
            namespace=namespace,
            task_id=task_id,
            sync_to_vector=True
        )
        
        # Batch ingest the first 100 rows as sub-nodes if small, or just reference the table
        # For "Big Data", we standardly index the schema and a sample.
        
        return {
            "success": True, 
            "node_id": node_id, 
            "row_count": row_count, 
            "namespace": namespace,
            "message": "Dataset indexed. Large tables are stored as summary nodes with vectorized samples."
        }
        
    except Exception as e:
        return {"error": str(e)}


@server.tool()
async def add_observations(name: str, observations: List[str]) -> Dict[str, Any]:
    """
    Add new observations to an existing entity.

    Args:
        name: The name of the entity to update
        observations: List of new observation strings to add
    """
    name = str(name or "").strip()
    if not name:
        return {"error": "name is required"}
    
    await db_manager.initialize()
    node_id = _get_id(name)
    
    # Get existing
    from sqlalchemy import select
    from brain.db.schema import KGNode
    
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode).where(KGNode.id == node_id)
        res = await session.execute(stmt)
        node = res.scalar()
        
        if not node:
            await session.close()
            return {"error": f"Entity '{name}' not found. Use create_entities first."}
        
        attr = node.attributes or {}
        cur_obs = attr.get("observations", [])
        new_obs = [str(o) for o in observations if str(o).strip()]
        merged = list(dict.fromkeys([*cur_obs, *new_obs]))
        
        attr["observations"] = merged
        attr["content"] = "\n".join(merged) # Sync for vector embedding
        
        await knowledge_graph.add_node(
            node_type="ENTITY",
            node_id=node_id,
            attributes=attr,
            sync_to_vector=True
        )
    finally:
        await session.close()

    return {"success": True, "name": name, "observations_count": len(merged)}


@server.tool()
async def get_entity(name: str) -> Dict[str, Any]:
    """
    Retrieve full details of a specific entity.
    """
    await db_manager.initialize()
    node_id = _get_id(name)
    
    from sqlalchemy import select
    from brain.db.schema import KGNode
    
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode).where(KGNode.id == node_id)
        res = await session.execute(stmt)
        node = res.scalar()
        
        if not node:
            return {"error": "not found"}
            
        return {
            "success": True,
            "name": name,
            "entityType": node.attributes.get("entity_type", "ENTITY"),
            "observations": node.attributes.get("observations", []),
            "last_updated": node.last_updated.isoformat() if node.last_updated else None
        }
    finally:
        await session.close()


@server.tool()
async def list_entities() -> Dict[str, Any]:
    """
    List all entity names in the knowledge graph.
    """
    await db_manager.initialize()
    
    from sqlalchemy import select
    from brain.db.schema import KGNode
    
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode.id).where(KGNode.type == "ENTITY")
        res = await session.execute(stmt)
        names = [str(row[0]).replace("entity:", "") for row in res.all()]
    finally:
        await session.close()
        
    return {"success": True, "names": sorted(names), "count": len(names)}


@server.tool()
async def search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Semantic search for entities matching a query string (via ChromaDB embeddings).

    Args:
        query: Text to search for within entity names, types, and observations
        limit: Maximum number of results to return (default: 10)
    """
    q = str(query or "").strip().lower()
    if not q:
        return {"error": "query is required"}

    lim = max(1, min(int(limit), 50))
    
    # 1. Semantic search via ChromaDB (Fastest and smartest)
    if long_term_memory.available:
        results = long_term_memory.knowledge.query(
            query_texts=[q],
            n_results=lim,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results and isinstance(results.get("documents"), list) and results.get("documents"):
            docs_list = results.get("documents") or [[]]
            docs = docs_list[0] if docs_list else []
            
            metas_list = results.get("metadatas") or [[]]
            metas = metas_list[0] if metas_list else []
            
            ids_list = results.get("ids") or [[]]
            ids = ids_list[0] if ids_list else []
            
            dists_list = results.get("distances") or [[]]
            dists = dists_list[0] if dists_list else []
            
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                # Filter to only show ENTITY types in this tool
                if isinstance(meta, dict) and meta.get("type") == "ENTITY":
                    formatted.append({
                        "name": str(ids[i]).replace("entity:", "") if i < len(ids) else "unknown",
                        "entityType": meta.get("entity_type", "ENTITY"),
                        "observations": meta.get("observations", []),
                        "score": 1.0 - (dists[i] if i < len(dists) else 0.5)
                    })
        
        return {"success": True, "results": formatted, "count": len(formatted), "method": "semantic"}

    # 2. Fallback to SQL ILIKE search if Chroma is down
    from sqlalchemy import select, or_
    from brain.db.schema import KGNode
    
    await db_manager.initialize()
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode).where(
            or_(
                KGNode.id.ilike(f"%{q}%"),
                KGNode.attributes["content"].astext.ilike(f"%{q}%")
            )
        ).limit(lim)
        res = await session.execute(stmt)
        nodes = res.scalars().all()
        
        results = []
        for n in nodes:
            results.append({
                "name": n.id.replace("entity:", ""),
                "entityType": n.attributes.get("entity_type", "ENTITY"),
                "observations": n.attributes.get("observations", [])
            })
    finally:
        await session.close()
            
    return {"success": True, "results": results, "count": len(results), "method": "sql_fallback"}


@server.tool()
async def create_relation(source: str, target: str, relation: str) -> Dict[str, Any]:
    """
    Create a relationship between two entities in the knowledge graph.

    Args:
        source: Name of the source entity
        target: Name of the target entity
        relation: Type of relationship (e.g. 'part_of', 'depends_on', 'works_with')
    """
    await db_manager.initialize()
    source_id = _get_id(source)
    target_id = _get_id(target)
    
    success = await knowledge_graph.add_edge(
        source_id=source_id,
        target_id=target_id,
        relation=relation
    )
    
    if not success:
        return {"error": "Failed to create relation. Ensure both entities exist first."}
        
    return {"success": True, "source": source, "target": target, "relation": relation}


@server.tool()
async def search_nodes(query: str, limit: int = 10) -> Dict[str, Any]:
    """Alias for search function to maintain compatibility"""
    return await search(query, limit)


@server.tool()
async def delete_entity(name: str) -> Dict[str, Any]:
    """
    Delete an entity from the knowledge graph (SQLite + ChromaDB).
    """
    name = str(name or "").strip()
    if not name:
        return {"error": "name is required"}

    await db_manager.initialize()
    node_id = _get_id(name)

    from brain.db.schema import KGNode
    from sqlalchemy import delete
    
    session = await db_manager.get_session()
    try:
        # Delete from structured DB (SQLite)
        stmt = delete(KGNode).where(KGNode.id == node_id)
        await session.execute(stmt)
        await session.commit()
    finally:
        await session.close()
    
    # Delete from ChromaDB
    if long_term_memory.available:
        try:
            long_term_memory.knowledge.delete(ids=[node_id])
        except Exception:
            pass

    return {"success": True, "deleted": True}


if __name__ == "__main__":
    import sys
    try:
        server.run()
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)
    except BaseException as e:
        def contains_broken_pipe(exc):
            if isinstance(exc, BrokenPipeError) or "Broken pipe" in str(exc):
                return True
            if hasattr(exc, "exceptions"):
                return any(contains_broken_pipe(inner) for inner in exc.exceptions)
            return False
        if contains_broken_pipe(e):
            sys.exit(0)
        raise
