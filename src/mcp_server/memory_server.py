import sys
from typing import Any, cast

from mcp.server import FastMCP

from src.brain.db.manager import db_manager
from src.brain.knowledge_graph import knowledge_graph
from src.brain.logger import logger
from src.brain.memory import long_term_memory

server = FastMCP("memory")


def _get_id(name: str) -> str:
    """Standardize entity ID format"""
    name = str(name).strip()
    if name.startswith("entity:"):
        return name
    return f"entity:{name}"


def _normalize_entity(ent: dict[str, Any]) -> dict[str, Any]:
    name = str(ent.get("name", "")).strip()
    entity_type = str(ent.get("entityType", "concept")).strip() or "concept"
    observations = ent.get("observations") or []
    if not isinstance(observations, list):
        observations = [str(observations)]
    observations = [str(o) for o in observations if str(o).strip()]
    return {"name": name, "entityType": entity_type, "observations": observations}


@server.tool()
async def create_entities(
    entities: list[dict[str, Any]], namespace: str = "global", task_id: str | None = None,
) -> dict[str, Any]:
    """Args:
    entities: List of entity dictionaries. Each must have a 'name' field.
    namespace: Isolation bucket for these entities.
    task_id: Optional task association.

    """
    if not isinstance(entities, list) or not entities:
        return {"error": "entities must be a non-empty list"}

    await db_manager.initialize()

    created: list[str] = []

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
            "content": "\n".join(n["observations"]),
        }

        success = await knowledge_graph.add_node(
            node_type="ENTITY",
            node_id=node_id,
            attributes=attributes,
            namespace=namespace,
            task_id=task_id,
            sync_to_vector=True,
        )

        if success:
            created.append(name)  # Simplification: SQLite upsert doesn't differentiate easily here

    return {"success": True, "created": created, "backend": "sqlite+chromadb"}


@server.tool()
async def batch_add_nodes(nodes: list[dict[str, Any]], namespace: str = "global") -> dict[str, Any]:
    """Optimized batch insertion of multiple nodes into the Knowledge Graph.

    Args:
        nodes: List of dicts, each with 'node_id', 'node_type', and 'attributes'.
        namespace: Isolation bucket for these nodes.

    """
    await db_manager.initialize()
    return await knowledge_graph.batch_add_nodes(nodes, namespace=namespace)


@server.tool()
async def bulk_ingest_table(
    file_path: str, table_name: str, namespace: str = "global", task_id: str | None = None,
) -> dict[str, Any]:
    """Ingest a large table (CSV/JSON/XLSX) into the Knowledge Graph as a DATASET node.
    This creates a summary node and indexes the content for semantic recall.

    Args:
        file_path: Path to the data file.
        table_name: Name of the dataset for the KG.
        namespace: Isolation bucket.
        task_id: Optional association with a task.

    """
    from pathlib import Path

    import pandas as pd

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
            "content": df.head(5).to_string(),  # Store preview
        }

        node_id = f"dataset:{table_name.lower().replace(' ', '_')}"
        await knowledge_graph.add_node(
            node_type="DATASET",
            node_id=node_id,
            attributes=attributes,
            namespace=namespace,
            task_id=task_id,
            sync_to_vector=True,
        )

        # Batch ingest the first 100 rows as sub-nodes if small, or just reference the table
        # For "Big Data", we standardly index the schema and a sample.

        return {
            "success": True,
            "node_id": node_id,
            "row_count": row_count,
            "namespace": namespace,
            "message": "Dataset indexed. Large tables are stored as summary nodes with vectorized samples.",
        }

    except Exception as e:
        return {"error": str(e)}


@server.tool()
async def add_observations(
    name: str, observations: list[str], namespace: str = "global", task_id: str | None = None,
) -> dict[str, Any]:
    """Args:
    name: The name of the entity to update
    observations: List of new observation strings to add
    namespace: Optional namespace filter.

    """
    name = str(name or "").strip()
    if not name:
        return {"error": "name is required"}

    await db_manager.initialize()
    node_id = _get_id(name)

    # Get existing
    from sqlalchemy import select

    from src.brain.db.schema import KGNode

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
        attr["content"] = "\n".join(merged)  # Sync for vector embedding

        await knowledge_graph.add_node(
            node_type="ENTITY",
            node_id=node_id,
            attributes=attr,
            namespace=node.namespace if hasattr(node, "namespace") else "global",
            sync_to_vector=True,
        )
    finally:
        await session.close()

    return {"success": True, "name": name, "observations_count": len(merged)}


@server.tool()
async def get_entity(name: str) -> dict[str, Any]:
    """Retrieve full details of a specific entity.
    """
    await db_manager.initialize()
    node_id = _get_id(name)

    from sqlalchemy import select

    from src.brain.db.schema import KGNode

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
            "last_updated": node.last_updated.isoformat() if node.last_updated else None,
        }
    finally:
        await session.close()


@server.tool()
async def list_entities() -> dict[str, Any]:
    """List all entity names in the knowledge graph.
    """
    await db_manager.initialize()

    from sqlalchemy import select

    from src.brain.db.schema import KGNode

    session = await db_manager.get_session()
    try:
        stmt = select(KGNode.id).where(KGNode.type == "ENTITY")
        res = await session.execute(stmt)
        names = [str(row[0]).replace("entity:", "") for row in res.all()]
    finally:
        await session.close()

    return {"success": True, "names": sorted(names), "count": len(names)}


@server.tool()
async def search(query: str, limit: int = 10, namespace: str | None = None) -> dict[str, Any]:
    """Args:
    query: Text to search for within entity names, types, and observations
    limit: Maximum number of results to return (default: 10)
    namespace: Optional filter (task_id or 'global')

    """
    q = str(query or "").strip().lower()
    if not q:
        return {"error": "query is required"}

    lim = max(1, min(int(limit), 50))

    # 1. Semantic search via ChromaDB (Fastest and smartest)
    if long_term_memory.available:
        where_filter = {"namespace": namespace} if namespace else None
        results = long_term_memory.knowledge.query(
            query_texts=[q],
            n_results=lim,
            include=["documents", "metadatas", "distances"],
            where=cast("Any", where_filter),
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

            for i, _doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                # Filter to only show ENTITY types in this tool
                if isinstance(meta, dict) and meta.get("type") == "ENTITY":
                    formatted.append(
                        {
                            "name": str(ids[i]).replace("entity:", "")
                            if i < len(ids)
                            else "unknown",
                            "entityType": meta.get("entity_type", "ENTITY"),
                            "observations": meta.get("observations", []),
                            "score": 1.0 - (dists[i] if i < len(dists) else 0.5),
                        },
                    )

        return {
            "success": True,
            "results": formatted,
            "count": len(formatted),
            "method": "semantic",
        }

    # 2. Fallback to SQL ILIKE search if Chroma is down
    from sqlalchemy import or_, select

    from src.brain.db.schema import KGNode

    await db_manager.initialize()
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode).where(
            or_(KGNode.id.ilike(f"%{q}%"), KGNode.attributes["content"].astext.ilike(f"%{q}%")),
        )
        if namespace:
            stmt = stmt.where(KGNode.namespace == namespace)

        stmt = stmt.limit(lim)
        res = await session.execute(stmt)
        nodes = res.scalars().all()

        results = []
        for n in nodes:
            results.append(
                {
                    "name": n.id.replace("entity:", ""),
                    "entityType": n.attributes.get("entity_type", "ENTITY"),
                    "observations": n.attributes.get("observations", []),
                },
            )
    finally:
        await session.close()

    return {"success": True, "results": results, "count": len(results), "method": "sql_fallback"}


@server.tool()
async def create_relation(
    source: str, target: str, relation: str, namespace: str | None = None,
) -> dict[str, Any]:
    """Create a relationship between two entities in the knowledge graph.

    Args:
        source: Name of the source entity
        target: Name of the target entity
        relation: Type of relationship (e.g. 'part_of', 'depends_on', 'works_with')
        namespace: Optional namespace bucket.

    """
    await db_manager.initialize()
    source_id = _get_id(source)
    target_id = _get_id(target)

    success = await knowledge_graph.add_edge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        namespace=str(namespace) if namespace else "global",
    )

    if not success:
        return {"error": "Failed to create relation. Ensure both entities exist first."}

    return {"success": True, "source": source, "target": target, "relation": relation}


@server.tool()
async def search_nodes(query: str, limit: int = 10, namespace: str | None = None) -> dict[str, Any]:
    """Alias for search function to maintain compatibility"""
    return await search(query, limit, namespace)


@server.tool()
async def delete_entity(name: str, namespace: str | None = None) -> dict[str, Any]:
    """Delete an entity from the knowledge graph (SQLite + ChromaDB).
    """
    name = str(name or "").strip()
    if not name:
        return {"error": "name is required"}

    await db_manager.initialize()
    node_id = _get_id(name)

    from sqlalchemy import delete

    from src.brain.db.schema import KGNode

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


@server.tool()
async def ingest_verified_dataset(
    file_path: str, dataset_name: str, namespace: str = "global", task_id: str | None = None,
) -> dict[str, Any]:
    """High-Precision Ingestion Pipeline:
    1. Validates data quality and integrity.
    2. Creates a dedicated SQLite table for structured storage.
    3. Indexes metadata in the Knowledge Graph.
    4. Syncs semantic preview to Vector Memory.
    """
    from pathlib import Path

    import pandas as pd

    from src.brain.data_guard import data_guard

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    # 1. Load Data
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix == ".json":
            df = pd.read_json(path)
        else:
            return {"error": f"Unsupported format: {path.suffix}"}
    except Exception as e:
        return {"error": f"Failed to load file: {e}"}

    # 2. Validate Quality
    validation = data_guard.validate_dataframe(df, dataset_name)
    if not validation["is_worthy"]:
        return {
            "success": False,
            "error": "Dataset rejected by Quality Guard",
            "validation_report": validation,
        }

    # 3. Create Structured Table
    await db_manager.initialize()
    table_name = f"dataset_{dataset_name.lower().replace(' ', '_').replace('-', '_')}"
    db_success = await db_manager.create_table_from_df(table_name, df)

    if not db_success:
        return {"success": False, "error": f"Failed to create structured table '{table_name}'"}

    # 4. Register in Knowledge Graph
    node_id = f"dataset:{dataset_name.lower().replace(' ', '_')}"
    attributes = {
        "description": f"Verified High-Precision Dataset: {dataset_name}",
        "table_name": table_name,
        "row_count": len(df),
        "columns": list(df.columns),
        "validation_score": 1.0 - validation["checks"]["null_ratio"],
        "content_preview": df.head(10).to_string(),  # Semantic preview for vector search
    }

    kg_success = await knowledge_graph.add_node(
        node_type="DATASET",
        node_id=node_id,
        attributes=attributes,
        namespace=namespace,
        task_id=task_id,
        sync_to_vector=True,
    )

    # 5. Automated Semantic Linking (New)
    from src.brain.semantic_linker import semantic_linker

    links = await semantic_linker.discover_links(df, node_id, namespace=namespace)
    for link in links:
        await knowledge_graph.add_edge(
            source_id=link["source"],
            target_id=link["target"],
            relation=link["relation"],
            attributes=link["attributes"],
            namespace=namespace,
        )

    return {
        "success": kg_success,
        "node_id": node_id,
        "table_name": table_name,
        "links_discovered": len(links),
        "validation_report": validation,
        "message": f"Dataset '{dataset_name}' ingested and semantically linked ({len(links)} links) in {namespace}.",
    }


@server.tool()
async def query_db(query: str) -> dict[str, Any]:
    """Execute a raw SQL query against the system database (READ-ONLY).
    Used for technical audits of 'tool_executions' and other tables.

    Args:
        query: SQL SELECT statement.
    """
    await db_manager.initialize()

    # Safety: enforce SELECT only
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed for safety reasons."}

    from sqlalchemy import text

    session = await db_manager.get_session()
    try:
        res = await session.execute(text(query))
        rows = res.fetchall()
        columns = res.keys()

        formatted_rows = []
        for r in rows:
            formatted_rows.append(dict(r._mapping))

        return {
            "success": True,
            "columns": list(columns),
            "rows": formatted_rows,
            "count": len(formatted_rows),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        await session.close()


@server.tool()
async def get_db_schema() -> dict[str, Any]:
    """Retrieve the database schema (tables and columns) for technical audits."""
    await db_manager.initialize()

    from sqlalchemy import inspect

    def _sync_get_schema(connection):
        ins = inspect(connection)
        tables = {}
        for table_name in ins.get_table_names():
            cols = ins.get_columns(table_name)
            tables[table_name] = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c["nullable"]} for c in cols
            ]
        return tables

    async with await db_manager.get_session() as session:
        # engine.connect().run_sync is deprecated, but DatabaseManager doesn't expose engine easily for run_sync
        # We can use the internal _engine
        tables = await db_manager._engine.run_sync(_sync_get_schema)

    return {"success": True, "tables": tables}


@server.tool()
async def trace_data_chain(
    start_value: Any, start_dataset_id: str, namespace: str = "global", max_depth: int = 3,
) -> dict[str, Any]:
    """Recursively follows LINKED_TO edges in the KG to reconstruct a unified record
    accross multiple datasets.
    """
    from sqlalchemy import text

    await db_manager.initialize()
    chain = []
    unified_record = {}
    visited_datasets = set()

    current_targets = [(start_dataset_id, start_value)]

    for _ in range(max_depth):
        next_targets = []
        for dataset_id, value in current_targets:
            if dataset_id in visited_datasets:
                continue
            visited_datasets.add(dataset_id)

            # A. Fetch dataset metadata
            async with await db_manager.get_session() as session:
                from src.brain.db.schema import KGNode

                ds_node = await session.get(KGNode, dataset_id)
                if not ds_node:
                    continue

                table_name = ds_node.attributes.get("table_name")
                if not table_name:
                    continue

                # B. Find the record in the SQLite table
                # We search all columns for the value (could be optimized)
                cols = ds_node.attributes.get("columns", [])
                for col in cols:
                    safe_col = str(col).lower().replace(" ", "_").replace("-", "_")
                    try:
                        sql = f'SELECT * FROM "{table_name}" WHERE "{safe_col}" = :val'
                        res = await session.execute(text(sql), {"val": value})
                        row = res.fetchone()
                        if row:
                            # Found it! Merge into unified record
                            row_dict = dict(row._mapping)
                            unified_record.update(row_dict)
                            chain.append({"dataset": dataset_id, "found_in_col": safe_col})

                            # C. Find linked datasets to jump further
                            graph = await knowledge_graph.get_graph_data(namespace=namespace)
                            edges = [
                                e
                                for e in graph["edges"]
                                if (e["source"] == dataset_id or e["target"] == dataset_id)
                                and e["relation"] == "LINKED_TO"
                            ]

                            for edge in edges:
                                other_ds = (
                                    edge["target"]
                                    if edge["source"] == dataset_id
                                    else edge["source"]
                                )
                                shared_key = edge.get("attributes", {}).get("shared_key")
                                if shared_key and shared_key in row_dict:
                                    next_targets.append((other_ds, row_dict[shared_key]))
                            break  # Move to next dataset jump
                    except Exception as e:
                        logger.warning(f"Error searching {table_name}: {e}")

        current_targets = next_targets
        if not current_targets:
            break

    return {
        "unified_record": unified_record,
        "chain_length": len(chain),
        "traversed_path": chain,
        "value_searched": start_value,
    }


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
