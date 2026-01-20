# Setup paths (for standalone run if needed)
import os
import sys
from typing import Any

from mcp.server import FastMCP

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, "..", "..")
sys.path.insert(0, os.path.abspath(root))

from src.brain.db.manager import db_manager  # noqa: E402
from src.brain.knowledge_graph import knowledge_graph  # noqa: E402

server = FastMCP("graph")


@server.tool()
async def get_graph_json(namespace: str | None = None) -> dict[str, Any]:
    """
    Returns the Knowledge Graph in JSON format.

    Args:
        namespace: Optional filter (e.g. task_id or 'global').
    """
    await db_manager.initialize()
    return await knowledge_graph.get_graph_data(namespace=namespace)


@server.tool()
async def generate_mermaid(node_type: str | None = None, namespace: str | None = None) -> str:
    """
    Generates a Mermaid.js flowchart representation of the current Knowledge Graph.

    Args:
        node_type: Optional filter (FILE, TASK, TOOL, ENTITY)
        namespace: Optional filter (task_id or 'global')
    """
    await db_manager.initialize()
    data = await knowledge_graph.get_graph_data(namespace=namespace)

    if "error" in data:
        return f"Error: {data['error']}"

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if node_type:
        node_type = node_type.upper()
        nodes = [n for n in nodes if n["type"] == node_type]
        valid_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in valid_ids or e["target"] in valid_ids]

    if not nodes:
        return f"graph TD\n  Empty[No {node_type or 'nodes'} found in graph]"

    mermaid = "graph TD\n"

    # 1. Add Nodes with types as styles
    # Clean IDs for Mermaid (no special chars, use aliases)
    node_map = {}
    for i, n in enumerate(nodes):
        alias = f"N{i}"
        node_map[n["id"]] = alias
        label = n["id"].split("/")[-1] or n["id"]
        # Limit label length
        if len(label) > 30:
            label = label[:27] + "..."

        type_icon = {
            "FILE": "📄",
            "TASK": "🎯",
            "TOOL": "🛠️",
            "USER": "👤",
            "CONCEPT": "💡",
            "ENTITY": "🧠",
        }.get(n["type"], "⚪")

        mermaid += f'  {alias}["{type_icon} {label}"]\n'

    # 2. Add Edges
    for e in edges:
        source_id = e.get("source")
        target_id = e.get("target")
        relation = e.get("relation", "rel")

        if source_id in node_map and target_id in node_map:
            mermaid += f'  {node_map[source_id]} -- "{relation}" --> {node_map[target_id]}\n'

    return mermaid


@server.tool()
async def get_node_details(node_id: str) -> dict[str, Any]:
    """Retrieve all attributes of a specific node."""
    from sqlalchemy import select

    from src.brain.db.schema import KGNode

    await db_manager.initialize()
    session = await db_manager.get_session()
    try:
        stmt = select(KGNode).where(KGNode.id == node_id)
        res = await session.execute(stmt)
        node = res.scalar()
        if not node:
            return {"error": "Node not found"}
        return {
            "id": node.id,
            "type": node.type,
            "attributes": node.attributes,
            "last_updated": node.last_updated.isoformat() if node.last_updated else None,
        }
    finally:
        await session.close()


@server.tool()
async def get_related_nodes(node_id: str) -> dict[str, Any]:
    """Find all nodes directly connected to the specified node."""
    from sqlalchemy import or_, select

    from src.brain.db.schema import KGEdge

    await db_manager.initialize()
    session = await db_manager.get_session()
    try:
        stmt = select(KGEdge).where(or_(KGEdge.source_id == node_id, KGEdge.target_id == node_id))
        res = await session.execute(stmt)
        edges = res.scalars().all()

        return {
            "node_id": node_id,
            "relations": [
                {"source": e.source_id, "target": e.target_id, "relation": e.relation}
                for e in edges
            ],
        }
    finally:
        await session.close()


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
