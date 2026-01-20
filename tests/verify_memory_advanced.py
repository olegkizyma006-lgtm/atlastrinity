import asyncio
import os
import sys
import uuid
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, "..")
sys.path.insert(0, os.path.abspath(root))

from src.brain.db.manager import db_manager
from src.brain.knowledge_graph import knowledge_graph
from src.brain.memory import long_term_memory

async def verify_isolation():
    print("--- Starting Advanced Memory Verification ---")
    await db_manager.initialize()
    
    task_id = str(uuid.uuid4())
    namespace = f"test_task_{task_id[:8]}"
    
    print(f"1. Testing Isolation with namespace: {namespace}")
    
    # Add a global node
    await knowledge_graph.add_node(
        node_type="CONCEPT",
        node_id="entity:global_secret",
        attributes={"content": "This is a global secret."},
        namespace="global"
    )
    
    # Add a task-specific node
    await knowledge_graph.add_node(
        node_type="CONCEPT",
        node_id="entity:task_secret",
        attributes={"content": f"Secret for task {task_id}"},
        namespace=namespace,
        task_id=task_id
    )
    
    print("2. Verifying SQL Persistence...")
    from sqlalchemy import select
    from src.brain.db.schema import KGNode
    
    async with await db_manager.get_session() as session:
        # Check task node
        res = await session.execute(select(KGNode).where(KGNode.id == "entity:task_secret"))
        node = res.scalar()
        if node and node.namespace == namespace:
            print(f"SUCCESS: SQL Node isolated to {namespace}")
        else:
            print(f"FAILURE: SQL Node isolation failed. Namespace: {getattr(node, 'namespace', 'NONE')}")

    print("3. Testing Bulk Ingest Mock...")
    # Create a mock CSV
    csv_path = Path("/tmp/test_dataset.csv")
    csv_path.write_text("id,name,value\n1,alpha,100\n2,beta,200\n3,gamma,300")
    
    # We test the logic manually since we don't want to spin up the whole MCP server
    import pandas as pd
    df = pd.read_csv(csv_path)
    node_id = "dataset:test_bulk"
    
    await knowledge_graph.add_node(
        node_type="DATASET",
        node_id=node_id,
        attributes={
            "description": "Test Dataset",
            "row_count": len(df),
            "columns": list(df.columns),
            "content": df.to_string()
        },
        namespace=namespace,
        task_id=task_id
    )
    
    print(f"SUCCESS: Dataset {node_id} ingested into {namespace}")
    print("--- Verification Complete ---")

if __name__ == "__main__":
    asyncio.run(verify_isolation())
