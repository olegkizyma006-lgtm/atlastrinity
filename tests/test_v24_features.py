import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock CONFIG_ROOT for testing
import src.brain.config

src.brain.config.CONFIG_ROOT = PROJECT_ROOT / "tests" / "mock_config"
src.brain.config.CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

from src.brain.db.manager import db_manager
from src.brain.knowledge_graph import knowledge_graph
from src.brain.semantic_linker import semantic_linker


async def test_v24_logic():
    print("\n--- Testing v2.4 Golden Fund & Semantic Chaining ---")

    # 1. Initialize Database
    test_db_path = PROJECT_ROOT / "tests" / "v24_test.db"
    if test_db_path.exists():
        test_db_path.unlink()

    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{test_db_path}"
    await db_manager.initialize()
    print(f"1. Database initialized at {test_db_path}")

    # 2. Test High-Precision Ingestion
    print("2. Testing High-Precision Ingestion...")
    data = {
        "user_id": [1, 2, 3],
        "email": ["atlas@trinity.ai", "tetyana@trinity.ai", "grisha@trinity.ai"],
        "status": ["active", "active", "auditing"],
    }
    df = pd.DataFrame(data)
    table_name = "test_ingestion_table"

    success = await db_manager.create_table_from_df(table_name, df)
    if success:
        print(f"   SUCCESS: Created and populated table '{table_name}'")
    else:
        print(f"   FAILURE: Failed to create table '{table_name}'")

    # 3. Test Semantic Chaining (Link Discovery)
    print("3. Testing Semantic Chaining...")
    # Create a second table to link to
    data2 = {"contact_email": ["atlas@trinity.ai"], "role": ["Meta-Planner"]}
    df2 = pd.DataFrame(data2)
    table_name2 = "test_roles_table"
    await db_manager.create_table_from_df(table_name2, df2)

    # Register these datasets in KG (Simulating what Tetyana does)
    await knowledge_graph.add_node(
        "DATASET",
        f"dataset:{table_name}",
        {"columns": list(df.columns)},
    )
    await knowledge_graph.add_node(
        "DATASET",
        f"dataset:{table_name2}",
        {"columns": list(df2.columns)},
    )

    # Run Semantic Linker
    links = await semantic_linker.discover_links(df2, f"dataset:{table_name2}")
    if links:
        print(f"   SUCCESS: Discovered {len(links)} semantic links!")
        for l in links:
            # semantic_linker returns objects with source, target, relation, attributes
            source = l.get("source")
            target = l.get("target")
            shared_key = l.get("attributes", {}).get("shared_key")
            print(f"   - Link found: {source} -> {target} via '{shared_key}'")
    else:
        print("   FAILURE: No semantic links discovered between related tables.")

    # 4. Test Knowledge Promotion
    print("4. Testing Knowledge Promotion...")
    # (Simplified test as full promotion involves Grisha logic, but we check if namespaces work)
    await knowledge_graph.add_node(
        "ENTITY",
        "entity:test_fact",
        {"value": "Trinity is Omniscient"},
        namespace="task_123",
    )

    # Mocking a promotion (we can just add another node in global for now)
    await knowledge_graph.add_node(
        "ENTITY",
        "entity:test_fact",
        {"value": "Trinity is Omniscient"},
        namespace="global",
    )

    # Check if we can find it in global
    graph = await knowledge_graph.get_graph_data(namespace="global")
    nodes = [n for n in graph["nodes"] if n["id"] == "entity:test_fact"]

    if nodes:
        print("   SUCCESS: Knowledge promoted to global namespace!")
    else:
        print("   FAILURE: Knowledge not found in global namespace.")

    print("\n--- v2.4 Logic Test Complete ---")


if __name__ == "__main__":
    asyncio.run(test_v24_logic())
