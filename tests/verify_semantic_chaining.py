import asyncio
import sys
from pathlib import Path

import pandas as pd

# Setup paths
PROJECT_ROOT = Path("/Users/dev/Documents/GitHub/atlastrinity")
sys.path.insert(0, str(PROJECT_ROOT))

# Mock environment
from src.brain.db.manager import db_manager
from src.mcp_server.memory_server import ingest_verified_dataset, trace_data_chain


async def verify_semantic_chaining():
    print("--- Starting Semantic Dataset Chaining Verification ---")
    await db_manager.initialize()

    # 0. Cleanup from previous runs
    from sqlalchemy import text

    async with await db_manager.get_session() as session:
        await session.execute(text("DROP TABLE IF EXISTS dataset_alpha_pricing"))
        await session.execute(text("DROP TABLE IF EXISTS dataset_beta_logistics"))
        await session.execute(text("DELETE FROM kg_nodes WHERE id LIKE 'dataset:%'"))
        await session.execute(text("DELETE FROM kg_edges WHERE relation='LINKED_TO'"))
        await session.commit()

    # 1. Prepare Datasets
    # Dataset A: Product -> Price
    path_a = Path("/tmp/alpha_pricing.csv")
    df_a = pd.DataFrame({"product_id": ["P001", "P002", "P003"], "price": [100.5, 200.0, 50.75]})
    df_a.to_csv(path_a, index=False)

    # Dataset B: Product -> Manufacturer
    # Note: Column is named 'product_id' to test exact match first
    path_b = Path("/tmp/beta_logistics.csv")
    df_b = pd.DataFrame(
        {
            "product_id": ["P001", "P002", "P004"],
            "manufacturer": ["TechCorp", "NeuroDynamics", "SpaceX"],
        },
    )
    df_b.to_csv(path_b, index=False)

    print("1. Test Datasets prepared.")

    # 2. Ingest Dataset A
    print("2. Ingesting Dataset A...")
    res_a = await ingest_verified_dataset(
        file_path=str(path_a),
        dataset_name="Alpha_Pricing",
        namespace="global",
    )
    print(f"   Dataset A node: {res_a.get('node_id')}")

    # 3. Ingest Dataset B
    print("3. Ingesting Dataset B...")
    res_b = await ingest_verified_dataset(
        file_path=str(path_b),
        dataset_name="Beta_Logistics",
        namespace="global",
    )
    print(f"   Dataset B node: {res_b.get('node_id')}")
    print(f"   Links discovered: {res_b.get('links_discovered')}")

    # 4. Verify Chaining Logic
    print("4. Verifying record tracing across datasets...")
    # Expected: P001 should have price from A and manufacturer from B
    trace_res = await trace_data_chain(start_value="P001", start_dataset_id=res_a["node_id"])

    record = trace_res.get("unified_record", {})
    print(f"   Tracing result for P001: {record}")

    if "price" in record and "manufacturer" in record:
        print("SUCCESS: Full data chain reconstructed across tables!")
    else:
        print("FAILURE: Data chain incomplete.")
        print(f"   Chain length: {trace_res.get('chain_length')}")
        print(f"   Traversed Path: {trace_res.get('traversed_path')}")

    print("--- Verification Complete ---")


if __name__ == "__main__":
    asyncio.run(verify_semantic_chaining())
