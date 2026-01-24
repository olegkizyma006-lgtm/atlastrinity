import asyncio
import sys
from pathlib import Path

import pandas as pd

# Setup paths
PROJECT_ROOT = Path("/Users/dev/Documents/GitHub/atlastrinity")
sys.path.insert(0, str(PROJECT_ROOT))

# Mock environment
from src.brain.db.manager import db_manager


async def verify_ingestion_pipeline():
    print("--- Starting High-Precision Ingestion Verification ---")
    await db_manager.initialize()

    # 0. Cleanup from previous runs
    from sqlalchemy import text

    async with await db_manager.get_session() as session:
        await session.execute(text("DROP TABLE IF EXISTS dataset_clean_tech"))
        await session.execute(text("DELETE FROM kg_nodes WHERE id='dataset:clean_tech'"))
        await session.commit()

    # 1. Prepare TEST DATA
    dirty_path = Path("/tmp/dirty_dataset.csv")
    dirty_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alpha", None, None, "delta", "trash"],
            "junk": [None, None, None, None, None],  # 100% null
            "nonsense": ["valid", "valid", "NaN", "NULL", "undefined"],  # Trash markers
        },
    )
    dirty_df.to_csv(dirty_path, index=False)

    clean_path = Path("/tmp/clean_dataset.csv")
    clean_df = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "product": ["Quantum Processor", "Neural Interface", "Gravity Stabilizer"],
            "price": [1500.0, 2400.5, 9800.0],
            "status": ["active", "active", "prototype"],
        },
    )
    clean_df.to_csv(clean_path, index=False)

    print("1. Test Datasets prepared.")

    # 2. Test REJECTION (Dirty Data)
    print("2. Testing REJECTION of dirty data...")
    from src.mcp_server.memory_server import ingest_verified_dataset

    dirty_res = await ingest_verified_dataset(
        file_path=str(dirty_path),
        dataset_name="Dirty_Test",
        namespace="test_isolation",
    )

    if dirty_res.get("success") is False and "rejected" in dirty_res.get("error", "").lower():
        print("SUCCESS: Dirty dataset correctly rejected by Quality Guard.")
        # print(f"   Reason: {dirty_res['validation_report']['issues']}")
    else:
        print("FAILURE: Dirty dataset was NOT rejected.")

    # 3. Test ACCEPTANCE (Clean Data)
    print("3. Testing ACCEPTANCE of clean data...")
    clean_res = await ingest_verified_dataset(
        file_path=str(clean_path),
        dataset_name="Clean_Tech",
        namespace="global",
    )

    table_name = None
    if clean_res.get("success") is True:
        print("SUCCESS: Clean dataset accepted and ingested.")
        table_name = clean_res.get("table_name")
        print(f"   Table created: {table_name}")
    else:
        print(f"FAILURE: Clean dataset rejected. Error: {clean_res.get('error')}")
        if "validation_report" in clean_res:
            print(f"   Validation Report: {clean_res['validation_report']}")

    # 4. Verify SQL Persistence & KG Registration
    print("4. Verifying Persistence & Knowledge Graph...")
    if clean_res.get("success") and table_name:
        # Check KG Node
        node_id = clean_res["node_id"]
        from sqlalchemy import text

        async with await db_manager.get_session() as session:
            node_res = await session.execute(
                text("SELECT * FROM kg_nodes WHERE id=:node_id"),
                {"node_id": node_id},
            )
            node = node_res.fetchone()

            if node:
                print(f"SUCCESS: KG Node '{node_id}' registered correctly.")
            else:
                print(f"FAILURE: KG Node '{node_id}' missing.")

            # Check Structured Table contents
            raw_data_res = await session.execute(text(f"SELECT * FROM {table_name}"))
            raw_data = raw_data_res.fetchall()
            if len(raw_data) == 3:
                print(
                    f"SUCCESS: Structured table '{table_name}' contains all {len(raw_data)} rows.",
                )
            else:
                print(
                    f"FAILURE: Structured table '{table_name}' has incorrect row count: {len(raw_data)}",
                )

    print("--- Verification Complete ---")


if __name__ == "__main__":
    asyncio.run(verify_ingestion_pipeline())
