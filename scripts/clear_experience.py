import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

from src.brain.db.manager import db_manager
from src.brain.memory import long_term_memory


async def clear_experience():
    print("🧹 Clearing all AtlasTrinity experience...")

    # 1. Clear PostgreSQL
    if not db_manager.available:
        await db_manager.initialize()

    if db_manager.available and db_manager._engine:
        try:
            async with db_manager._engine.begin() as conn:
                print("Deleting from PostgreSQL tables...")
                # Order matters due to FKs
                tables = [
                    "kg_edges",
                    "kg_nodes",
                    "tool_executions",
                    "task_steps",
                    "tasks",
                    "sessions",
                    "logs",
                ]
                for table in tables:
                    await conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                print("✅ PostgreSQL tables cleared.")
        except Exception as e:
            print(f"❌ Failed to clear PostgreSQL: {e}")

    # 2. Clear ChromaDB
    if long_term_memory.available:
        try:
            print("Deleting ChromaDB collections...")
            # We recreate the collections to clear them
            for collection_name in ["lessons", "strategies", "knowledge_graph_nodes"]:
                try:
                    long_term_memory.client.delete_collection(collection_name)
                    print(f"✅ Deleted collection {collection_name}")
                except Exception:
                    print(f"ℹ️ Collection {collection_name} does not exist or already deleted.")

            # Re-initialize collections
            print("Re-initializing collections...")
            long_term_memory.lessons = long_term_memory.client.get_or_create_collection(
                name="lessons",
                metadata={"description": "Error patterns and solutions"},
            )
            long_term_memory.strategies = long_term_memory.client.get_or_create_collection(
                name="strategies",
                metadata={"description": "Successful task execution strategies"},
            )
            long_term_memory.knowledge = long_term_memory.client.get_or_create_collection(
                name="knowledge_graph_nodes",
                metadata={"description": "Semantic embedding of Knowledge Graph nodes"},
            )
            print("✅ ChromaDB collections reset.")
        except Exception as e:
            print(f"❌ Failed to clear ChromaDB: {e}")
    else:
        print("ℹ️ ChromaDB not available, skipping.")

    print("\n✨ All experience has been reset. Atlas is now a blank slate.")


if __name__ == "__main__":
    asyncio.run(clear_experience())
