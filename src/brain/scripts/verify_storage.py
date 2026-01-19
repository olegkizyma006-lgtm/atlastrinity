
import asyncio
import sys
import os
import logging
import json
from sqlalchemy import select, func, text

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_storage")

# Import system managers
try:
    from src.brain.memory import long_term_memory
    from src.brain.db.manager import db_manager
    from src.brain.state_manager import state_manager
    from src.brain.db.schema import (
        KGNode, KGEdge, Session, Task, TaskStep, ToolExecution, 
        LogEntry, AgentMessage, RecoveryAttempt
    )
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

async def verify_chromadb():
    print("\n--- 1. Semantic/Vector Storage (ChromaDB) ---")
    if not long_term_memory.available:
        print("❌ ChromaDB is NOT available.")
        return

    print("✅ ChromaDB is available.")
    print(f"Path: {long_term_memory.get_stats().get('path')}")
    
    collections = {
        "lessons": long_term_memory.lessons,
        "strategies": long_term_memory.strategies,
        "knowledge_graph_nodes": long_term_memory.knowledge
    }

    for name, col in collections.items():
        count = col.count()
        print(f"   - Collection '{name}': {count} documents")
        if count > 0:
            # Peek at one item to confirm content
            peek = col.peek(limit=1)
            print(f"     Sample ID: {peek['ids'][0]}")

async def verify_database():
    print("\n--- 2. Structured Storage (SQLite) ---")
    await db_manager.initialize()
    if not db_manager.available:
        print("❌ Structured DB is NOT available.")
        return

    try:
        async with await db_manager.get_session() as session:
            # Check connection
            res = await session.execute(text("SELECT 1"))
                if res.scalar() == 1:
                    print("✅ Database Connection: OK")
            
            # Count rows in key tables
            tables = [
                ("KGNode (Semantic Nodes)", KGNode),
                ("KGEdge (Relations)", KGEdge),
                ("Session", Session),
                ("Task", Task),
                ("TaskStep", TaskStep),
                ("ToolExecution", ToolExecution),
                ("LogEntry", LogEntry),
                ("AgentMessage", AgentMessage),
                ("RecoveryAttempt", RecoveryAttempt)
            ]

            for name, model in tables:
                try:
                    count_res = await session.execute(select(func.count()).select_from(model))
                    count = count_res.scalar()
                    print(f"   - Table '{name}': {count} rows")
                except Exception as e:
                    print(f"   ❌ Table '{name}': Error/Missing - {e}")

    except Exception as e:
        print(f"❌ Database Verification Failed: {e}")

async def verify_redis():
    print("\n--- 3. State & Cache (Redis) ---")
    if not state_manager.available:
        print("❌ Redis is NOT available (via StateManager).")
        return

    try:
        # Perform a write/read test
        test_key = "verify_storage_test_key"
        test_val = "working"
        
        state_manager.redis.set(test_key, test_val, ex=10)
        val = state_manager.redis.get(test_key)
        
        if val and val == test_val:
             print("✅ Redis Connection: OK (Write/Read success)")
             
             # Check distinct active sessions if possible
             # Iterate keys safely
             keys = state_manager.redis.keys("session:*")
             print(f"   - Active Session Keys in Redis: {len(keys)}")
             
        else:
             print(f"❌ Redis Write/Read Mismatch. Got: {val}")

    except Exception as e:
        print(f"❌ Redis Verification Failed: {e}")

async def main():
    await verify_chromadb()
    await verify_database()
    await verify_redis()

if __name__ == "__main__":
    asyncio.run(main())
