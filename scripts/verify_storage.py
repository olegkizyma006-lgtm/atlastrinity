
import asyncio
import os
import sys
from sqlalchemy import select, func

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from brain.db.manager import db_manager, DB_URL
from brain.db.schema import Session, Task, TaskStep, ToolExecution, KGNode, ConversationSummary
from brain.memory import long_term_memory

async def verify_storage():
    print(f"🔍 Verifying Storage Systems...")
    print(f"   Database URL: {DB_URL}")
    print(f"   ChromaDB Path: {long_term_memory.get_stats().get('path', 'Unknown')}")
    print("-" * 50)

    # 1. Verify PostgreSQL
    print("🐘 PostgreSQL Verification:")
    try:
        await db_manager.initialize()
        if not db_manager.available:
            print("   ❌ Database Connection Failed!")
        else:
            async with (await db_manager.get_session()) as session:
                async def count(model):
                    res = await session.execute(select(func.count(model.id)))
                    return res.scalar()

                session_count = await count(Session)
                task_count = await count(Task)
                step_count = await count(TaskStep)
                tool_count = await count(ToolExecution)
                node_count = await count(KGNode)
                conv_sum_count = await count(ConversationSummary)

                print(f"   ✅ Connection Established")
                print(f"   Sessions: {session_count}")
                print(f"   Tasks: {task_count}")
                print(f"   Task Steps: {step_count}")
                print(f"   Tool Executions: {tool_count}")
                print(f"   Knowledge Graph Nodes (DB): {node_count}")
                print(f"   Conversation Summaries (DB): {conv_sum_count}")

                if session_count > 0:
                    print("   ✅ Data is being persisted to Postgres.")
                else:
                    print("   ⚠️ Tables appear empty (might be a fresh install or saving issue).")

    except Exception as e:
        print(f"   ❌ Error verifying Postgres: {e}")

    print("-" * 50)

    # 2. Verify ChromaDB
    print("🧬 ChromaDB (Semantic Memory) Verification:")
    try:
        stats = long_term_memory.get_stats()
        if not stats.get("available"):
             print("   ❌ ChromaDB Unavailable!")
        else:
             print(f"   ✅ Connection Established")
             print(f"   Lessons (Errors/Solutions): {stats.get('lessons_count')}")
             print(f"   Strategies (Plans): {stats.get('strategies_count')}")
             
             # Check Knowledge Graph Nodes in Chroma
             kg_nodes_count = long_term_memory.knowledge.count()
             print(f"   Knowledge Graph Nodes (Chroma): {kg_nodes_count}")
             
             # Check Conversations in Chroma
             conv_count = long_term_memory.conversations.count()
             print(f"   Conversation Summaries (Chroma): {conv_count}")
             
             # Try simple query to ensure embeddings work
             print("   🧪 Testing vector query...")
             try:
                 results = long_term_memory.recall_similar_tasks("test task", n_results=1)
                 print(f"   ✅ Vector query successful (Found {len(results)} matches)")
             except Exception as e:
                 print(f"   ❌ Vector query failed: {e}")

    except Exception as e:
        print(f"   ❌ Error verifying ChromaDB: {e}")

    print("-" * 50)
    print("🏁 Verification Complete")

if __name__ == "__main__":
    asyncio.run(verify_storage())
