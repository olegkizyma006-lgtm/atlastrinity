
import asyncio
import sys
import os

# Ensure src module is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from sqlalchemy import text
from src.brain.db.manager import db_manager

async def migrate():
    print("Initializing database connection...")
    await db_manager.initialize()
    
    if not db_manager.available:
        print("❌ Database connection failed.")
        return

    print("Attempting to add 'task_id' column to 'tool_executions'...")
    try:
        async with await db_manager.get_session() as session:
            # Check if column exists strictly if needed, but IF NOT EXISTS is cleaner
            await session.execute(text("ALTER TABLE tool_executions ADD COLUMN IF NOT EXISTS task_id UUID REFERENCES tasks(id);"))
            await session.commit()
            print("✅ Migration successful: 'task_id' column added (or already existed).")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(migrate())
