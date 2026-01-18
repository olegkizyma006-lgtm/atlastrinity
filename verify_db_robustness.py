
import asyncio
import os
import sys
from sqlalchemy import text

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.brain.db.manager import db_manager

async def chaos_test_schema():
    print("🧨 Starting Schema Chaos Test...")
    
    # 1. Initialize
    await db_manager.initialize()
    if not db_manager.available:
        print("❌ DB not available")
        return

    # 2. Drop a column manually
    print("🔥 Dropping column 'vibe_text' from 'recovery_attempts'...")
    async with await db_manager.get_session() as session:
        await session.execute(text('ALTER TABLE recovery_attempts DROP COLUMN IF EXISTS vibe_text;'))
        await session.commit()
    
    # 3. Verify it's gone
    async with await db_manager.get_session() as session:
        try:
            await session.execute(text('SELECT vibe_text FROM recovery_attempts LIMIT 1;'))
            print("❌ Column STILL EXISTS? Drop failed.")
            return
        except Exception:
            print("✅ Column successfully dropped.")

    # 4. Re-run initialization (which should fix it)
    print("🛠 Re-running DatabaseManager.initialize()...")
    # Close and re-init to simulate a fresh system start
    await db_manager.close()
    await db_manager.initialize()
    
    # 5. Verify it's back
    async with await db_manager.get_session() as session:
        try:
            await session.execute(text('SELECT vibe_text FROM recovery_attempts LIMIT 1;'))
            print("🏆 SUCCESS: Column 'vibe_text' was automatically restored by DatabaseManager!")
        except Exception as e:
            print(f"❌ FAILURE: Column was NOT restored. Error: {e}")

if __name__ == "__main__":
    asyncio.run(chaos_test_schema())
