#!/usr/bin/env python3
"""Test synchronization of behavioral deviations between ChromaDB and SQL"""
import asyncio
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.memory import long_term_memory
from src.brain.db.manager import db_manager
from src.brain.db.schema import BehavioralDeviation, Session
from sqlalchemy import select

async def test_deviation_sync():
    print("[TEST] Testing Behavioral Deviation Sync (Syncing ChromaDB and SQL)...")
    
    await db_manager.initialize()
    
    # 1. Create a dummy session for the deviation
    session_id = uuid.uuid4()
    async with await db_manager.get_session() as session:
        db_sess = Session(id=session_id)
        session.add(db_sess)
        await session.commit()
    
    # 2. Add a deviation via memory.py (this should trigger sync)
    intent = "Test original intent"
    deviation = "Test deviation action"
    reason = "Test reason"
    result = "Test result"
    context = {"session_id": session_id}
    factors = {"test_factor": "high"}
    
    print(f"[TEST] Storing deviation for session {session_id}...")
    success = long_term_memory.remember_behavioral_change(
        intent, deviation, reason, result, context, factors
    )
    
    if not success:
        print("[TEST] FAILED: remember_behavioral_change returned False")
        return False
        
    print("[TEST] Waiting for async sync to complete...")
    await asyncio.sleep(2) # Wait for the background task
    
    # 3. Verify in ChromaDB
    print("[TEST] Verifying in ChromaDB...")
    recalls = long_term_memory.recall_behavioral_logic(intent)
    if not recalls:
        print("[TEST] FAILED: Could not recall deviation from ChromaDB")
        return False
    print(f"[TEST] Recalled from ChromaDB: {recalls[0]['document'][:50]}...")
    
    # 4. Verify in SQL
    print("[TEST] Verifying in SQL...")
    async with await db_manager.get_session() as session:
        stmt = select(BehavioralDeviation).where(BehavioralDeviation.session_id == session_id)
        res = await session.execute(stmt)
        entry = res.scalar()
        
        if not entry:
            print("[TEST] FAILED: Could not find deviation in SQL")
            return False
            
        print(f"[TEST] Success! Found entry in SQL table 'behavioral_deviations'")
        print(f"      Original Intent: {entry.original_intent}")
        print(f"      Deviation: {entry.deviation}")
        print(f"      Factors: {entry.decision_factors}")
        
    await db_manager.close()
    return True

if __name__ == "__main__":
    if asyncio.run(test_deviation_sync()):
        print("\n✅ Behavioral Deviation Sync Test PASSED!")
    else:
        print("\n❌ Behavioral Deviation Sync Test FAILED!")
        sys.exit(1)
