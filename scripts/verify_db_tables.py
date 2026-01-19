#!/usr/bin/env python3
"""Standalone script to verify database tables and counts during setup"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.db.manager import db_manager
from src.brain.db.schema import Base
from sqlalchemy import select, func

async def verify_database_tables():
    """Detailed verification of database tables and counts"""
    try:
        await db_manager.initialize()
        
        async with await db_manager.get_session() as session:
            print(f"[DB] Found {len(Base.metadata.tables)} tables in schema.")
            for table_name in Base.metadata.tables.keys():
                try:
                    table = Base.metadata.tables[table_name]
                    stmt = select(func.count()).select_from(table)
                    res = await session.execute(stmt)
                    count = res.scalar()
                    print(f"[DB] Table '{table_name}': {count} records")
                except Exception as e:
                    print(f"[DB] WARNING: Could not verify table '{table_name}': {e}")
                    
        await db_manager.close()
        return True
    except Exception as e:
        print(f"[DB] ERROR during table verification: {e}")
        return False

if __name__ == "__main__":
    if not asyncio.run(verify_database_tables()):
        sys.exit(1)
