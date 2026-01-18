"""
Database Connection Manager
"""

import asyncio
import os
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .schema import Base

# Default connection string (can be overridden by Env)
# Using 'postgresql+asyncpg' driver
DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://dev:postgres@localhost/atlastrinity_db")


class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._session_maker = None
        self._semaphore = asyncio.Semaphore(15)  # Limit concurrent connections
        self.available = False

    async def initialize(self):
        """Initialize DB connection and create tables if missing."""
        try:
            self._engine = create_async_engine(
                DB_URL, 
                echo=False,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True
            )

            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # NEW: Verify and fix schema inconsistencies (missing columns)
            await self.verify_schema(fix=True)

            self._session_maker = async_sessionmaker(self._engine, expire_on_commit=False)
            self.available = True
            print("[DB] Database initialized successfully.")
        except Exception as e:
            print(f"[DB] Failed to initialize database: {e}")
            self.available = False

    async def verify_schema(self, fix: bool = True):
        """
        Check for missing columns in existing tables and optionally add them.
        """
        if not self._engine:
            return

        def _sync_verify(connection):
            from sqlalchemy import inspect, text
            inspector = inspect(connection)
            
            # 1. Get existing tables
            existing_tables = inspector.get_table_names()
            
            for table_name, table in Base.metadata.tables.items():
                if table_name not in existing_tables:
                    continue # create_all already handles missing tables
                
                # 2. Get existing columns
                existing_cols = {c['name'] for c in inspector.get_columns(table_name)}
                
                for column in table.columns:
                    if column.name not in existing_cols:
                        print(f"[DB] Mismatch: Missing column '{column.name}' in table '{table_name}'")
                        if fix:
                            # 3. Add column
                            try:
                                col_type = column.type.compile(connection.dialect)
                                nullable = "NULL" if column.nullable else "NOT NULL"
                                sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type} {nullable};'
                                connection.execute(text(sql))
                                print(f"[DB] FIXED: Added column '{column.name}' to '{table_name}'")
                            except Exception as e:
                                print(f"[DB] FAILED to fix column '{column.name}': {e}")

        async with self._engine.begin() as conn:
            await conn.run_sync(_sync_verify)

    async def get_session(self) -> AsyncSession:
        """Get a new async session."""
        if not self.available or not self._session_maker:
            raise RuntimeError("Database not initialized")
        return self._session_maker()

    async def close(self):
        if self._engine:
            await self._engine.dispose()


db_manager = DatabaseManager()
