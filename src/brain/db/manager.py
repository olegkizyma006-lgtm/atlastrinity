"""Database Connection Manager"""

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.brain.config import CONFIG_ROOT
from src.brain.config_loader import config
from src.brain.db.schema import Base


class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._session_maker = None
        self._semaphore = asyncio.Semaphore(15)
        self.available = False

        # Resolve DB_URL dynamically
        url = config.get(
            "database.url",
            os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{CONFIG_ROOT}/atlastrinity.db"),
        )
        # Handle placeholders: ${CONFIG_ROOT}, ${HOME}, ${PROJECT_ROOT}
        from src.brain.config import PROJECT_ROOT

        placeholders = {
            "${CONFIG_ROOT}": str(CONFIG_ROOT),
            "${HOME}": str(Path.home()),
            "${PROJECT_ROOT}": str(PROJECT_ROOT),
        }
        for k, v in placeholders.items():
            url = url.replace(k, v)

        self.db_url = url

    async def initialize(self):
        """Initialize DB connection and create tables if missing."""
        try:
            self._engine = create_async_engine(
                self.db_url,
                echo=False,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
            )

            # Enable Foreign Key support for SQLite
            if "sqlite" in self.db_url:
                from sqlalchemy import event

                @event.listens_for(self._engine.sync_engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()

            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # NEW: Verify and fix schema inconsistencies (missing columns, mismatched types, missing indexes/FKs)
            await self.verify_schema(fix=True)

            self._session_maker = async_sessionmaker(self._engine, expire_on_commit=False)
            self.available = True

            # Ensure seed data exists
            await self.ensure_seed_data()

            print("[DB] Database initialized successfully.")
        except Exception as e:
            # Helpful guidance for common driver issues (aiosqlite missing when using sqlite+aiosqlite)
            err_str = str(e)
            if "No module named 'aiosqlite'" in err_str or "aiosqlite" in err_str:
                print(
                    f"[DB] Failed to initialize database: {e}\n[DB] Hint: The async SQLite driver 'aiosqlite' is not installed. Install it via 'pip install aiosqlite' or set DATABASE_URL to a supported DB backend.",
                )
            else:
                print(f"[DB] Failed to initialize database: {e}")
            self.available = False

    async def verify_schema(self, fix: bool = True):
        """Comprehensive schema verification:
        1. Checks for missing columns and adds them.
        2. Checks for column type mismatches and alters them.
        3. Checks for missing indexes and creates them.
        """
        if not self._engine:
            return

        def _sync_verify(connection):
            from sqlalchemy import inspect, text

            inspector = inspect(connection)
            if not inspector:
                print("[DB] Critical Error: SQLAlchemy inspector is None")
                return

            # 1. Get existing tables
            existing_tables = inspector.get_table_names()

            for table_name, table in Base.metadata.tables.items():
                if table_name not in existing_tables:
                    continue  # create_all already handles missing tables

                # 2. Get existing columns and types
                existing_columns = inspector.get_columns(table_name)
                existing_col_map = {c["name"]: c for c in existing_columns}

                # 3. Check for missing or mismatched columns
                for column in table.columns:
                    if column.name not in existing_col_map:
                        print(
                            f"[DB] Mismatch: Missing column '{column.name}' in table '{table_name}'",
                        )
                        if fix:
                            try:
                                col_type = column.type.compile(connection.dialect)
                                nullable = "NOT NULL" if not column.nullable else "NULL"
                                default_val = ""
                                if not column.nullable:
                                    # Heuristic: find a safe default
                                    if (
                                        "VARCHAR" in str(col_type).upper()
                                        or "TEXT" in str(col_type).upper()
                                    ):
                                        default_val = (
                                            " DEFAULT 'global'"
                                            if column.name == "namespace"
                                            else " DEFAULT ''"
                                        )
                                    elif (
                                        "INT" in str(col_type).upper()
                                        or "BOOLEAN" in str(col_type).upper()
                                    ):
                                        default_val = " DEFAULT 0"

                                sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type} {nullable}{default_val};'
                                connection.execute(text(sql))
                                print(f"[DB] FIXED: Added column '{column.name}' to '{table_name}'")
                            except Exception as e:
                                print(f"[DB] FAILED to add column '{column.name}': {e}")
                    else:
                        # TYPE CHECK (Improved)
                        # We compare the compiled string representation of types and handle common aliases.
                        raw_existing_type = str(existing_col_map[column.name]["type"]).upper()

                        # Compile our expected type to the current dialect
                        try:
                            expected_type = column.type.compile(connection.dialect).upper()
                        except Exception:
                            expected_type = str(column.type).upper()

                        # Normalize types (handle variants like TIMESTAMP vs TIMESTAMP WITHOUT TIME ZONE)
                        def normalize(t):
                            t = t.replace("WITHOUT TIME ZONE", "").strip()
                            t = t.replace("WITH TIME ZONE", "").strip()
                            # Common mappings
                            mapping = {
                                "DATETIME": "TIMESTAMP",
                                "VARCHAR": "CHARACTER VARYING",
                                "JSONB": "JSONB",
                                "UUID": "UUID",
                                "BOOLEAN": "BOOLEAN",
                                "INTEGER": "INTEGER",
                            }
                            for k, v in mapping.items():
                                if k in t:
                                    return v
                            return t

                        norm_existing = normalize(raw_existing_type)
                        norm_expected = normalize(expected_type)

                        # Special case for JSONB/JSON
                        if "JSON" in norm_expected and "JSON" in norm_existing:
                            continue

                        if norm_expected != norm_existing and not (
                            norm_expected in norm_existing or norm_existing in norm_expected
                        ):
                            print(
                                f"[DB] Type Warning: Column '{column.name}' in '{table_name}' type mismatch. Found: {raw_existing_type} (norm: {norm_existing}), Expected: {expected_type} (norm: {norm_expected})",
                            )
                            if fix:
                                try:
                                    col_type = column.type.compile(connection.dialect)
                                    sql = f'ALTER TABLE "{table_name}" ALTER COLUMN "{column.name}" TYPE {col_type} USING "{column.name}"::{col_type};'
                                    connection.execute(text(sql))
                                    print(
                                        f"[DB] FIXED: Altered column '{column.name}' type to {col_type}",
                                    )
                                except Exception as e:
                                    print(f"[DB] FAILED to alter type: {e}")

                # 4. Check for missing indexes
                existing_indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
                for index in table.indexes:
                    if index.name not in existing_indexes:
                        print(
                            f"[DB] Mismatch: Missing index '{index.name}' on table '{table_name}'",
                        )
                        if fix:
                            try:
                                index.create(connection)
                                print(f"[DB] FIXED: Created index '{index.name}'")
                            except Exception as e:
                                print(f"[DB] FAILED to create index: {e}")

                # 5. Check for missing Foreign Keys
                existing_fks = {
                    (
                        tuple(fk["constrained_columns"]),
                        fk["referred_table"],
                        tuple(fk["referred_columns"]),
                    )
                    for fk in inspector.get_foreign_keys(table_name)
                }

                for fk in table.foreign_key_constraints:
                    fk_data = (
                        tuple(c.name for c in fk.columns),
                        fk.referred_table.name,
                        tuple(c.name for c in fk.referred_table.primary_key.columns),
                    )
                    if fk_data not in existing_fks:
                        print(
                            f"[DB] Mismatch: Missing Foreign Key on '{table_name}' referencing '{fk_data[1]}'",
                        )
                        if fix:
                            if connection.dialect.name == "sqlite":
                                print(
                                    f"[DB] Info: Foreign Key missing on '{table_name}' referencing '{fk_data[1]}'. SQLite doesn't support adding FKs via ALTER TABLE. This is normal for initial setups.",
                                )
                                continue
                            try:
                                # Constructing ALTER TABLE ADD CONSTRAINT manually
                                cols = ", ".join(f'"{c}"' for c in fk_data[0])
                                ref_cols = ", ".join(f'"{c}"' for c in fk_data[2])
                                constraint_name = f"fk_{table_name}_{fk_data[0][0]}"
                                sql = f'ALTER TABLE "{table_name}" ADD CONSTRAINT "{constraint_name}" FOREIGN KEY ({cols}) REFERENCES "{fk_data[1]}" ({ref_cols});'
                                connection.execute(text(sql))
                                print(
                                    f"[DB] FIXED: Added Foreign Key constraint '{constraint_name}'",
                                )
                            except Exception as e:
                                print(f"[DB] FAILED to add Foreign Key: {e}")

        async with self._engine.begin() as conn:
            await conn.run_sync(_sync_verify)

    async def ensure_seed_data(self):
        """Ensure mandatory initial rows exist in tables (Seed Data)."""
        if not self.available:
            return

        from sqlalchemy import select

        from src.brain.db.schema import KGNode

        async with await self.get_session() as session:
            # 1. Ensure core system node exists in Knowledge Graph
            # Use cast(Any, ...) to satisfy linter regarding SQLAlchemy operator overloading
            stmt = select(KGNode).where(cast("Any", KGNode.id == "entity:trinity"))
            res = await session.execute(stmt)
            if not res.scalar():
                print("[DB] Seeding core Knowledge Graph nodes...")
                trinity = KGNode(
                    id="entity:trinity",
                    type="CONCEPT",
                    attributes={
                        "description": "AtlasTrinity Core System. The root of all knowledge and strategy.",
                        "version": "4.2",
                    },
                )
                session.add(trinity)
                await session.commit()
                print("[DB] Seeding complete.")

    async def create_table_from_df(self, table_name: str, df: pd.DataFrame) -> bool:
        """Dynamically create a table in the database matching the DataFrame schema.
        Used for High-Precision Ingestion.
        """
        if not self.available:
            return False

        from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, Table, Text

        # 1. Map pandas dtypes to SQLAlchemy types
        def map_dtype(col):
            t = df[col].dtype
            if pd.api.types.is_integer_dtype(t):
                return Integer
            elif pd.api.types.is_float_dtype(t):
                return Float
            elif pd.api.types.is_datetime64_any_dtype(t):
                return DateTime
            elif pd.api.types.is_bool_dtype(t):
                return Boolean
            else:
                return Text  # Default to Text for objects/strings

        # 2. Define the new table
        metadata = MetaData()
        columns = [Column("row_id", Integer, primary_key=True, autoincrement=True)]
        for col_name in df.columns:
            safe_col = str(col_name).lower().replace(" ", "_").replace("-", "_")
            # If the CSV has an 'id' or 'row_id' column, we rename the existing one to avoid collision
            if safe_col in ["row_id"]:
                safe_col = f"original_{safe_col}"

            # Explicitly cast to Any to satisfy strict linters regarding the positional type argument
            col_type = map_dtype(col_name)
            columns.append(Column(safe_col, cast("Any", col_type)))

        # 3. Create the table sync-style via engine.run_sync
        def sync_create(connection):
            Table(table_name, metadata, *columns, extend_existing=True)
            metadata.create_all(connection)

        try:
            if not self._engine:
                print("[DB] Error: Cannot create table, engine not initialized.")
                return False

            async with self._engine.begin() as conn:
                await conn.run_sync(sync_create)

            # 4. Bulk insert data
            async with await self.get_session() as session:
                # Convert DF to list of dicts with sanitized keys
                sanitized_data = []
                for _, row in df.iterrows():
                    d = {}
                    for col_name in df.columns:
                        safe_col = str(col_name).lower().replace(" ", "_").replace("-", "_")
                        d[safe_col] = row[col_name] if pd.notnull(row[col_name]) else None
                    sanitized_data.append(d)

                # Dynamic insert is tricky with SQLAlchemy async,
                # we'll use raw SQL text for simplicity and performance in bulk ingestion
                from sqlalchemy import text

                cols = ", ".join(
                    [f'"{str(c).lower().replace(" ", "_").replace("-", "_")}"' for c in df.columns],
                )
                placeholders = ", ".join(
                    [f":{str(c).lower().replace(' ', '_').replace('-', '_')}" for c in df.columns],
                )
                sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders})'

                await session.execute(text(sql), sanitized_data)
                await session.commit()

            print(f"[DB] Dynamic table '{table_name}' created and populated with {len(df)} rows.")
            return True
        except Exception as e:
            print(f"[DB] Failed to create dynamic table '{table_name}': {e}")
            return False

    async def get_session(self) -> AsyncSession:
        """Get a new async session."""
        if not self.available or not self._session_maker:
            raise RuntimeError("Database not initialized")
        return self._session_maker()

    async def close(self):
        if self._engine:
            await self._engine.dispose()


db_manager = DatabaseManager()
