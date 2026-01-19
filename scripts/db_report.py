#!/usr/bin/env python3
"""Simple DB reporter: lists problematic tasks and recent ERROR logs.

Run with: .venv/bin/python scripts/db_report.py
"""
import asyncio
import sys
import traceback
from datetime import datetime

from sqlalchemy import select

from src.brain.db.manager import db_manager
from src.brain.db.schema import LogEntry, Task, TaskStep


async def report(limit_tasks: int = 50, limit_logs: int = 200):
    try:
        await db_manager.initialize()
    except Exception as e:
        print("[DB] initialize() failed: {e}")
        traceback.print_exc()
        return 1

    if not db_manager.available:
        print("[DB] Database not available. Ensure DATABASE_URL is set and the configured DB driver (e.g., aiosqlite for SQLite) is installed and accessible.")
        return 2

    try:
        async with await db_manager.get_session() as s:
            # Tasks
            print("\n=== Tasks (PENDING / RUNNING / FAILED) ===\n")
            q = select(Task).where(Task.status.in_(["PENDING", "RUNNING", "FAILED"]))
            q = q.order_by(Task.created_at.desc()).limit(limit_tasks)
            tasks = await s.execute(q)
            tasks = tasks.scalars().all()
            if not tasks:
                print("No PENDING/RUNNING/FAILED tasks found.")
            for t in tasks:
                print(
                    f"TASK {t.id}\n  goal: {t.goal}\n  status: {t.status}\n  created_at: {t.created_at}\n  completed_at: {t.completed_at}"
                )
                steps = await s.execute(
                    select(TaskStep)
                    .where(TaskStep.task_id == t.id)
                    .order_by(TaskStep.sequence_number)
                )
                for st in steps.scalars().all():
                    if st.status != "SUCCESS":
                        print(
                            f"    STEP {st.sequence_number}: {st.action}\n      status: {st.status}\n      error: {st.error_message}\n      duration_ms: {st.duration_ms}"
                        )

            # Recent ERROR logs
            print("\n=== Recent ERROR / CRITICAL logs ===\n")
            logs = await s.execute(
                select(LogEntry)
                .where(LogEntry.level.in_(["ERROR", "CRITICAL", "FATAL"]))
                .order_by(LogEntry.timestamp.desc())
                .limit(limit_logs)
            )
            logs = logs.scalars().all()
            if not logs:
                print("No ERROR/CRITICAL logs found.")
            for l in logs:
                ts = l.timestamp.isoformat() if isinstance(l.timestamp, datetime) else l.timestamp
                print(f"{ts} | {l.source} | {l.level} | {l.message}")

    except Exception as e:
        print("[DB] Query failed: {e}")
        traceback.print_exc()
        return 3

    return 0


def main():
    try:
        rc = asyncio.run(report())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
