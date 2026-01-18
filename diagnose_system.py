
import asyncio
import sys
import os
from pathlib import Path

# Fix path to include src/
sys.path.append(str(Path(__file__).parent / "src"))

from brain.db.manager import db_manager
from brain.db.schema import Session, Task, TaskStep, ToolExecution, AgentMessage
from brain.memory import long_term_memory
from sqlalchemy import select, func

async def diagnose_db():
    print(" === 1. DATABASE DIAGNOSTICS ===")
    if not db_manager.available:
        await db_manager.initialize()
    
    if not db_manager.available:
        print("[FAIL] Database is NOT available.")
        return
    
    async with await db_manager.get_session() as session:
        # Count sessions
        session_count = await session.scalar(select(func.count(Session.id)))
        print(f"[OK] Sessions count: {session_count}")
        
        # Count tasks
        task_count = await session.scalar(select(func.count(Task.id)))
        print(f"[OK] Tasks count: {task_count}")
        
        # Count steps
        step_count = await session.scalar(select(func.count(TaskStep.id)))
        print(f"[OK] Task Steps count: {step_count}")
        
        # Count tool executions
        exec_count = await session.scalar(select(func.count(ToolExecution.id)))
        print(f"[OK] Tool Executions count: {exec_count}")
        
        # Check for recursion (steps with dots)
        result = await session.execute(select(TaskStep.sequence_number).filter(TaskStep.sequence_number.contains(".")))
        recursive_steps = result.scalars().all()
        print(f"[RECURSION] Found {len(recursive_steps)} recursive steps in history.")
        if recursive_steps:
             print(f"            Examples: {recursive_steps[:5]}")

        # Check Inter-Agent Messaging
        msg_count = await session.scalar(select(func.count(AgentMessage.id)))
        print(f"[MESSAGING] Agent Messages count: {msg_count}")

async def diagnose_memory():
    print("\n === 2. VECTOR MEMORY DIAGNOSTICS ===")
    stats = long_term_memory.get_stats()
    if stats.get("available"):
        print(f"[OK] ChromaDB is available at {stats.get('path')}")
        print(f"[OK] Lessons: {stats.get('lessons_count')}")
        print(f"[OK] Strategies: {stats.get('strategies_count')}")
        print(f"[OK] Conversations: {stats.get('conversations_count')}")
    else:
        print("[FAIL] Vector memory (ChromaDB) is NOT available.")

async def diagnose_browser():
    print("\n === 3. BROWSER/MCP DIAGNOSTICS ===")
    # Check if puppeteer is in MCP config and reachable
    from brain.mcp_manager import MCPManager
    mcp = MCPManager()
    catalog = await mcp.get_mcp_catalog()
    
    # Check if puppeteer is in catalog
    if "puppeteer" in catalog:
        print("[OK] Puppeteer server is registered in catalog.")
    else:
        print("[WARN] Puppeteer server NOT found in catalog.")
    
    if "macos-use" in catalog:
        print("[OK] macos-use server is registered in catalog.")

async def main():
    await diagnose_db()
    await diagnose_memory()
    await diagnose_browser()
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
