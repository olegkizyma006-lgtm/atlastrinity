"""
AtlasTrinity Consolidation Module

Sleep & Consolidation - Nightly learning process that:
1. Reads audit logs
2. Extracts patterns using LLM
3. Compresses into lessons for ChromaDB
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .logger import logger
from .memory import long_term_memory


class ConsolidationModule:
    """
    Handles nightly learning and memory consolidation.

    Based on TDD spec:
    - Trigger: Scheduled (03:00 AM) or idle > 2 hours
    - Process: Read logs -> LLM analysis -> Extract rules -> Update ChromaDB
    """

    def __init__(self):
        self.last_consolidation: Optional[datetime] = None
        self.idle_threshold = timedelta(hours=2)
        self.log_path = os.path.join(os.path.expanduser("~/.config/atlastrinity/logs"), "brain.log")

    async def run_consolidation(self, llm=None) -> Dict[str, Any]:
        """
        Main consolidation process using DB data and LLM.
        """
        from .db.manager import db_manager

        if not db_manager.available:
            logger.warning("[CONSOLIDATION] DB unavailable, skipping.")
            return {"error": "DB unavailable"}

        logger.info("[CONSOLIDATION] Starting structured memory consolidation...")

        try:
            # 1. Fetch recent tasks (last 24h)
            cutoff = datetime.utcnow() - timedelta(hours=24)
            tasks_data = await self._fetch_tasks_from_db(cutoff)
            logger.info(f"[CONSOLIDATION] Fetched {len(tasks_data)} tasks from DB")

            if not tasks_data:
                return {"message": "No new tasks to consolidate"}

            # 2. LLM Analysis (Batch or individual)
            # We'll use the provided LLM or a default Atlas instance
            if not llm:
                from .agents import Atlas

                atlas = Atlas(model_name="raptor-mini")
                llm = atlas.llm

            # Extract Lessons from failures
            lessons_added = 0
            for task in [t for t in tasks_data if t["status"] == "FAILED"]:
                lesson = await self._distill_lesson_via_llm(llm, task)
                if lesson and long_term_memory.remember_error(
                    error=lesson["error"],
                    solution=lesson["rule"],
                    context=task,
                    task_description=task["goal"],
                ):
                    lessons_added += 1

            # 3. Consolidate successes into Best Practices
            # (Simplification: just register them if golden_path is true,
            # we've already done this in orchestrator, so here we might group them)
            # For now, let's just log stats

            self.last_consolidation = datetime.utcnow()

            stats = {
                "timestamp": self.last_consolidation.isoformat(),
                "tasks_processed": len(tasks_data),
                "lessons_added": lessons_added,
                "memory_stats": long_term_memory.get_stats(),
            }

            logger.info(
                f"[CONSOLIDATION] Complete: {lessons_added} new lessons derived from failures."
            )
            return stats

        except Exception as e:
            logger.error(f"[CONSOLIDATION] Failed: {e}")
            return {"error": str(e)}

    async def _fetch_tasks_from_db(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """Fetch tasks and their steps from the configured SQL database (SQLite by default)."""
        from sqlalchemy import select

        from .db.manager import db_manager
        from .db.schema import Task as DBTask
        from .db.schema import TaskStep as DBStep

        results = []
        async with await db_manager.get_session() as session:
            stmt = select(DBTask).where(DBTask.created_at > cutoff)
            res = await session.execute(stmt)
            tasks = res.scalars().all()

            for task in tasks:
                # Load steps
                step_stmt = (
                    select(DBStep).where(DBStep.task_id == task.id).order_by(DBStep.sequence_number)
                )
                step_res = await session.execute(step_stmt)
                steps = step_res.scalars().all()

                results.append(
                    {
                        "id": str(task.id),
                        "goal": task.goal,
                        "status": task.status,
                        "steps": [
                            {
                                "action": s.action,
                                "status": s.status,
                                "error": s.error_message,
                            }
                            for s in steps
                        ],
                    }
                )
        return results

    async def _distill_lesson_via_llm(
        self, llm, task_data: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Uses LLM to turn a failure into a generalized rule/lesson."""
        import json

        from langchain_core.messages import HumanMessage, SystemMessage

        history = "\n".join(
            [f"- {s['action']}: {s['status']} (Error: {s['error']})" for s in task_data["steps"]]
        )

        prompt = """Analyze this failed task and extract a general 'Lesson' to prevent this in the future.

        TASK: {task_data['goal']}
        EXECUTION:
        {history}

        Respond in JSON:
        {{
            "error": "The core technical reason for failure (English)",
            "rule": "Generalized rule or best practice to avoid this next time (English)",
            "analysis": "Technical analysis in English"
        }}
        """

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a Memory Consolidation Expert."),
                    HumanMessage(content=prompt),
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            # Basic JSON extraction
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            logger.warning(f"LLM Lesson distillation failed: {e}")
            return None

    def should_consolidate(self, last_activity: Optional[datetime] = None) -> bool:
        """Check if consolidation should run."""
        now = datetime.now()

        # Never consolidated
        if not self.last_consolidation:
            return True

        # Been more than 24 hours
        if now - self.last_consolidation > timedelta(hours=24):
            return True

        # Idle for 2+ hours
        if last_activity and now - last_activity > self.idle_threshold:
            return True

        # Nighttime (3 AM)
        if now.hour == 3 and (now - self.last_consolidation).total_seconds() > 3600:
            return True

        return False


# Singleton instance
consolidation_module = ConsolidationModule()
