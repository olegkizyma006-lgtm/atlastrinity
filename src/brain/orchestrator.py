"""
AtlasTrinity Orchestrator
LangGraph-based state machine that coordinates Agents (Atlas, Tetyana, Grisha)
"""

import ast
import asyncio
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

try:
    from langgraph.graph import add_messages
except ImportError:
    pass

import uuid

from src.brain.agents import Atlas, Grisha, Tetyana
from src.brain.agents.tetyana import StepResult
from src.brain.config import IS_MACOS, PLATFORM_NAME
from src.brain.config_loader import config
from src.brain.consolidation import consolidation_module
from src.brain.context import shared_context
from src.brain.db.manager import db_manager
from src.brain.db.schema import ConversationSummary as DBConvSummary
from src.brain.db.schema import LogEntry as DBLog
from src.brain.db.schema import RecoveryAttempt
from src.brain.db.schema import Session as DBSession
from src.brain.db.schema import Task as DBTask
from src.brain.db.schema import TaskStep as DBStep
from src.brain.db.schema import ToolExecution as DBToolExecution
from src.brain.knowledge_graph import knowledge_graph
from src.brain.logger import logger
from src.brain.mcp_manager import mcp_manager
from src.brain.message_bus import AgentMsg, MessageType, message_bus
from src.brain.metrics import metrics_collector
from src.brain.notifications import notifications
from src.brain.state_manager import state_manager
from src.brain.voice.tts import VoiceManager
from src.brain.adaptive_behavior import adaptive_behavior


class SystemState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CHAT = "CHAT"


class TrinityState(TypedDict):
    messages: list[BaseMessage]
    system_state: str
    current_plan: Any | None
    step_results: list[dict[str, Any]]
    error: str | None
    logs: list[dict[str, Any]]
    session_id: str | None
    db_session_id: str | None
    db_task_id: str | None
    _theme: str | None


class Trinity:
    def __init__(self):
        self.atlas = Atlas()
        self.tetyana = Tetyana()
        self.grisha = Grisha()
        self.voice = VoiceManager()

        # Ensure global singletons are loaded

        # Initialize graph
        self.graph = self._build_graph()
        self._log_lock = asyncio.Lock()
        self.current_session_id = "current_session"  # Default alias for the last active session
        self._resumption_pending = False
        self._user_node_created = False

    async def initialize(self):
        """Async initialization of system components"""
        # Initialize state if not exists
        if not hasattr(self, "state") or not self.state:
            self.state = {
                "messages": [],
                "system_state": SystemState.IDLE.value,
                "current_plan": None,
                "step_results": [],
                "error": None,
                "logs": [],
            }
            logger.info("[ORCHESTRATOR] State initialized during initialize()")

        # Start MCP health check loop
        mcp_manager.start_health_monitoring(interval=60)

        # Capture MCP server log notifications for UI visibility
        async def mcp_log_forwarder(message, source, level="info"):
            await self._log(message, source=source, type=level)

        mcp_manager.register_log_callback(mcp_log_forwarder)
        # Initialize DB
        await db_manager.initialize()

        # Check for pending restart state
        await self._resume_after_restart()

        # If resumption is pending, trigger the run() in background after a short delay
        if getattr(self, "_resumption_pending", False):

            async def auto_resume():
                await asyncio.sleep(5)  # Wait for all components to stabilize
                messages = self.state.get("messages", [])
                if messages:
                    # Get the original request from the first HumanMessage
                    original_request = ""
                    for m in messages:
                        if "HumanMessage" in str(type(m)) or (
                            isinstance(m, dict) and m.get("type") == "human"
                        ):
                            if hasattr(m, "content"):
                                original_request = str(m.content)
                            elif isinstance(m, dict):
                                original_request = str(m.get("content", ""))
                            else:
                                original_request = str(m)
                            break

                    if original_request:
                        logger.info(
                            f"[ORCHESTRATOR] Auto-resuming task: {original_request[:50]}..."
                        )
                        await self.run(original_request)

            asyncio.create_task(auto_resume())

        logger.info(f"[GRISHA] Auditor ready. Vision: {self.grisha.llm.model_name}")

    async def reset_session(self):
        """Reset the current session and start a fresh one"""
        self.state = {
            "messages": [],
            "system_state": SystemState.IDLE.value,
            "current_plan": None,
            "step_results": [],
            "error": None,
            "logs": [],
        }
        # Clear IDs so they are regenerated on next run
        if "db_session_id" in self.state:
            del self.state["db_session_id"]
        if "db_task_id" in self.state:
            del self.state["db_task_id"]
        # Auto-backup before clearing session
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            try:
                from scripts import setup_dev

                await asyncio.to_thread(setup_dev.backup_databases)
            except ImportError:
                # Handle non-package scripts folder
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "setup_dev", str(project_root / "scripts" / "setup_dev.py")
                )
                if spec and spec.loader:
                    setup_dev = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(setup_dev)
                    await asyncio.to_thread(setup_dev.backup_databases)
            await self._log("📦 Backup попередньої сесії...", "system")
        except Exception as e:
            logger.warning(f"[BACKUP] Не вдалося створити backup: {e}")

        if state_manager.available:
            await state_manager.clear_session(self.current_session_id)

        # Create a new unique session ID
        self.current_session_id = f"session_{uuid.uuid4().hex[:8]}"

        await self._log(f"Нова сесія розпочата ({self.current_session_id})", "system")
        return {"status": "success", "session_id": self.current_session_id}

    async def load_session(self, session_id: str):
        """Load a specific session from Redis"""
        if not state_manager.available:
            return {"status": "error", "message": "Persistence unavailable"}

        saved_state = await state_manager.restore_session(session_id)
        if saved_state:
            self.state = saved_state
            self.current_session_id = session_id
            await self._log(f"Сесія {session_id} відновлена", "system")
            return {"status": "success"}
        return {"status": "error", "message": "Session not found"}

    def _build_graph(self):
        workflow = StateGraph(TrinityState)

        # Define nodes
        workflow.add_node("atlas_planning", lambda state: self.planner_node(state))
        workflow.add_node("tetyana_execution", lambda state: self.executor_node(state))
        workflow.add_node("grisha_verification", lambda state: self.verifier_node(state))

        # Define edges
        workflow.set_entry_point("atlas_planning")

        workflow.add_edge("atlas_planning", "tetyana_execution")
        workflow.add_conditional_edges(
            "tetyana_execution",
            self.should_verify,
            {
                "verify": "grisha_verification",
                "continue": "tetyana_execution",
                "end": END,
            },
        )
        workflow.add_edge("grisha_verification", "tetyana_execution")

        return workflow.compile()

    def _mcp_result_to_text(self, res: Any) -> str:
        if isinstance(res, dict):
            try:
                return json.dumps(res, ensure_ascii=False)
            except Exception:
                return str(res)

        if hasattr(res, "content") and isinstance(res.content, list):
            parts: list[str] = []
            for item in res.content:
                txt = getattr(item, "text", None)
                if isinstance(txt, str) and txt:
                    parts.append(txt)
            if parts:
                return "".join(parts)
        return str(res)

    def _extract_vibe_payload(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        try:
            data = json.loads(t)
        except Exception:
            try:
                data = ast.literal_eval(t)
            except Exception:
                return t

        if isinstance(data, dict):
            stdout = data.get("stdout")
            stderr = data.get("stderr")
            if isinstance(stdout, str) and stdout.strip():
                return stdout.strip()
            if isinstance(stderr, str) and stderr.strip():
                return stderr.strip()
        return t

    async def _speak(self, agent_id: str, text: str):
        """Voice wrapper with smarter sanitization"""
        # 1. Clean up text for TTS (don't be too aggressive)
        # We only strip weird control chars or excessive whitespace
        import re

        clean_text = re.sub(r"\s+", " ", text).strip()

        # If text is empty or too short for TTS, skip
        if not clean_text or len(clean_text) < 2:
            return

        print(f"[{agent_id.upper()}] Speaking: {clean_text}")

        # 2. Synchronize with UI chat log
        if hasattr(self, "state") and self.state is not None:
            if "messages" not in self.state:
                self.state["messages"] = []

            # Avoid duplicate messages if this was already in the history (e.g. during resumption)
            # We only append if it's the latest message (real-time generated)
            self.state["messages"].append(AIMessage(content=text, name=agent_id.upper()))

        await self._log(text, source=agent_id, type="voice")
        try:
            # Pass ORIGINAL text to voice, let engine handle it
            await self.voice.speak(agent_id, text)
        except Exception as e:
            print(f"TTS Error: {e}")

    async def _log(self, text: str, source: str = "system", type: str = "info"):
        """Log wrapper with message types and DB persistence"""
        # Ensure text is a string to prevent React "Objects are not valid as a React child" error
        text_str = str(text)
        logger.info(f"[{source.upper()}] {text_str}")

        # DB Persistence
        if db_manager.available:
            async with self._log_lock:
                try:
                    async with await db_manager.get_session() as session:
                        entry = DBLog(
                            level=type.upper(),
                            source=source,
                            message=text_str,
                            metadata_blob={"type": type},
                        )
                        session.add(entry)
                        await session.commit()
                except Exception as e:
                    logger.error(f"DB Log failed: {e}")

        if self.state:
            # Basic log format for API
            import time

            entry = {
                "id": f"log-{len(self.state.get('logs') or [])}-{time.time()}",
                "timestamp": time.time(),
                "agent": source.upper(),
                "message": text_str,
                "type": type,
            }
            if "logs" not in self.state:
                self.state["logs"] = []
            self.state["logs"].append(entry)

            # 3. Publish to Redis for real-time UI updates
            if state_manager.available:
                try:
                    asyncio.create_task(state_manager.publish_event("logs", entry))
                except Exception as e:
                    logger.warning(f"Failed to publish log to Redis: {e}")

    async def _resume_after_restart(self):
        """Check if we are recovering from a restart and resume state"""
        if not state_manager.available:
            return

        try:
            # Check for restart flag in Redis
            restart_key = state_manager._key("restart_pending")
            data = None
            if state_manager.redis:
                data = await state_manager.redis.get(restart_key)

            if data:
                restart_info = json.loads(cast(str, data))
                reason = restart_info.get("reason", "Unknown reason")
                session_id = restart_info.get("session_id", "current_session")

                logger.info(
                    f"[ORCHESTRATOR] Recovering from self-healing restart. Reason: {reason}"
                )

                if session_id == "current":
                    # Use the most recent session from list_sessions
                    sessions = await state_manager.list_sessions()
                    if sessions:
                        session_id = sessions[0]["id"]

                saved_state = await state_manager.restore_session(session_id)
                if saved_state:
                    self.state = saved_state
                    self.current_session_id = session_id

                    # Clear the flag
                    if state_manager.redis:
                        await state_manager.redis.delete(restart_key)
                    self._resumption_pending = True

                    await self._log(
                        f"Система успішно перезавантажилася та відновила стан. Причина: {reason}",
                        "system",
                    )
                    await self._speak(
                        "atlas", "Я повернувся. Продовжую виконання завдання з того ж місця."
                    )
        except Exception as e:
            logger.error(f"Failed to resume after restart: {e}")

            # Implementation: querying Redis for a specific "system:restart_pending" key
            # which we would have set in tool_dispatcher (we need to update tool_dispatcher to set this!)
            # WAIT: tool_dispatcher doesn't have access to state_manager directly usually.
            # We should probably update tool_dispatcher to use state_manager if available.

            # ALTERNATIVE: orchestrator handles the restart tool?
            # No, dispatcher handles it. Dispatcher needs access to state_manager to set the flag.

            # For now, let's just log that we are booting up.
            await self._log("System booted. Checking for pending tasks...", "system")

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Resume check failed: {e}")

    async def _verify_db_ids(self):
        """Verify that restored DB IDs exist. If not, clear them."""
        try:
            from src.brain.db.manager import db_manager

            if not db_manager or not getattr(db_manager, "available", False):
                return
        except (ImportError, NameError):
            return

        session_id_str = self.state.get("db_session_id")
        task_id_str = self.state.get("db_task_id")

        async with await db_manager.get_session() as db_sess:
            import uuid

            from sqlalchemy import select

            if session_id_str and isinstance(session_id_str, str):
                try:
                    session_id = uuid.UUID(session_id_str)
                    result = await db_sess.execute(
                        select(DBSession).where(DBSession.id == session_id)
                    )
                    if not result.scalar():
                        logger.warning(
                            f"[ORCHESTRATOR] Restored session_id {session_id_str} not found in DB. Clearing."
                        )
                        del self.state["db_session_id"]
                        if "db_task_id" in self.state:
                            del self.state["db_task_id"]
                        return  # If session is gone, task is definitely gone
                except Exception as e:
                    logger.error(f"Error verifying session_id {session_id_str}: {e}")
                    # If it's not a valid UUID, it's definitely stale/junk
                    del self.state["db_session_id"]

            if task_id_str and isinstance(task_id_str, str):
                try:
                    task_id = uuid.UUID(task_id_str)
                    result = await db_sess.execute(select(DBTask).where(DBTask.id == task_id))
                    if not result.scalar():
                        logger.warning(
                            f"[ORCHESTRATOR] Restored task_id {task_id_str} not found in DB. Clearing."
                        )
                        del self.state["db_task_id"]
                except Exception as e:
                    logger.error(f"Error verifying task_id {task_id_str}: {e}")
                    del self.state["db_task_id"]

    def get_state(self) -> dict[str, Any]:
        """Return current system state for API"""
        if not hasattr(self, "state") or not self.state:
            logger.warning("[ORCHESTRATOR] State not initialized, returning default state")
            return {
                "system_state": SystemState.IDLE.value,
                "current_task": "Waiting for input...",
                "active_agent": "ATLAS",
                "logs": [],
                "step_results": [],
            }

        # Determine active agent based on system state
        active_agent = "ATLAS"
        sys_state = self.state.get("system_state", SystemState.IDLE.value)

        if sys_state == SystemState.EXECUTING.value:
            active_agent = "TETYANA"
        elif sys_state == SystemState.VERIFYING.value:
            active_agent = "GRISHA"

        plan = self.state.get("current_plan")

        # Handle plan being either object or string (from Redis/JSON serialization)
        if plan:
            if isinstance(plan, str):
                task_summary = plan
            elif hasattr(plan, "goal"):
                task_summary = plan.goal
            else:
                task_summary = str(plan)
        else:
            task_summary = "IDLE"

        # Prepare messages for frontend
        messages = []
        from datetime import datetime

        msg_list = self.state.get("messages")
        if isinstance(msg_list, list):
            for m in msg_list:
                if isinstance(m, HumanMessage):
                    messages.append(
                        {
                            "agent": "USER",
                            "text": m.content,
                            "timestamp": datetime.now().timestamp(),
                            "type": "text",
                        }
                    )
                elif isinstance(m, AIMessage):
                    # Support custom agent names (e.g. TETYANA, GRISHA) stored in .name
                    agent_name = m.name if hasattr(m, "name") and m.name else "ATLAS"
                    messages.append(
                        {
                            "agent": agent_name,
                            "text": m.content,
                            "timestamp": datetime.now().timestamp(),
                            "type": "voice",
                        }
                    )

        return {
            "system_state": sys_state,
            "current_task": task_summary,
            "active_agent": active_agent,
            "session_id": self.current_session_id,
            "messages": messages[-50:],
            "logs": (self.state.get("logs") or [])[-100:],
            "step_results": self.state.get("step_results") or [],
            "metrics": metrics_collector.get_metrics(),
        }

    async def run(self, user_request: str) -> dict[str, Any]:
        """
        Main orchestration loop with advanced persistence and memory
        """
        self.voice.stop()  # Stop any current speech when a new request arrives
        start_time = asyncio.get_event_loop().time()
        session_id = self.current_session_id

        # 0. Platform Insurance Check
        if not IS_MACOS:
            await self._log(
                f"WARNING: Running on {PLATFORM_NAME}. AtlasTrinity is optimized for macOS. Some tools may fail.",
                "system",
                type="warning",
            )
            # If the user strictly wants it to stop, we could raise an error here.
            # But for development/testing, a warning is safer and more informative.
        is_subtask = (
            hasattr(self, "state")
            and self.state is not None
            and hasattr(self, "_in_subtask")
            and self._in_subtask
        )

        if not is_subtask:
            # Initialize or restore state
            if not hasattr(self, "state") or self.state is None:
                self.state = {
                    "messages": [],
                    "system_state": SystemState.IDLE.value,
                    "current_plan": None,
                    "step_results": [],
                    "error": None,
                    "logs": [],
                }

            # Restore from Redis if available and we are starting fresh
            try:
                from src.brain.state_manager import state_manager

                if (
                    state_manager
                    and getattr(state_manager, "available", False)
                    and not self.state["messages"]
                    and session_id == "current_session"
                ):
                    saved_state = await state_manager.restore_session(session_id)
                    if saved_state:
                        self.state = saved_state
                        logger.info("[STATE] Successfully restored last active session")
            except (ImportError, NameError):
                pass

            # Update session ID if we were using the alias
            if session_id == "current_session" and isinstance(self.state.get("session_id"), str):
                session_id = self.state["session_id"]
                self.current_session_id = session_id
            elif "session_id" not in self.state:
                self.state["session_id"] = session_id

            # Set theme if this is the first message
            if "_theme" not in self.state or self.state["_theme"] == "Untitled Session":
                self.state["_theme"] = user_request[:40] + ("..." if len(user_request) > 40 else "")

            # CRITICAL: Verify that the DB IDs in the restored state actually exist
            await self._verify_db_ids()
            logger.info("[ORCHESTRATOR] State validation complete")

            # Append the new user message
            self.state["messages"].append(HumanMessage(content=user_request))
            self.state["system_state"] = SystemState.PLANNING.value
            self.state["error"] = None

            # DB Session Creation (Only for top-level)
            try:
                from src.brain.db.manager import db_manager

                if (
                    db_manager
                    and getattr(db_manager, "available", False)
                    and "db_session_id" not in self.state
                ):
                    async with await db_manager.get_session() as db_sess:
                        new_session = DBSession(started_at=datetime.now(UTC))
                        db_sess.add(new_session)
                        await db_sess.commit()
                        self.state["db_session_id"] = str(new_session.id)
            except Exception as e:
                logger.error(f"DB Session creation failed: {e}")
                if "db_session_id" in self.state:
                    del self.state["db_session_id"]

        try:
            from src.brain.state_manager import state_manager

            if not is_subtask and state_manager and getattr(state_manager, "available", False):
                await state_manager.publish_event(
                    "tasks",
                    {
                        "type": "task_started",
                        "request": user_request,
                        "session_id": session_id,
                    },
                )
        except (ImportError, NameError):
            pass

        await self._log(f"New Request: {user_request}", "system")

        # 1. Push Global Goal to Shared Context
        shared_context.push_goal(user_request)

        # 1.5. CHECK IF WE CAN SKIP PLANNING (Resumption Path)
        plan = None
        if self.state.get("current_plan") and getattr(self, "_resumption_pending", False):
            logger.info("[ORCHESTRATOR] Plan already exists. skipping Atlas planning phase.")
            plan_obj = self.state["current_plan"]
            # Cast plan if it's a dict (from Redis restoration)
            if isinstance(plan_obj, dict):
                from src.brain.agents.atlas import TaskPlan

                # Clean steps if they are already success: True
                plan = TaskPlan(
                    id=plan_obj.get("id", "resumed"),
                    goal=plan_obj.get("goal", user_request),
                    steps=plan_obj.get("steps", []),
                )
            else:
                plan = plan_obj
            self._resumption_pending = False
        else:
            # 2. Atlas Planning
            try:
                try:
                    from src.brain.state_manager import state_manager

                    if state_manager and getattr(state_manager, "available", False):
                        await state_manager.publish_event(
                            "tasks", {"type": "planning_started", "request": user_request}
                        )
                except (ImportError, NameError):
                    pass

                # 1.6 Pass history to Atlas for context
                messages_raw = self.state.get("messages")
                history = []
                if isinstance(messages_raw, list):
                    history = messages_raw[-25:-1] if len(messages_raw) > 1 else []

                analysis = await self.atlas.analyze_request(user_request, history=history)

                if analysis.get("intent") == "chat":
                    response = analysis.get("initial_response") or await self.atlas.chat(
                        user_request,
                        history=history,
                        use_deep_persona=analysis.get("use_deep_persona", False),
                    )
                    await self._speak("atlas", response)
                    try:
                        from src.brain.state_manager import state_manager

                        if state_manager and getattr(state_manager, "available", False):
                            await state_manager.save_session(session_id, self.state)
                    except (ImportError, NameError):
                        pass
                    self.state["system_state"] = SystemState.IDLE.value
                    return {"status": "completed", "result": response, "type": "chat"}

                self.state["system_state"] = SystemState.PLANNING.value

                # Fetch dynamic MCP Catalog
                mcp_catalog = await mcp_manager.get_mcp_catalog()
                shared_context.available_mcp_catalog = mcp_catalog

                spoken_text = analysis.get("voice_response") or "Аналізую ваш запит..."
                await self._speak("atlas", spoken_text)

                _keep_alive_last_log = [0.0]

                async def keep_alive_logging():
                    import time

                    while True:
                        await asyncio.sleep(15)
                        current_time = time.time()
                        if current_time - _keep_alive_last_log[0] >= 10:
                            _keep_alive_last_log[0] = current_time
                            await self._log("Atlas is thinking... (Planning logic flow)", "system")

                planning_task = asyncio.create_task(self.atlas.create_plan(analysis))
                logger_task = asyncio.create_task(keep_alive_logging())
                try:
                    plan = await asyncio.wait_for(
                        planning_task,
                        timeout=config.get("orchestrator", {}).get("task_timeout", 1200.0),
                    )
                finally:
                    logger_task.cancel()
                    try:
                        await logger_task
                    except asyncio.CancelledError:
                        pass

                if not plan or not plan.steps:
                    msg = self.atlas.get_voice_message("no_steps")
                    await self._speak("atlas", msg)

                    # Trigger fallback response if Atlas thought it was a task but produced no steps
                    fallback_chat = await self.atlas.chat(
                        user_request, history=history, use_deep_persona=True
                    )
                    await self._speak("atlas", fallback_chat)
                    self.state["system_state"] = SystemState.IDLE.value
                    return {"status": "completed", "result": fallback_chat, "type": "chat"}

                self.state["current_plan"] = plan

                # DB Task Creation
                try:
                    from src.brain.db.manager import db_manager

                    if (
                        db_manager
                        and getattr(db_manager, "available", False)
                        and self.state.get("db_session_id")
                    ):
                        async with await db_manager.get_session() as db_sess:
                            new_task = DBTask(
                                session_id=self.state["db_session_id"],
                                goal=user_request,
                                status="PENDING",
                            )
                            db_sess.add(new_task)
                            await db_sess.commit()
                            self.state["db_task_id"] = str(new_task.id)

                            await knowledge_graph.add_node(
                                node_type="TASK",
                                node_id=f"task:{new_task.id}",
                                attributes={
                                    "goal": user_request,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "steps_count": len(plan.steps),
                                },
                            )
                except Exception as e:
                    logger.error(f"DB Task creation failed: {e}")

                try:
                    from src.brain.state_manager import state_manager

                    if state_manager and getattr(state_manager, "available", False):
                        await state_manager.save_session(session_id, self.state)
                except (ImportError, NameError):
                    pass

                try:
                    from src.brain.state_manager import state_manager

                    if state_manager and getattr(state_manager, "available", False):
                        await state_manager.publish_event(
                            "tasks",
                            {
                                "type": "planning_finished",
                                "session_id": session_id,
                                "steps_count": len(plan.steps),
                            },
                        )
                except (ImportError, NameError):
                    pass

                await self._speak(
                    "atlas",
                    self.atlas.get_voice_message("plan_created", steps=len(plan.steps)),
                )

            except Exception as e:
                import traceback

                logger.error(f"[ORCHESTRATOR] Planning error: {e}")
                logger.error(traceback.format_exc())
                self.state["system_state"] = SystemState.ERROR.value
                return {"status": "error", "error": str(e)}

        # 3. Execution Loop (Tetyana) - Recursive Execution
        self.state["system_state"] = SystemState.EXECUTING.value

        try:
            # Initial numbering is 1, 2, 3...
            if plan and plan.steps:
                await self._execute_steps_recursive(plan.steps)
            else:
                logger.warning("[ORCHESTRATOR] No steps to execute. Ending task.")

        except Exception as e:
            await self._log(f"Critical error: {e}", "error")
            return {"status": "error", "error": str(e)}

        # 4. Success Tasks: Memory & Cleanup
        duration = asyncio.get_event_loop().time() - start_time
        notifications.show_completion(user_request, True, duration)

        # Atlas Verification Gate & Memory
        try:
            from src.brain.memory import long_term_memory

            if (
                long_term_memory
                and getattr(long_term_memory, "available", False)
                and not is_subtask
                and self.state["system_state"] != SystemState.ERROR.value
            ):
                # Atlas reviews the execution
                evaluation = await self.atlas.evaluate_execution(
                    user_request, self.state["step_results"]
                )

                # Speak Final Report if available and goal achieved
                final_report = evaluation.get("final_report")
                if final_report and evaluation.get("achieved"):
                    await self._speak("atlas", final_report)
                elif evaluation.get("achieved"):
                    # Fallback if no specific report generated
                    await self._speak("atlas", "Завдання успішно виконано.")

                if evaluation.get("should_remember") and evaluation.get("quality_score", 0) >= 0.7:
                    await self._log(
                        f"Verification Pass: Score {evaluation.get('quality_score')} ({evaluation.get('analysis')})",
                        "atlas",
                    )

                    strategy_steps = evaluation.get(
                        "compressed_strategy"
                    ) or self._extract_golden_path(self.state["step_results"])

                    try:
                        from src.brain.memory import long_term_memory

                        if long_term_memory and getattr(long_term_memory, "available", False):
                            long_term_memory.remember_strategy(
                                task=user_request,
                                plan_steps=strategy_steps,
                                outcome="SUCCESS",
                                success=True,
                            )
                    except (ImportError, NameError):
                        pass
                    await self._log(f"Brain saved {len(strategy_steps)} steps to memory", "system")

                # Update DB Task with quality metric
                try:
                    from src.brain.db.manager import db_manager

                    if (
                        db_manager
                        and getattr(db_manager, "available", False)
                        and self.state.get("db_task_id")
                    ):
                        async with await db_manager.get_session() as db_sess:
                            from sqlalchemy import update

                            await db_sess.execute(
                                update(DBTask)
                                .where(DBTask.id == self.state["db_task_id"])
                                .values(golden_path=True)
                            )
                            await db_sess.commit()
                except Exception as e:
                    logger.error(f"Failed to mark golden path in DB: {e}")
        except Exception as e:
            logger.error(f"Post-execution verification/memory stage failed: {e}")

        # Nightly/End-of-task consolidation check
        if not is_subtask and consolidation_module.should_consolidate():
            asyncio.create_task(consolidation_module.run_consolidation())

        self.state["system_state"] = SystemState.COMPLETED.value

        # 4. Pro-Memory: Summarize and persist session context for semantic search
        msg_list_summary = self.state.get("messages")
        if not is_subtask and isinstance(msg_list_summary, list) and len(msg_list_summary) > 2:
            asyncio.create_task(self._persist_session_summary(session_id))

        # Pop Global Goal
        shared_context.pop_goal()

        try:
            from src.brain.state_manager import state_manager

            if state_manager and getattr(state_manager, "available", False):
                await state_manager.clear_session(session_id)
        except (ImportError, NameError):
            pass

        try:
            from src.brain.state_manager import state_manager

            if state_manager and getattr(state_manager, "available", False):
                await state_manager.publish_event(
                    "tasks",
                    {"type": "task_finished", "status": "completed", "session_id": session_id},
                )
        except (ImportError, NameError):
            pass

        try:
            from src.brain.state_manager import state_manager

            if state_manager and getattr(state_manager, "available", False):
                try:
                    await state_manager.publish_event(
                        "sessions",
                        {"type": "session_started", "session_id": session_id},
                    )
                except (ImportError, NameError):
                    pass
        except (ImportError, NameError):
            pass

        # Auto-backup databases after session completion
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            from scripts.setup_dev import backup_databases

            asyncio.create_task(asyncio.to_thread(backup_databases))
            await self._log("📦 Автоматичний backup баз даних...", "system")
        except Exception as e:
            logger.warning(f"[BACKUP] Не вдалося створити backup: {e}")

        return {"status": "completed", "result": self.state["step_results"]}

    async def _persist_session_summary(self, session_id: str):
        """Generates a professional summary and stores it in DB and Vector memory."""
        try:
            messages = self.state.get("messages")
            if not isinstance(messages, list) or not messages:
                return

            summary_data = await self.atlas.summarize_session(messages)
            summary = summary_data.get("summary", "No summary generated")
            entities = summary_data.get("entities", [])

            # A. Store in Vector Memory (ChromaDB)
            try:
                from src.brain.memory import long_term_memory

                if long_term_memory and getattr(long_term_memory, "available", False):
                    long_term_memory.remember_conversation(
                        session_id=session_id, summary=summary, metadata={"entities": entities}
                    )
            except (ImportError, NameError):
                pass

            # B. Store in Structured DB (SQLite)
            try:
                from src.brain.db.manager import db_manager

                if db_manager and getattr(db_manager, "available", False):
                    async with await db_manager.get_session() as db_sess:
                        new_summary = DBConvSummary(
                            session_id=session_id, summary=summary, key_entities=entities
                        )
                        db_sess.add(new_summary)
                        await db_sess.commit()
            except Exception as e:
                logger.error(f"Failed to store summary in DB: {e}")

            # C. Add entities to Knowledge Graph
            for ent_name in entities:
                await knowledge_graph.add_node(
                    node_type="CONCEPT",
                    node_id=f"concept:{ent_name.lower().replace(' ', '_')}",
                    attributes={
                        "description": f"Entity mentioned in session {session_id}",
                        "source": "session_summary",
                    },
                    namespace="global",  # Concepts from summaries are often global-worthy, or could stay in session?
                    # For now, let's keep session concepts in global as "anchors" or maybe we decide later.
                )

            logger.info(f"[ORCHESTRATOR] Persisted professional session summary for {session_id}")

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Failed to persist session summary: {e}")

    def _extract_golden_path(self, raw_results: list[dict[str, Any]]) -> list[str]:
        """
        Extracts only the successful actions that led to the solution.
        Smartly filters out:
        - Failed attempts
        - Steps replaced by recovery actions
        - Repair loops (e.g. Step 3 failed -> 3.1 fixed -> Step 3 success)
        """
        golden_path = []

        # 1. Deduplicate by step_id, keeping only the LATEST attempt
        # This handles retries automatically (Attempt 1 fail, Attempt 2 success -> keeps Attempt 2)
        latest_results = {}
        for res in raw_results:
            step_id = res.get("step_id")
            latest_results[step_id] = res

        # 2. Sort by step ID to respect execution order
        # We need a robust sort for "1", "2", "2.1", "2.2", "3"
        def parse_step_id(sid):
            try:
                return [int(p) for p in str(sid).split(".")]
            except:
                return [float("inf")]  # Put weird IDs at current level end

        sorted_steps = sorted(
            latest_results.values(), key=lambda x: parse_step_id(x.get("step_id", "0"))
        )

        # 3. Filter for SUCCESS only
        # If a step failed but the task continued, it means it was critical to fix it?
        # No, if it failed and we moved on, usually means recovery handled it.
        # We want to capture the recovery steps (e.g. 2.1) if they succeeded.

        for item in sorted_steps:
            if item.get("success"):
                # Clean up action text
                action = item.get("action", "")

                # Remove ID prefix if present for cleaner reading e.g. "[3.1] Fix code" -> "Fix code"
                if action.startswith("[") and "]" in action:
                    try:
                        action = action.split("]", 1)[1].strip()
                    except:
                        pass

                if not action:
                    action = str(item.get("result", ""))[:100]

                golden_path.append(action)

        return golden_path

    async def _execute_steps_recursive(
        self, steps: list[dict], parent_prefix: str = "", depth: int = 0
    ) -> bool:
        """
        Recursively executes a list of steps.
        Supports hierarchical numbering (e.g. 3.1, 3.2) and deep recovery.
        """
        MAX_RECURSION_DEPTH = 5
        BACKOFF_BASE_MS = 500  # Exponential backoff between depths

        if depth > MAX_RECURSION_DEPTH:
            raise RecursionError("Max task recursion depth reached. Failing task.")

        # Exponential backoff on deeper recursion to prevent overwhelming the system
        if depth > 1:
            backoff_ms = BACKOFF_BASE_MS * (2 ** (depth - 1))
            await self._log(
                f"Recursion depth {depth}: applying {backoff_ms}ms backoff", "orchestrator"
            )
            await asyncio.sleep(backoff_ms / 1000)

        # Track recursion metrics for analytics
        metrics_collector.record("recursion_depth", depth, tags={"parent": parent_prefix or "root"})

        for i, step in enumerate(steps):
            # Generate hierarchical ID: "1", "2" or "3.1", "3.2"
            if parent_prefix:
                step_id = f"{parent_prefix}.{i + 1}"
            else:
                step_id = str(i + 1)

            # Update step object with this dynamic ID (for logging/recovery context)
            step["id"] = step_id

            notifications.show_progress(i + 1, len(steps), f"[{step_id}] {step.get('action')}")

            # Push step goal to context
            try:
                from src.brain.context import shared_context

                shared_context.push_goal(step.get("action", "Working..."), total_steps=len(steps))
                shared_context.current_step_id = i + 1
            except (ImportError, NameError, AttributeError):
                pass

            # --- SKIP ALREADY COMPLETED STEPS ---
            step_results = self.state.get("step_results") or []
            if any(
                isinstance(res, dict)
                and str(res.get("step_id")) == str(step_id)
                and res.get("success")
                for res in step_results
            ):
                logger.info(f"[ORCHESTRATOR] Skipping already completed step {step_id}")
                continue

            # Retry loop with Dynamic Temperature
            max_step_retries = 3
            last_error = ""

            for attempt in range(1, max_step_retries + 1):
                db_step_id = None
                step_result: StepResult | None = None
                await self._log(
                    f"Step {step_id}, Attempt {attempt}: {step.get('action')}",
                    "orchestrator",
                )

                try:
                    step_result = await asyncio.wait_for(
                        self.execute_node(
                            cast(TrinityState, self.state), step, step_id, attempt=attempt
                        ),
                        timeout=float(config.get("orchestrator", {}).get("task_timeout", 1200.0))
                        + 60.0,
                    )
                    if step_result and step_result.success:
                        break
                    elif step_result:
                        last_error = step_result.error
                    else:
                        last_error = "Unknown execution error (no result)"

                    db_step_id = self.state.get("db_step_id")
                    await self._log(
                        f"Step {step_id} Attempt {attempt} failed: {last_error}",
                        "warning",
                    )

                except Exception as e:
                    last_error = f"{type(e).__name__}: {e!s}"
                    await self._log(
                        f"Step {step_id} Attempt {attempt} crashed: {last_error}",
                        "error",
                    )

                # RECOVERY LOGIC
                validate_with_grisha = bool(
                    config.get("orchestrator", {}).get("validate_failed_steps_with_grisha", False)
                )
                recovery_agent = config.get("orchestrator", {}).get("recovery_voice_agent", "atlas")

                if validate_with_grisha:
                    try:
                        await self._log(
                            f"Requesting Grisha validation for failed step {step_id}...",
                            "orchestrator",
                        )
                        screenshot = None
                        expected = step.get("expected_result", "").lower()
                        if any(
                            k in expected
                            for k in ["visual", "screenshot", "ui", "interface", "window"]
                        ):
                            screenshot = await self.grisha.take_screenshot()

                        try:
                            from src.brain.context import shared_context

                            goal_ctx = str(shared_context.get_goal_context() or "")
                        except (ImportError, NameError):
                            goal_ctx = ""

                        verify_result = await self.grisha.verify_step(
                            step=step,
                            result=step_result
                            if step_result is not None
                            else StepResult(
                                step_id=step_id, success=False, result="", error=last_error
                            ),
                            screenshot_path=screenshot,
                            goal_context=goal_ctx,
                            task_id=str(self.state.get("db_task_id") or ""),
                        )

                        if verify_result.verified:
                            await self._log(
                                f"Grisha verified step {step_id} despite reporting failure. Marking success.",
                                "orchestrator",
                            )
                            break
                        else:
                            await self._speak(
                                recovery_agent,
                                verify_result.voice_message or "Крок потребує відновлення.",
                            )
                    except Exception as e:
                        logger.warning(f"Grisha validation failed: {e}")

                notifications.send_stuck_alert(
                    int(str(step_id).split(".")[-1])
                    if "." in str(step_id)
                    else (
                        int(step_id)
                        if isinstance(step_id, int | str) and str(step_id).isdigit()
                        else 0
                    ),
                    str(last_error or "Unknown error"),
                    max_step_retries,
                )

                await self._log(
                    f"Recovery for Step {step_id} (announced by {recovery_agent})...",
                    "orchestrator",
                )
                if recovery_agent == "atlas":
                    await self._speak(
                        "atlas", self.atlas.get_voice_message("recovery_started", step_id=step_id)
                    )
                else:
                    await self._speak(
                        recovery_agent, "Крок зупинився — починаю процедуру відновлення."
                    )

                # DB: Track Recovery Attempt
                try:
                    from src.brain.db.manager import db_manager

                    if db_manager and getattr(db_manager, "available", False) and db_step_id:
                        async with await db_manager.get_session() as db_sess:
                            rec_attempt = RecoveryAttempt(
                                step_id=db_step_id,
                                depth=depth,
                                recovery_method="vibe",
                                success=False,
                                error_before=str(last_error)[:5000],
                            )
                            db_sess.add(rec_attempt)
                            await db_sess.commit()
                except Exception as e:
                    logger.error(f"Failed to log recovery attempt start: {e}")

                # Collect context for Vibe
                recent_logs = []
                if self.state and "logs" in self.state:
                    recent_logs = [
                        f"[{l.get('agent', 'SYS')}] {l.get('message', '')}"
                        for l in self.state["logs"][-20:]
                    ]
                log_context = "\n".join(recent_logs)
                error_context = f"Step ID: {step_id}\nAction: {step.get('action', '')}\n"

                # Vibe diagnostics & multi-agent healing
                vibe_text = None
                try:
                    await self._log("[VIBE] Diagnostic Phase...", "vibe")
                    vibe_res = await asyncio.wait_for(
                        mcp_manager.call_tool(
                            "vibe",
                            "vibe_analyze_error",
                            {
                                "error_message": f"{error_context}\n{last_error}",
                                "log_context": log_context,
                                "auto_fix": False,
                            },
                        ),
                        timeout=300,
                    )
                    vibe_text = self._extract_vibe_payload(self._mcp_result_to_text(vibe_res))

                    if vibe_text:
                        grisha_audit = await self.grisha.audit_vibe_fix(str(last_error), vibe_text)
                        healing_decision = await self.atlas.evaluate_healing_strategy(
                            str(last_error), vibe_text, grisha_audit
                        )
                        await self._speak(
                            "atlas", healing_decision.get("voice_message", "Я знайшов рішення.")
                        )

                        if healing_decision.get("decision") == "PROCEED":
                            await mcp_manager.call_tool(
                                "vibe",
                                "vibe_prompt",
                                {
                                    "prompt": f"EXECUTE FIX: {healing_decision.get('instructions_for_vibe')}",
                                    "auto_approve": True,
                                },
                            )
                            logger.info(f"[ORCHESTRATOR] Vibe healing applied for {step_id}")
                except Exception as ve:
                    logger.warning(f"Vibe self-healing failed: {ve}")

                # Standard Atlas help as fallback
                try:
                    recovery = await asyncio.wait_for(
                        self.atlas.help_tetyana(str(step_id), str(last_error)), timeout=60.0
                    )
                    await self._speak(
                        "atlas", recovery.get("voice_message", "Альтернативний шлях.")
                    )
                    alt_steps = recovery.get("alternative_steps", [])
                    if alt_steps:
                        await self._execute_steps_recursive(
                            alt_steps, parent_prefix=step_id, depth=depth + 1
                        )
                        # If recursion returns, we consider the step fixed
                        break
                except Exception as r_err:
                    logger.error(f"Atlas recovery failed: {r_err}")
                    raise Exception(
                        f"Task failed at step {step_id} after retries and recovery attempts."
                    )

            # End of retry loop for THIS step
            try:
                from src.brain.context import shared_context

                shared_context.pop_goal()
            except:
                pass

        return True

    async def execute_node(
        self, state: TrinityState, step: dict[str, Any], step_id: str, attempt: int = 1
    ) -> StepResult:
        """Atomic execution logic with recursion and dynamic temperature"""
        # Starting message logic
        # Simple heuristic: If it's a top level step (no dots) and first attempt
        if "." not in str(step_id) and attempt == 1:
            # Use voice_action from plan if available, else fallback to generic
            msg = step.get("voice_action")
            if not msg:
                msg = self.tetyana.get_voice_message(
                    "starting", step=step_id, description=step.get("action", "")
                )
            await self._speak("tetyana", msg)
        elif "." in str(step_id):
            # It's a sub-step/recovery step
            pass

        try:
            from src.brain.state_manager import state_manager

            if state_manager and getattr(state_manager, "available", False):
                await state_manager.publish_event(
                    "steps",
                    {
                        "type": "step_started",
                        "step_id": str(step_id),
                        "action": step.get("action", "Working..."),
                        "attempt": attempt,
                    },
                )
        except (ImportError, NameError):
            pass
        # DB Step logging
        db_step_id = None
        self.state["db_step_id"] = None
        try:
            from src.brain.db.manager import db_manager

            if (
                db_manager
                and getattr(db_manager, "available", False)
                and self.state.get("db_task_id")
            ):
                async with await db_manager.get_session() as db_sess:
                    new_step = DBStep(
                        task_id=self.state["db_task_id"],
                        sequence_number=str(step_id),
                        action=f"[{step_id}] {step.get('action', '')}",
                        tool=step.get("tool", ""),
                        status="RUNNING",
                    )
                    db_sess.add(new_step)
                    await db_sess.commit()
                    db_step_id = str(new_step.id)
                    self.state["db_step_id"] = db_step_id
        except Exception as e:
            logger.error(f"DB Step creation failed: {e}")

        step_start_time = asyncio.get_event_loop().time()

        if step.get("type") == "subtask" or step.get("tool") == "subtask":
            self._in_subtask = True
            try:
                sub_result = await self.run(str(step.get("action") or ""))
            finally:
                self._in_subtask = False

            result = StepResult(
                step_id=str(step.get("id") or step_id),
                success=sub_result.get("status") == "completed",
                result="Subtask completed",
                error=str(sub_result.get("error") or ""),
            )
        else:
            try:
                # Inject context results (last 10 for better relevance)
                step_copy = step.copy()
                if self.state and "step_results" in self.state:
                    step_copy["previous_results"] = self.state["step_results"][-10:]

                # Full plan for sequence context
                plan = self.state.get("current_plan")
                if plan:
                    # Convert plan steps to a readable summary
                    step_list = []
                    plan_steps = getattr(plan, "steps", [])
                    if isinstance(plan_steps, list):
                        for s in plan_steps:
                            s_dict = s if isinstance(s, dict) else {}
                            step_results = self.state.get("step_results") or []
                            status = (
                                "DONE"
                                if any(
                                    isinstance(res, dict)
                                    and str(res.get("step_id")) == str(s_dict.get("id"))
                                    and res.get("success")
                                    for res in step_results
                                )
                                else "PENDING"
                            )
                            step_list.append(
                                f"Step {s_dict.get('id')}: {s_dict.get('action')} [{status}]"
                            )
                    step_copy["full_plan"] = "\n".join(step_list)

                # Check message bus for specific feedback from other agents
                bus_messages = await message_bus.receive("tetyana", mark_read=True)
                if bus_messages:
                    step_copy["bus_messages"] = [m.to_dict() for m in bus_messages]

                result = await self.tetyana.execute_step(step_copy, attempt=attempt)

                # --- RESTART DETECTION ---
                try:
                    from src.brain.state_manager import state_manager

                    if state_manager and getattr(state_manager, "available", False):
                        restart_key = state_manager._key("restart_pending")
                        try:
                            if state_manager.redis and await state_manager.redis.exists(
                                restart_key
                            ):
                                logger.warning(
                                    "[ORCHESTRATOR] Imminent application restart detected. Saving session state immediately."
                                )
                                await state_manager.save_session(
                                    self.current_session_id, self.state
                                )
                                # We stop here. The process replacement (execv) will happen in ToolDispatcher task
                                # and this orchestrator task will either be killed or return soon.
                        except Exception:
                            pass
                except (ImportError, NameError):
                    pass

                # --- DYNAMIC AGENCY: Check for Strategy Deviation ---
                # --- DYNAMIC AGENCY: Check for Strategy Deviation ---
                if getattr(result, "is_deviation", False) or result.error == "strategy_deviation":
                    try:
                        proposal_text = (
                            result.deviation_info.get("analysis")
                            if getattr(result, "deviation_info", None)
                            else result.result
                        )
                        p_text = str(proposal_text)
                        logger.warning(
                            f"[ORCHESTRATOR] Tetyana proposed a deviation: {p_text[:200]}..."
                        )

                        # Consult Atlas
                        evaluation = await self.atlas.evaluate_deviation(
                            step,
                            str(proposal_text),
                            getattr(self.state.get("current_plan"), "steps", []),
                        )

                        voice_msg = evaluation.get("voice_message", "")
                        if voice_msg:
                            await self._speak("atlas", voice_msg)

                        if evaluation.get("approved"):
                            logger.info("[ORCHESTRATOR] Deviation APPROVED. Adjusting plan...")
                            result.success = True
                            result.result = f"Strategy Deviated: {evaluation.get('reason')}"
                            result.error = None

                            # Mark for behavioral learning after successful verification
                            result.is_deviation = True
                            result.deviation_info = evaluation

                            # PERSISTENCE: Remember this approved deviation immediately
                            try:
                                from src.brain.memory import long_term_memory

                                if long_term_memory and getattr(
                                    long_term_memory, "available", False
                                ):
                                    reason_text = str(evaluation.get("reason", "Unknown"))
                                    long_term_memory.remember_behavioral_change(
                                        original_intent=step.get("action", ""),
                                        deviation=p_text[:300],
                                        reason=reason_text,
                                        result="Deviated plan approved",
                                        context={"step_id": str(step_id)},
                                        decision_factors={
                                            "original_step": step,
                                            "analysis": proposal_text,
                                        },
                                    )
                                    logger.info(
                                        "[ORCHESTRATOR] Learned and memorized new behavioral deviation strategy."
                                    )
                            except (ImportError, NameError) as mem_err:
                                logger.warning(f"Failed to memorize deviation: {mem_err}")

                        else:
                            logger.info("[ORCHESTRATOR] Deviation REJECTED. Forcing original plan.")
                            step["grisha_feedback"] = (
                                f"Strategy Deviation Rejected: {evaluation.get('reason')}. Stick to the plan."
                            )
                            result.success = False
                    except Exception as eval_err:
                        logger.error(f"[ORCHESTRATOR] Deviation evaluation failed: {eval_err}")
                        result.success = False
                        result.error = "evaluation_error"
                        return result

                # Handle need_user_input signal (New Autonomous Timeout Logic)
                if result.error == "need_user_input":
                    # Speak Tetyana's request BEFORE waiting to inform the user immediately
                    if result.voice_message:
                        await self._speak("tetyana", result.voice_message)
                        result.voice_message = (
                            None  # Clear it so it won't be spoken again at the end of node
                        )

                    timeout_val = float(config.get("orchestrator.user_input_timeout", 60.0))
                    await self._log(
                        f"User input needed for step {step_id}. Waiting {timeout_val} seconds...",
                        "orchestrator",
                    )

                    # Display the question to the user in the logs/UI
                    await self._log(f"[REQUEST] {result.result}", "system", type="warning")

                    # Wait for user message on the bus or timeout
                    user_response = None
                    try:
                        # Wait for a 'user_response' message type specifically for this step
                        start_wait = asyncio.get_event_loop().time()
                        while asyncio.get_event_loop().time() - start_wait < timeout_val:
                            bus_msgs = await message_bus.receive("orchestrator", mark_read=True)
                            for m in bus_msgs:
                                if m.message_type == MessageType.CHAT and m.from_agent == "USER":
                                    user_response = m.payload.get("text")
                                    break
                            if user_response:
                                break
                            await asyncio.sleep(0.5)

                    except Exception as wait_err:
                        logger.warning(f"Error during user wait: {wait_err}")

                    if user_response:
                        await self._log(f"User responded: {user_response}", "system")
                        messages = self.state.get("messages")
                        if messages is not None and isinstance(messages, list):
                            messages.append(HumanMessage(content=user_response))
                            self.state["messages"] = messages
                        try:
                            from src.brain.state_manager import state_manager

                            if state_manager and getattr(state_manager, "available", False):
                                await state_manager.save_session("current_session", self.state)
                        except (ImportError, NameError):
                            pass

                        # Direct feedback for the next retry
                        await message_bus.send(
                            AgentMsg(
                                from_agent="USER",
                                to_agent="tetyana",
                                message_type=MessageType.FEEDBACK,
                                payload={"user_response": user_response},
                                step_id=step.get("id"),
                            )
                        )
                        result.success = False
                        result.error = "user_input_received"
                    else:
                        # TIMEOUT: Atlas ONLY speaks if user was truly silent
                        await self._log(
                            "User silent for timeout. Atlas deciding...",
                            "orchestrator",
                            type="warning",
                        )

                        def _get_msg_content(m):
                            if hasattr(m, "content"):
                                return m.content
                            if isinstance(m, dict):
                                return m.get("content", str(m))
                            return str(m)

                        messages = self.state.get("messages", [])
                        goal_msg = messages[0] if messages else HumanMessage(content="Unknown")

                        autonomous_decision = await self.atlas.decide_for_user(
                            str(result.result or ""),
                            {
                                "goal": _get_msg_content(goal_msg),
                                "current_step": str(step.get("action") or ""),
                                "history": [
                                    _get_msg_content(m) for m in (messages[-5:] if messages else [])
                                ],
                            },
                        )

                        await self._log(
                            f"Atlas Autonomous Decision (Timeout): {autonomous_decision}", "atlas"
                        )
                        await self._speak(
                            "atlas", f"Оскільки ви не відповіли, я вирішив: {autonomous_decision}"
                        )

                        # Inject decision as feedback
                        await message_bus.send(
                            AgentMsg(
                                from_agent="atlas",
                                to_agent="tetyana",
                                message_type=MessageType.FEEDBACK,
                                payload={
                                    "user_response": f"(Autonomous Decision): {autonomous_decision}"
                                },
                                step_id=step.get("id"),
                            )
                        )
                        result.success = False
                        result.error = "autonomous_decision_made"

                # Log tool execution to DB for Grisha's audit
                try:
                    from src.brain.db.manager import db_manager

                    if (
                        db_manager
                        and getattr(db_manager, "available", False)
                        and db_step_id
                        and result.tool_call
                    ):
                        async with await db_manager.get_session() as db_sess:
                            tool_exec = DBToolExecution(
                                step_id=db_step_id,
                                task_id=self.state.get("db_task_id"),  # Direct link for analytics
                                server_name=result.tool_call.get("server")
                                or result.tool_call.get("realm")
                                or "unknown",
                                tool_name=result.tool_call.get("name") or "unknown",
                                arguments=result.tool_call.get("args") or {},
                                result=str(result.result)[:10000],  # Cap size
                            )
                            db_sess.add(tool_exec)
                            await db_sess.commit()
                            logger.info(
                                f"[ORCHESTRATOR] Logged tool execution: {tool_exec.tool_name}"
                            )
                except Exception as e:
                    logger.error(f"Failed to log tool execution to DB: {e}")

                # Handle proactive help requested by Tetyana
                if result.error == "proactive_help_requested":
                    await self._log(
                        f"Tetyana requested proactive help: {result.result}", "orchestrator"
                    )
                    # Atlas help logic
                    help_resp = await self.atlas.help_tetyana(
                        str(step.get("id") or step_id), str(result.result or "")
                    )

                    # Extract voice message or reason from Atlas response
                    voice_msg = ""
                    if isinstance(help_resp, dict):
                        voice_msg = (
                            help_resp.get("voice_message")
                            or help_resp.get("reason")
                            or str(help_resp)
                        )
                    else:
                        voice_msg = str(help_resp)

                    await self._speak("atlas", voice_msg)
                    # Re-run the step with Atlas's guidance as bus feedback

                    await message_bus.send(
                        AgentMsg(
                            from_agent="atlas",
                            to_agent="tetyana",
                            message_type=MessageType.FEEDBACK,
                            payload={"guidance": help_resp},
                            step_id=step.get("id"),
                        )
                    )
                    # Mark result as "Help pending" so retry loop can pick it up
                    result.success = False
                    result.error = "help_pending"

                # Log interaction to Knowledge Graph if successful
                if result.success and result.tool_call:
                    await knowledge_graph.add_node(
                        node_type="TOOL",
                        node_id=f"tool:{result.tool_call.get('name')}",
                        attributes={"last_used_step": str(step_id), "success": True},
                    )
                    await knowledge_graph.add_edge(
                        source_id=f"task:{self.state.get('db_task_id', 'unknown')}",
                        target_id=f"tool:{result.tool_call.get('name')}",
                        relation="USED",
                    )
                if result.voice_message:
                    await self._speak("tetyana", result.voice_message)
            except Exception as e:
                logger.exception("Tetyana execution crashed")
                result = StepResult(
                    step_id=str(step.get("id") or step_id),
                    success=False,
                    result="Crashed",
                    error=str(e),
                )

        # Update DB Step
        try:
            from src.brain.db.manager import db_manager

            if db_manager and getattr(db_manager, "available", False) and db_step_id:
                try:
                    duration_ms = int((asyncio.get_event_loop().time() - step_start_time) * 1000)
                    async with await db_manager.get_session() as db_sess:
                        from sqlalchemy import update

                        await db_sess.execute(
                            update(DBStep)
                            .where(DBStep.id == db_step_id)
                            .values(
                                status="SUCCESS" if result.success else "FAILED",
                                error_message=result.error,
                                duration_ms=duration_ms,
                            )
                        )
                        await db_sess.commit()
                except Exception as e:
                    logger.error(f"DB Step update failed: {e}")
        except (ImportError, NameError):
            pass

        # Check verification
        if step.get("requires_verification"):
            self.state["system_state"] = SystemState.VERIFYING.value
            # Removed redundant speak call here.
            # Tetyana's execute_step already provides result.voice_message if successful.

            try:
                # OPTIMIZATION: Reduced delay from 2.5s to 0.5s
                await self._log("Preparing verification...", "system")
                await asyncio.sleep(0.5)

                # Only take screenshot if visual verification is needed
                expected = step.get("expected_result", "").lower()
                visual_verification_needed = (
                    "visual" in expected
                    or "screenshot" in expected
                    or "ui" in expected
                    or "interface" in expected
                    or "window" in expected
                )

                screenshot = None
                if visual_verification_needed:
                    screenshot = await self.grisha.take_screenshot()

                # GRISHA'S AWARENESS: Pass the full result (including thoughts) and the goal
                verify_result = await self.grisha.verify_step(
                    step=step,
                    result=result,
                    screenshot_path=screenshot,
                    goal_context=shared_context.get_goal_context(),
                    task_id=str(self.state.get("db_task_id") or ""),
                )
                if not verify_result.verified:
                    result.success = False
                    result.error = f"Grisha rejected: {verify_result.description}"
                    if verify_result.issues:
                        result.error += f" Issues: {', '.join(verify_result.issues)}"

                    await self._speak(
                        "grisha",
                        verify_result.voice_message or "Результат не прийнято.",
                    )
                else:
                    await self._speak(
                        "grisha",
                        verify_result.voice_message or "Підтверджую виконання.",
                    )

                    # --- BEHAVIORAL LEARNING: Commit successful deviations ---
                    if result.is_deviation and result.success and result.deviation_info:
                        evaluation = result.deviation_info
                        factors = evaluation.get("decision_factors", {})

                        # 1. Vector Memory
                        try:
                            from src.brain.memory import long_term_memory

                            if long_term_memory and getattr(long_term_memory, "available", False):
                                long_term_memory.remember_behavioral_change(
                                    original_intent=str(step.get("action") or "Unknown"),
                                    deviation=str(result.result),
                                    reason=str(evaluation.get("reason") or "Unknown"),
                                    result="Verified Success",
                                    context={"step_id": str(step.get("id") or step_id)},
                                    decision_factors=factors,
                                )
                        except (ImportError, NameError):
                            pass

                        # 2. Knowledge Graph (Structured Factors)
                        if knowledge_graph:
                            try:
                                lesson_id = f"lesson:{int(datetime.now().timestamp())}"
                                await knowledge_graph.add_node(
                                    node_type="LESSON",
                                    node_id=lesson_id,
                                    attributes={
                                        "name": f"Successful Deviation: {str(evaluation.get('reason') or '')[:50]}",
                                        "intent": str(step.get("action") or ""),
                                        "outcome": "Verified Success",
                                        "reason": str(evaluation.get("reason") or ""),
                                    },
                                )
                                # Link to task
                                if self.state.get("db_task_id"):
                                    await knowledge_graph.add_edge(
                                        f"task:{self.state.get('db_task_id')}",
                                        lesson_id,
                                        "learned_lesson",
                                    )

                                # Structured Factor Nodes
                                for f_name, f_val in factors.items():
                                    factor_node_id = (
                                        f"factor:{f_name}:{str(f_val).lower().replace(' ', '_')}"
                                    )
                                    await knowledge_graph.add_node(
                                        "FACTOR",
                                        factor_node_id,
                                        {
                                            "name": f_name,
                                            "value": f_val,
                                            "type": "environmental_factor",
                                        },
                                    )
                                    await knowledge_graph.add_edge(
                                        lesson_id, factor_node_id, "CONTINGENT_ON"
                                    )

                            except Exception as g_err:
                                logger.error(
                                    f"[ORCHESTRATOR] Error linking factors in graph: {g_err}"
                                )
            except Exception as e:
                print(f"[ERROR] Verification failed: {e}")
                await self._log(f"Verification crashed: {e}", "error")
                result.success = False
                result.error = f"Verification system error: {e!s}"

            self.state["system_state"] = SystemState.EXECUTING.value

        # Store final result
        self.state["step_results"].append(
            {
                "step_id": str(result.step_id),  # Ensure string
                "action": f"[{step_id}] {step.get('action')}",  # Adding ID context
                "success": result.success,
                "result": result.result,
                "error": result.error,
            }
        )

        try:
            from src.brain.state_manager import state_manager

            if state_manager and getattr(state_manager, "available", False):
                await state_manager.publish_event(
                    "steps",
                    {
                        "type": "step_finished",
                        "step_id": str(step_id),
                        "success": result.success,
                        "error": result.error,
                        "result": result.result,
                    },
                )
        except (ImportError, NameError):
            pass

        # Knowledge Graph Sync
        asyncio.create_task(self._update_knowledge_graph(step_id, result))

        return result

    async def planner_node(self, state: TrinityState) -> dict[str, Any]:
        return {"system_state": SystemState.PLANNING.value}

    async def executor_node(self, state: TrinityState) -> dict[str, Any]:
        return {"system_state": SystemState.EXECUTING.value}

    async def verifier_node(self, state: TrinityState) -> dict[str, Any]:
        return {"system_state": SystemState.VERIFYING.value}

    def should_verify(self, state: TrinityState) -> str:
        return "continue"

    async def shutdown(self):
        """Clean shutdown of system components"""
        logger.info("[ORCHESTRATOR] Shutting down...")
        try:
            from src.brain.mcp_manager import mcp_manager

            await mcp_manager.shutdown()
        except:
            pass
        try:
            from src.brain.db.manager import db_manager

            await db_manager.close()
        except:
            pass
        try:
            await self.voice.close()
        except Exception:
            pass
        logger.info("[ORCHESTRATOR] Shutdown complete.")

    async def _update_knowledge_graph(self, step_id: str, result: StepResult):
        """Background task to sync execution results to Knowledge Graph"""
        try:
            from src.brain.knowledge_graph import knowledge_graph

            if knowledge_graph:
                await knowledge_graph.add_node(
                    node_type="STEP_EXECUTION",
                    node_id=f"exec:{self.state.get('db_task_id')}:{step_id}",
                    attributes={
                        "success": result.success,
                        "error": result.error,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
