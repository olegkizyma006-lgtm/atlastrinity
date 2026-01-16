"""
AtlasTrinity Orchestrator
LangGraph-based state machine that coordinates Agents (Atlas, Tetyana, Grisha)
"""

import ast
import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

try:
    from langgraph.graph import add_messages
except ImportError:
    from langgraph.graph.message import add_messages

from .agents import Atlas, Grisha, Tetyana
from .agents.tetyana import StepResult
from .config import IS_MACOS, PLATFORM_NAME, PROJECT_ROOT
from .consolidation import consolidation_module
from .context import shared_context
from .db.manager import db_manager
from .db.schema import LogEntry as DBLog
from .db.schema import Session as DBSession
from .db.schema import Task as DBTask
from .db.schema import TaskStep as DBStep
from .knowledge_graph import knowledge_graph
from .logger import logger
from .mcp_manager import mcp_manager
from .memory import long_term_memory
from .metrics import metrics_collector
from .notifications import notifications
from .state_manager import state_manager
from .voice.tts import VoiceManager


class SystemState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CHAT = "CHAT"


class TrinityState(TypedDict):
    messages: List[BaseMessage]
    system_state: str
    current_plan: Optional[Any]
    step_results: List[Dict[str, Any]]
    error: Optional[str]


class Trinity:
    def __init__(self):
        self.atlas = Atlas()
        self.tetyana = Tetyana()
        self.grisha = Grisha()
        self.voice = VoiceManager()

        # Initialize graph
        self.graph = self._build_graph()
        self._log_lock = asyncio.Lock()

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
        # Initialize DB
        await db_manager.initialize()

        logger.info(f"[GRISHA] Auditor ready. Vision: {self.grisha.llm.model_name}")

    def _build_graph(self):
        workflow = StateGraph(TrinityState)

        # Define nodes
        workflow.add_node("atlas_planning", self.planner_node)
        workflow.add_node("tetyana_execution", self.executor_node)
        workflow.add_node("grisha_verification", self.verifier_node)

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
            parts: List[str] = []
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
        """Voice wrapper"""
        print(f"[{agent_id.upper()}] Speaking: {text}")

        # Synchronize with UI chat log (MESSAGES, NOT JUST LOGS)
        if hasattr(self, "state") and self.state is not None:
            from langchain_core.messages import AIMessage

            if "messages" not in self.state:
                self.state["messages"] = []
            # Append to chat history
            self.state["messages"].append(AIMessage(content=text, name=agent_id.upper()))

        await self._log(text, source=agent_id, type="voice")
        try:
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
                "id": f"log-{len(self.state.get('logs', []))}-{time.time()}",
                "timestamp": time.time(),
                "agent": source.upper(),
                "message": text_str,
                "type": type,
            }
            if "logs" not in self.state:
                self.state["logs"] = []
            self.state["logs"].append(entry)

    async def _update_knowledge_graph(self, step_id: str, result: StepResult):
        """Update KG with entities found in the step results"""
        if not knowledge_graph:
            return

        task_id = self.state.get("db_task_id")
        if not task_id:
            return

        task_node_id = f"task:{task_id}"

        # 1. TOOL node
        if result.tool_call:
            tool_name = result.tool_call.get("name")
            if tool_name:
                tool_node_id = f"tool:{tool_name}"
                await knowledge_graph.add_node(
                    "TOOL",
                    tool_node_id,
                    {"description": f"MCP Tool or Wrapper: {tool_name}"},
                )
                await knowledge_graph.add_edge(task_node_id, tool_node_id, "USES")

        # 2. FILE nodes (from shared_context)
        for file_path in shared_context.recent_files:
            file_node_id = f"file://{file_path}"
            await knowledge_graph.add_node(
                "FILE",
                file_node_id,
                {"description": "File used or created by task", "path": file_path},
            )
            relation = "MODIFIED" if "write" in shared_context.last_operation else "ACCESSED"
            await knowledge_graph.add_edge(task_node_id, file_node_id, relation)

        # 3. USER node (only create once per session to avoid duplicates)
        if not getattr(self, "_user_node_created", False):
            user_node_id = "user:dev"
            await knowledge_graph.add_node("USER", user_node_id, {"name": "Developer"})
            await knowledge_graph.add_edge(task_node_id, user_node_id, "ASSIGNED_BY")
            self._user_node_created = True

    async def _verify_db_ids(self):
        """Verify that restored DB IDs exist. If not, clear them."""
        if not db_manager.available:
            return

        session_id_str = self.state.get("db_session_id")
        task_id_str = self.state.get("db_task_id")

        async with await db_manager.get_session() as db_sess:
            import uuid

            from sqlalchemy import select

            if session_id_str:
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

            if task_id_str:
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

    def get_state(self) -> Dict[str, Any]:
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

        for m in self.state.get("messages", []):
            if isinstance(m, HumanMessage):
                messages.append(
                    {
                        "agent": "USER",
                        "text": m.content,
                        "timestamp": datetime.now().timestamp(),
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
                    }
                )

        return {
            "system_state": sys_state,
            "current_task": task_summary,
            "active_agent": active_agent,
            "messages": messages,
            "logs": self.state.get("logs", [])[-50:],
            "step_results": self.state.get("step_results", []),
            "metrics": metrics_collector.get_metrics(),
        }

    async def run(self, user_request: str) -> Dict[str, Any]:
        """
        Main orchestration loop with advanced persistence and memory
        """
        start_time = asyncio.get_event_loop().time()
        session_id = "current_session"

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
            if state_manager.available and not self.state["messages"]:
                saved_state = state_manager.restore_session(session_id)
                if saved_state:
                    self.state = saved_state
                    # Ensure critical fields exist after restoration
                    if "step_results" not in self.state:
                        self.state["step_results"] = []
                    if "logs" not in self.state:
                        self.state["logs"] = []
                    if "error" not in self.state:
                        self.state["error"] = None
                    logger.info("[ORCHESTRATOR] State restored from Redis")
                    # CRITICAL: Verify that the DB IDs in the restored state actually exist
                    await self._verify_db_ids()

            # Append the new user message
            self.state["messages"].append(HumanMessage(content=user_request))
            self.state["system_state"] = SystemState.PLANNING.value
            self.state["error"] = None

            # DB Session Creation (Only for top-level)
            if db_manager.available and "db_session_id" not in self.state:
                try:
                    async with await db_manager.get_session() as db_sess:
                        new_session = DBSession(started_at=datetime.utcnow())
                        db_sess.add(new_session)
                        await db_sess.commit()
                        self.state["db_session_id"] = str(new_session.id)
                except Exception as e:
                    logger.error(f"DB Session creation failed: {e}")
                    if "db_session_id" in self.state:
                        del self.state["db_session_id"]

        if not is_subtask:
            state_manager.publish_event(
                "tasks",
                {
                    "type": "task_started",
                    "request": user_request,
                    "session_id": session_id,
                },
            )

        await self._log(f"New Request: {user_request}", "system")

        # 2. Atlas Planning
        try:
            state_manager.publish_event(
                "tasks", {"type": "planning_started", "request": user_request}
            )
            # Pass history to Atlas for context (Last 10 messages to avoid context pollution)
            history = self.state.get("messages", [])[-10:-1]

            # Fetch dynamic MCP Catalog (concise servers list)
            mcp_catalog = await mcp_manager.get_mcp_catalog()

            # Inject catalog into shared context
            shared_context.available_mcp_catalog = mcp_catalog

            analysis = await self.atlas.analyze_request(user_request, history=history)

            if analysis.get("intent") == "chat":
                response = analysis.get("initial_response") or await self.atlas.chat(
                    user_request, history=history
                )
                # Note: _speak already appends the message to history
                await self._speak("atlas", response)
                self.state["system_state"] = SystemState.IDLE.value

                # Save state for UI but don't clear entire session unless requested
                if state_manager.available:
                    state_manager.save_session(session_id, self.state)

                return {"status": "completed", "result": response, "type": "chat"}

            await self._speak("atlas", analysis.get("reason", "Аналізую..."))

            # Keep-alive logger to show activity in UI during long LLM calls
            # Added rate limiting to prevent log spam
            _keep_alive_last_log = [0.0]  # Use list for mutable closure

            async def keep_alive_logging():
                import time

                while True:
                    await asyncio.sleep(15)  # Wait 15 seconds
                    current_time = time.time()
                    # Rate limit: don't log if less than 10 seconds since last log
                    if current_time - _keep_alive_last_log[0] >= 10:
                        _keep_alive_last_log[0] = current_time
                        await self._log("Atlas is thinking... (Planning logic flow)", "system")

            planning_task = asyncio.create_task(self.atlas.create_plan(analysis))
            logger_task = asyncio.create_task(keep_alive_logging())

            try:
                plan = await asyncio.wait_for(planning_task, timeout=300.0)
            finally:
                logger_task.cancel()
                try:
                    await logger_task
                except asyncio.CancelledError:
                    pass

            if not plan or not plan.steps:
                msg = self.atlas.get_voice_message("no_steps")
                await self._speak("atlas", msg)
                return {"status": "completed", "result": msg, "type": "chat"}

            self.state["current_plan"] = plan

            # DB Task Creation
            if db_manager.available and self.state.get("db_session_id"):
                try:
                    async with await db_manager.get_session() as db_sess:
                        new_task = DBTask(
                            session_id=self.state["db_session_id"],
                            goal=user_request,
                            status="PENDING",
                        )
                        db_sess.add(new_task)
                        await db_sess.commit()
                        self.state["db_task_id"] = str(new_task.id)

                        # GraphChain: Add Task Node and sync to vector
                        await knowledge_graph.add_node(
                            node_type="TASK",
                            node_id=f"task:{new_task.id}",
                            attributes={
                                "goal": user_request,
                                "timestamp": datetime.utcnow().isoformat(),
                                "steps_count": len(plan.steps),
                            },
                        )
                except Exception as e:
                    logger.error(f"DB Task creation failed: {e}")
                    # Clear ID if it failed to persist
                    if "db_task_id" in self.state:
                        del self.state["db_task_id"]

            if state_manager.available:
                state_manager.save_session(session_id, self.state)

            state_manager.publish_event(
                "tasks",
                {
                    "type": "planning_finished",
                    "session_id": session_id,
                    "steps_count": len(plan.steps),
                },
            )

            await self._speak(
                "atlas",
                self.atlas.get_voice_message("plan_created", steps=len(plan.steps)),
            )

        except Exception as e:
            import traceback

            logger.error(f"[ORCHESTRATOR] Planning error: {e}")
            logger.error(traceback.format_exc())
            self.state["system_state"] = SystemState.ERROR.value
            state_manager.publish_event(
                "tasks",
                {
                    "type": "task_finished",
                    "status": "error",
                    "error": str(e),
                    "session_id": session_id,
                },
            )
            return {"status": "error", "error": str(e)}

        # 3. Execution Loop (Tetyana)
        self.state["system_state"] = SystemState.EXECUTING.value

        # 3. Execution Loop (Tetyana) - RECURSIVE
        self.state["system_state"] = SystemState.EXECUTING.value

        try:
            # Initial numbering is 1, 2, 3...
            await self._execute_steps_recursive(plan.steps)

        except Exception as e:
            await self._log(f"Critical error: {e}", "error")
            return {"status": "error", "error": str(e)}

        # 4. Success Tasks: Memory & Cleanup
        duration = asyncio.get_event_loop().time() - start_time
        notifications.show_completion(user_request, True, duration)

        # Atlas Verification Gate & Memory
        if (
            long_term_memory.available
            and not is_subtask
            and self.state["system_state"] != SystemState.ERROR.value
        ):
            # Atlas reviews the execution
            evaluation = await self.atlas.evaluate_execution(
                user_request, self.state["step_results"]
            )

            if evaluation.get("should_remember") and evaluation.get("quality_score", 0) >= 0.7:
                await self._log(
                    f"Verification Pass: Score {evaluation.get('quality_score')} ({evaluation.get('analysis')})",
                    "atlas",
                )

                strategy_steps = evaluation.get("compressed_strategy") or self._extract_golden_path(
                    self.state["step_results"]
                )

                long_term_memory.remember_strategy(
                    task=user_request,
                    plan_steps=strategy_steps,
                    outcome="SUCCESS",
                    success=True,
                )
                await self._log(f"Brain saved {len(strategy_steps)} steps to memory", "system")

                # Update DB Task with quality metric
                if db_manager.available and self.state.get("db_task_id"):
                    try:
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
            else:
                await self._log(
                    f"Verification Fail: Score {evaluation.get('quality_score', 0)}. Analysis: {evaluation.get('analysis', 'No analysis')}",
                    "atlas",
                    type="warning",
                )

        # Nightly/End-of-task consolidation check
        if not is_subtask and consolidation_module.should_consolidate():
            asyncio.create_task(consolidation_module.run_consolidation())

        self.state["system_state"] = SystemState.COMPLETED.value
        if state_manager.available:
            state_manager.clear_session(session_id)

        state_manager.publish_event(
            "tasks",
            {"type": "task_finished", "status": "completed", "session_id": session_id},
        )

        return {"status": "completed", "result": self.state["step_results"]}

    def _extract_golden_path(self, raw_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts only the successful actions that led to the solution.
        Filters out:
        - Failed attempts
        - Steps that were retried
        - Steps replaced by recovery actions
        """
        golden_path = []

        # Create map of step_id to LAST result (overwriting previous attempts)
        final_outcomes = {}
        for res in raw_results:
            step_id = res.get("step_id")
            # If we retried a step, this overwrite ensures we only keep the latest
            final_outcomes[step_id] = res

        # Sort by step sequence (assuming ID is number or sortable)
        # Note: IDs might be "1", "2" or "2.1" (if subtasks).
        # We'll rely on the order they appear in the final_outcomes keys if feasible,
        # but sorted is safer.
        try:
            sorted_keys = sorted(final_outcomes.keys(), key=lambda x: str(x))
        except Exception:
            sorted_keys = list(final_outcomes.keys())

        for key in sorted_keys:
            item = final_outcomes[key]
            if item.get("success"):
                # Format: "Action description"
                # Ideally we want the intent/action name, but result might just contain output.
                # We need to map back to the Plan if possible, but the 'result' dict here
                # might not have the original action text.
                # Use action description if available
                val = item.get("action")
                if not val:
                    val = str(item.get("result", ""))[:200]
                golden_path.append(val)

        return golden_path

    async def _execute_steps_recursive(
        self, steps: List[Dict], parent_prefix: str = "", depth: int = 0
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
                f"Recursion depth {depth}: applying {backoff_ms}ms backoff",
                "orchestrator"
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

            # Retry loop with Dynamic Temperature
            max_step_retries = 3
            step_success = False
            last_error = ""

            for attempt in range(1, max_step_retries + 1):
                await self._log(
                    f"Step {step_id}, Attempt {attempt}: {step.get('action')}",
                    "orchestrator",
                )

                try:
                    step_result = await asyncio.wait_for(
                        self.execute_node(self.state, step, step_id, attempt=attempt),
                        timeout=300.0,
                    )
                    if step_result.success:
                        step_success = True
                        break
                    else:
                        last_error = step_result.error
                        await self._log(
                            f"Step {step_id} Attempt {attempt} failed: {last_error}",
                            "warning",
                        )

                except Exception as e:
                    last_error = str(e)
                    await self._log(
                        f"Step {step_id} Attempt {attempt} crashed: {last_error}",
                        "error",
                    )

            if not step_success:
                # RECOVERY LOGIC
                notifications.send_stuck_alert(step_id, last_error, max_step_retries)

                await self._log(f"Atlas Recovery for Step {step_id}...", "orchestrator")
                await self._speak(
                    "atlas", self.atlas.get_voice_message("recovery_started", step_id=step_id)
                )
                try:

                    # Collect recent logs for context
                    recent_logs = []
                    if self.state and "logs" in self.state:
                        recent_logs = [
                            f"[{l.get('agent', 'SYS')}] {l.get('message', '')}"
                            for l in self.state["logs"][-20:]
                        ]
                    log_context = "\n".join(recent_logs)

                    # Construct detailed error context for Vibe
                    error_context = f"Step ID: {step_id}\n" f"Action: {step.get('action', '')}\n"

                    await self._log(
                        f"Engaging Vibe Self-Healing for Step {step_id} (Timeout: 300s)...",
                        "orchestrator",
                    )
                    await self._log(f"[VIBE] Error to analyze: {last_error[:200]}...", "vibe")
                    await self._speak(
                        "atlas", self.atlas.get_voice_message("vibe_engaged", step_id=step_id)
                    )

                    # Use vibe_analyze_error for programmatic CLI mode with full logging
                    vibe_res = await asyncio.wait_for(
                        mcp_manager.call_tool(
                            "vibe",
                            "vibe_analyze_error",
                            {
                                "error_message": f"{error_context}\n{last_error}",
                                "log_context": log_context,
                                "cwd": str(PROJECT_ROOT),
                                "timeout_s": 300,  # 5 minutes for deep debugging
                                "auto_fix": True,  # Enable auto-fixing
                            },
                        ),
                        timeout=310.0,
                    )
                    vibe_text = self._extract_vibe_payload(self._mcp_result_to_text(vibe_res))
                    if vibe_text:
                        last_error = last_error + "\n\nVIBE_FIX_REPORT:\n" + vibe_text[:4000]
                        await self._log(f"Vibe completed self-healing for step {step_id}", "system")
                except Exception as ve:
                    await self._log(f"Vibe self-healing failed: {ve}", "error")

                # RECURSION SAFEGUARD
                if depth >= 3:
                     # Stop infinite recovery loops
                     raise Exception(f"Max recovery depth ({depth}) reached for {step_id}. Aborting.")

                try:
                    # Ask Atlas for help
                    recovery = await asyncio.wait_for(
                        self.atlas.help_tetyana(step.get("action", f"Step {step_id}"), last_error),
                        timeout=45.0,
                    )

                    voice_msg = recovery.get("voice_message", "Знайшов альтернативний шлях.")
                    await self._speak("atlas", voice_msg)

                    alt_steps = recovery.get("alternative_steps", [])
                    if alt_steps and len(alt_steps) > 0:
                        # RECURSIVE CALL for alternative steps
                        # These become sub-steps: 3.1, 3.2 etc.
                        await self._log(
                            f"Executing {len(alt_steps)} recovery steps for {step_id}",
                            "system",
                        )

                        # Pass current step_id as parent_prefix
                        await self._execute_steps_recursive(
                            alt_steps, parent_prefix=step_id, depth=depth + 1
                        )

                        # If recursion returned without exception, it means success.
                        # We consider the original Step X as "fixed" by its children X.1, X.2.
                        continue
                    else:
                        raise Exception(f"No alternative steps provided for {step_id}")

                except Exception as rec_err:
                    error_msg = f"Recovery failed for {step_id}: {rec_err}"
                    await self._log(error_msg, "error")
                    raise Exception(error_msg)

        return True

    async def execute_node(
        self, state: TrinityState, step: Dict[str, Any], step_id: str, attempt: int = 1
    ) -> StepResult:
        """Atomic execution logic with recursion and dynamic temperature"""
        # Starting message logic
        # Simple heuristic: If it's a top level step (no dots) and first attempt
        if "." not in str(step_id) and attempt == 1 and str(step_id) == "1":
            await self._speak("tetyana", self.tetyana.get_voice_message("starting"))
        elif "." in str(step_id):
            # It's a sub-step/recovery step
            pass

        state_manager.publish_event(
            "steps",
            {
                "type": "step_started",
                "step_id": str(step_id),
                "action": step.get("action", "Working..."),
                "attempt": attempt,
            },
        )
        # DB Step logging
        db_step_id = None
        if db_manager.available and self.state.get("db_task_id"):
            try:
                # We try to convert step_id (e.g. "3.2.1") to a sequence number?
                # DB schema expects integer sequence?
                # Ideally DB should support string sequence or we map it.
                # Schema definition said 'sequence_number' is Integer.
                # We'll use a hash or just simple sequential mapping if we can't store "3.1".
                # WORKAROUND: For now, we store the hierarchical ID in the 'action' or 'tool' field prefix?
                # OR, we just log it as-is and let Integer fail?
                # Let's assume we want to store it.
                # FIX: We'll modify the DB schema later if strictly integer.
                # For now, let's just use a simple counter or 0 for subtasks if it fails validation,
                # BUT wait, the schema defines it as Integer.
                # I will create a hash/counter for the DB or just 0.
                seq_num = 0
                try:
                    seq_num = int(float(step_id))  # works for "1", "3.0" but not "3.1"
                except Exception:
                    seq_num = 0  # Subtasks represent 0

                async with await db_manager.get_session() as db_sess:
                    new_step = DBStep(
                        task_id=self.state["db_task_id"],
                        sequence_number=seq_num,
                        action=f"[{step_id}] {step.get('action', '')}",
                        tool=step.get("tool", ""),
                        status="RUNNING",
                    )
                    db_sess.add(new_step)
                    await db_sess.commit()
                    db_step_id = str(new_step.id)
            except Exception as e:
                logger.error(f"DB Step creation failed: {e}")

        step_start_time = asyncio.get_event_loop().time()

        if step.get("type") == "subtask" or step.get("tool") == "subtask":
            self._in_subtask = True
            try:
                sub_result = await self.run(step.get("action"))
            finally:
                self._in_subtask = False

            result = StepResult(
                step_id=step.get("id"),
                success=sub_result["status"] == "completed",
                result="Subtask completed",
                error=sub_result.get("error"),
            )
        else:
            try:
                # Inject context results (last 3 for relevance) so Tetyana knows what happened
                step_copy = step.copy()
                if self.state and "step_results" in self.state:
                    step_copy["previous_results"] = self.state["step_results"][-3:]

                result = await self.tetyana.execute_step(step_copy, attempt=attempt)
                if result.voice_message:
                    await self._speak("tetyana", result.voice_message)
            except Exception as e:
                logger.exception("Tetyana execution crashed")
                result = StepResult(
                    step_id=step.get("id"),
                    success=False,
                    result="Crashed",
                    error=str(e),
                )

        # Update DB Step
        if db_manager.available and db_step_id:
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

        # Check verification
        if step.get("requires_verification"):
            self.state["system_state"] = SystemState.VERIFYING.value
            await self._speak("tetyana", self.tetyana.get_voice_message("asking_verification"))

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
                    overall_goal=(
                        self.state["current_plan"].goal
                        if self.state.get("current_plan")
                        else "Task"
                    ),
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
            except Exception as e:
                print(f"[ERROR] Verification failed: {e}")
                await self._log(f"Verification crashed: {e}", "error")
                result.success = False
                result.error = f"Verification system error: {str(e)}"

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

        state_manager.publish_event(
            "steps",
            {
                "type": "step_finished",
                "step_id": str(step_id),
                "success": result.success,
                "error": result.error,
                "result": result.result,
            },
        )

        # Knowledge Graph Sync
        asyncio.create_task(self._update_knowledge_graph(step_id, result))

        return result

    # Placeholder graph nodes (not used in direct loop but required for graph structure)
    async def planner_node(self, state: TrinityState):
        return {"system_state": SystemState.PLANNING.value}

    async def executor_node(self, state: TrinityState):
        return {"system_state": SystemState.EXECUTING.value}

    async def verifier_node(self, state: TrinityState):
        return {"system_state": SystemState.VERIFYING.value}

    def should_verify(self, state: TrinityState):
        return "continue"
