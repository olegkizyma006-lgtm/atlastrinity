"""
AtlasTrinity State Manager

Redis-based state persistence for:
- Surviving restarts
- Checkpointing task progress
- Session recovery
"""

import json
import os
from datetime import datetime
from typing import Any

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis

        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

from .logger import logger


class StateManager:
    """
    Manages orchestrator state persistence using Redis.

    Features:
    - Save/restore task state
    - Checkpointing during execution
    - Session recovery after restart
    """

    def __init__(self, host: str = "localhost", port: int = 6379, prefix: str = "atlastrinity"):
        from .config_loader import config

        self.prefix = prefix
        self.available = False

        if not REDIS_AVAILABLE:
            logger.warning("[STATE] Redis not installed. Running without persistence.")
            return

        # Priority: EnvVar > Config > Default Host/Port
        redis_url = os.getenv("REDIS_URL") or config.get("state.redis_url")

        if redis_url:
            self.redis: Any | None = redis.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=2
            )
            logger.info("[STATE] Redis connected via URL")
        else:
            self.redis: Any | None = redis.Redis(
                host=host, port=port, decode_responses=True, socket_connect_timeout=2
            )
            logger.info(f"[STATE] Redis connected at {host}:{port}")

        # Connection will be tested lazily or in initialize
        self.available = True

    def _key(self, *parts: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self.prefix}:{':'.join(parts)}"

    def _serialize_state(self, state: dict[str, Any]) -> str:
        """Serialize state, handling LangChain messages correctly."""
        from langchain_core.messages import message_to_dict

        serialized = state.copy()
        if "messages" in serialized:
            serialized["messages"] = [
                message_to_dict(m) if not isinstance(m, dict) else m for m in serialized["messages"]
            ]

        return json.dumps(serialized, default=str)

    def _deserialize_state(self, data: str) -> dict[str, Any]:
        """Deserialize state, reconstructing LangChain messages defensively."""
        from langchain_core.messages import messages_from_dict

        try:
            state = json.loads(data)
            if "messages" in state:
                msgs = state["messages"]
                # DEFENSIVE: Only attempt to restore if we have a list of dicts
                if isinstance(msgs, list):
                    valid_dicts = [m for m in msgs if isinstance(m, dict) and "type" in m]

                    if valid_dicts:
                        # Reconstruct objects
                        state["messages"] = messages_from_dict(valid_dicts)
                    else:
                        # Legacy data or empty - start fresh for messages
                        logger.warning(
                            f"[STATE] No valid dict messages found. Found {len(msgs)} legacy/string items. Start fresh history."
                        )
                        state["messages"] = []
                else:
                    state["messages"] = []

            return state
        except Exception as e:
            logger.error(f"[STATE] Deserialization failed: {e}")
            # Return a minimal valid state so orchestrator doesn't crash
            return {"messages": [], "system_state": "IDLE"}

    async def save_session(self, session_id: str, state: dict[str, Any]) -> bool:
        """
        Save full session state.
        """
        if not self.available or not self.redis:
            return False

        try:
            key = self._key("session", session_id)
            state_to_save = state.copy()
            state_to_save["_saved_at"] = datetime.now().isoformat()

            await self.redis.set(key, self._serialize_state(state_to_save))
            await self.redis.expire(key, 86400 * 7)  # 7 days TTL
            logger.info(f"[STATE] Session saved: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to save session: {e}")
            return False

    async def restore_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Restore session state.
        """
        if not self.available or not self.redis:
            return None

        try:
            key = self._key("session", session_id)
            data = await self.redis.get(key)
            if data:
                state = self._deserialize_state(data)
                logger.info(f"[STATE] Session restored: {session_id}")
                return state
        except Exception as e:
            logger.error(f"[STATE] Failed to restore session: {e}")

        return None

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Directly get session data by ID."""
        return await self.restore_session(session_id)

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions with summaries from Redis and DB."""
        sessions_map = {}

        # 1. Fetch from Redis (Active but ephemeral)
        if self.available and self.redis:
            try:
                pattern = self._key("session", "*")
                async for key in self.redis.scan_iter(pattern):
                    data = await self.redis.get(key)
                    if data:
                        try:
                            state = json.loads(str(data))
                            session_id = key.split(":")[-1]
                            theme = state.get("_theme", "Untitled Session")
                            if theme == "Untitled Session":
                                msgs = state.get("messages", [])
                                for m in msgs:
                                    if isinstance(m, dict) and m.get("type") == "human":
                                        theme = m.get("content", "")[:40] + "..."
                                        break

                            sessions_map[session_id] = {
                                "id": session_id,
                                "theme": theme,
                                "saved_at": state.get("_saved_at", datetime.now().isoformat()),
                                "source": "redis",
                            }
                        except Exception as e:
                            logger.error(f"[STATE] Error parsing redis session {key}: {e}")
            except Exception as e:
                logger.error(f"[STATE] Failed to list redis sessions: {e}")

        # 2. Fetch from DB (Persistent History)
        from .db.manager import db_manager

        if db_manager.available:
            try:
                from sqlalchemy import select

                from .db.schema import Session as DBSession

                async with await db_manager.get_session() as session:
                    stmt = select(DBSession).order_by(DBSession.started_at.desc()).limit(50)
                    result = await session.execute(stmt)
                    db_sessions = result.scalars().all()

                    for ds in db_sessions:
                        sid = str(ds.id)
                        # We prefer Redis state if it exists (might be newer)
                        if sid not in sessions_map:
                            meta = ds.metadata_blob or {}
                            sessions_map[sid] = {
                                "id": sid,
                                "theme": meta.get("theme")
                                or f"Session {ds.started_at.strftime('%Y-%m-%d %H:%M')}",
                                "saved_at": ds.started_at.isoformat(),
                                "source": "db",
                            }
            except Exception as e:
                logger.error(f"[STATE] Failed to list DB sessions: {e}")

        # Sort combined list by saved_at desc
        sessions = list(sessions_map.values())
        sessions.sort(key=lambda x: x["saved_at"], reverse=True)
        return sessions

    async def checkpoint(self, session_id: str, step_id: int, step_result: dict[str, Any]) -> bool:
        """
        Save checkpoint for a specific step.

        Args:
            session_id: Current session
            step_id: Step number
            step_result: Result of the step
        """
        if not self.available or not self.redis:
            return False

        try:
            key = self._key("checkpoint", session_id, str(step_id))
            checkpoint = {
                "step_id": step_id,
                "result": step_result,
                "timestamp": datetime.now().isoformat(),
            }
            await self.redis.set(key, json.dumps(checkpoint, default=str))
            await self.redis.expire(key, 86400)  # 1 day TTL
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to checkpoint: {e}")
            return False

    async def get_last_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Get the most recent checkpoint for a session."""
        if not self.available or not self.redis:
            return None

        try:
            # Get all checkpoints for session
            pattern = self._key("checkpoint", session_id, "*")
            keys = []
            async for key in self.redis.scan_iter(pattern):
                keys.append(key)

            if not keys:
                return None

            # Find the latest
            latest = None
            latest_id = -1

            for key in keys:
                data = await self.redis.get(key)
                if data:
                    checkpoint = json.loads(str(data))
                    if checkpoint.get("step_id", 0) > latest_id:
                        latest = checkpoint
                        latest_id = checkpoint.get("step_id", 0)

            return latest
        except Exception as e:
            logger.error(f"[STATE] Failed to get checkpoint: {e}")
            return None

    async def set_current_task(self, task_description: str, task_id: str | None = None) -> bool:
        """Save the current active task."""
        if not self.available or not self.redis:
            return False

        try:
            key = self._key("current_task")
            task = {
                "id": task_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                "description": task_description,
                "started_at": datetime.now().isoformat(),
            }
            await self.redis.set(key, json.dumps(task))
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to set current task: {e}")
            return False

    async def get_current_task(self) -> dict[str, Any] | None:
        """Get the current active task (for recovery after restart)."""
        if not self.available or not self.redis:
            return None

        try:
            key = self._key("current_task")
            data = await self.redis.get(key)
            if data:
                return json.loads(str(data))
        except Exception as e:
            logger.error(f"[STATE] Failed to get current task: {e}")

        return None

    async def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session."""
        if not self.available or not self.redis:
            return False

        try:
            # Delete session
            await self.redis.delete(self._key("session", session_id))

            # Delete checkpoints
            pattern = self._key("checkpoint", session_id, "*")
            async for key in self.redis.scan_iter(pattern):
                await self.redis.delete(key)

            logger.info(f"[STATE] Session cleared: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to clear session: {e}")
            return False

    async def publish_event(self, channel: str, data: dict[str, Any]) -> bool:
        """
        Broadcast an event via Redis Pub/Sub.

        Args:
            channel: The channel name (e.g., 'tasks', 'steps')
            data: Event payload
        """
        if not self.available or not self.redis:
            return False

        try:
            full_channel = self._key("events", channel)
            data["timestamp"] = datetime.now().isoformat()
            await self.redis.publish(full_channel, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to publish event: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get state manager statistics."""
        if not self.available or not self.redis:
            return {"available": False}

        try:
            info = await self.redis.info("keyspace")
            return {"available": True, "connected": True, "keyspace": info}
        except Exception as e:
            return {"available": True, "connected": False, "error": str(e)}


# Singleton instance
state_manager = StateManager()
