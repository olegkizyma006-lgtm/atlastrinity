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
from typing import Any, Dict, List, Optional

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

        self.prefix = prefix
        self.available = False

        if not REDIS_AVAILABLE:
            logger.warning("[STATE] Redis not installed. Running without persistence.")
            return

        try:
            self.redis = redis.Redis(
                host=host, port=port, decode_responses=True, socket_connect_timeout=2
            )
            # Test connection
            self.redis.ping()
            self.available = True
            logger.info(f"[STATE] Redis connected at {host}:{port}")
        except redis.ConnectionError:
            logger.warning("[STATE] Redis not running. State persistence disabled.")
            self.redis = None
        except Exception as e:
            logger.warning(f"[STATE] Redis error: {e}. State persistence disabled.")
            self.redis = None

    def _key(self, *parts: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self.prefix}:{':'.join(parts)}"

    def save_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """
        Save full session state.

        Args:
            session_id: Unique session identifier
            state: Full state dict to save
        """
        if not self.available:
            return False

        try:
            key = self._key("session", session_id)
            state["_saved_at"] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(state, default=str))
            self.redis.expire(key, 86400 * 7)  # 7 days TTL
            logger.info(f"[STATE] Session saved: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to save session: {e}")
            return False

    def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore session state.

        Args:
            session_id: Session to restore

        Returns:
            State dict or None if not found
        """
        if not self.available:
            return None

        try:
            key = self._key("session", session_id)
            data = self.redis.get(key)
            if data:
                state = json.loads(data)
                logger.info(f"[STATE] Session restored: {session_id}")
                return state
        except Exception as e:
            logger.error(f"[STATE] Failed to restore session: {e}")

        return None

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Directly get session data by ID."""
        return self.restore_session(session_id)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions with summaries from Redis and DB."""
        sessions_map = {}

        # 1. Fetch from Redis (Active but ephemeral)
        if self.available:
            try:
                pattern = self._key("session", "*")
                for key in self.redis.scan_iter(pattern):
                    data = self.redis.get(key)
                    if data:
                        try:
                            state = json.loads(data)
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
                                "source": "redis"
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
                                "theme": meta.get("theme") or f"Session {ds.started_at.strftime('%Y-%m-%d %H:%M')}",
                                "saved_at": ds.started_at.isoformat(),
                                "source": "db"
                            }
            except Exception as e:
                logger.error(f"[STATE] Failed to list DB sessions: {e}")

        # Sort combined list by saved_at desc
        sessions = list(sessions_map.values())
        sessions.sort(key=lambda x: x["saved_at"], reverse=True)
        return sessions

    def checkpoint(self, session_id: str, step_id: int, step_result: Dict[str, Any]) -> bool:
        """
        Save checkpoint for a specific step.

        Args:
            session_id: Current session
            step_id: Step number
            step_result: Result of the step
        """
        if not self.available:
            return False

        try:
            key = self._key("checkpoint", session_id, str(step_id))
            checkpoint = {
                "step_id": step_id,
                "result": step_result,
                "timestamp": datetime.now().isoformat(),
            }
            self.redis.set(key, json.dumps(checkpoint, default=str))
            self.redis.expire(key, 86400)  # 1 day TTL
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to checkpoint: {e}")
            return False

    def get_last_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint for a session."""
        if not self.available:
            return None

        try:
            # Get all checkpoints for session
            pattern = self._key("checkpoint", session_id, "*")
            keys = list(self.redis.scan_iter(pattern))

            if not keys:
                return None

            # Find the latest
            latest = None
            latest_id = -1

            for key in keys:
                data = self.redis.get(key)
                if data:
                    checkpoint = json.loads(data)
                    if checkpoint.get("step_id", 0) > latest_id:
                        latest = checkpoint
                        latest_id = checkpoint.get("step_id", 0)

            return latest
        except Exception as e:
            logger.error(f"[STATE] Failed to get checkpoint: {e}")
            return None

    def set_current_task(self, task_description: str, task_id: str = None) -> bool:
        """Save the current active task."""
        if not self.available:
            return False

        try:
            key = self._key("current_task")
            task = {
                "id": task_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                "description": task_description,
                "started_at": datetime.now().isoformat(),
            }
            self.redis.set(key, json.dumps(task))
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to set current task: {e}")
            return False

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Get the current active task (for recovery after restart)."""
        if not self.available:
            return None

        try:
            key = self._key("current_task")
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"[STATE] Failed to get current task: {e}")

        return None

    def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session."""
        if not self.available:
            return False

        try:
            # Delete session
            self.redis.delete(self._key("session", session_id))

            # Delete checkpoints
            pattern = self._key("checkpoint", session_id, "*")
            for key in self.redis.scan_iter(pattern):
                self.redis.delete(key)

            logger.info(f"[STATE] Session cleared: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to clear session: {e}")
            return False

    def publish_event(self, channel: str, data: Dict[str, Any]) -> bool:
        """
        Broadcast an event via Redis Pub/Sub.

        Args:
            channel: The channel name (e.g., 'tasks', 'steps')
            data: Event payload
        """
        if not self.available:
            return False

        try:
            full_channel = self._key("events", channel)
            data["timestamp"] = datetime.now().isoformat()
            self.redis.publish(full_channel, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.error(f"[STATE] Failed to publish event: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        if not self.available:
            return {"available": False}

        try:
            info = self.redis.info("keyspace")
            return {"available": True, "connected": True, "keyspace": info}
        except Exception as e:
            return {"available": True, "connected": False, "error": str(e)}


# Singleton instance
state_manager = StateManager()
