"""AtlasTrinity Message Bus

Typed inter-agent communication system with DB persistence.
Provides reliable message passing between Atlas, Tetyana, and Grisha.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

logger = logging.getLogger("brain.message_bus")


class MessageType(Enum):
    """Types of messages that can be sent between agents"""

    REJECTION = "rejection"  # Grisha -> Tetyana: step verification failed
    HELP_REQUEST = "help_request"  # Tetyana -> Atlas: need assistance
    CHAT = "chat"  # User <-> Agents: conversational messages
    FEEDBACK = "feedback"  # Any agent: general feedback
    RECOVERY_COMPLETE = "recovery_complete"  # Vibe/Atlas: fix applied
    STEP_COMPLETE = "step_complete"  # Tetyana -> Grisha: ready for verification
    VOICE = "voice"  # Any agent: signal for orchestrator to speak
    LOG = "log"  # Any agent: signal for custom UI log
    # Enhanced communication types for self-healing
    ERROR_ANALYSIS = "error_analysis"  # Vibe -> Grisha/Atlas: detailed error analysis
    VERIFICATION_REQUEST = "verification_request"  # Tetyana -> Grisha: verify result
    APPROVAL = "approval"  # Grisha/Atlas -> Tetyana: proceed with action
    CORRECTION = "correction"  # Grisha -> Tetyana: specific fix instruction
    HEALING_STATUS = "healing_status"  # Vibe -> All: self-healing progress update


@dataclass
class AgentMsg:
    """Typed message for inter-agent communication"""

    from_agent: str  # atlas, tetyana, grisha, vibe
    to_agent: str  # atlas, tetyana, grisha, or "all"
    message_type: MessageType
    payload: dict[str, Any]
    step_id: str | None = None
    session_id: UUID | None = None
    message_id: UUID | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    read_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DB storage"""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "step_id": self.step_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "timestamp": self.timestamp.isoformat(),
        }


class MessageBus:
    """Typed inter-agent communication bus with optional DB persistence.

    Features:
    - Typed messages with MessageType enum
    - In-memory queue for fast access
    - Optional DB persistence for recovery
    - Read tracking for message acknowledgment
    """

    def __init__(self):
        self._queue: dict[str, list[AgentMsg]] = {
            "atlas": [],
            "tetyana": [],
            "grisha": [],
            "orchestrator": [],
            "all": [],
        }
        self._db_available = False
        self._init_db()

    def _init_db(self):
        """Initialize DB connection for persistence"""
        try:
            from .db.manager import db_manager

            self._db_available = db_manager.available
            if self._db_available:
                logger.info("[MESSAGE_BUS] DB persistence enabled")
        except Exception as e:
            logger.warning(f"[MESSAGE_BUS] DB not available: {e}")
            self._db_available = False

    async def send(self, msg: AgentMsg) -> bool:
        """Send message to target agent.

        Args:
            msg: The message to send

        Returns:
            True if sent successfully

        """
        try:
            # Add to in-memory queue
            target = msg.to_agent.lower()
            if target in self._queue:
                self._queue[target].append(msg)
            else:
                self._queue["all"].append(msg)

            logger.info(
                f"[MESSAGE_BUS] {msg.from_agent} -> {msg.to_agent}: "
                f"{msg.message_type.value} (step={msg.step_id})",
            )

            # Persist to DB if available
            if self._db_available:
                await self._persist_message(msg)

            return True
        except Exception as e:
            logger.error(f"[MESSAGE_BUS] Send failed: {e}")
            return False

    async def receive(
        self,
        agent: str,
        message_type: MessageType | None = None,
        mark_read: bool = True,
    ) -> list[AgentMsg]:
        """Receive pending messages for an agent.

        Args:
            agent: The receiving agent name
            message_type: Optional filter by message type
            mark_read: Whether to mark messages as read

        Returns:
            List of pending messages

        """
        agent = agent.lower()
        messages = []

        # Get from specific queue
        if agent in self._queue:
            messages.extend(self._queue[agent])
            if mark_read:
                self._queue[agent] = []

        # Also get broadcast messages
        messages.extend(self._queue.get("all", []))
        if mark_read:
            self._queue["all"] = []

        # Filter by type if specified
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        # Mark as read
        if mark_read:
            now = datetime.now()
            for msg in messages:
                msg.read_at = now

        return messages

    async def get_unread_count(self, agent: str) -> int:
        """Get count of unread messages for an agent"""
        agent = agent.lower()
        count = len(self._queue.get(agent, []))
        count += len([m for m in self._queue.get("all", []) if not m.read_at])
        return count

    async def _persist_message(self, msg: AgentMsg) -> bool:
        """Persist message to database"""
        try:
            from .db.manager import db_manager
            from .db.schema import AgentMessage

            async with await db_manager.get_session() as session:
                db_msg = AgentMessage(
                    session_id=msg.session_id,
                    from_agent=msg.from_agent,
                    to_agent=msg.to_agent,
                    message_type=msg.message_type.value,
                    step_id=msg.step_id,
                    payload=msg.payload,
                )
                session.add(db_msg)
                await session.commit()
                msg.message_id = db_msg.id
            return True
        except Exception as e:
            logger.warning(f"[MESSAGE_BUS] DB persist failed: {e}")
            return False

    async def clear(self, agent: str | None = None):
        """Clear message queue for agent or all"""
        if agent:
            self._queue[agent.lower()] = []
        else:
            for key in self._queue:
                self._queue[key] = []


# Singleton instance
message_bus = MessageBus()
