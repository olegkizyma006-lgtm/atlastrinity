"""
AtlasTrinity Database Schema
Uses SQLAlchemy 2.0+ (Async)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Database-agnostic UUID and JSON support
from sqlalchemy.types import TypeDecorator, CHAR
import uuid

class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses CHAR(36) for SQLite (default), or PostgreSQL's native UUID type.
    Stores UUIDs as canonical strings with hyphens.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import UUID as PG_UUID
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            else:
                return value


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    metadata_blob: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    tasks: Mapped[List["Task"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"))

    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(50), default="PENDING"
    )  # PENDING, RUNNING, COMPLETED, FAILED
    golden_path: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    session: Mapped["Session"] = relationship(back_populates="tasks")
    steps: Mapped[List["TaskStep"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class TaskStep(Base):
    __tablename__ = "task_steps"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"))

    sequence_number: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(Text)
    tool: Mapped[str] = mapped_column(String(100))

    status: Mapped[str] = mapped_column(String(50))  # SUCCESS, FAILED
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    task: Mapped["Task"] = relationship(back_populates="steps")
    tool_executions: Mapped[List["ToolExecution"]] = relationship(back_populates="step")


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("task_steps.id"))
    
    # Direct task association for faster audits
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=True)

    server_name: Mapped[str] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100))
    arguments: Mapped[Dict[str, Any]] = mapped_column(JSON)
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    step: Mapped["TaskStep"] = relationship(back_populates="tool_executions")


class LogEntry(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    level: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(Text)
    metadata_blob: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


# Knowledge Graph Nodes (Vertices)
class KGNode(Base):
    __tablename__ = "kg_nodes"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # URI: file://..., task:uuid
    type: Mapped[str] = mapped_column(String(50))  # FILE, TASK, TOOL, CONCEPT, DATASET
    namespace: Mapped[str] = mapped_column(String(100), default="global", index=True)
    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("tasks.id"), nullable=True)
    attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Knowledge Graph Edges (Relationships)
class KGEdge(Base):
    __tablename__ = "kg_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("kg_nodes.id"))
    target_id: Mapped[str] = mapped_column(ForeignKey("kg_nodes.id"))
    relation: Mapped[str] = mapped_column(String(50))  # CREATED, MODIFIED, READ, USED
    namespace: Mapped[str] = mapped_column(String(100), default="global", index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Agent Message Bus - Typed inter-agent communication
class AgentMessage(Base):
    """Typed messages between agents for reliable communication"""
    __tablename__ = "agent_messages"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"))

    from_agent: Mapped[str] = mapped_column(String(20))  # atlas, tetyana, grisha
    to_agent: Mapped[str] = mapped_column(String(20))
    message_type: Mapped[str] = mapped_column(String(50))  # rejection, help_request, feedback
    step_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# Analytics for recursive healing
class RecoveryAttempt(Base):
    """Track recursive healing attempts for analytics"""
    __tablename__ = "recovery_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("task_steps.id"))

    depth: Mapped[int] = mapped_column(Integer)  # recursion depth
    recovery_method: Mapped[str] = mapped_column(String(50))  # vibe, atlas_help, retry
    success: Mapped[bool] = mapped_column(Boolean)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    vibe_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_before: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ConversationSummary(Base):
    """Stores professional summaries of chat sessions for semantic recall"""
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    
    summary: Mapped[str] = mapped_column(Text)
    key_entities: Mapped[List[str]] = mapped_column(JSON, default=[]) # List of names/concepts
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_blob: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

class BehavioralDeviation(Base):
    """
    Stores logic deviations from original plans for auditing and analytics.
    Complements the vector-based memory in ChromaDB.
    """
    __tablename__ = "behavioral_deviations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"))
    step_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("task_steps.id"), nullable=True)

    original_intent: Mapped[str] = mapped_column(Text)
    deviation: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text)
    decision_factors: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
