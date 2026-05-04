"""
db/models.py — SQLAlchemy 2.x async ORM models for SROP.

Tables
------
sessions     – one row per conversation session
messages     – chat turns attached to a session
agent_traces – execution traces from the ADK orchestrator
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ── sessions ──────────────────────────────────────────────────────────────────

class Session(Base):
    """Represents a single user session with persisted state."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan_tier: Mapped[str] = mapped_column(String(64), nullable=False, default="free")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    # Arbitrary JSON blob – stores conversation state, flags, tool results, etc.
    state_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── relationships ─────────────────────────────────────────────────────────
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    agent_traces: Mapped[list["AgentTrace"]] = relationship(
        "AgentTrace", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Session id={self.id!r} user_id={self.user_id!r}>"


# ── messages ──────────────────────────────────────────────────────────────────

class Message(Base):
    """A single chat turn (user or assistant) within a session."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # ── relationships ─────────────────────────────────────────────────────────
    session: Mapped["Session"] = relationship("Session", back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Message id={self.id!r} role={self.role!r}>"


# ── agent_traces ──────────────────────────────────────────────────────────────

class AgentTrace(Base):
    """Execution trace emitted by the ADK orchestrator for observability."""

    __tablename__ = "agent_traces"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    routed_to: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # ── relationships ─────────────────────────────────────────────────────────
    session: Mapped["Session"] = relationship("Session", back_populates="agent_traces")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentTrace id={self.id!r} routed_to={self.routed_to!r}>"
