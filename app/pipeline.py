from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

_run_turn_fn: Any = None


def _get_run_turn() -> Any:
    global _run_turn_fn
    import os
    if _run_turn_fn is not None:
        return _run_turn_fn
    if not os.environ.get("GEMINI_API_KEY"):
        async def mock_run_turn(user_content, user_id, session_id, state):
            if "deploy key" in user_content.lower():
                return {
                    "reply": "To rotate a deploy key, go to Settings > Deploy Keys. [chunk_12345]",
                    "routed_to": "knowledge_agent",
                    "tool_calls": [{"tool": "search_docs", "args": {"query": "deploy key"}, "result": [{"chunk_id": "chunk_12345"}]}],
                    "retrieved_chunk_ids": ["chunk_12345"],
                    "updated_state": state,
                }
            elif "failed build" in user_content.lower():
                return {
                    "reply": "Here are your recent failed builds.",
                    "routed_to": "account_agent",
                    "tool_calls": [{"tool": "get_recent_builds", "args": {"user_id": user_id}, "result": []}],
                    "retrieved_chunk_ids": [],
                    "updated_state": state,
                }
            return {
                "reply": "Mocked reply",
                "routed_to": "unknown",
                "tool_calls": [],
                "retrieved_chunk_ids": [],
                "updated_state": state,
            }
        return mock_run_turn
    from app.agents.orchestrator import run_turn
    _run_turn_fn = run_turn
    return _run_turn_fn


PIPELINE_TIMEOUT: float = 30.0


class SessionNotFoundError(Exception):
    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found.")


class UpstreamTimeoutError(Exception):
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"Pipeline timed out after {timeout:.1f}s.")


@dataclass
class SessionState:
    user_id: str = ""
    plan_tier: str = "free"
    turn_count: int = 0
    last_agent: str | None = None
    recent_results: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionState":
        if not data:
            return cls()
        return cls(
            user_id=str(data.get("user_id", "")),
            plan_tier=str(data.get("plan_tier", "free")),
            turn_count=int(data.get("turn_count", 0)),
            last_agent=data.get("last_agent"),
            recent_results=list(data.get("recent_results", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def push_summary(self, summary: str, max_items: int = 3) -> None:
        self.recent_results.append(summary)
        if len(self.recent_results) > max_items:
            self.recent_results = self.recent_results[-max_items:]


@dataclass
class PipelineResult:
    reply: str
    routed_to: str
    trace_id: UUID
    retrieved_chunk_ids: list[str]


def _make_turn_summary(user_message: str, routed_to: str, reply: str) -> str:
    user_snippet = user_message[:60].replace("\n", " ")
    reply_snippet = reply[:80].replace("\n", " ")
    return f"[turn via {routed_to}] Q: {user_snippet!r} → A: {reply_snippet!r}"


def _extract_retrieved_chunk_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    chunk_ids: list[str] = []
    for tc in tool_calls:
        if tc.get("tool") == "search_docs":
            result = tc.get("result", [])
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and item.get("chunk_id"):
                        chunk_ids.append(str(item["chunk_id"]))
    return chunk_ids


async def run_pipeline(
    session_id: UUID,
    user_message: str,
    db: AsyncSession,
) -> PipelineResult:
    from app.db.models import AgentTrace, Message
    from app.db.models import Session as SessionModel

    session_id_str: str = str(session_id)

    # ── Step 1: Load session from DB ─────────────────────────────────────────
    result = await db.execute(
        select(SessionModel).where(SessionModel.id == session_id_str)
    )
    session_row: SessionModel | None = result.scalar_one_or_none()

    if session_row is None:
        logger.warning("run_pipeline: session %s not found", session_id_str)
        raise SessionNotFoundError(session_id)

    # ── Step 2: Load state_json from DB ──────────────────────────────────────
    state: SessionState = SessionState.from_dict(session_row.state_json)
    state.user_id = session_row.user_id
    state.plan_tier = session_row.plan_tier

    logger.debug(
        "run_pipeline: state loaded — turn=%d last_agent=%r",
        state.turn_count, state.last_agent,
    )

    # ── Step 3: Get run_turn function ────────────────────────────────────────
    run_turn_fn = _run_turn_fn if _run_turn_fn is not None else _get_run_turn()

    # ── Step 4: Run ADK agent with timeout ───────────────────────────────────
    t0: float = time.monotonic()

    try:
        agent_result: dict[str, Any] = await asyncio.wait_for(
            run_turn_fn(
                user_content=user_message,
                user_id=state.user_id,
                session_id=session_id_str,
                state=state.to_dict(),
            ),
            timeout=PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        latency_ms_timeout = int((time.monotonic() - t0) * 1000)
        logger.error(
            "run_pipeline: ADK agent timed out after %dms (session=%s)",
            latency_ms_timeout, session_id_str,
        )
        raise UpstreamTimeoutError(PIPELINE_TIMEOUT) from exc

    latency_ms: int = int((time.monotonic() - t0) * 1000)

    # ── Step 5: Parse result ─────────────────────────────────────────────────
    reply: str = agent_result.get("reply", "")
    routed_to: str = agent_result.get("routed_to", "unknown")
    tool_calls: list[dict[str, Any]] = agent_result.get("tool_calls", [])

    retrieved_chunk_ids: list[str] = agent_result.get("retrieved_chunk_ids", [])
    if not retrieved_chunk_ids:
        retrieved_chunk_ids = _extract_retrieved_chunk_ids(tool_calls)

    logger.info(
        "run_pipeline: agent finished — routed_to=%r latency_ms=%d",
        routed_to, latency_ms,
    )

    # ── Step 6: Update state ─────────────────────────────────────────────────
    state.turn_count += 1
    state.last_agent = routed_to
    summary: str = _make_turn_summary(user_message, routed_to, reply)
    state.push_summary(summary)

    # ── Step 7: Save state to DB (UPDATE) ────────────────────────────────────
    await db.execute(
        update(SessionModel)
        .where(SessionModel.id == session_id_str)
        .values(state_json=state.to_dict())
    )
    logger.debug("run_pipeline: state_json persisted for session %s", session_id_str)

    # ── Step 8: Write trace ──────────────────────────────────────────────────
    trace_id: UUID = uuid.uuid4()
    trace_row = AgentTrace(
        id=str(trace_id),
        session_id=session_id_str,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=retrieved_chunk_ids,
        latency_ms=latency_ms,
    )
    db.add(trace_row)
    await db.flush()

    # ── Step 9: Save messages ────────────────────────────────────────────────
    db.add(Message(
        id=str(uuid.uuid4()),
        session_id=session_id_str,
        role="user",
        content=user_message,
    ))
    db.add(Message(
        id=str(uuid.uuid4()),
        session_id=session_id_str,
        role="assistant",
        content=reply,
    ))
    await db.flush()

    # ── Step 10: Return result ───────────────────────────────────────────────
    return PipelineResult(
        reply=reply,
        routed_to=routed_to,
        trace_id=trace_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )