"""
api/routes.py — SROP REST endpoints.

Routes
------
POST /v1/sessions              → create a new session
POST /v1/chat/{session_id}     → send a message, run the SROP pipeline
GET  /v1/traces/{trace_id}     → retrieve an agent trace

Error contract
--------------
All 4xx / 5xx responses carry a JSON body with at least a "code" field:
  {"code": "SESSION_NOT_FOUND"}
  {"code": "UPSTREAM_TIMEOUT"}
  {"code": "TRACE_NOT_FOUND"}
  {"code": "STORE_NOT_READY"}
"""

from __future__ import annotations

import logging
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    TraceResponse,
)
from app.db.models import AgentTrace
from app.db.models import Session as SessionModel
from app.db.session import get_db
from app.pipeline import (
    PipelineResult,
    SessionNotFoundError,
    UpstreamTimeoutError,
    run_pipeline,
)
from app.rag.store import VectorStoreEmptyError

logger = logging.getLogger(__name__)

router = APIRouter()


# ── POST /v1/sessions ─────────────────────────────────────────────────────────

@router.post(
    "/v1/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    tags=["sessions"],
)
async def create_session(
    payload: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Insert a new row into the *sessions* table and return the generated UUID.

    The initial ``state_json`` is populated with the user_id and plan_tier so
    the pipeline can read them on the first turn without a separate query.
    """
    new_id: UUID = uuid.uuid4()

    # Seed state_json with the values pipeline.SessionState expects.
    initial_state: dict = {
        "user_id":        payload.user_id,
        "plan_tier":      payload.plan_tier,
        "turn_count":     0,
        "last_agent":     None,
        "recent_results": [],
    }

    session_row = SessionModel(
        id=str(new_id),
        user_id=payload.user_id,
        plan_tier=payload.plan_tier,
        state_json=initial_state,
    )
    db.add(session_row)
    await db.flush()

    logger.info(
        "create_session: id=%s user=%r tier=%s",
        new_id, payload.user_id, payload.plan_tier,
    )
    return CreateSessionResponse(session_id=new_id)


# ── POST /v1/chat/{session_id} ────────────────────────────────────────────────

@router.post(
    "/v1/chat/{session_id}",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and receive an AI response",
    tags=["chat"],
)
async def chat(
    session_id: UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Run one SROP pipeline turn.

    Delegates entirely to ``pipeline.run_pipeline`` which owns:
      - session loading from DB
      - state deserialisation / update / re-persistence
      - ADK agent execution (with 30-second timeout)
      - message + trace persistence

    Error mapping
    -------------
    SessionNotFoundError  → 404 SESSION_NOT_FOUND
    UpstreamTimeoutError  → 504 UPSTREAM_TIMEOUT
    VectorStoreEmptyError → 503 STORE_NOT_READY
    """
    try:
        result: PipelineResult = await run_pipeline(
            session_id=session_id,
            user_message=payload.message,
            db=db,
        )

    except SessionNotFoundError as exc:
        logger.warning("chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND"},
        ) from exc

    except UpstreamTimeoutError as exc:
        logger.error("chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"code": "UPSTREAM_TIMEOUT"},
        ) from exc

    except VectorStoreEmptyError as exc:
        logger.warning("chat: vector store empty — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "STORE_NOT_READY",
                "hint": "Run: python -m app.rag.ingest --path docs/",
            },
        ) from exc

    logger.info(
        "chat: session=%s routed_to=%s trace=%s",
        session_id, result.routed_to, result.trace_id,
    )

    return ChatResponse(
        reply=result.reply,
        routed_to=result.routed_to,
        trace_id=result.trace_id,
    )


# ── GET /v1/traces/{trace_id} ─────────────────────────────────────────────────

@router.get(
    "/v1/traces/{trace_id}",
    response_model=TraceResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve an agent trace by ID",
    tags=["traces"],
)
async def get_trace(
    trace_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    """
    Fetch an :class:`AgentTrace` row by primary key.

    Returns 404 ``TRACE_NOT_FOUND`` if no matching row exists.
    """
    result = await db.execute(
        select(AgentTrace).where(AgentTrace.id == str(trace_id))
    )
    trace: AgentTrace | None = result.scalar_one_or_none()

    if trace is None:
        logger.warning("get_trace: trace %s not found", trace_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRACE_NOT_FOUND"},
        )

    return TraceResponse.model_validate(trace)
