"""
agents/orchestrator.py — Root SROP orchestrator (google-adk LlmAgent).

Architecture
------------
                      ┌─────────────────┐
  user message ──────▶│   srop_root     │
                      │  (LlmAgent)     │
                      └────────┬────────┘
                               │ decides via LLM (no keyword routing)
                    ┌──────────┴──────────┐
                    ▼                     ▼
           AgentTool(knowledge_agent)  AgentTool(account_agent)
                    │                     │
              KnowledgeAgent         AccountAgent
              (search_docs)    (get_recent_builds,
                                get_account_status)

Session state injection
-----------------------
The root agent's instruction is rendered at call time with live session values:
  {plan_tier}   – user's subscription tier
  {turn_count}  – conversation turn index (incremented each call)
  {last_agent}  – name of sub-agent used on the previous turn

These are stored in / read from the ADK session state dict
(``session.state``), which is persisted by the SROP session service.

Running
-------
Call ``run_turn(user_content, user_id, session_id, state)`` for each
chat turn.  The function manages a per-process Runner + SessionService
singleton to avoid re-creating them on every request.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

# -- Monkey patch google.genai.types.AvatarConfig to fix google.adk import --
import sys
try:
    import google.genai.types
    if not hasattr(google.genai.types, "AvatarConfig"):
        class AvatarConfig:
            pass
        google.genai.types.AvatarConfig = AvatarConfig
except ImportError:
    pass

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
from google.adk.runners import Runner  # type: ignore[import-untyped]
from google.adk.sessions import InMemorySessionService  # type: ignore[import-untyped]
from google.adk.tools import AgentTool  # type: ignore[import-untyped]
from google.genai import types as genai_types  # type: ignore[import-untyped]

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.config import settings

logger = logging.getLogger(__name__)

# ── System prompt template ────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE: str = """\
You are an AI support concierge for Helix.

Route to knowledge_agent for documentation questions \
(how-to, feature explanations, config).
Route to account_agent for account-specific queries \
(builds, deploys, account status).

Current session context:
  - The user's plan tier is {plan_tier}.
  - This is turn {turn_count}.
  - Last agent used: {last_agent}.

Always route to the most appropriate sub-agent. \
Do NOT answer directly from memory — always delegate.\
"""

_APP_NAME: str = "srop"


# ── Root agent factory ────────────────────────────────────────────────────────

def _build_root_agent(instruction: str) -> LlmAgent:
    """Build a root LlmAgent with both sub-agents wrapped as AgentTools."""
    return LlmAgent(
        name="srop_root",
        model="gemini-2.0-flash",
        instruction=instruction,
        description="Root SROP concierge agent that routes to knowledge or account sub-agents.",
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )


# ── Singleton runtime ─────────────────────────────────────────────────────────
# Re-creating Runner + SessionService on every request is expensive.
# We hold one per-process instance and recycle it across calls.

_session_service: InMemorySessionService | None = None
_runner: Runner | None = None
_current_instruction: str = ""


def _get_runtime(instruction: str) -> tuple[Runner, InMemorySessionService]:
    """Return (or lazily create) the process-lifetime Runner and SessionService.

    The Runner is recreated whenever the instruction changes so the root
    agent's system prompt always reflects the current session context
    (plan_tier, turn_count, last_agent).  The SessionService is reused
    across all turns so ADK conversation history is preserved.
    """
    global _session_service, _runner, _current_instruction

    if _session_service is None:
        _session_service = InMemorySessionService()
        logger.info("Initialised ADK InMemorySessionService.")

    # Rebuild the runner whenever the instruction string changes.
    if _runner is None or instruction != _current_instruction:
        root_agent = _build_root_agent(instruction)
        _runner = Runner(
            agent=root_agent,
            app_name=_APP_NAME,
            session_service=_session_service,
        )
        _current_instruction = instruction
        logger.debug("ADK Runner (re)built with updated instruction.")

    return _runner, _session_service


# ── Public async interface ────────────────────────────────────────────────────

async def run_turn(
    *,
    user_content: str,
    user_id: str,
    session_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one chat turn through the SROP root agent.

    Parameters
    ----------
    user_content:
        The raw user message for this turn.
    user_id:
        Stable user identifier (used by the ADK session service).
    session_id:
        SROP session UUID; maps to an ADK session.
    state:
        Current persisted SROP session state dict.
        Expected keys (all optional, with defaults):
            plan_tier  (str)  – "free" | "pro" | "enterprise"
            turn_count (int)  – monotonically increasing turn index
            last_agent (str)  – name of the previously used sub-agent

    Returns
    -------
    dict with keys:
        reply       (str)           – final assistant text
        routed_to   (str)           – sub-agent name extracted from events
        tool_calls  (list[dict])    – tool invocations recorded during the turn
        updated_state (dict)        – updated state to persist back to the DB
    """
    # ── Resolve session state values ──────────────────────────────────────────
    plan_tier:   str = state.get("plan_tier",  "free")
    turn_count:  int = state.get("turn_count", 0) + 1
    last_agent:  str = state.get("last_agent", "none")

    # ── Build the instruction with live state injected ────────────────────────
    instruction: str = _SYSTEM_PROMPT_TEMPLATE.format(
        plan_tier=plan_tier,
        turn_count=turn_count,
        last_agent=last_agent,
    )

    runner, session_service = _get_runtime(instruction)

    # ── Ensure the ADK session exists ─────────────────────────────────────────
    # ADK's InMemorySessionService.get_session raises / returns None if absent.
    existing = await session_service.get_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if existing is None:
        await session_service.create_session(
            app_name=_APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state=state,
        )
        logger.debug("Created ADK session %s for user %s.", session_id, user_id)

    # ── Format the user message ───────────────────────────────────────────────
    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_content)],
    )

    # ── Stream events from the runner ─────────────────────────────────────────
    reply:               str              = ""
    routed_to:           str              = "unknown"
    tool_calls:          list[dict]       = []
    retrieved_chunk_ids: list[str]        = []

    # Map tool name → tool_calls entry so function_response can add "result".
    _pending: dict[str, dict] = {}

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        # ── Capture final text response ───────────────────────────────────────
        if event.is_final_response():
            if event.content and event.content.parts:
                reply = "".join(
                    part.text
                    for part in event.content.parts
                    if hasattr(part, "text") and part.text
                )

        # ── Capture tool calls and responses ──────────────────────────────────
        if hasattr(event, "content") and event.content:
            for part in event.content.parts or []:

                # function_call → record the invocation
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_name: str  = fc.name or ""
                    tool_args: dict = dict(fc.args) if fc.args else {}
                    entry: dict = {
                        "tool_name": tool_name,   # CHECK 9.3 field
                        "tool":      tool_name,   # backward-compat alias
                        "args":      tool_args,
                        "result":    None,         # filled in by function_response
                    }
                    tool_calls.append(entry)
                    _pending[tool_name] = entry

                    # AgentTool names == sub-agent names
                    if tool_name in ("knowledge_agent", "account_agent"):
                        routed_to = tool_name

                # function_response → attach result + extract chunk_ids
                if hasattr(part, "function_response") and part.function_response:
                    fr        = part.function_response
                    fr_name:  str = getattr(fr, "name", "") or ""
                    fr_value: Any = getattr(fr, "response", None)

                    if fr_name in _pending:
                        _pending[fr_name]["result"] = fr_value

                    # Extract chunk_ids from search_docs results
                    if fr_name == "search_docs" and isinstance(fr_value, list):
                        for item in fr_value:
                            cid = item.get("chunk_id", "") if isinstance(item, dict) else ""
                            if cid:
                                retrieved_chunk_ids.append(cid)

    logger.info(
        "run_turn: session=%s turn=%d routed_to=%r reply_len=%d chunks=%d",
        session_id, turn_count, routed_to, len(reply), len(retrieved_chunk_ids),
    )

    # ── Build updated state ───────────────────────────────────────────────────
    updated_state: dict[str, Any] = {
        **state,
        "plan_tier":  plan_tier,
        "turn_count": turn_count,
        "last_agent": routed_to,
    }

    return {
        "reply":               reply,
        "routed_to":           routed_to,
        "tool_calls":          tool_calls,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "updated_state":       updated_state,
    }


# ── Backwards-compatible class wrapper ────────────────────────────────────────
# The pipeline.py calls RootOrchestrator(...).run(...) — we keep that contract.

class RootOrchestrator:
    """
    Thin wrapper around ``run_turn`` preserving the interface expected by
    ``app/pipeline.py``.

    The heavy lifting (Runner, SessionService, ADK agents) is done by the
    module-level ``run_turn`` coroutine above.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        plan_tier: str,
        state: dict[str, Any],
    ) -> None:
        self.session_id = session_id
        self.user_id    = user_id
        self.plan_tier  = plan_tier
        self.state      = {**state, "plan_tier": plan_tier}

    async def run(
        self,
        user_content: str,
        context: str,  # RAG context — prepended to the user message
    ) -> dict[str, Any]:
        """
        Execute one turn and return the pipeline result dict.

        The RAG context is prepended to the user message so the root agent
        and whichever sub-agent it delegates to can see the retrieved chunks.
        """
        # Prepend RAG context so sub-agents benefit from retrieved docs
        augmented_content: str = (
            f"[Retrieved context]\n{context}\n\n[User message]\n{user_content}"
            if context.strip()
            else user_content
        )

        return await run_turn(
            user_content=augmented_content,
            user_id=self.user_id,
            session_id=self.session_id,
            state=self.state,
        )
