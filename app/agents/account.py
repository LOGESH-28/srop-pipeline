"""
agents/account.py — AccountAgent (google-adk LlmAgent).

Equipped with two tools:
  • get_recent_builds   → list[dict]  – last N build records for a user
  • get_account_status  → dict        – plan tier, active deploys, quota

The agent is created as a module-level singleton so it can be wrapped in an
AgentTool by the root orchestrator without re-instantiation overhead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = (
    "You are an account specialist. Answer questions about builds, deploys, "
    "and account status using the provided tools. "
    "Always use tool results, never invent data."
)

# ── Mock data fixtures ────────────────────────────────────────────────────────

_MOCK_BUILDS: list[dict] = [
    {
        "build_id":  "bld_a1b2c3",
        "status":    "failed",
        "timestamp": "2025-05-03T22:14:00Z",
        "branch":    "feature/rag-pipeline",
    },
    {
        "build_id":  "bld_d4e5f6",
        "status":    "success",
        "timestamp": "2025-05-03T20:05:00Z",
        "branch":    "main",
    },
    {
        "build_id":  "bld_g7h8i9",
        "status":    "failed",
        "timestamp": "2025-05-03T18:33:00Z",
        "branch":    "fix/timeout-handling",
    },
    {
        "build_id":  "bld_j1k2l3",
        "status":    "success",
        "timestamp": "2025-05-03T16:00:00Z",
        "branch":    "main",
    },
    {
        "build_id":  "bld_m4n5o6",
        "status":    "success",
        "timestamp": "2025-05-02T09:45:00Z",
        "branch":    "release/v0.2",
    },
    {
        "build_id":  "bld_p7q8r9",
        "status":    "failed",
        "timestamp": "2025-05-01T14:20:00Z",
        "branch":    "experiment/vector-index",
    },
]

_MOCK_ACCOUNT: dict = {
    "plan_tier":       "pro",
    "active_deploys":  3,
    "quota_used_pct":  67,
}


# ── Tool definitions ──────────────────────────────────────────────────────────

def get_recent_builds(user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the most recent CI/CD build records for the given user.

    Args:
        user_id: The unique identifier of the user whose builds to fetch.
        limit:   Maximum number of build records to return (default 5).

    Returns:
        A list of dicts, each containing:
            build_id  – unique build identifier (str)
            status    – "success" | "failed" | "running" | "cancelled"
            timestamp – ISO-8601 UTC completion time (str)
            branch    – git branch that triggered the build (str)
    """
    logger.debug("get_recent_builds(user_id=%r, limit=%d)", user_id, limit)
    return _MOCK_BUILDS[:max(1, limit)]


def get_account_status(user_id: str) -> dict:
    """
    Return current account and quota information for the given user.

    Args:
        user_id: The unique identifier of the user to look up.

    Returns:
        A dict containing:
            user_id         – echoed user identifier (str)
            plan_tier       – current subscription tier (str)
            active_deploys  – number of running deployments (int)
            quota_used_pct  – percentage of monthly API quota consumed (int)
    """
    logger.debug("get_account_status(user_id=%r)", user_id)
    return {
        "user_id": user_id,
        **_MOCK_ACCOUNT,
    }


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_account_agent() -> LlmAgent:
    """Construct and return the AccountAgent LlmAgent instance."""
    return LlmAgent(
        name="account_agent",
        model="gemini-2.0-flash",
        instruction=_SYSTEM_PROMPT,
        description=(
            "Handles all account-specific queries: build history, deploy status, "
            "and account quota. Use this agent when the user asks about their "
            "builds, deployments, plan tier, or quota usage."
        ),
        tools=[get_recent_builds, get_account_status],
    )


# Module-level singleton — shared by the orchestrator's AgentTool wrapper.
account_agent: LlmAgent = build_account_agent()
