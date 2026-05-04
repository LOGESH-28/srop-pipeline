"""
agents/knowledge.py — KnowledgeAgent (google-adk LlmAgent).

Equipped with one tool:
  • search_docs(query, k) → list[dict]  – delegates to app.rag.store.search_docs

The agent cites chunk IDs in every answer per its system prompt.
The tool wraps the async store function in asyncio.run_coroutine_threadsafe
via an inner async def — ADK supports async tool functions natively.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = (
    "You are a documentation specialist. Answer questions using search_docs. "
    "You MUST cite chunk IDs in every answer using the format [chunk_abc123]. "
    "Never answer from memory if search_docs can be called."
)

# ── Tool definition ───────────────────────────────────────────────────────────

async def search_docs(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Search the SROP documentation vector store and return the most relevant chunks.

    Call this tool for every question about features, configuration, how-to
    guides, API references, or any other product documentation.

    Args:
        query: The search question or keywords to look up in the documentation.
        k:     Maximum number of document chunks to return (default 5, max 20).

    Returns:
        A list of result dicts, each containing:
            chunk_id  – unique identifier to cite in your answer, e.g. [chunk_abc123]
            score     – cosine similarity score in [0.0, 1.0]; higher = more relevant
            text      – the raw document chunk text
            metadata  – dict with keys: source_file, product_area, title, chunk_index
    """
    from app.rag.store import search_docs as _store_search_docs

    k = min(max(1, k), 20)   # clamp to [1, 20]
    logger.debug("search_docs tool called: query=%r k=%d", query[:80], k)

    try:
        results: list[dict[str, Any]] = await _store_search_docs(query, k=k)
    except Exception as exc:
        # Surface the error as a structured dict so the LLM can handle it gracefully
        logger.warning("search_docs tool error: %s", exc)
        return [{"error": str(exc), "chunk_id": "", "score": 0.0, "text": "", "metadata": {}}]

    logger.debug("search_docs: returned %d result(s)", len(results))
    return results


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_knowledge_agent() -> LlmAgent:
    """Construct and return the KnowledgeAgent LlmAgent instance."""
    return LlmAgent(
        name="knowledge_agent",
        model="gemini-2.0-flash",
        instruction=_SYSTEM_PROMPT,
        description=(
            "Answers documentation questions about Helix features, configuration, "
            "how-to guides, API references, and product explanations by searching "
            "the vector store. Use this agent for any 'how does X work' or "
            "'how do I configure Y' type questions."
        ),
        tools=[search_docs],
    )


# Module-level singleton — shared by the orchestrator's AgentTool wrapper.
knowledge_agent: LlmAgent = build_knowledge_agent()
