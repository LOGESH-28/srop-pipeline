from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

_KNOWLEDGE_REPLY = (
    "To rotate a deploy key, navigate to Settings → Deploy Keys. "
    "Revoke the existing key and generate a new one. "
    "See [chunk_a1b2c3] for the full procedure and [chunk_d4e5f6] for "
    "permission requirements."
)

_ACCOUNT_REPLY = (
    "Your last 3 failed builds are: "
    "bld_a1b2c3 (feature/rag-pipeline, 2025-05-03T22:14:00Z), "
    "bld_g7h8i9 (fix/timeout-handling, 2025-05-03T18:33:00Z), "
    "bld_p7q8r9 (experiment/vector-index, 2025-05-01T14:20:00Z)."
)


class TestConversationFlow:

    async def test_turn1_knowledge_routing(self, client: AsyncClient) -> None:
        sess_resp = await client.post(
            "/v1/sessions",
            json={"user_id": "test_user_001", "plan_tier": "pro"},
        )
        assert sess_resp.status_code == 201, sess_resp.text
        session_id: str = sess_resp.json()["session_id"]

        mock = AsyncMock(return_value={
            "reply": "Rotate deploy key in Settings > Deploy Keys. See [chunk_abc123].",
            "routed_to": "knowledge_agent",
            "tool_calls": [{"tool_name": "search_docs", "tool": "search_docs",
                            "args": {"query": "rotate deploy key"}, "result": []}],
            "retrieved_chunk_ids": ["abc123def456"],
            "updated_state": {},
        })

        with patch("app.pipeline._run_turn_fn", mock):
            chat_resp = await client.post(
                f"/v1/chat/{session_id}",
                json={"message": "How do I rotate a deploy key?"},
            )

        assert chat_resp.status_code == 200, chat_resp.text
        data = chat_resp.json()
        assert data["routed_to"] == "knowledge_agent"
        assert "[chunk_" in data["reply"], (
            f"Expected citation '[chunk_...' in reply, got: {data['reply']!r}"
        )

    async def test_turn2_account_routing_and_state_persistence(
        self, client: AsyncClient
    ) -> None:
        sess_resp = await client.post(
            "/v1/sessions",
            json={"user_id": "test_user_002", "plan_tier": "enterprise"},
        )
        assert sess_resp.status_code == 201
        session_id: str = sess_resp.json()["session_id"]

        mock_t1 = AsyncMock(return_value={
            "reply": _KNOWLEDGE_REPLY,
            "routed_to": "knowledge_agent",
            "tool_calls": [],
            "retrieved_chunk_ids": [],
            "updated_state": {},
        })
        with patch("app.pipeline._run_turn_fn", mock_t1):
            r1 = await client.post(
                f"/v1/chat/{session_id}",
                json={"message": "How do I rotate a deploy key?"},
            )
        assert r1.status_code == 200, r1.text

        captured_state: dict[str, Any] = {}

        async def _mock_turn_2(user_content, user_id, session_id, state) -> dict[str, Any]:
            captured_state.update(state or {})
            return {
                "reply": _ACCOUNT_REPLY,
                "routed_to": "account_agent",
                "tool_calls": [],
                "retrieved_chunk_ids": [],
                "updated_state": {},
            }

        with patch("app.pipeline._run_turn_fn", side_effect=_mock_turn_2):
            r2 = await client.post(
                f"/v1/chat/{session_id}",
                json={"message": "Show me my last 3 failed builds"},
            )

        assert r2.status_code == 200, r2.text
        data2 = r2.json()
        assert data2["routed_to"] == "account_agent"

        assert captured_state.get("turn_count") == 1, (
            f"Expected turn_count=1 (from DB after turn 1), "
            f"got: {captured_state.get('turn_count')!r}. "
            "This means state was NOT persisted/loaded correctly between turns."
        )


class TestErrorHandling:

    async def test_chat_unknown_session_returns_404(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/chat/00000000-0000-0000-0000-000000000000",
            json={"message": "hello"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "SESSION_NOT_FOUND"

    async def test_get_trace_unknown_returns_404(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/traces/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "TRACE_NOT_FOUND"

    async def test_healthz(self, client: AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"