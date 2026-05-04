"""
tests/test_rag.py — Unit tests for app/rag/store.py search_docs.

Skip policy
-----------
If the vector store has zero documents (ingest hasn't been run), these tests
are skipped with a clear message.  The skip is performed inside each test via
``pytest.skip()`` after an async count check — this avoids running ``asyncio.run``
at module-collection time, which conflicts with pytest-asyncio's event-loop
management.

To run these tests:
    1.  Create docs/ with at least one .md file.
    2.  python -m app.rag.ingest --path docs/
    3.  pytest tests/test_rag.py -v
"""

from __future__ import annotations

import pytest

from app.rag.store import VectorStore, VectorStoreEmptyError, search_docs


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _skip_if_empty() -> None:
    """Skip the calling test if the vector store has no documents."""
    store = VectorStore()
    count: int = await store.count()
    if count == 0:
        pytest.skip(
            "Vector store is empty. "
            "Run: python -m app.rag.ingest --path docs/  then re-run tests."
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSearchDocs:
    """Unit tests for search_docs against a live (ingested) vector store."""

    async def test_returns_correct_count(self) -> None:
        """search_docs("rotate deploy key", k=3) returns exactly 3 results."""
        await _skip_if_empty()

        results = await search_docs("rotate deploy key", k=3)

        assert len(results) == 3, (
            f"Expected 3 results, got {len(results)}. "
            "The store may have fewer than 3 chunks."
        )

    async def test_all_chunk_ids_non_empty(self) -> None:
        """Every result has a non-empty chunk_id string."""
        await _skip_if_empty()

        results = await search_docs("rotate deploy key", k=3)

        assert all(
            isinstance(r.get("chunk_id"), str) and r["chunk_id"] != ""
            for r in results
        ), f"One or more results have an empty chunk_id: {results}"

    async def test_scores_in_unit_interval(self) -> None:
        """Every result score is a float in [0.0, 1.0]."""
        await _skip_if_empty()

        results = await search_docs("rotate deploy key", k=3)

        for r in results:
            score = r.get("score")
            assert isinstance(score, float), f"score is not float: {score!r}"
            assert 0.0 <= score <= 1.0, (
                f"score {score} is outside [0.0, 1.0] for chunk_id={r.get('chunk_id')!r}"
            )

    async def test_result_shape(self) -> None:
        """Each result dict contains the required keys with correct types."""
        await _skip_if_empty()

        results = await search_docs("deploy key rotation", k=2)

        required_keys = {"chunk_id", "score", "text", "metadata"}
        for i, r in enumerate(results):
            missing = required_keys - r.keys()
            assert not missing, f"Result {i} missing keys: {missing}"
            assert isinstance(r["text"], str) and r["text"], (
                f"Result {i} has empty text"
            )
            assert isinstance(r["metadata"], dict), (
                f"Result {i} metadata is not a dict: {r['metadata']!r}"
            )

    async def test_empty_store_raises_clear_error(self, monkeypatch) -> None:
        """search_docs on an empty store raises VectorStoreEmptyError.

        Simulated by monkeypatching _sync_query (called inside asyncio.to_thread)
        to raise VectorStoreEmptyError — no real ChromaDB or embedder needed.
        """
        from app.rag.store import VectorStore, VectorStoreEmptyError

        empty_store = VectorStore()

        def _raise_empty(*args, **kwargs):
            raise VectorStoreEmptyError()

        # Patch both the embed and query sync methods so nothing real is called.
        monkeypatch.setattr(empty_store, "_sync_embed", lambda q: [0.0] * 384)
        monkeypatch.setattr(empty_store, "_sync_query", _raise_empty)

        with pytest.raises(VectorStoreEmptyError):
            await empty_store.search_docs("anything")
