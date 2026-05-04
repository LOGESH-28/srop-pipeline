"""
rag/store.py — Async vector store wrapper for SROP.

Public API
----------
search_docs(query, k=5) -> list[dict]
    Embed query with sentence-transformers and query ChromaDB.
    Each result dict: {chunk_id, score, text, metadata}.
    Scores are cosine similarities normalised to [0, 1].

Raises
------
VectorStoreEmptyError
    Raised when the ChromaDB collection exists but contains no documents.
    This tells the caller that ingest has not been run yet.
"""

from __future__ import annotations

import asyncio
import logging
from functools import cached_property
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME: str = "srop_chunks"
MODEL_NAME: str = "all-MiniLM-L6-v2"


# ── Custom exceptions ─────────────────────────────────────────────────────────

class VectorStoreEmptyError(RuntimeError):
    """
    Raised when the vector store collection exists but contains zero documents.

    The caller should surface this as a 503 / 412 and ask the operator to
    run ``python -m app.rag.ingest --path docs/``.
    """

    def __init__(self) -> None:
        super().__init__(
            f"Vector store collection '{COLLECTION_NAME}' is empty. "
            "Run 'python -m app.rag.ingest --path docs/' to populate it."
        )


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin async wrapper around a persistent ChromaDB collection.

    All blocking operations (ChromaDB I/O, model inference) are offloaded
    with ``asyncio.to_thread`` so they never block the FastAPI event loop.

    The model and ChromaDB client are lazily initialised on first use and
    then cached via ``@cached_property`` for the lifetime of the object.
    """

    # ── Lazy initialisation ───────────────────────────────────────────────────

    @cached_property
    def _embedder(self) -> Any:
        """Load the sentence-transformers model exactly once."""
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        logger.info("Loading embedding model '%s' …", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME)

    @cached_property
    def _client(self) -> Any:
        """Open the persistent ChromaDB client exactly once."""
        import chromadb  # type: ignore[import-untyped]

        path = Path(settings.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Opening ChromaDB at '%s'.", path)
        return chromadb.PersistentClient(path=str(path))

    @cached_property
    def _collection(self) -> Any:
        """Get or create the collection with cosine distance metric."""
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Internal sync helpers (run inside asyncio.to_thread) ──────────────────

    def _sync_embed(self, text: str) -> list[float]:
        """Embed a single query string synchronously."""
        vector = self._embedder.encode(
            text,
            normalize_embeddings=True,   # keeps scores in [0, 1]
            convert_to_numpy=True,
        )
        return vector.tolist()

    def _sync_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def _sync_query(
        self, embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """
        Run a nearest-neighbour query and return normalised results.

        ChromaDB with ``hnsw:space = "cosine"`` returns:
            distance = 1 − cosine_similarity   ∈ [0, 2]

        For L2-normalised embeddings (which sentence-transformers produces
        with ``normalize_embeddings=True``), cosine_similarity is in [0, 1]
        for semantically similar texts, so distance ∈ [0, 1].

        Conversion:  score = 1 − distance   →  score ∈ [0, 1]
        We clamp with max(0.0, ...) as a safety net for floating-point noise.
        """
        count = self._sync_count()
        if count == 0:
            raise VectorStoreEmptyError()

        # Request at most as many results as the collection has
        n_results = min(k, count)

        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        ids:       list[str]              = raw.get("ids", [[]])[0]
        documents: list[str]              = raw.get("documents", [[]])[0]
        distances: list[float]            = raw.get("distances", [[]])[0]
        metadatas: list[dict[str, Any]]   = raw.get("metadatas", [[]])[0]

        results: list[dict[str, Any]] = []
        for chunk_id, text, dist, meta in zip(ids, documents, distances, metadatas):
            score = max(0.0, 1.0 - dist)   # cosine similarity in [0, 1]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score":    round(score, 6),
                    "text":     text,
                    "metadata": meta,
                }
            )

        return results

    # ── Public async API ──────────────────────────────────────────────────────

    async def search_docs(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the ingested document store.

        Parameters
        ----------
        query:
            Natural-language question or search string.
        k:
            Maximum number of results to return.  If the store contains
            fewer than *k* documents, all available documents are returned.

        Returns
        -------
        list[dict] — each element has keys:
            ``chunk_id``  (str)   — deterministic 16-char hex ID
            ``score``     (float) — cosine similarity ∈ [0, 1]
            ``text``      (str)   — chunk content
            ``metadata``  (dict)  — source_file, product_area, title, chunk_index

        Raises
        ------
        VectorStoreEmptyError
            If the collection is empty (ingest not yet run).
        ValueError
            If *k* is not a positive integer.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")

        # Offload blocking ops to thread pool — never block the event loop
        embedding: list[float] = await asyncio.to_thread(self._sync_embed, query)
        results: list[dict[str, Any]] = await asyncio.to_thread(
            self._sync_query, embedding, k
        )

        logger.debug(
            "search_docs: query=%r k=%d → %d result(s)", query[:60], k, len(results)
        )
        return results

    async def count(self) -> int:
        """Return the total number of chunks currently in the store."""
        return await asyncio.to_thread(self._sync_count)

    async def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """
        Upsert pre-embedded chunks.  Each dict must have:
            ``id``         (str)
            ``text``       (str)
            ``embedding``  (list[float])
            ``metadata``   (dict)  — optional

        Use ``ingest.py`` for the full CLI pipeline; this method is provided
        for programmatic / test use.
        """
        if not chunks:
            return

        def _sync_upsert() -> None:
            self._collection.upsert(
                ids=[c["id"] for c in chunks],
                embeddings=[c["embedding"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c.get("metadata", {}) for c in chunks],
            )

        await asyncio.to_thread(_sync_upsert)
        logger.debug("Upserted %d chunk(s).", len(chunks))


# ── Module-level convenience function ─────────────────────────────────────────
# Allows `from app.rag.store import search_docs` without managing a VectorStore
# instance manually.

_default_store: VectorStore | None = None


def _get_default_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


async def search_docs(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Module-level shortcut for ``VectorStore().search_docs(query, k)``.

    Uses a process-lifetime singleton so the model is only loaded once.
    """
    return await _get_default_store().search_docs(query, k)
