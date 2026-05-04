"""
rag/ingest.py — CLI document ingestion pipeline for SROP.

Usage
-----
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --batch-size 64 --dry-run

Chunking strategy (justified)
------------------------------
1. Split on paragraph boundaries (\\n\\n) first.
   Paragraphs are natural semantic units in Markdown; keeping them intact
   preserves context that a sentence-level split would fragment.

2. Greedily pack paragraphs into windows of <= max_tokens (400).
   When a paragraph would overflow the window, emit the current chunk
   and seed the next one with the last `overlap` (80) tokens — ensuring
   cross-chunk context is not lost at boundaries.

3. If a *single* paragraph exceeds max_tokens (e.g. a big code block),
   fall back to pure token-level sliding window within that paragraph.

Token counting uses whitespace splitting (proxy for BPE tokens).
For all-MiniLM-L6-v2 (max 256 subword tokens) a 400-word window is
safely within model capacity after subword expansion of typical prose.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "srop_chunks"
MODEL_NAME: str = "all-MiniLM-L6-v2"
MAX_TOKENS: int = 400
OVERLAP: int = 80


# ── Chunk ID ───────────────────────────────────────────────────────────────────

def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """
    Deterministic 16-char hex ID: sha256(file_path + str(chunk_index))[:16].

    Re-ingesting the same file at the same index produces the same ID,
    so ChromaDB's upsert de-duplicates automatically.
    """
    raw: str = file_path + str(chunk_index)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Chunking ───────────────────────────────────────────────────────────────────

def _token_len(text: str) -> int:
    """Whitespace-split word count as a BPE token proxy."""
    return len(text.split())


def _slide_tokens(tokens: list[str], max_tokens: int, overlap: int) -> list[str]:
    """Pure token-level sliding window — fallback for overlong paragraphs."""
    chunks: list[str] = []
    start: int = 0
    while start < len(tokens):
        end: int = min(start + max_tokens, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def chunk_text(
    text: str,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Sliding-window chunker with paragraph-boundary awareness.

    Parameters
    ----------
    text:       Raw document text (Markdown body, frontmatter already stripped).
    max_tokens: Target maximum tokens per chunk (word count proxy).
    overlap:    Number of tokens to carry over into the next chunk.

    Returns
    -------
    Non-empty list of chunk strings.
    """
    # Split into non-empty paragraphs on 2+ consecutive newlines
    paragraphs: list[str] = [
        p.strip() for p in re.split(r"\n{2,}", text) if p.strip()
    ]

    chunks: list[str] = []
    buf: list[str] = []  # token accumulation buffer

    for para in paragraphs:
        para_tokens: list[str] = para.split()
        if not para_tokens:
            continue

        # ── Overlong paragraph → token-level sliding window ─────────────────
        if len(para_tokens) > max_tokens:
            # Flush existing buffer as its own chunk first
            if buf:
                chunks.append(" ".join(buf))
                buf = []

            chunks.extend(_slide_tokens(para_tokens, max_tokens, overlap))
            # Seed next buffer with the overlap tail of this paragraph
            buf = para_tokens[-overlap:]
            continue

        # ── Normal paragraph: check if it fits in the current buffer ─────────
        if buf and len(buf) + len(para_tokens) > max_tokens:
            chunks.append(" ".join(buf))
            # Preserve overlap for cross-chunk context
            buf = buf[-overlap:] if len(buf) > overlap else buf[:]

        buf.extend(para_tokens)

    # Flush the final buffer
    if buf:
        chunks.append(" ".join(buf))

    # Guarantee at least one chunk even for empty docs
    return chunks if chunks else [""]


# ── Embedding ──────────────────────────────────────────────────────────────────

def load_embedder():  # type: ignore[return]
    """Load the sentence-transformers model (cached after first call)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    logger.info("Loading embedding model '%s' …", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_batch(texts: list[str], embedder: Any) -> list[list[float]]:
    """Return L2-normalised embeddings for *texts*."""
    vectors = embedder.encode(
        texts,
        normalize_embeddings=True,   # cosine similarity = dot product
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# ── ChromaDB client ────────────────────────────────────────────────────────────

def get_collection(vector_store_path: str):  # type: ignore[return]
    """Create or open the persistent ChromaDB collection."""
    import chromadb  # type: ignore[import-untyped]

    path = Path(vector_store_path)
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # distances = 1 − cosine_similarity
    )
    return collection


# ── Ingest pipeline ────────────────────────────────────────────────────────────

def ingest(
    docs_path: str,
    vector_store_path: str,
    batch_size: int = 64,
    dry_run: bool = False,
) -> None:
    """
    Walk *docs_path* for ``.md`` files, chunk and embed them, then upsert
    into the ChromaDB collection.

    Parameters
    ----------
    docs_path:          Root directory to scan recursively.
    vector_store_path:  Directory passed to ChromaDB PersistentClient.
    batch_size:         Number of chunks to embed + upsert per batch.
    dry_run:            If True, log what would be done but skip all writes.
    """
    root = Path(docs_path)
    md_files: list[Path] = sorted(root.rglob("*.md"))

    if not md_files:
        logger.warning("No .md files found under '%s'.", docs_path)
        return

    logger.info("Found %d Markdown file(s) under '%s'.", len(md_files), docs_path)

    embedder = load_embedder()
    collection = None if dry_run else get_collection(vector_store_path)

    # Accumulate all chunks before batched embedding
    all_ids:        list[str]         = []
    all_texts:      list[str]         = []
    all_metadatas:  list[dict[str, Any]] = []

    for md_file in md_files:
        # ── Parse frontmatter ─────────────────────────────────────────────────
        try:
            post = frontmatter.load(str(md_file))
        except Exception as exc:
            logger.warning("Skipping '%s' — frontmatter parse error: %s", md_file, exc)
            continue

        body: str          = post.content
        product_area: str  = str(post.metadata.get("product_area", ""))
        title: str         = str(post.metadata.get("title", md_file.stem))

        # ── Chunk ─────────────────────────────────────────────────────────────
        file_path_str = str(md_file.resolve())
        text_chunks   = chunk_text(body)

        logger.info(
            "  %s → %d chunk(s) [product_area=%r, title=%r]",
            md_file.name, len(text_chunks), product_area, title,
        )

        for idx, chunk in enumerate(text_chunks):
            chunk_id = make_chunk_id(file_path_str, idx)
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "chunk_id":     chunk_id,
                    "source_file":  file_path_str,
                    "product_area": product_area,
                    "title":        title,
                    "chunk_index":  idx,
                }
            )

    total = len(all_ids)
    logger.info("Total chunks to ingest: %d", total)

    if dry_run:
        logger.info("[dry-run] Skipping embedding and upsert.")
        return

    # ── Batched embed + upsert ────────────────────────────────────────────────
    for batch_start in range(0, total, batch_size):
        batch_end   = min(batch_start + batch_size, total)
        b_ids       = all_ids[batch_start:batch_end]
        b_texts     = all_texts[batch_start:batch_end]
        b_metadatas = all_metadatas[batch_start:batch_end]

        logger.info(
            "Embedding + upserting chunks %d–%d …", batch_start + 1, batch_end
        )
        b_embeddings = embed_batch(b_texts, embedder)

        collection.upsert(
            ids=b_ids,
            embeddings=b_embeddings,
            documents=b_texts,
            metadatas=b_metadatas,
        )

    logger.info("✓ Ingest complete — %d chunk(s) upserted into '%s'.", total, COLLECTION_NAME)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.ingest",
        description="Ingest Markdown documents into the SROP vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python -m app.rag.ingest --path docs/
              python -m app.rag.ingest --path docs/ --batch-size 32 --dry-run
            """
        ),
    )
    parser.add_argument(
        "--path",
        default="docs/",
        help="Root directory to walk for .md files (default: docs/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Chunks per embed+upsert batch (default: 64).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Parse and chunk without embedding or writing to the store.",
    )
    return parser


def main() -> None:
    from app.config import settings

    args = _build_parser().parse_args()
    ingest(
        docs_path=args.path,
        vector_store_path=settings.vector_store_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
