"""
conftest.py — shared pytest fixtures for SROP.

Key decisions
-------------
* DATABASE_URL is set to an in-memory-equivalent test file BEFORE any app
  module is imported, so the global SQLAlchemy engine and Settings singleton
  both point to the test database.
* Tables are created fresh per test via `test_engine` (function scope).
* `get_db` is overridden via FastAPI dependency_overrides so the app uses
  the same test engine as the fixtures — the override commits on success and
  rolls back on exception, matching the production behaviour.
* httpx.AsyncClient + ASGITransport sends real HTTP to the ASGI app without
  starting a network server.  The ASGI lifespan runs and is idempotent
  (create_all is a no-op when tables already exist).
"""

from __future__ import annotations

import os

# ── Override settings BEFORE any app import ───────────────────────────────────
_TEST_DB_URL = "sqlite+aiosqlite:///./srop_test.db"
os.environ.setdefault("DATABASE_URL", _TEST_DB_URL)
os.environ.setdefault("GEMINI_API_KEY", "test-not-a-real-key")
os.environ.setdefault("VECTOR_STORE_PATH", "./test_vector_store")
os.environ.setdefault("DEBUG", "false")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app


# ── Test engine (function-scoped: fresh schema per test) ──────────────────────

@pytest_asyncio.fixture
async def test_engine():
    """Create a fresh async engine + schema for one test, then tear it down."""
    engine = create_async_engine(
        _TEST_DB_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── AsyncClient fixture ───────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(test_engine):
    """
    httpx.AsyncClient wired to the FastAPI app with the test DB.

    ``get_db`` is overridden so every request inside the test uses the same
    engine that the test fixtures control.
    """
    TestSessionLocal = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async def override_get_db():
        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
