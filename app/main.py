"""
main.py — FastAPI application factory with lifespan and health endpoint.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.schemas import HealthResponse
from app.config import settings
from app.db.models import Base
from app.db.session import engine


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Start-up: create all ORM tables (idempotent).
    Shut-down: dispose the async engine connection pool.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield  # application runs here

    await engine.dispose()


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Stateful RAG Orchestration Pipeline — async FastAPI backend.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (permissive for local dev; tighten for production) ───────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router)   # paths already include /v1/... prefix

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Health check",
    )
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


app = create_app()
