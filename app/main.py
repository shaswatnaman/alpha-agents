"""
AlphaAgents FastAPI application entry point.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import RequestIDMiddleware
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.research import router as research_router
from app.cache.redis_client import close_redis
from app.config.settings import get_settings
from app.observability.logging import configure_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    settings = get_settings()
    log.info(
        "startup", app=settings.app_name, version=settings.app_version, env=settings.environment
    )
    yield
    await close_redis()
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AlphaAgents",
        description="Production-grade multi-agent financial research platform",
        version=settings.app_version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "request_id": request.headers.get("X-Request-ID"),
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(research_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")

    return app


app = create_app()
