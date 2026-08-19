"""
FastAPI middleware: request IDs, latency tracking, structured logging.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.observability.metrics import http_latency_histogram, http_request_counter

log = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID into every request and response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind to structlog context so all logs within this request carry the ID
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t0 = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - t0

        path = request.url.path
        method = request.method
        status = response.status_code

        http_request_counter.labels(method=method, path=path, status_code=str(status)).inc()
        http_latency_histogram.labels(method=method, path=path).observe(elapsed)

        log.info(
            "http_request",
            method=method,
            path=path,
            status_code=status,
            latency_ms=int(elapsed * 1000),
        )

        response.headers["X-Request-ID"] = request_id
        return response
