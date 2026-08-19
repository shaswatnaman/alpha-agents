"""
Idempotency key management.

When a client submits a research request with an idempotency key,
we store the result in Redis for 24 hours.  A second request with
the same key within that window returns the cached result immediately
without re-running the expensive pipeline.

Key format: idempotency:{key}
Value: JSON of the existing research_id
"""

from __future__ import annotations

import json

import structlog

from app.cache.redis_client import get_redis
from app.config.settings import get_settings

log = structlog.get_logger(__name__)


async def get_idempotent_result(idempotency_key: str) -> str | None:
    """Return the research_id stored for this key, or None."""
    redis = await get_redis()
    raw = await redis.get(f"idempotency:{idempotency_key}")
    if raw:
        log.info("idempotency_hit", key=idempotency_key)
        return json.loads(raw)["research_id"]
    return None


async def store_idempotent_result(idempotency_key: str, research_id: str) -> None:
    """Persist the mapping for the configured TTL."""
    redis = await get_redis()
    ttl = get_settings().idempotency_key_ttl_seconds
    await redis.setex(
        f"idempotency:{idempotency_key}",
        ttl,
        json.dumps({"research_id": research_id}),
    )
    log.info("idempotency_stored", key=idempotency_key, research_id=research_id, ttl=ttl)
