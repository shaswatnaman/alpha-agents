"""
Redis client singleton with connection pooling.

Usage:
    from app.cache.redis_client import get_redis
    redis = await get_redis()
    await redis.set("key", "value", ex=300)
"""
from __future__ import annotations

import redis.asyncio as aioredis
from app.config.settings import get_settings

_pool: aioredis.ConnectionPool | None = None


def _build_pool() -> aioredis.ConnectionPool:
    settings = get_settings()
    return aioredis.ConnectionPool.from_url(
        str(settings.redis_url),
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )


async def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = _build_pool()
    return aioredis.Redis(connection_pool=_pool)


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
