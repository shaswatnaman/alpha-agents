"""
Redis-backed sliding-window rate limiter.

Uses a sorted-set per (identifier, endpoint) pair.
Members are timestamps; we remove entries outside the window and count remaining.

Why sliding window over fixed window:
- Fixed windows allow burst at the boundary (2x rate in 2 seconds)
- Sliding window enforces a smooth limit regardless of timing
"""
from __future__ import annotations

import time

import structlog
from fastapi import HTTPException, Request, status

from app.cache.redis_client import get_redis
from app.config.settings import get_settings

log = structlog.get_logger(__name__)


class RateLimiter:
    def __init__(
        self,
        requests: int,
        window_seconds: int,
        identifier: str = "global",
    ) -> None:
        self._requests = requests
        self._window = window_seconds
        self._identifier = identifier

    async def check(self, key: str) -> None:
        """
        Raise HTTP 429 if the key has exceeded the rate limit.
        key should be something like the API key or IP address.
        """
        redis = await get_redis()
        now = time.time()
        window_start = now - self._window
        bucket = f"rl:{self._identifier}:{key}"

        async with redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(bucket, "-inf", window_start)
            pipe.zadd(bucket, {str(now): now})
            pipe.zcard(bucket)
            pipe.expire(bucket, self._window + 1)
            results = await pipe.execute()

        count = results[2]
        if count > self._requests:
            log.warning(
                "rate_limit_exceeded",
                key=key,
                identifier=self._identifier,
                count=count,
                limit=self._requests,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit: {self._requests} per {self._window}s",
                    "retry_after": self._window,
                },
                headers={"Retry-After": str(self._window)},
            )


# Pre-built limiters for the two main surfaces
api_limiter = RateLimiter(
    requests=get_settings().rate_limit_requests_per_minute,
    window_seconds=60,
    identifier="api",
)

research_limiter = RateLimiter(
    requests=get_settings().rate_limit_research_per_hour,
    window_seconds=3600,
    identifier="research",
)
