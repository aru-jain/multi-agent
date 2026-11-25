"""Redis client and caching operations for workflow persistence and budget tracking."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from src.core.config import get_settings
from src.core.models import BudgetUsage, TraceEntry

logger = logging.getLogger(__name__)

__all__ = [
    "get_redis",
    "close_redis",
    "RedisCache",
]

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis | None:
    """Get or create the Redis client.

    Returns:
        aioredis.Redis | None: An active Redis connection or None if unavailable.
    """
    global _redis_client
    settings = get_settings()

    if _redis_client is None and settings.redis_url:
        try:
            _redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await _redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning("Redis connection failed: %s. Caching disabled.", e)
            _redis_client = None

    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.close()
        except Exception as e:
            logger.debug("Error while closing Redis connection: %s", e)
        _redis_client = None


class RedisCache:
    """Redis-backed caching, stream tracing, and budget tracking."""

    @staticmethod
    async def cache_tool_result(
        run_id: str, tool_name: str, args_hash: str, result: str, ttl: int = 300
    ) -> None:
        """Cache a tool result with TTL.

        Args:
            run_id: Unique workflow run identifier.
            tool_name: Name of the executed tool.
            args_hash: Deterministic hash of the tool invocation arguments.
            result: Serialized result string.
            ttl: Time-to-live expiration in seconds (default 300s).
        """
        client = await get_redis()
        if client:
            key = f"tool_cache:{run_id}:{tool_name}:{args_hash}"
            try:
                await client.setex(key, ttl, result)
            except Exception as e:
                logger.warning("Cache write failed: %s", e)

    @staticmethod
    async def get_cached_result(
        run_id: str, tool_name: str, args_hash: str
    ) -> str | None:
        """Get a cached tool result.

        Args:
            run_id: Unique workflow run identifier.
            tool_name: Name of the executed tool.
            args_hash: Deterministic hash of the tool invocation arguments.

        Returns:
            str | None: Cached result string if found, otherwise None.
        """
        client = await get_redis()
        if client:
            key = f"tool_cache:{run_id}:{tool_name}:{args_hash}"
            try:
                return await client.get(key)
            except Exception:
                pass
        return None

    @staticmethod
    async def update_budget(run_id: str, tokens: int, cost: float) -> dict[str, float]:
        """Atomically update budget counters in Redis.

        Args:
            run_id: Unique workflow run identifier.
            tokens: Number of tokens consumed in this step.
            cost: Monetary cost in USD consumed in this step.

        Returns:
            dict[str, float]: Dictionary with current total_tokens and total_cost.
        """
        client = await get_redis()
        if client:
            try:
                pipe = client.pipeline()
                pipe.incrbyfloat(f"budget:{run_id}:tokens", float(tokens))
                pipe.incrbyfloat(f"budget:{run_id}:cost", float(cost))
                results = await pipe.execute()
                return {"total_tokens": float(results[0]), "total_cost": float(results[1])}
            except Exception as e:
                logger.warning("Budget update failed: %s", e)
        return {"total_tokens": 0.0, "total_cost": 0.0}

    @staticmethod
    async def append_trace(run_id: str, entry: TraceEntry) -> None:
        """Append a trace entry to the Redis stream for real-time monitoring.

        Args:
            run_id: Unique workflow run identifier.
            entry: Structured TraceEntry model instance.
        """
        client = await get_redis()
        if client:
            try:
                await client.xadd(
                    f"trace:{run_id}",
                    {
                        "type": entry.entry_type,
                        "agent": entry.agent or "",
                        "content": entry.content,
                        "timestamp": entry.timestamp.isoformat(),
                    },
                    maxlen=1000,
                )
            except Exception as e:
                logger.warning("Trace append failed: %s", e)

    @staticmethod
    async def get_trace(run_id: str, count: int = 100) -> list[dict[str, Any]]:
        """Get recent trace entries from the Redis stream.

        Args:
            run_id: Unique workflow run identifier.
            count: Maximum number of entries to retrieve.

        Returns:
            list[dict[str, Any]]: List of trace entry dictionaries.
        """
        client = await get_redis()
        if client:
            try:
                entries = await client.xrange(f"trace:{run_id}", count=count)
                return [entry[1] for entry in entries]
            except Exception:
                pass
        return []
