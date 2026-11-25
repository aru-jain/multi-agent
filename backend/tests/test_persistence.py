"""Unit tests for persistence modules: checkpointer, redis_client, and database."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from langgraph.checkpoint.memory import MemorySaver

from src.core.models import TraceEntry
from src.persistence.checkpointer import (
    cleanup_checkpointer,
    create_checkpointer,
    get_checkpointer,
)
from src.persistence.database import _seed_sample_data, init_database
from src.persistence.redis_client import (
    RedisCache,
    close_redis,
    get_redis,
)


class TestCheckpointer:
    """Tests for LangGraph checkpointer creation and cleanup."""

    @pytest.mark.asyncio
    async def test_create_checkpointer_fallback_to_memory(self) -> None:
        """Verify checkpointer falls back to MemorySaver when postgres fails or is unset."""
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
            checkpointer = await create_checkpointer()
            assert isinstance(checkpointer, MemorySaver)
            assert get_checkpointer() is checkpointer

    @pytest.mark.asyncio
    async def test_create_checkpointer_postgres_failure_fallback(self) -> None:
        """Verify PostgresSaver error leads to MemorySaver fallback."""
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql://invalid_user:invalid_pwd@localhost:5432/invalid_db"},
            clear=False,
        ):
            checkpointer = await create_checkpointer()
            assert isinstance(checkpointer, MemorySaver)

    @pytest.mark.asyncio
    async def test_cleanup_checkpointer(self) -> None:
        """Verify checkpointer cleanup resets global instance."""
        await create_checkpointer()
        assert get_checkpointer() is not None
        await cleanup_checkpointer()
        # After cleanup, get_checkpointer creates a new MemorySaver
        new_saver = get_checkpointer()
        assert isinstance(new_saver, MemorySaver)


class TestRedisClient:
    """Tests for Redis client and RedisCache methods."""

    @pytest.mark.asyncio
    async def test_get_redis_connection_failure_graceful(self) -> None:
        """Verify get_redis handles connection failure gracefully."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://invalid-host:6379/0"}, clear=False):
            with patch("src.persistence.redis_client._redis_client", None):
                client = await get_redis()
                # Should return None on connection failure
                assert client is None

    @pytest.mark.asyncio
    async def test_close_redis_resets_client(self) -> None:
        """Verify close_redis closes active client."""
        mock_client = AsyncMock()
        with patch("src.persistence.redis_client._redis_client", mock_client):
            await close_redis()
            mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_cache_without_connection(self) -> None:
        """Verify RedisCache functions return safe defaults when Redis is offline."""
        with patch("src.persistence.redis_client.get_redis", AsyncMock(return_value=None)):
            # Caching write
            await RedisCache.cache_tool_result("run-1", "web_search", "hash1", "result")
            # Caching read
            res = await RedisCache.get_cached_result("run-1", "web_search", "hash1")
            assert res is None
            # Budget update
            budget = await RedisCache.update_budget("run-1", 100, 0.05)
            assert budget == {"total_tokens": 0.0, "total_cost": 0.0}
            # Append trace
            entry = TraceEntry(entry_type="tool_call", content="Running query")
            await RedisCache.append_trace("run-1", entry)
            # Get trace
            traces = await RedisCache.get_trace("run-1")
            assert traces == []

    @pytest.mark.asyncio
    async def test_redis_cache_with_mock_client(self) -> None:
        """Verify RedisCache methods interact correctly with Redis when online."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "cached_output"

        # Pipeline mock: pipeline() is sync in redis, returns pipeline object
        # incrbyfloat is sync (chaining), execute() is async
        mock_pipe = MagicMock()
        mock_pipe.incrbyfloat.return_value = mock_pipe  # chaining
        mock_pipe.execute = AsyncMock(return_value=[150.0, 0.02])
        # Explicitly set pipeline as MagicMock so it doesn't auto-become async
        mock_client.pipeline = MagicMock(return_value=mock_pipe)
        mock_client.xrange.return_value = [
            ("1-0", {"type": "plan", "agent": "planner", "content": "Started", "timestamp": "2026-10-19T00:00:00Z"})
        ]

        with patch("src.persistence.redis_client.get_redis", AsyncMock(return_value=mock_client)):
            # Write cache
            await RedisCache.cache_tool_result("run-2", "web_search", "args_hash", "result_data", ttl=60)
            mock_client.setex.assert_awaited_once_with("tool_cache:run-2:web_search:args_hash", 60, "result_data")

            # Read cache
            cached = await RedisCache.get_cached_result("run-2", "web_search", "args_hash")
            assert cached == "cached_output"

            # Budget update
            budget = await RedisCache.update_budget("run-2", 150, 0.02)
            assert budget == {"total_tokens": 150.0, "total_cost": 0.02}

            # Append trace
            entry = TraceEntry(entry_type="agent_decision", content="Decided next step", agent="coder")
            await RedisCache.append_trace("run-2", entry)
            mock_client.xadd.assert_awaited_once()

            # Get trace
            traces = await RedisCache.get_trace("run-2")
            assert len(traces) == 1
            assert traces[0]["type"] == "plan"


class TestDatabase:
    """Tests for PostgreSQL database initialization."""

    @pytest.mark.asyncio
    async def test_init_database_handles_exception(self) -> None:
        """Verify init_database catches errors without raising."""
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql://invalid:invalid@localhost:5432/invalid"},
            clear=False,
        ):
            # Should not raise exception
            await init_database()

    @pytest.mark.asyncio
    async def test_seed_sample_data_mocked(self) -> None:
        """Verify _seed_sample_data executes insert queries when table is empty."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch("psycopg.connect", return_value=mock_conn):
            await _seed_sample_data("postgresql://postgres:postgres@localhost:5432/multi_agent")
            assert mock_cursor.execute.call_count >= 10
            mock_conn.commit.assert_called_once()
