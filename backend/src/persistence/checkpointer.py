"""LangGraph checkpointer persistence layer."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from src.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "create_checkpointer",
    "get_checkpointer",
    "cleanup_checkpointer",
]

# Global checkpointer instance
_checkpointer: BaseCheckpointSaver | None = None


async def create_checkpointer() -> BaseCheckpointSaver:
    """Create and initialize the appropriate checkpointer.
    
    Uses AsyncPostgresSaver if DATABASE_URL is configured and postgresql is present,
    otherwise falls back to MemorySaver.

    Returns:
        BaseCheckpointSaver: An initialized checkpointer instance.
    """
    global _checkpointer
    settings = get_settings()
    
    if settings.database_url and "postgresql" in settings.database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            # Convert asyncpg URL format to psycopg format if needed
            db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
            checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
            await checkpointer.setup()
            _checkpointer = checkpointer
            logger.info("Using AsyncPostgresSaver for workflow persistence")
            return checkpointer
        except Exception as e:
            logger.warning("Failed to initialize Postgres checkpointer: %s. Falling back to MemorySaver.", e)
    
    _checkpointer = MemorySaver()
    logger.info("Using MemorySaver for workflow persistence (non-persistent)")
    return _checkpointer


def get_checkpointer() -> BaseCheckpointSaver:
    """Get the global checkpointer instance.

    Returns:
        BaseCheckpointSaver: The current checkpointer, defaulting to MemorySaver.
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


async def cleanup_checkpointer() -> None:
    """Clean up checkpointer resources."""
    global _checkpointer
    if _checkpointer and hasattr(_checkpointer, "conn"):
        try:
            await _checkpointer.conn.close()
        except Exception as e:
            logger.debug("Error while closing checkpointer connection: %s", e)
    _checkpointer = None
