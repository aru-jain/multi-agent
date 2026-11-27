"""FastAPI application for Multi-Agent AI Workflow system."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import router as auth_router
from src.api.routes import router
from src.core.config import get_settings
from src.persistence.checkpointer import cleanup_checkpointer, create_checkpointer
from src.persistence.database import init_database
from src.persistence.redis_client import close_redis, get_redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

__all__ = ["app", "lifespan", "root", "health"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and cleanup resources."""
    logger.info("Starting Multi-Agent Workflow API...")

    # Initialize persistence
    try:
        await create_checkpointer()
        logger.info("Checkpointer initialized")
    except Exception as e:
        logger.warning("Checkpointer init failed: %s", e)

    try:
        await get_redis()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis init failed: %s", e)

    try:
        await init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init failed: %s", e)

    logger.info("Multi-Agent Workflow API is ready!")
    yield

    # Cleanup
    logger.info("Shutting down...")
    await cleanup_checkpointer()
    await close_redis()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Multi-Agent AI Workflow API",
    description="An agentic system where specialized agents plan, call real tools, and complete multi-step tasks with human-in-the-loop checkpoints.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware (explicit origins required when allow_credentials=True)
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        _settings.frontend_url.rstrip("/"),
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(auth_router)
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "name": "Multi-Agent AI Workflow API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "info": "Use /workflow/run to start a workflow",
        },
    }


if __name__ == "__main__":
    import uvicorn
    from src.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
