"""Configuration settings for the Multi-Agent AI Workflow system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

__all__ = ["Settings", "get_settings"]

# Load environment variables from .env file if present
load_dotenv()


def _get_int(key: str, default: int) -> int:
    """Safely parse integer environment variable with fallback."""
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    """Safely parse float environment variable with fallback."""
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Global configuration settings for the Multi-Agent AI Workflow.
    
    Attributes:
        groq_api_key: API key for Groq service.
        groq_model: Default Groq model identifier.
        tavily_api_key: API key for Tavily search provider.
        database_url: Connection URL for PostgreSQL database.
        redis_url: Connection URL for Redis cache/message broker.
        max_steps: Maximum execution steps allowed per workflow.
        max_tokens: Maximum token budget across a workflow run.
        max_cost_usd: Maximum total dollar spend allowed per workflow run.
        api_host: Host binding for the API service.
        api_port: Port binding for the API service.
    """

    # LLM
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    # Tools
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/multi_agent"
        )
    )

    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # Budget
    max_steps: int = field(default_factory=lambda: _get_int("MAX_STEPS", 50))
    max_tokens: int = field(default_factory=lambda: _get_int("MAX_TOKENS", 100000))
    max_cost_usd: float = field(default_factory=lambda: _get_float("MAX_COST_USD", 5.0))

    # Server
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _get_int("API_PORT", 8000))


def get_settings() -> Settings:
    """Get a Settings instance populated from environment variables.
    
    Returns:
        Settings: The instantiated application settings.
    """
    return Settings()
