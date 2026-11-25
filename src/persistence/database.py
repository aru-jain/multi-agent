"""Database initialization and schema management for workflow tracking and evaluation."""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = ["init_database"]

_pool = None


async def init_database() -> None:
    """Initialize the database and create tables for workflow tracking and evaluation data."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    try:
        import psycopg

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Create tables for workflow tracking and evaluation
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_runs (
                        id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'running',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        total_steps INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        total_cost_usd NUMERIC(10, 6) DEFAULT 0,
                        error TEXT,
                        result_summary TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS evaluation_results (
                        id SERIAL PRIMARY KEY,
                        scenario_id INTEGER NOT NULL,
                        scenario_name TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        steps INTEGER DEFAULT 0,
                        tokens INTEGER DEFAULT 0,
                        cost_usd NUMERIC(10, 6) DEFAULT 0,
                        wall_time_seconds NUMERIC(10, 2) DEFAULT 0,
                        error TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    
                    CREATE TABLE IF NOT EXISTS sample_data (
                        id SERIAL PRIMARY KEY,
                        category TEXT,
                        name TEXT NOT NULL,
                        value NUMERIC(10, 2),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
            conn.commit()

        logger.info("Database tables initialized")

        # Seed sample data if empty
        await _seed_sample_data(db_url)

    except Exception as e:
        logger.warning("Database initialization failed: %s", e)


async def _seed_sample_data(db_url: str) -> None:
    """Seed sample data for evaluation scenarios."""
    try:
        import psycopg

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sample_data")
                row = cur.fetchone()
                count = row[0] if row else 0
                if count == 0:
                    sample_rows = [
                        ("sales", "Widget A", 1250.00),
                        ("sales", "Widget B", 890.50),
                        ("sales", "Widget C", 2100.75),
                        ("sales", "Gadget X", 3450.00),
                        ("sales", "Gadget Y", 1800.25),
                        ("users", "Alice", 1.00),
                        ("users", "Bob", 1.00),
                        ("users", "Charlie", 1.00),
                        ("inventory", "Part 1", 500.00),
                        ("inventory", "Part 2", 250.00),
                    ]
                    for category, name, value in sample_rows:
                        cur.execute(
                            "INSERT INTO sample_data (category, name, value) VALUES (%s, %s, %s)",
                            (category, name, value),
                        )
                conn.commit()
                logger.info("Sample data seeded")
    except Exception as e:
        logger.warning("Sample data seeding failed: %s", e)
