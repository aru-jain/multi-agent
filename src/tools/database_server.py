"""MCP Database Server providing read and write database operations."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

__all__ = ["mcp", "query_db", "write_db"]

mcp = FastMCP("DatabaseServer")

# Module-level connection pool
_pool: Any = None


def _get_pool() -> Any:
    """Initialize or return the cached psycopg connection pool."""
    global _pool
    if _pool is None:
        import psycopg_pool

        db_url = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/multi_agent"
        )
        # Convert asyncpg URL to psycopg format if needed
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        _pool = psycopg_pool.ConnectionPool(db_url, min_size=1, max_size=5)
    return _pool


@mcp.tool()
def query_db(sql: str, params: str = "[]") -> str:
    """Execute a READ-ONLY SQL query on the database.

    Args:
        sql: SQL SELECT query to execute.
        params: JSON-encoded list of query parameters (default empty list).

    Returns:
        JSON string with query results or error details.
    """
    start = time.time()
    try:
        # Validate read-only
        sql_upper = sql.strip().upper()
        if any(
            keyword in sql_upper
            for keyword in [
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "ALTER",
                "TRUNCATE",
                "CREATE",
            ]
        ):
            return json.dumps({
                "success": False,
                "tool_name": "query_db",
                "data": None,
                "error": "Only SELECT queries are allowed. Use write_db for mutations.",
                "execution_time_ms": (time.time() - start) * 1000,
            })

        parsed_params = json.loads(params) if params else []
        pool = _get_pool()

        with pool.connection() as conn:
            conn.execute("SET statement_timeout = '10s'")
            with conn.cursor() as cur:
                cur.execute(sql, parsed_params)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                else:
                    columns = []
                    rows = []

                return json.dumps(
                    {
                        "success": True,
                        "tool_name": "query_db",
                        "data": {
                            "columns": columns,
                            "rows": rows,
                            "row_count": len(rows),
                        },
                        "error": None,
                        "execution_time_ms": (time.time() - start) * 1000,
                    },
                    default=str,
                )
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "query_db",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


@mcp.tool()
def write_db(sql: str, params: str = "[]") -> str:
    """Execute a WRITE SQL statement (INSERT/UPDATE/DELETE). THIS IS A DESTRUCTIVE OPERATION requiring human approval.

    Args:
        sql: SQL write statement to execute.
        params: JSON-encoded list of query parameters (default empty list).

    Returns:
        JSON string with write result or error details.
    """
    start = time.time()
    try:
        parsed_params = json.loads(params) if params else []
        pool = _get_pool()

        with pool.connection() as conn:
            conn.execute("SET statement_timeout = '10s'")
            with conn.cursor() as cur:
                cur.execute(sql, parsed_params)
                rows_affected = cur.rowcount
            conn.commit()

            return json.dumps({
                "success": True,
                "tool_name": "write_db",
                "data": {"rows_affected": rows_affected},
                "error": None,
                "execution_time_ms": (time.time() - start) * 1000,
                "metadata": {"is_destructive": True},
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "write_db",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")
