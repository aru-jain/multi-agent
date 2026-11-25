"""MCP Search Server providing web search and URL fetching tools."""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP

__all__ = ["fetch_url", "mcp", "web_search"]

mcp = FastMCP("SearchServer")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date information.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string with search results or error details.
    """
    start = time.time()
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return json.dumps({
                "success": False,
                "tool_name": "web_search",
                "data": None,
                "error": "TAVILY_API_KEY not set",
                "execution_time_ms": (time.time() - start) * 1000,
            })
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)
        return json.dumps({
            "success": True,
            "tool_name": "web_search",
            "data": results,
            "error": None,
            "execution_time_ms": (time.time() - start) * 1000,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "web_search",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


@mcp.tool()
def fetch_url(url: str, timeout_seconds: int = 10) -> str:
    """Fetch the content of a URL and return its text.

    Args:
        url: The URL to fetch.
        timeout_seconds: Request timeout in seconds (default 10).

    Returns:
        JSON string with page content or error details.
    """
    start = time.time()
    try:
        import httpx

        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            # Truncate very long pages
            content = response.text[:10000]
            return json.dumps({
                "success": True,
                "tool_name": "fetch_url",
                "data": {
                    "url": url,
                    "status_code": response.status_code,
                    "content": content,
                    "truncated": len(response.text) > 10000,
                },
                "error": None,
                "execution_time_ms": (time.time() - start) * 1000,
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "fetch_url",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")
