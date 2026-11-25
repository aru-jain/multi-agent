"""MCP Code Server providing sandboxed execution and code linting tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from mcp.server.fastmcp import FastMCP

__all__ = ["execute_python", "lint_code", "mcp"]

mcp = FastMCP("CodeServer")


@mcp.tool()
def execute_python(code: str, timeout_seconds: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess and return the output.

    Args:
        code: Python code to execute.
        timeout_seconds: Maximum execution time in seconds (default 30).

    Returns:
        JSON string with execution output or error details.
    """
    start = time.time()
    temp_path: str | None = None
    try:
        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tempfile.gettempdir(),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output = result.stdout
            errors = result.stderr
            success = result.returncode == 0

            return json.dumps({
                "success": success,
                "tool_name": "execute_python",
                "data": {
                    "stdout": output[:5000],
                    "stderr": errors[:2000] if errors else None,
                    "return_code": result.returncode,
                    "truncated": len(output) > 5000,
                },
                "error": errors[:500] if not success and errors else None,
                "execution_time_ms": (time.time() - start) * 1000,
            })
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "tool_name": "execute_python",
            "data": None,
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "execution_time_ms": (time.time() - start) * 1000,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "execute_python",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


@mcp.tool()
def lint_code(code: str) -> str:
    """Lint Python code using ruff and return any issues found.

    Args:
        code: Python code to lint.

    Returns:
        JSON string with lint results or error details.
    """
    start = time.time()
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", temp_path, "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            issues = []
            if result.stdout.strip():
                try:
                    issues = json.loads(result.stdout)
                except json.JSONDecodeError:
                    issues = [{"message": result.stdout}]

            return json.dumps({
                "success": True,
                "tool_name": "lint_code",
                "data": {"issues": issues, "issue_count": len(issues), "clean": len(issues) == 0},
                "error": None,
                "execution_time_ms": (time.time() - start) * 1000,
            })
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        return json.dumps({
            "success": False,
            "tool_name": "lint_code",
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "execution_time_ms": (time.time() - start) * 1000,
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")
