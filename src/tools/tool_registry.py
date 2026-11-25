"""Central tool registry that manages MCP servers and loads tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any

from langchain_core.tools import BaseTool, tool as langchain_tool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

__all__ = [
    "AGENT_TOOL_MAPPING",
    "DESTRUCTIVE_TOOLS",
    "TOOL_SERVERS",
    "ToolRegistry",
    "ToolServerConfig",
    "tool_registry",
]

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).parent


@dataclass
class ToolServerConfig:
    """Configuration for an MCP tool server."""

    name: str
    script_path: str
    tool_names: list[str]
    is_destructive_tools: list[str] = field(default_factory=list)
    timeout_seconds: int = 30


# Registry of all tool servers
TOOL_SERVERS: list[ToolServerConfig] = [
    ToolServerConfig(
        name="SearchServer",
        script_path=str(TOOLS_DIR / "search_server.py"),
        tool_names=["web_search", "fetch_url"],
    ),
    ToolServerConfig(
        name="CodeServer",
        script_path=str(TOOLS_DIR / "code_server.py"),
        tool_names=["execute_python", "lint_code"],
    ),
    ToolServerConfig(
        name="DatabaseServer",
        script_path=str(TOOLS_DIR / "database_server.py"),
        tool_names=["query_db", "write_db"],
        is_destructive_tools=["write_db"],
    ),
]

# Tool grouping by agent role
AGENT_TOOL_MAPPING: dict[str, list[str]] = {
    "researcher": ["web_search", "fetch_url"],
    "coder": ["execute_python", "lint_code"],
    "db_analyst": ["query_db", "write_db"],
}

# Set of tools requiring human approval
DESTRUCTIVE_TOOLS: set[str] = {"write_db"}


class ToolRegistry:
    """Manages MCP tool servers and provides tools to agents.

    This registry starts MCP servers as subprocesses, loads their tools
    via langchain-mcp-adapters, and provides filtered tool lists per agent role.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._sessions: list[Any] = []  # Keep references to prevent GC
        self._loaded: bool = False

    async def load_tools_for_role(self, role: str) -> list[BaseTool]:
        """Get the list of LangChain tools available for a given agent role.

        If tools are not yet loaded, this will load them.
        For simplicity in the initial version, we use direct function tools instead
        of MCP subprocess servers to avoid session management complexity.
        """
        if not self._loaded:
            await self._load_all_tools()

        tool_names = AGENT_TOOL_MAPPING.get(role, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def load_tools_for_role_sync(self, role: str) -> list[BaseTool]:
        """Synchronously get the list of LangChain tools available for a given agent role."""
        if not self._loaded:
            self._load_all_tools_sync()

        tool_names = AGENT_TOOL_MAPPING.get(role, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    async def _load_all_tools(self) -> None:
        """Load tools from all configured MCP servers.

        Uses direct LangChain tool wrappers around the MCP server functions
        to avoid subprocess lifecycle complexity. In production, these would
        connect to long-running MCP server processes.
        """
        self._load_all_tools_sync()

    def _load_all_tools_sync(self) -> None:
        """Synchronously load tools from all configured MCP server modules."""
        for server_config in TOOL_SERVERS:
            try:
                # Load the server module to get the functions directly
                spec = importlib.util.spec_from_file_location(
                    server_config.name, server_config.script_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for tool_name in server_config.tool_names:
                        if hasattr(module, tool_name):
                            fn = getattr(module, tool_name)
                            # Wrap as LangChain tool
                            wrapped = langchain_tool(fn)
                            self._tools[tool_name] = wrapped
                            logger.info(
                                "Loaded tool: %s from %s",
                                tool_name,
                                server_config.name,
                            )

            except Exception as e:
                logger.error(
                    "Failed to load tools from %s: %s",
                    server_config.name,
                    e,
                )

        self._loaded = True
        logger.info(
            "Tool registry loaded %d tools: %s",
            len(self._tools),
            list(self._tools.keys()),
        )

    def get_all_tools(self) -> list[BaseTool]:
        """Get all loaded tools."""
        if not self._loaded:
            self._load_all_tools_sync()
        return list(self._tools.values())

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a specific tool by name."""
        if not self._loaded:
            self._load_all_tools_sync()
        return self._tools.get(name)

    def is_destructive(self, tool_name: str) -> bool:
        """Check if a tool is marked as destructive (requires approval)."""
        return tool_name in DESTRUCTIVE_TOOLS


# Global registry instance
tool_registry = ToolRegistry()
