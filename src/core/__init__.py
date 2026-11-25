"""Core module providing configuration, data models, and state schemas."""

from __future__ import annotations

from src.core.config import Settings, get_settings
from src.core.models import (
    AgentRole,
    ApprovalRequest,
    BudgetUsage,
    SubTask,
    TaskStatus,
    ToolResult,
    TraceEntry,
    WorkflowConfig,
    WorkflowStatus,
)
from src.core.state import (
    WorkflowState,
    _increment,
    _merge_results,
    _merge_trace,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Models & Enums
    "AgentRole",
    "ApprovalRequest",
    "BudgetUsage",
    "SubTask",
    "TaskStatus",
    "ToolResult",
    "TraceEntry",
    "WorkflowConfig",
    "WorkflowStatus",
    # State
    "WorkflowState",
    "_increment",
    "_merge_results",
    "_merge_trace",
]
