"""Pydantic models and data structures for the Multi-Agent AI Workflow system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

__all__ = [
    "TaskStatus",
    "WorkflowStatus",
    "AgentRole",
    "SubTask",
    "ToolResult",
    "TraceEntry",
    "BudgetUsage",
    "ApprovalRequest",
    "WorkflowConfig",
]


class TaskStatus(str, Enum):
    """Execution status for an individual subtask."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Lifecycle status of a multi-agent workflow run."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class AgentRole(str, Enum):
    """Specialized agent roles available in the workflow system."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    DB_ANALYST = "db_analyst"
    EVALUATOR = "evaluator"


class SubTask(BaseModel):
    """A single subtask in the execution plan.
    
    Attributes:
        id: Unique 8-character identifier for the task.
        description: Detailed instruction of what this task entails.
        assigned_agent: Role of the agent responsible for executing this task.
        requires_approval: Whether human-in-the-loop sign-off is required before execution.
        dependencies: Task IDs that must be completed prior to executing this task.
        status: Current execution status of the subtask.
        result: String output or summary from the task execution if successful.
        error: Error message if the task failed.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    assigned_agent: AgentRole
    requires_approval: bool = False
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None


class ToolResult(BaseModel):
    """Standardized result from any tool call. Tools NEVER raise — they return this.
    
    Attributes:
        success: Whether the tool operation completed without errors.
        tool_name: Name of the invoked tool.
        data: Arbitrary structured data returned on success.
        error: Error message if the tool execution failed.
        execution_time_ms: Elapsed execution time in milliseconds.
        metadata: Additional contextual information about the execution.
    """

    success: bool
    tool_name: str
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEntry(BaseModel):
    """A single entry in the execution trace log.
    
    Attributes:
        timestamp: UTC timestamp when the event occurred.
        entry_type: Categorization of the trace event.
        agent: Identifier or role of the agent generating the trace event.
        content: Detailed textual representation of the event.
        metadata: Additional key-value properties describing the trace context.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entry_type: Literal[
        "agent_decision",
        "tool_call",
        "tool_result",
        "handoff",
        "plan",
        "guard",
        "approval",
        "error",
    ]
    agent: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetUsage(BaseModel):
    """Tracks resource consumption for a workflow run.
    
    Attributes:
        total_tokens: Sum of prompt and completion tokens.
        prompt_tokens: Number of prompt/input tokens consumed.
        completion_tokens: Number of completion/output tokens consumed.
        total_cost_usd: Estimated monetary cost in USD.
        total_steps: Total number of graph execution steps taken.
        tool_calls: Total number of tool invocations performed.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_steps: int = 0
    tool_calls: int = 0


class ApprovalRequest(BaseModel):
    """Represents a pending human approval request.
    
    Attributes:
        id: Unique 8-character identifier for the approval request.
        workflow_run_id: ID of the workflow run awaiting human input.
        tool_name: The tool that requires authorization before running.
        tool_args: Arguments that would be passed to the tool.
        agent: Agent role requesting approval.
        reason: Justification or context explaining why approval is sought.
        created_at: UTC timestamp when the request was generated.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_run_id: str
    tool_name: str
    tool_args: dict[str, Any]
    agent: AgentRole
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowConfig(BaseModel):
    """Configuration for a single workflow run.
    
    Attributes:
        max_steps: Maximum graph steps before terminating.
        max_tokens: Maximum token limit for the entire run.
        max_cost_usd: Maximum dollar cost limit before terminating.
        max_retries_per_task: Retry budget per failed subtask.
        max_replans: Maximum number of times the planner can replan.
        tool_timeout_seconds: Timeout per tool invocation in seconds.
        require_approval_for_writes: Enforce human approval for write/destructive actions.
    """

    max_steps: int = 50
    max_tokens: int = 100_000
    max_cost_usd: float = 5.0
    max_retries_per_task: int = 3
    max_replans: int = 2
    tool_timeout_seconds: int = 30
    require_approval_for_writes: bool = True
