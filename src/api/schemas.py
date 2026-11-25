"""Pydantic schemas for the FastAPI workflow API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

__all__ = [
    "WorkflowRunRequest",
    "WorkflowStatusResponse",
    "WorkflowRunResponse",
    "ApprovalActionRequest",
    "TraceResponse",
]


class WorkflowRunRequest(BaseModel):
    """Request to start a new workflow."""

    goal: str = Field(..., description="The goal or task to accomplish")
    max_steps: int = Field(default=50, ge=1, le=200, description="Maximum steps for this run")
    max_tokens: int = Field(default=100_000, ge=1000, description="Maximum token budget")
    max_cost_usd: float = Field(default=5.0, ge=0.01, description="Maximum spend in USD")


class WorkflowStatusResponse(BaseModel):
    """Response with workflow status."""

    run_id: str
    goal: str
    status: str
    current_task_index: int = 0
    total_tasks: int = 0
    step_count: int = 0
    budget: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    plan: list[dict[str, Any]] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunResponse(BaseModel):
    """Response after starting a workflow."""

    run_id: str
    status: str
    message: str


class ApprovalActionRequest(BaseModel):
    """Request to approve or reject a pending action."""

    feedback: str = Field(default="", description="Optional feedback or reason")


class TraceResponse(BaseModel):
    """Response with workflow trace."""

    run_id: str
    entries: list[dict[str, Any]] = Field(default_factory=list)
    total_entries: int = 0
