"""Central state definition and reducers for LangGraph multi-agent workflows."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from langgraph.graph import add_messages

from src.core.models import BudgetUsage, SubTask, TraceEntry, WorkflowStatus

__all__ = [
    "WorkflowState",
    "_merge_trace",
    "_merge_results",
    "_increment",
]


def _merge_trace(
    left: list[TraceEntry] | None, right: list[TraceEntry] | None
) -> list[TraceEntry]:
    """Reducer that appends new trace entries.

    Args:
        left: Existing trace entries.
        right: New trace entries to append.

    Returns:
        Combined list of trace entries.
    """
    if left is None:
        return list(right) if right is not None else []
    if right is None:
        return list(left)
    return left + right


def _merge_results(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer that merges result dictionaries.

    Args:
        left: Existing result dictionary.
        right: New result dictionary to merge.

    Returns:
        Merged result dictionary.
    """
    if left is None:
        return dict(right) if right is not None else {}
    if right is None:
        return dict(left)
    return {**left, **right}


def _increment(left: int | None, right: int | None) -> int:
    """Reducer that sums integers (for step counters).

    Args:
        left: Current integer count.
        right: Integer delta to add.

    Returns:
        Summed integer count.
    """
    return (left or 0) + (right or 0)


class WorkflowState(TypedDict):
    """Central state for the multi-agent workflow graph.

    Attributes:
        messages: LangGraph managed conversation messages.
        plan: Ordered list of subtasks defining the execution plan.
        current_task_index: Index of the subtask currently being processed.
        results: Dictionary mapping task IDs to their execution outputs.
        step_count: Cumulative number of graph transitions.
        budget: Resource and token consumption tracking.
        status: Overall workflow lifecycle status.
        error: Top-level error description if workflow encountered a fatal error.
        trace_log: Append-only audit log of workflow events.
        goal: Original user request or high-level objective.
        run_id: Unique execution identifier for this workflow instance.
    """

    # Message history (LangGraph managed)
    messages: Annotated[list, add_messages]

    # Plan
    plan: list[SubTask]
    current_task_index: int

    # Results collected from agents
    results: Annotated[dict[str, Any], _merge_results]

    # Counters and budget
    step_count: Annotated[int, _increment]
    budget: BudgetUsage

    # Status
    status: WorkflowStatus
    error: str | None

    # Trace log
    trace_log: Annotated[list[TraceEntry], _merge_trace]

    # Workflow config
    goal: str
    run_id: str
