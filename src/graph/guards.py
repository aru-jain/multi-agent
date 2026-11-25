"""Safety guardrails for multi-agent workflow execution."""

from __future__ import annotations

import logging
from typing import Any

from src.core.models import BudgetUsage, WorkflowConfig
from src.core.state import WorkflowState

__all__ = [
    "GuardTripped",
    "check_step_limit",
    "check_budget",
    "check_loop_detection",
    "run_all_guards",
]

logger = logging.getLogger(__name__)


class GuardTripped(Exception):
    """Raised when a safety guard is triggered.

    Attributes:
        guard_name: Identifier of the tripped guard.
        message: Descriptive explanation of why the guard was tripped.
    """

    def __init__(self, guard_name: str, message: str) -> None:
        self.guard_name = guard_name
        self.message = message
        super().__init__(f"Guard '{guard_name}' tripped: {message}")


def check_step_limit(state: WorkflowState, config: WorkflowConfig) -> bool:
    """Check if step count exceeds the maximum allowed.

    Args:
        state: Current workflow execution state.
        config: Workflow configuration defining limits.

    Returns:
        True if safe to continue, False if limit exceeded.
    """
    step_count = state.get("step_count", 0)
    if step_count >= config.max_steps:
        logger.warning(
            "Step limit exceeded: %d >= %d", step_count, config.max_steps
        )
        return False
    return True


def check_budget(state: WorkflowState, config: WorkflowConfig) -> bool:
    """Check if token/cost budget is within limits.

    Args:
        state: Current workflow execution state.
        config: Workflow configuration defining limits.

    Returns:
        True if safe, False if budget exceeded.
    """
    budget = state.get("budget", BudgetUsage())
    if isinstance(budget, dict):
        budget = BudgetUsage(**budget)

    if budget.total_tokens >= config.max_tokens:
        logger.warning(
            "Token budget exceeded: %d >= %d",
            budget.total_tokens,
            config.max_tokens,
        )
        return False
    if budget.total_cost_usd >= config.max_cost_usd:
        logger.warning(
            "Cost budget exceeded: $%s >= $%s",
            budget.total_cost_usd,
            config.max_cost_usd,
        )
        return False
    return True


def check_loop_detection(state: WorkflowState, window: int = 6) -> bool:
    """Detect if the agent is stuck in a loop by checking recent messages.

    Args:
        state: Current workflow execution state.
        window: Number of recent messages to inspect for repetition.

    Returns:
        True if safe, False if loop detected.
    """
    messages = state.get("messages", [])
    if len(messages) < window:
        return True

    recent = messages[-window:]
    contents: list[str] = []
    for msg in recent:
        if hasattr(msg, "content"):
            contents.append(str(msg.content)[:200])

    if len(contents) >= 4:
        # Check if the last N messages are very similar (repetitive pattern)
        unique = set(contents)
        if len(unique) <= 2:
            logger.warning(
                "Loop detected: %d unique messages in last %d",
                len(unique),
                window,
            )
            return False
    return True


def run_all_guards(
    state: WorkflowState, config: WorkflowConfig | None = None
) -> tuple[bool, str | None]:
    """Run all safety guards.

    Args:
        state: Current workflow execution state.
        config: Optional configuration limits; defaults to standard WorkflowConfig.

    Returns:
        Tuple of (is_safe, error_message). If is_safe is True, error_message is None.
    """
    if config is None:
        config = WorkflowConfig()

    if not check_step_limit(state, config):
        return (
            False,
            f"Step limit exceeded ({state.get('step_count', 0)} >= {config.max_steps})",
        )

    if not check_budget(state, config):
        budget = state.get("budget", BudgetUsage())
        if isinstance(budget, dict):
            budget = BudgetUsage(**budget)
        return (
            False,
            f"Budget exceeded (tokens: {budget.total_tokens}, cost: ${budget.total_cost_usd})",
        )

    if not check_loop_detection(state):
        return False, "Loop detected — agent appears stuck in repetitive cycle"

    return True, None
