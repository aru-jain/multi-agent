"""Evaluator agent that assesses subtask execution results and directs workflow flow."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import Command

from src.core.config import get_settings
from src.core.models import TaskStatus, TraceEntry, WorkflowStatus
from src.core.state import WorkflowState

__all__ = [
    "EVALUATOR_SYSTEM_PROMPT",
    "evaluator_node",
]

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an evaluator agent. Your job is to review the result of a completed subtask and decide the next action.

You will be given:
- The original goal
- The current subtask description
- The result/output from the worker agent

Decide ONE of:
1. "complete" — The subtask was completed successfully. Move to the next task.
2. "retry" — The subtask failed or produced poor results. Retry it (if retries remain).
3. "replan" — The approach is fundamentally wrong. Ask the planner to create a new plan.

Respond with a JSON object:
{"decision": "complete|retry|replan", "reason": "explanation"}

Only respond with the JSON, no other text."""


def evaluator_node(state: WorkflowState) -> Command:
    """Evaluator agent that reviews subtask results and decides next steps.

    Args:
        state: Current workflow execution state.

    Returns:
        Command advancing to next task ('router'), retrying ('router'),
        replanning ('planner'), or completing workflow ('__end__').
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    plan = state.get("plan", [])
    task_index = state.get("current_task_index", 0)
    current_task = plan[task_index] if 0 <= task_index < len(plan) else None
    goal = state.get("goal", "")
    results = state.get("results", {})

    # Get the latest result
    task_id = current_task.id if current_task else "unknown"
    latest_result = results.get(task_id, "No result available")

    task_desc = current_task.description if current_task else "unknown"
    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Goal: {goal}\nCurrent subtask: {task_desc}\nResult: {latest_result}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                item if isinstance(item, str) else str(item) for item in content
            )
        content = content.strip()

        # Handle markdown blocks
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        decision = json.loads(content)
        action = str(decision.get("decision", "complete")).lower().strip()
        reason = decision.get("reason", "")

        trace = TraceEntry(
            entry_type="agent_decision",
            agent="evaluator",
            content=f"Decision: {action} — {reason}",
            metadata={"decision": action, "task_id": task_id, "reason": reason},
        )

        if action == "complete":
            # Mark task complete, record result, and advance index
            if current_task:
                current_task.status = TaskStatus.COMPLETED
                current_task.result = str(latest_result)

            new_index = task_index + 1
            if new_index >= len(plan):
                # All tasks done
                return Command(
                    update={
                        "plan": plan,
                        "current_task_index": new_index,
                        "status": WorkflowStatus.COMPLETED,
                        "step_count": 1,
                        "trace_log": [trace],
                        "messages": [
                            AIMessage(
                                content=f"Workflow completed! All {len(plan)} tasks done."
                            )
                        ],
                    },
                    goto="__end__",
                )
            else:
                # Move to next task
                return Command(
                    update={
                        "plan": plan,
                        "current_task_index": new_index,
                        "step_count": 1,
                        "trace_log": [trace],
                        "messages": [
                            AIMessage(
                                content=f"Task {task_id} completed. Moving to next task."
                            )
                        ],
                    },
                    goto="router",
                )
        elif action == "retry":
            if current_task:
                current_task.status = TaskStatus.PENDING
            return Command(
                update={
                    "plan": plan,
                    "step_count": 1,
                    "trace_log": [trace],
                    "messages": [AIMessage(content=f"Retrying task {task_id}: {reason}")],
                },
                goto="router",
            )
        else:  # replan
            return Command(
                update={
                    "step_count": 1,
                    "trace_log": [trace],
                    "messages": [AIMessage(content=f"Replanning: {reason}")],
                },
                goto="planner",
            )
    except Exception as e:
        logger.error("Evaluator failed: %s", e)
        # On evaluator failure, safely advance to next task or complete
        if current_task:
            current_task.status = TaskStatus.COMPLETED
            current_task.result = str(latest_result)

        new_index = task_index + 1
        if new_index >= len(plan):
            return Command(
                update={
                    "plan": plan,
                    "current_task_index": new_index,
                    "status": WorkflowStatus.COMPLETED,
                    "step_count": 1,
                    "trace_log": [
                        TraceEntry(
                            entry_type="error",
                            agent="evaluator",
                            content=str(e),
                        )
                    ],
                    "messages": [
                        AIMessage(content="Workflow completed (evaluator error, advancing).")
                    ],
                },
                goto="__end__",
            )
        return Command(
            update={
                "plan": plan,
                "current_task_index": new_index,
                "step_count": 1,
                "trace_log": [
                    TraceEntry(
                        entry_type="error",
                        agent="evaluator",
                        content=str(e),
                    )
                ],
                "messages": [
                    AIMessage(content=f"Evaluator error, advancing to next task: {e}")
                ],
            },
            goto="router",
        )
