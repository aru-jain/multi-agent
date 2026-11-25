"""Planner agent that decomposes a user goal into ordered SubTasks."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import Command

from src.core.config import get_settings
from src.core.models import AgentRole, SubTask, TaskStatus, TraceEntry, WorkflowStatus
from src.core.state import WorkflowState

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "planner_node",
]

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a planning agent. Given a user's goal, decompose it into ordered subtasks.

Each subtask must specify:
- description: Clear instruction for what needs to be done
- assigned_agent: One of "researcher", "coder", or "db_analyst"
- requires_approval: true only if the task involves database writes or destructive actions
- dependencies: list of subtask IDs this task depends on (use the IDs you assign)

Agent capabilities:
- researcher: Can search the web and fetch URLs. Use for information gathering.
- coder: Can execute Python code and lint code. Use for computation, data processing, code generation.
- db_analyst: Can query and write to databases. Use for data storage and retrieval.

Respond with a JSON array of subtasks. Example:
[
  {"id": "task_1", "description": "Search for...", "assigned_agent": "researcher", "requires_approval": false, "dependencies": []},
  {"id": "task_2", "description": "Write Python code to...", "assigned_agent": "coder", "requires_approval": false, "dependencies": ["task_1"]}
]

Keep plans concise — aim for 2-5 subtasks. Only add more if genuinely needed.
Always respond with ONLY the JSON array, no other text."""


def planner_node(state: WorkflowState) -> Command:
    """Planner agent that decomposes a goal into ordered subtasks.

    Reads the 'goal' from state, calls the LLM to generate a plan,
    and writes the plan into state. Routes to 'router' node.

    Args:
        state: Current workflow execution state.

    Returns:
        Command updating state with the generated plan and routing to 'router',
        or routing to '__end__' on unrecoverable planning failure.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    goal = state.get("goal", "")
    existing_results = state.get("results", {})

    # If replanning after a failure, include context
    context = ""
    if existing_results:
        context = (
            f"\n\nPrevious attempt results (some tasks may have failed):\n"
            f"{json.dumps(existing_results, indent=2, default=str)}"
        )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Goal: {goal}{context}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                item if isinstance(item, str) else str(item) for item in content
            )
        content = content.strip()

        # Parse the JSON plan; handle markdown code blocks if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        tasks_data = json.loads(content)
        if not isinstance(tasks_data, list):
            raise ValueError("Planner output must be a JSON array of subtasks")

        plan: list[SubTask] = []
        for task_data in tasks_data:
            assigned_raw = str(task_data.get("assigned_agent", "researcher")).lower().strip()
            if assigned_raw.startswith("agentrole."):
                assigned_raw = assigned_raw.split(".", 1)[1]

            try:
                assigned_agent = AgentRole(assigned_raw)
            except ValueError:
                assigned_agent = AgentRole.RESEARCHER

            subtask = SubTask(
                id=task_data.get("id") or str(uuid.uuid4())[:8],
                description=task_data.get("description", ""),
                assigned_agent=assigned_agent,
                requires_approval=bool(task_data.get("requires_approval", False)),
                dependencies=list(task_data.get("dependencies", [])),
                status=TaskStatus.PENDING,
            )
            plan.append(subtask)

        trace = TraceEntry(
            entry_type="plan",
            agent="planner",
            content=f"Created plan with {len(plan)} subtasks for goal: {goal}",
            metadata={"plan": [t.model_dump() for t in plan]},
        )

        logger.info("Planner created %d subtasks", len(plan))

        return Command(
            update={
                "plan": plan,
                "current_task_index": 0,
                "step_count": 1,
                "trace_log": [trace],
                "messages": [
                    AIMessage(
                        content=f"Plan created with {len(plan)} subtasks: {[t.description for t in plan]}"
                    )
                ],
            },
            goto="router",
        )
    except Exception as e:
        logger.error("Planner failed: %s", e)
        trace = TraceEntry(
            entry_type="error",
            agent="planner",
            content=f"Planning failed: {str(e)}",
        )
        return Command(
            update={
                "plan": [],
                "status": WorkflowStatus.FAILED,
                "error": f"Planning failed: {str(e)}",
                "step_count": 1,
                "trace_log": [trace],
                "messages": [AIMessage(content=f"Planning failed: {str(e)}")],
            },
            goto="__end__",
        )
