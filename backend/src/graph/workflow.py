"""LangGraph multi-agent workflow construction and routing graph."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from src.agents.coder import create_coder_node
from src.agents.db_analyst import create_db_analyst_node
from src.agents.evaluator import evaluator_node
from src.agents.planner import planner_node
from src.agents.researcher import create_researcher_node
from src.core.models import (
    AgentRole,
    BudgetUsage,
    TraceEntry,
    WorkflowConfig,
    WorkflowStatus,
)
from src.core.state import WorkflowState
from src.graph.guards import run_all_guards
from src.tools.tool_registry import tool_registry

__all__ = [
    "_guard_node",
    "_router_node",
    "build_workflow",
    "get_initial_state",
]

logger = logging.getLogger(__name__)


def _guard_node(state: WorkflowState) -> Command:
    """Safety guard node — checks step limits, budget, and loops before proceeding.

    Args:
        state: Current workflow execution state.

    Returns:
        Command advancing to 'planner' or terminating at '__end__' on failure.
    """
    config = WorkflowConfig()  # Uses defaults, can be overridden
    safe, error_msg = run_all_guards(state, config)

    if not safe:
        trace = TraceEntry(
            entry_type="guard",
            agent="guard",
            content=f"Guard tripped: {error_msg}",
        )
        return Command(
            update={
                "status": WorkflowStatus.FAILED,
                "error": error_msg,
                "trace_log": [trace],
                "messages": [
                    AIMessage(content=f"Workflow stopped by safety guard: {error_msg}")
                ],
            },
            goto="__end__",
        )

    return Command(
        update={"step_count": 1},
        goto="planner",
    )


def _router_node(state: WorkflowState) -> Command:
    """Route to the appropriate worker agent based on the current subtask.

    Args:
        state: Current workflow execution state.

    Returns:
        Command routing to the appropriate agent node or terminating at '__end__'.
    """
    plan = state.get("plan", [])
    task_index = state.get("current_task_index", 0)

    # Run guards before routing
    config = WorkflowConfig()
    safe, error_msg = run_all_guards(state, config)
    if not safe:
        return Command(
            update={
                "status": WorkflowStatus.FAILED,
                "error": error_msg,
                "trace_log": [
                    TraceEntry(
                        entry_type="guard",
                        agent="router",
                        content=f"Guard tripped: {error_msg}",
                    )
                ],
                "messages": [AIMessage(content=f"Workflow stopped: {error_msg}")],
            },
            goto="__end__",
        )

    if task_index >= len(plan):
        return Command(
            update={
                "status": WorkflowStatus.COMPLETED,
                "step_count": 1,
                "messages": [AIMessage(content="All tasks completed.")],
            },
            goto="__end__",
        )

    current_task = plan[task_index]
    if isinstance(current_task, dict):
        agent_role = current_task.get("assigned_agent")
        task_desc = str(current_task.get("description", ""))
        task_id = str(current_task.get("id", ""))
    else:
        agent_role = current_task.assigned_agent
        task_desc = current_task.description
        task_id = current_task.id

    if isinstance(agent_role, str):
        agent_role = AgentRole(agent_role)

    # Map agent role to node name
    role_to_node = {
        AgentRole.RESEARCHER: "researcher",
        AgentRole.CODER: "coder",
        AgentRole.DB_ANALYST: "db_analyst",
    }

    target = role_to_node.get(agent_role)
    if not target:
        return Command(
            update={
                "step_count": 1,
                "trace_log": [
                    TraceEntry(
                        entry_type="error",
                        agent="router",
                        content=f"Unknown agent role: {agent_role}",
                    )
                ],
                "messages": [
                    AIMessage(content=f"Unknown agent: {agent_role}, skipping task")
                ],
                "current_task_index": task_index + 1,
            },
            goto="router",  # Try next task
        )

    trace = TraceEntry(
        entry_type="handoff",
        agent="router",
        content=f"Routing task '{task_desc}' to {target}",
        metadata={"task_id": task_id, "task_index": task_index},
    )

    return Command(
        update={
            "step_count": 1,
            "trace_log": [trace],
        },
        goto=target,
    )


async def build_workflow(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the multi-agent workflow graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
            If None, uses MemorySaver for in-memory state.

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Load tools for each agent role
    researcher_tools = await tool_registry.load_tools_for_role("researcher")
    coder_tools = await tool_registry.load_tools_for_role("coder")
    db_tools = await tool_registry.load_tools_for_role("db_analyst")

    # Create agent nodes
    researcher_node = create_researcher_node(researcher_tools)
    coder_node = create_coder_node(coder_tools)
    db_analyst_node = create_db_analyst_node(db_tools)

    # Build graph
    builder = StateGraph(WorkflowState)

    # Add nodes
    builder.add_node("guard", _guard_node)
    builder.add_node("planner", planner_node)
    builder.add_node("router", _router_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("db_analyst", db_analyst_node)
    builder.add_node("evaluator", evaluator_node)

    # Add tool nodes
    if researcher_tools:
        builder.add_node("researcher_tools", ToolNode(researcher_tools))
        builder.add_edge("researcher_tools", "researcher")
    if coder_tools:
        builder.add_node("coder_tools", ToolNode(coder_tools))
        builder.add_edge("coder_tools", "coder")
    if db_tools:
        builder.add_node("db_tools", ToolNode(db_tools))
        builder.add_edge("db_tools", "db_analyst")

    # Entry point
    builder.add_edge(START, "guard")

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    # interrupt_before on destructive tool nodes for human approval
    interrupt_nodes = ["db_tools"] if "db_tools" in builder.nodes else []
    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if interrupt_nodes:
        compile_kwargs["interrupt_before"] = interrupt_nodes

    graph = builder.compile(**compile_kwargs)

    return graph


def get_initial_state(goal: str, run_id: str | None = None) -> dict[str, Any]:
    """Create the initial state for a new workflow run.

    Args:
        goal: Target objective or prompt that initiates the workflow.
        run_id: Optional unique identifier for this workflow execution.

    Returns:
        Initial state dictionary compliant with WorkflowState.
    """
    return {
        "messages": [],
        "plan": [],
        "current_task_index": 0,
        "results": {},
        "step_count": 0,
        "budget": BudgetUsage(),
        "status": WorkflowStatus.RUNNING,
        "error": None,
        "trace_log": [],
        "goal": goal,
        "run_id": run_id or str(uuid.uuid4()),
    }
