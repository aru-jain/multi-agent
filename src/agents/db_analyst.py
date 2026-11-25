"""Database analyst agent that inspects, queries, and modifies databases."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import Command

from src.core.config import get_settings
from src.core.models import TraceEntry
from src.core.state import WorkflowState

__all__ = [
    "DB_ANALYST_SYSTEM_PROMPT",
    "create_db_analyst_node",
]

logger = logging.getLogger(__name__)

DB_ANALYST_SYSTEM_PROMPT = """You are a database analyst agent. Your job is to query and manage relational database data.

Guidelines:
- Always explain your query intent before calling database tools.
- Use query_db for read-only queries (SELECT, schema inspection, aggregations).
- Use write_db ONLY when data modification (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER) is strictly necessary.
- Note: write_db operations are destructive and require human approval before execution.
- Format tabular outputs cleanly and provide clear summaries of rows returned or affected.
- If a query fails due to missing tables or syntax, inspect the schema first and adjust the query.
- When done, provide a clear final summary of your findings and database actions."""


def create_db_analyst_node(tools: list[Any]) -> Callable[[WorkflowState], Command]:
    """Create a database analyst agent node with the given tools.

    Args:
        tools: List of LangChain tools (query_db, write_db) available to the agent.

    Returns:
        A callable LangGraph node function for the database analyst agent.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    ).bind_tools(tools)

    def db_analyst_node(state: WorkflowState) -> Command:
        """Database analyst agent node that queries and writes to databases.

        Args:
            state: Current workflow execution state.

        Returns:
            Command routing to 'db_tools' if tool calls are requested,
            or to 'evaluator' when processing is complete or encounters an error.
        """
        plan = state.get("plan", [])
        task_index = state.get("current_task_index", 0)
        current_task = plan[task_index] if 0 <= task_index < len(plan) else None

        task_desc = current_task.description if current_task else "Perform database operations"

        # Build context from previous results
        results = state.get("results", {})
        context = ""
        if results:
            context = f"\n\nContext from previous tasks:\n{', '.join(f'{k}: {v}' for k, v in results.items())}"

        messages = [
            SystemMessage(content=DB_ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task_desc}{context}"),
        ] + state.get("messages", [])[-5:]

        try:
            response = llm.invoke(messages)

            tool_names = [tc.get("name", "") for tc in getattr(response, "tool_calls", [])]
            requires_approval = any(name == "write_db" for name in tool_names)

            trace = TraceEntry(
                entry_type="agent_decision",
                agent="db_analyst",
                content=f"DB analyst processing: {task_desc}",
                metadata={
                    "has_tool_calls": bool(response.tool_calls),
                    "tool_calls": tool_names,
                    "requires_approval": requires_approval,
                },
            )

            if response.tool_calls:
                return Command(
                    update={
                        "messages": [response],
                        "step_count": 1,
                        "trace_log": [trace],
                    },
                    goto="db_tools",
                )
            else:
                task_id = current_task.id if current_task else "unknown"
                result_content = response.content
                if isinstance(result_content, list):
                    result_content = " ".join(
                        item if isinstance(item, str) else str(item) for item in result_content
                    )
                return Command(
                    update={
                        "messages": [response],
                        "results": {task_id: result_content},
                        "step_count": 1,
                        "trace_log": [trace],
                    },
                    goto="evaluator",
                )
        except Exception as e:
            logger.error("DB analyst failed: %s", e)
            task_id = current_task.id if current_task else "unknown"
            error_msg = f"Database operation failed: {str(e)}"
            return Command(
                update={
                    "messages": [AIMessage(content=error_msg)],
                    "results": {task_id: error_msg},
                    "step_count": 1,
                    "trace_log": [
                        TraceEntry(
                            entry_type="error",
                            agent="db_analyst",
                            content=error_msg,
                        )
                    ],
                },
                goto="evaluator",
            )

    return db_analyst_node
