"""Coder agent that writes, tests, and debugs Python code using execution and linting tools."""

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
    "CODER_SYSTEM_PROMPT",
    "create_coder_node",
]

logger = logging.getLogger(__name__)

CODER_SYSTEM_PROMPT = """You are a senior Python software engineer. Your job is to write, test, and debug Python code to accomplish tasks.

Guidelines:
- Use execute_python to run Python code in an isolated environment.
- Use lint_code to verify code syntax, style, and potential issues before or after execution.
- If code fails with an error or incorrect output, inspect the error message carefully, fix the code, and re-execute.
- Iterate and fix errors up to 3 times if needed.
- Write clean, type-annotated, Python 3.11+ compliant code.
- Always check that imports and functions are properly defined.
- When done, provide a clear final summary of your solution and execution output."""


def create_coder_node(tools: list[Any]) -> Callable[[WorkflowState], Command]:
    """Create a coder agent node with the given tools.

    Args:
        tools: List of LangChain tools (execute_python, lint_code) available to the agent.

    Returns:
        A callable LangGraph node function for the coder agent.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    ).bind_tools(tools)

    def coder_node(state: WorkflowState) -> Command:
        """Coder agent node that writes and executes Python code.

        Args:
            state: Current workflow execution state.

        Returns:
            Command routing to 'coder_tools' if tool calls are requested,
            or to 'evaluator' when processing is complete or encounters an error.
        """
        plan = state.get("plan", [])
        task_index = state.get("current_task_index", 0)
        current_task = plan[task_index] if 0 <= task_index < len(plan) else None

        task_desc = current_task.description if current_task else "Implement the requested code"

        # Build context from previous results
        results = state.get("results", {})
        context = ""
        if results:
            context = f"\n\nContext from previous tasks:\n{', '.join(f'{k}: {v}' for k, v in results.items())}"

        messages = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task_desc}{context}"),
        ] + state.get("messages", [])[-5:]

        try:
            response = llm.invoke(messages)

            trace = TraceEntry(
                entry_type="agent_decision",
                agent="coder",
                content=f"Coder processing: {task_desc}",
                metadata={
                    "has_tool_calls": bool(response.tool_calls),
                    "tool_calls": [tc.get("name", "") for tc in getattr(response, "tool_calls", [])],
                },
            )

            if response.tool_calls:
                return Command(
                    update={
                        "messages": [response],
                        "step_count": 1,
                        "trace_log": [trace],
                    },
                    goto="coder_tools",
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
            logger.error("Coder failed: %s", e)
            task_id = current_task.id if current_task else "unknown"
            error_msg = f"Coding task failed: {str(e)}"
            return Command(
                update={
                    "messages": [AIMessage(content=error_msg)],
                    "results": {task_id: error_msg},
                    "step_count": 1,
                    "trace_log": [
                        TraceEntry(
                            entry_type="error",
                            agent="coder",
                            content=error_msg,
                        )
                    ],
                },
                goto="evaluator",
            )

    return coder_node
