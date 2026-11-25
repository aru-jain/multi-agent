"""Researcher agent that gathers information using web search and URL fetching tools."""

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
    "RESEARCHER_SYSTEM_PROMPT",
    "create_researcher_node",
]

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a research agent. Your job is to find information using web search and URL fetching tools.

Guidelines:
- Use web_search to find relevant information.
- Use fetch_url to get detailed content from specific pages when needed.
- Summarize your findings clearly and concisely.
- If a search returns no useful results, try rephrasing the query.
- Always provide structured, factual summaries of what you find.
- When done, provide a clear final summary of all findings."""


def create_researcher_node(tools: list[Any]) -> Callable[[WorkflowState], Command]:
    """Create a researcher agent node with the given tools.

    Args:
        tools: List of LangChain tools (web_search, fetch_url) available to the agent.

    Returns:
        A callable LangGraph node function for the researcher agent.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    ).bind_tools(tools)

    def researcher_node(state: WorkflowState) -> Command:
        """Research agent node that searches the web for information.

        Args:
            state: Current workflow execution state.

        Returns:
            Command routing to 'researcher_tools' if tool calls are requested,
            or to 'evaluator' when processing is complete or encounters an error.
        """
        plan = state.get("plan", [])
        task_index = state.get("current_task_index", 0)
        current_task = plan[task_index] if 0 <= task_index < len(plan) else None

        task_desc = current_task.description if current_task else "Research the topic"

        # Build context from previous results
        results = state.get("results", {})
        context = ""
        if results:
            context = f"\n\nContext from previous tasks:\n{', '.join(f'{k}: {v}' for k, v in results.items())}"

        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task_desc}{context}"),
        ] + state.get("messages", [])[-5:]

        try:
            response = llm.invoke(messages)

            trace = TraceEntry(
                entry_type="agent_decision",
                agent="researcher",
                content=f"Researcher processing: {task_desc}",
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
                    goto="researcher_tools",
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
            logger.error("Researcher failed: %s", e)
            task_id = current_task.id if current_task else "unknown"
            error_msg = f"Research failed: {str(e)}"
            return Command(
                update={
                    "messages": [AIMessage(content=error_msg)],
                    "results": {task_id: error_msg},
                    "step_count": 1,
                    "trace_log": [
                        TraceEntry(
                            entry_type="error",
                            agent="researcher",
                            content=error_msg,
                        )
                    ],
                },
                goto="evaluator",
            )

    return researcher_node
