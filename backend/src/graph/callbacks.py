"""Token counting, cost tracking, and execution tracing callbacks for workflows."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.core.models import BudgetUsage, TraceEntry

__all__ = [
    "MODEL_PRICING",
    "WorkflowCallbackHandler",
]

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (Groq — free tier has $0 cost)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.40},
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.60},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


class WorkflowCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks tokens, costs, and creates trace entries."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b") -> None:
        """Initialize the callback handler with the target model pricing.

        Args:
            model_name: Name of the LLM model to use for pricing lookup.
        """
        super().__init__()
        self.model_name = model_name
        self.budget = BudgetUsage()
        self.trace_entries: list[TraceEntry] = []
        self._call_start_times: dict[str, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record the start time of an LLM call.

        Args:
            serialized: Serialized representation of the LLM.
            prompts: List of prompt strings.
            run_id: Unique execution identifier for this LLM invocation.
            **kwargs: Additional callback arguments.
        """
        self._call_start_times[str(run_id)] = time.time()

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Extract token usage, compute cost, and append trace entry on LLM completion.

        Args:
            response: Execution result from the LLM.
            run_id: Unique execution identifier for this LLM invocation.
            **kwargs: Additional callback arguments.
        """
        elapsed = time.time() - self._call_start_times.pop(str(run_id), time.time())

        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0

        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            # Try to get from generation info
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        usage = gen.generation_info.get("usage") or gen.generation_info.get("token_usage") or {}
                        prompt_tokens += usage.get("prompt_tokens", 0)
                        completion_tokens += usage.get("completion_tokens", 0)

        self.budget.prompt_tokens += prompt_tokens
        self.budget.completion_tokens += completion_tokens
        self.budget.total_tokens += prompt_tokens + completion_tokens

        # Calculate cost
        pricing = MODEL_PRICING.get(self.model_name, MODEL_PRICING["openai/gpt-oss-120b"])
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000
        self.budget.total_cost_usd += cost

        self.trace_entries.append(
            TraceEntry(
                entry_type="tool_result",
                agent="llm",
                content=f"LLM call completed in {elapsed:.2f}s",
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": round(cost, 6),
                    "model": self.model_name,
                },
            )
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record start time and create a trace entry for tool invocation.

        Args:
            serialized: Serialized tool dictionary.
            input_str: Input arguments passed to the tool.
            run_id: Unique execution identifier for this tool call.
            **kwargs: Additional callback arguments.
        """
        tool_name = serialized.get("name", "unknown")
        self._call_start_times[str(run_id)] = time.time()
        self.budget.tool_calls += 1

        self.trace_entries.append(
            TraceEntry(
                entry_type="tool_call",
                content=f"Calling tool: {tool_name}",
                metadata={
                    "tool": tool_name,
                    "input_preview": str(input_str)[:200],
                },
            )
        )

    def on_tool_end(
        self, output: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Record execution duration and result preview on tool completion.

        Args:
            output: Output string returned by the tool.
            run_id: Unique execution identifier for this tool call.
            **kwargs: Additional callback arguments.
        """
        elapsed = time.time() - self._call_start_times.pop(str(run_id), time.time())
        self.trace_entries.append(
            TraceEntry(
                entry_type="tool_result",
                content=f"Tool completed in {elapsed:.2f}s",
                metadata={"output_preview": str(output)[:300]},
            )
        )

    def get_budget(self) -> BudgetUsage:
        """Retrieve the current cumulative budget usage.

        Returns:
            BudgetUsage model instance.
        """
        return self.budget

    def get_trace(self) -> list[TraceEntry]:
        """Retrieve a copy of all accumulated trace entries.

        Returns:
            List of TraceEntry objects.
        """
        return list(self.trace_entries)

    def reset(self) -> None:
        """Reset all tracked metrics, traces, and timing records."""
        self.budget = BudgetUsage()
        self.trace_entries.clear()
        self._call_start_times.clear()
