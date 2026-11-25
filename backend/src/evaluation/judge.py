"""LLM-as-judge evaluator for multi-agent workflow outputs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.core.config import get_settings
from src.evaluation.scenarios import EvalScenario

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "judge_result",
    "judge_all_results",
    "format_judge_report",
]

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator judging the output of an AI agent workflow.

You will be given:
- The original goal/task
- The success criteria
- The agent's output/results

Score the output on three dimensions (1-5 scale each):
1. **Relevance**: Does the output address the original goal? (1=completely off-topic, 5=perfectly relevant)
2. **Completeness**: Is the output thorough and complete? (1=missing everything, 5=fully complete)
3. **Correctness**: Is the information/code factually correct? (1=entirely wrong, 5=perfectly correct)

Respond with ONLY a JSON object:
{
    "relevance": <1-5>,
    "completeness": <1-5>,
    "correctness": <1-5>,
    "overall_pass": <true/false>,
    "explanation": "<brief explanation>"
}

Set overall_pass to true if the output meaningfully addresses the goal (even if not perfect)."""


async def judge_result(
    scenario: EvalScenario,
    output: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Use an LLM to judge the quality of a scenario's output.

    Uses a separate LLM call to avoid self-evaluation bias.

    Args:
        scenario: The evaluation scenario definition.
        output: The agent workflow's output to evaluate.
        model: Override model name (defaults to settings.openai_model).

    Returns:
        Dictionary with scores (relevance, completeness, correctness),
        overall_pass boolean flag, and explanation.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=model or settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Goal: {scenario.goal}\n\n"
                f"Success Criteria: {scenario.success_criteria}\n\n"
                f"Agent Output:\n{output[:3000]}"
            )
        ),
    ]

    try:
        invoked = llm.invoke(messages)
        if inspect.isawaitable(invoked):
            response = await invoked
        else:
            response = invoked

        # Extract text content
        if isinstance(response.content, str):
            content = response.content.strip()
        elif isinstance(response.content, list):
            content = "".join(
                p if isinstance(p, str) else p.get("text", "") for p in response.content
            ).strip()
        else:
            content = str(response.content).strip()

        # Parse JSON, handling optional markdown code fences
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

        result = json.loads(content)

        # Validate scores
        for key in ["relevance", "completeness", "correctness"]:
            if key in result:
                result[key] = max(1, min(5, int(result[key])))

        if "overall_pass" in result:
            result["overall_pass"] = bool(result["overall_pass"])

        return result
    except Exception as e:
        logger.error("Judge evaluation failed for scenario %d: %s", scenario.id, e)
        return {
            "relevance": 0,
            "completeness": 0,
            "correctness": 0,
            "overall_pass": False,
            "explanation": f"Judge evaluation failed: {str(e)}",
        }


async def judge_all_results(
    scenarios_and_outputs: list[tuple[EvalScenario, str]],
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Judge multiple scenario results sequentially.

    Args:
        scenarios_and_outputs: List of (scenario, output_text) tuples.
        model: Optional model identifier override.

    Returns:
        List of score dictionaries augmented with scenario metadata.
    """
    results: list[dict[str, Any]] = []
    for scenario, output in scenarios_and_outputs:
        result = await judge_result(scenario, output, model=model)
        result["scenario_id"] = scenario.id
        result["scenario_name"] = scenario.name
        results.append(result)
    return results


def format_judge_report(results: list[dict[str, Any]]) -> str:
    """Format judge evaluation results as a markdown table.

    Args:
        results: List of result dictionaries from judge_result or judge_all_results.

    Returns:
        Markdown table string summarizing the evaluations.
    """
    lines = [
        "# LLM Judge Evaluation\n",
        "| # | Scenario | Relevance | Completeness | Correctness | Pass | Explanation |",
        "|---|---------|-----------|--------------|-------------|------|-------------|",
    ]

    for r in results:
        emoji = "✅" if r.get("overall_pass") else "❌"
        explanation = (r.get("explanation", "") or "")[:60]
        lines.append(
            f"| {r.get('scenario_id', '?')} | {r.get('scenario_name', '?')} | "
            f"{r.get('relevance', 0)}/5 | {r.get('completeness', 0)}/5 | "
            f"{r.get('correctness', 0)}/5 | {emoji} | {explanation} |"
        )

    # Averages
    if results:
        avg_rel = sum(r.get("relevance", 0) for r in results) / len(results)
        avg_comp = sum(r.get("completeness", 0) for r in results) / len(results)
        avg_corr = sum(r.get("correctness", 0) for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.get("overall_pass")) / len(results) * 100

        lines.append("|---|---------|-----------|--------------|-------------|------|-------------|")
        lines.append(
            f"| | **AVERAGE** | **{avg_rel:.1f}/5** | **{avg_comp:.1f}/5** | "
            f"**{avg_corr:.1f}/5** | **{pass_rate:.0f}%** | |"
        )

    return "\n".join(lines)
