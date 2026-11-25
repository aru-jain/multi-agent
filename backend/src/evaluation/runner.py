"""Evaluation runner for executing multi-agent workflow test scenarios."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from src.core.models import BudgetUsage, WorkflowConfig, WorkflowStatus
from src.evaluation.scenarios import SCENARIOS, EvalScenario
from src.graph.workflow import build_workflow, get_initial_state

__all__ = [
    "ScenarioResult",
    "EvaluationReport",
    "run_scenario",
    "run_all_scenarios",
    "format_report_markdown",
    "format_report_json",
    "main",
]

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a single evaluation scenario.

    Attributes:
        scenario_id: Identifier of the evaluated scenario.
        scenario_name: Human-readable scenario name.
        success: Whether the scenario completed successfully or met criteria.
        steps: Total workflow execution steps taken.
        tokens: Total LLM tokens consumed.
        cost_usd: Estimated financial cost in USD.
        wall_time_seconds: Elapsed execution time in seconds.
        error: Error message if execution encountered a failure.
        output_summary: Truncated JSON summary of results.
        trace_count: Number of trace entries recorded.
    """

    scenario_id: int
    scenario_name: str
    success: bool
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    error: str | None = None
    output_summary: str = ""
    trace_count: int = 0


@dataclass
class EvaluationReport:
    """Full evaluation report across all scenarios.

    Attributes:
        results: List of individual scenario results.
        total_scenarios: Number of scenarios evaluated.
        successful: Number of scenarios passing criteria.
        failed: Number of scenarios failing criteria.
        success_rate: Percentage of scenarios that passed.
        avg_steps: Average execution steps across all scenarios.
        avg_tokens: Average token usage across all scenarios.
        avg_cost_usd: Average monetary cost in USD across all scenarios.
        avg_wall_time: Average execution time in seconds across all scenarios.
        total_cost_usd: Total monetary cost across all scenarios.
    """

    results: list[ScenarioResult] = field(default_factory=list)
    total_scenarios: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_tokens: float = 0.0
    avg_cost_usd: float = 0.0
    avg_wall_time: float = 0.0
    total_cost_usd: float = 0.0


async def run_scenario(scenario: EvalScenario, timeout: int = 120) -> ScenarioResult:
    """Run a single evaluation scenario and return its result.

    Args:
        scenario: The evaluation scenario definition to execute.
        timeout: Maximum duration in seconds allowed before aborting.

    Returns:
        A ScenarioResult summarizing the outcome and performance metrics.
    """
    logger.info("Running scenario %d: %s", scenario.id, scenario.name)
    start_time = time.time()

    # Determine if workflow config overrides are needed (e.g. for stress tests)
    config_overrides: dict[str, Any] = {}
    if scenario.max_steps_override is not None:
        config_overrides["max_steps"] = scenario.max_steps_override
    if scenario.max_tokens_override is not None:
        config_overrides["max_tokens"] = scenario.max_tokens_override

    patcher = None
    if config_overrides:
        patcher = patch(
            "src.graph.workflow.WorkflowConfig",
            lambda **kw: WorkflowConfig(**{**config_overrides, **kw}),
        )
        patcher.start()

    try:
        # Build a fresh workflow for each scenario
        graph = await build_workflow()
        initial_state = get_initial_state(goal=scenario.goal)

        config = {"configurable": {"thread_id": f"eval-{scenario.id}"}}

        # Run with timeout
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=timeout,
        )

        wall_time = time.time() - start_time

        # Extract metrics from result
        status = result.get("status", WorkflowStatus.FAILED)
        budget = result.get("budget", BudgetUsage())
        if isinstance(budget, dict):
            budget = BudgetUsage(**budget)

        step_count = result.get("step_count", 0)
        trace_log = result.get("trace_log", [])
        error = result.get("error")
        results_data = result.get("results", {})

        # Determine success
        is_success = (
            str(status) in ("completed", "WorkflowStatus.COMPLETED")
            and error is None
        )

        # For stress tests, guard trips are expected successes
        if scenario.difficulty == "stress" and str(status) in ("failed", "WorkflowStatus.FAILED"):
            if error and ("Step limit" in error or "Budget exceeded" in error or "budget" in error.lower()):
                is_success = True

        output_summary = json.dumps(results_data, default=str)[:500] if results_data else ""

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            success=is_success,
            steps=step_count,
            tokens=budget.total_tokens,
            cost_usd=round(budget.total_cost_usd, 6),
            wall_time_seconds=round(wall_time, 2),
            error=error,
            output_summary=output_summary,
            trace_count=len(trace_log),
        )
    except asyncio.TimeoutError:
        wall_time = time.time() - start_time
        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            success=False,
            wall_time_seconds=round(wall_time, 2),
            error=f"Scenario timed out after {timeout}s",
        )
    except Exception as e:
        wall_time = time.time() - start_time
        logger.error("Scenario %d error: %s", scenario.id, e)
        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            success=False,
            wall_time_seconds=round(wall_time, 2),
            error=str(e),
        )
    finally:
        if patcher is not None:
            patcher.stop()


async def run_all_scenarios(
    scenario_ids: list[int] | None = None,
    timeout_per_scenario: int = 120,
) -> EvaluationReport:
    """Run all (or selected) evaluation scenarios and produce a report.

    Args:
        scenario_ids: Optional list of scenario IDs to filter by.
        timeout_per_scenario: Maximum seconds per scenario execution.

    Returns:
        EvaluationReport containing aggregated and per-scenario metrics.
    """
    scenarios = SCENARIOS
    if scenario_ids:
        scenarios = [s for s in SCENARIOS if s.id in scenario_ids]

    results: list[ScenarioResult] = []

    for scenario in scenarios:
        result = await run_scenario(scenario, timeout=timeout_per_scenario)
        results.append(result)

        status_emoji = "✅" if result.success else "❌"
        logger.info(
            "%s Scenario %d: %s | steps=%d tokens=%d cost=$%s time=%ss",
            status_emoji,
            result.scenario_id,
            result.scenario_name,
            result.steps,
            result.tokens,
            result.cost_usd,
            result.wall_time_seconds,
        )

    # Compute aggregates
    total = len(results)
    successful = sum(1 for r in results if r.success)

    report = EvaluationReport(
        results=results,
        total_scenarios=total,
        successful=successful,
        failed=total - successful,
        success_rate=round((successful / total * 100) if total > 0 else 0, 1),
        avg_steps=round(sum(r.steps for r in results) / total if total > 0 else 0, 1),
        avg_tokens=round(sum(r.tokens for r in results) / total if total > 0 else 0, 0),
        avg_cost_usd=round(sum(r.cost_usd for r in results) / total if total > 0 else 0, 4),
        avg_wall_time=round(sum(r.wall_time_seconds for r in results) / total if total > 0 else 0, 1),
        total_cost_usd=round(sum(r.cost_usd for r in results), 4),
    )

    return report


def format_report_markdown(report: EvaluationReport) -> str:
    """Format an evaluation report as a markdown table.

    Args:
        report: EvaluationReport instance to format.

    Returns:
        A Markdown-formatted string with a summary table and statistics.
    """
    lines = [
        "# Evaluation Results\n",
        "| # | Scenario | Success | Steps | Tokens | Cost ($) | Time (s) | Error |",
        "|---|---------|---------|-------|--------|----------|----------|-------|",
    ]

    for r in report.results:
        emoji = "✅" if r.success else "❌"
        error_short = (r.error or "")[:40]
        lines.append(
            f"| {r.scenario_id} | {r.scenario_name} | {emoji} | {r.steps} | "
            f"{r.tokens:,} | {r.cost_usd:.4f} | {r.wall_time_seconds} | {error_short} |"
        )

    lines.append("|---|---------|---------|-------|--------|----------|----------|-------|")
    lines.append(
        f"| | **AGGREGATE** | **{report.success_rate}%** | **{report.avg_steps}** | "
        f"**{report.avg_tokens:,.0f}** | **{report.avg_cost_usd:.4f}** | "
        f"**{report.avg_wall_time}** | |\n"
    )

    lines.append(f"\n**Total scenarios:** {report.total_scenarios}")
    lines.append(f"**Passed:** {report.successful} | **Failed:** {report.failed}")
    lines.append(f"**Success rate:** {report.success_rate}%")
    lines.append(f"**Total cost:** ${report.total_cost_usd}")

    return "\n".join(lines)


def format_report_json(report: EvaluationReport) -> str:
    """Format an evaluation report as JSON.

    Args:
        report: EvaluationReport instance to format.

    Returns:
        Indented JSON string representation of the report.
    """
    data = {
        "summary": {
            "total_scenarios": report.total_scenarios,
            "successful": report.successful,
            "failed": report.failed,
            "success_rate_pct": report.success_rate,
            "avg_steps": report.avg_steps,
            "avg_tokens": report.avg_tokens,
            "avg_cost_usd": report.avg_cost_usd,
            "avg_wall_time_seconds": report.avg_wall_time,
            "total_cost_usd": report.total_cost_usd,
        },
        "results": [
            {
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "success": r.success,
                "steps": r.steps,
                "tokens": r.tokens,
                "cost_usd": r.cost_usd,
                "wall_time_seconds": r.wall_time_seconds,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    return json.dumps(data, indent=2)


async def main() -> None:
    """CLI entry point to run evaluations."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Parse optional scenario IDs from CLI args
    scenario_ids = None
    if len(sys.argv) > 1:
        scenario_ids = [int(x) for x in sys.argv[1:]]

    report = await run_all_scenarios(scenario_ids=scenario_ids)

    # Print markdown report
    md = format_report_markdown(report)
    print(md)

    # Save reports
    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        f.write(format_report_json(report))

    print("\nReports saved to evaluation_results.md and evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
