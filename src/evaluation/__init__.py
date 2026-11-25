"""Evaluation framework for the Multi-Agent AI Workflow system."""

from __future__ import annotations

from src.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    format_judge_report,
    judge_all_results,
    judge_result,
)
from src.evaluation.runner import (
    EvaluationReport,
    ScenarioResult,
    format_report_json,
    format_report_markdown,
    main,
    run_all_scenarios,
    run_scenario,
)
from src.evaluation.scenarios import (
    SCENARIOS,
    EvalScenario,
    get_scenario,
    get_scenarios_by_difficulty,
)

__all__ = [
    # Scenarios
    "EvalScenario",
    "SCENARIOS",
    "get_scenario",
    "get_scenarios_by_difficulty",
    # Runner
    "ScenarioResult",
    "EvaluationReport",
    "run_scenario",
    "run_all_scenarios",
    "format_report_markdown",
    "format_report_json",
    "main",
    # Judge
    "JUDGE_SYSTEM_PROMPT",
    "judge_result",
    "judge_all_results",
    "format_judge_report",
]
