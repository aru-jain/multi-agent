"""Unit tests for the evaluation framework (scenarios, runner, judge)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from src.core.models import BudgetUsage, WorkflowStatus
from src.evaluation import (
    JUDGE_SYSTEM_PROMPT,
    SCENARIOS,
    EvalScenario,
    EvaluationReport,
    ScenarioResult,
    format_judge_report,
    format_report_json,
    format_report_markdown,
    get_scenario,
    get_scenarios_by_difficulty,
    judge_all_results,
    judge_result,
    run_all_scenarios,
    run_scenario,
)


class TestScenarios:
    """Tests for evaluation scenarios catalog and lookup helpers."""

    def test_total_scenario_count(self) -> None:
        """Verify exactly 20 scenarios are configured with IDs 1 to 20."""
        assert len(SCENARIOS) == 20
        ids = [s.id for s in SCENARIOS]
        assert ids == list(range(1, 21))

    def test_get_scenario_found(self) -> None:
        """Verify scenario lookup by existing ID."""
        s = get_scenario(1)
        assert s is not None
        assert s.id == 1
        assert s.name == "Simple factual query"
        assert s.difficulty == "easy"
        assert "Paris" in s.success_criteria

    def test_get_scenario_not_found(self) -> None:
        """Verify scenario lookup with non-existent ID returns None."""
        assert get_scenario(999) is None

    def test_get_scenarios_by_difficulty(self) -> None:
        """Verify filtering scenarios by difficulty level."""
        easy = get_scenarios_by_difficulty("easy")
        medium = get_scenarios_by_difficulty("medium")
        hard = get_scenarios_by_difficulty("hard")
        stress = get_scenarios_by_difficulty("stress")

        assert len(easy) == 5
        assert len(medium) == 6
        assert len(hard) == 7
        assert len(stress) == 2
        assert len(easy) + len(medium) + len(hard) + len(stress) == 20

    def test_stress_scenarios_have_overrides(self) -> None:
        """Verify stress scenarios specify limit overrides."""
        scenario_19 = get_scenario(19)
        assert scenario_19 is not None
        assert scenario_19.max_steps_override == 15

        scenario_20 = get_scenario(20)
        assert scenario_20 is not None
        assert scenario_20.max_tokens_override == 5000

    def test_injected_failure_scenario(self) -> None:
        """Verify scenario 17 defines inject_failure."""
        scenario_17 = get_scenario(17)
        assert scenario_17 is not None
        assert scenario_17.inject_failure == "web_search"


class TestRunner:
    """Tests for the evaluation runner and report generation."""

    def test_scenario_result_defaults(self) -> None:
        """Verify default field values in ScenarioResult."""
        res = ScenarioResult(scenario_id=1, scenario_name="Test", success=True)
        assert res.steps == 0
        assert res.tokens == 0
        assert res.cost_usd == 0.0
        assert res.error is None

    def test_format_report_markdown(self) -> None:
        """Verify markdown report generation."""
        report = EvaluationReport(
            results=[
                ScenarioResult(
                    scenario_id=1,
                    scenario_name="Test 1",
                    success=True,
                    steps=3,
                    tokens=150,
                    cost_usd=0.001,
                    wall_time_seconds=2.5,
                ),
                ScenarioResult(
                    scenario_id=2,
                    scenario_name="Test 2",
                    success=False,
                    steps=5,
                    tokens=250,
                    cost_usd=0.002,
                    wall_time_seconds=3.0,
                    error="Failed step",
                ),
            ],
            total_scenarios=2,
            successful=1,
            failed=1,
            success_rate=50.0,
            avg_steps=4.0,
            avg_tokens=200.0,
            avg_cost_usd=0.0015,
            avg_wall_time=2.8,
            total_cost_usd=0.003,
        )
        md = format_report_markdown(report)
        assert "# Evaluation Results" in md
        assert "Test 1" in md
        assert "Test 2" in md
        assert "✅" in md
        assert "❌" in md
        assert "50.0%" in md

    def test_format_report_json(self) -> None:
        """Verify JSON report formatting."""
        report = EvaluationReport(
            results=[
                ScenarioResult(
                    scenario_id=1,
                    scenario_name="Test 1",
                    success=True,
                    steps=3,
                    tokens=150,
                    cost_usd=0.001,
                    wall_time_seconds=2.5,
                )
            ],
            total_scenarios=1,
            successful=1,
            failed=0,
            success_rate=100.0,
            avg_steps=3.0,
            avg_tokens=150.0,
            avg_cost_usd=0.001,
            avg_wall_time=2.5,
            total_cost_usd=0.001,
        )
        json_str = format_report_json(report)
        parsed = json.loads(json_str)
        assert parsed["summary"]["total_scenarios"] == 1
        assert parsed["summary"]["successful"] == 1
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["scenario_name"] == "Test 1"

    @pytest.mark.asyncio
    @patch("src.evaluation.runner.build_workflow")
    async def test_run_scenario_success(self, mock_build_workflow: MagicMock) -> None:
        """Verify run_scenario succeeds when graph execution completes successfully."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "status": WorkflowStatus.COMPLETED,
                "step_count": 4,
                "budget": BudgetUsage(total_tokens=500, total_cost_usd=0.005),
                "trace_log": ["t1", "t2"],
                "results": {"answer": "Paris"},
                "error": None,
            }
        )
        mock_build_workflow.return_value = mock_graph

        scenario = get_scenario(1)
        assert scenario is not None
        res = await run_scenario(scenario)

        assert res.success is True
        assert res.steps == 4
        assert res.tokens == 500
        assert res.cost_usd == 0.005
        assert res.error is None
        assert "Paris" in res.output_summary

    @pytest.mark.asyncio
    @patch("src.evaluation.runner.build_workflow")
    async def test_run_scenario_stress_guard_pass(self, mock_build_workflow: MagicMock) -> None:
        """Verify stress tests consider tripped safety guards as passing."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "status": WorkflowStatus.FAILED,
                "step_count": 15,
                "budget": BudgetUsage(total_tokens=6000),
                "trace_log": [],
                "results": {},
                "error": "Step limit exceeded: 15 >= 15",
            }
        )
        mock_build_workflow.return_value = mock_graph

        scenario = get_scenario(19)
        assert scenario is not None
        res = await run_scenario(scenario)

        assert res.success is True
        assert "Step limit exceeded" in (res.error or "")

    @pytest.mark.asyncio
    @patch("src.evaluation.runner.build_workflow")
    async def test_run_scenario_timeout(self, mock_build_workflow: MagicMock) -> None:
        """Verify scenario timeout produces a failed ScenarioResult."""
        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(2)
            return {}

        mock_graph = MagicMock()
        mock_graph.ainvoke = slow_invoke
        mock_build_workflow.return_value = mock_graph

        scenario = get_scenario(1)
        assert scenario is not None
        res = await run_scenario(scenario, timeout=0.1)

        assert res.success is False
        assert "timed out" in (res.error or "")

    @pytest.mark.asyncio
    @patch("src.evaluation.runner.build_workflow")
    async def test_run_scenario_exception_handling(self, mock_build_workflow: MagicMock) -> None:
        """Verify unexpected errors in workflow execution are caught cleanly."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph compilation crashed"))
        mock_build_workflow.return_value = mock_graph

        scenario = get_scenario(1)
        assert scenario is not None
        res = await run_scenario(scenario)

        assert res.success is False
        assert "Graph compilation crashed" in (res.error or "")

    @pytest.mark.asyncio
    @patch("src.evaluation.runner.run_scenario")
    async def test_run_all_scenarios_filtered(self, mock_run_scenario: MagicMock) -> None:
        """Verify run_all_scenarios filters by scenario IDs and aggregates metrics."""
        mock_run_scenario.side_effect = [
            ScenarioResult(scenario_id=1, scenario_name="S1", success=True, steps=2, tokens=100, cost_usd=0.001),
            ScenarioResult(scenario_id=2, scenario_name="S2", success=False, steps=4, tokens=200, cost_usd=0.002),
        ]

        report = await run_all_scenarios(scenario_ids=[1, 2])
        assert report.total_scenarios == 2
        assert report.successful == 1
        assert report.failed == 1
        assert report.success_rate == 50.0
        assert report.avg_steps == 3.0
        assert report.total_cost_usd == 0.003


class TestJudge:
    """Tests for LLM-as-judge evaluation logic."""

    @pytest.mark.asyncio
    @patch("src.evaluation.judge.ChatGroq")
    async def test_judge_result_success(self, mock_llm_cls: MagicMock) -> None:
        """Verify judge_result parses clean JSON LLM evaluation."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        judge_json = json.dumps({
            "relevance": 5,
            "completeness": 4,
            "correctness": 5,
            "overall_pass": True,
            "explanation": "Excellent and accurate answer.",
        })
        mock_llm.invoke.return_value = AIMessage(content=judge_json)

        scenario = get_scenario(1)
        assert scenario is not None
        res = await judge_result(scenario, "Paris is the capital of France.")

        assert res["relevance"] == 5
        assert res["completeness"] == 4
        assert res["correctness"] == 5
        assert res["overall_pass"] is True
        assert "Excellent" in res["explanation"]

    @pytest.mark.asyncio
    @patch("src.evaluation.judge.ChatGroq")
    async def test_judge_result_markdown_fence_and_score_clamping(self, mock_llm_cls: MagicMock) -> None:
        """Verify judge_result handles markdown fences and clamps scores to 1-5."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        wrapped = "```json\n{\"relevance\": 10, \"completeness\": 0, \"correctness\": 4, \"overall_pass\": true, \"explanation\": \"Good\"}\n```"
        mock_llm.invoke.return_value = AIMessage(content=wrapped)

        scenario = get_scenario(1)
        assert scenario is not None
        res = await judge_result(scenario, "Some output")

        assert res["relevance"] == 5  # Clamped to 5
        assert res["completeness"] == 1  # Clamped to 1
        assert res["correctness"] == 4
        assert res["overall_pass"] is True

    @pytest.mark.asyncio
    @patch("src.evaluation.judge.ChatGroq")
    async def test_judge_result_failure_handling(self, mock_llm_cls: MagicMock) -> None:
        """Verify judge_result handles LLM invocation failure gracefully."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.side_effect = RuntimeError("OpenAI API rate limited")

        scenario = get_scenario(1)
        assert scenario is not None
        res = await judge_result(scenario, "Some output")

        assert res["relevance"] == 0
        assert res["overall_pass"] is False
        assert "failed" in res["explanation"].lower()

    @pytest.mark.asyncio
    @patch("src.evaluation.judge.judge_result")
    async def test_judge_all_results(self, mock_judge_result: MagicMock) -> None:
        """Verify judge_all_results iterates over items and attaches scenario metadata."""
        mock_judge_result.side_effect = [
            {"relevance": 5, "completeness": 5, "correctness": 5, "overall_pass": True, "explanation": "Great"},
            {"relevance": 2, "completeness": 2, "correctness": 2, "overall_pass": False, "explanation": "Poor"},
        ]

        s1 = get_scenario(1)
        s2 = get_scenario(2)
        assert s1 is not None and s2 is not None

        items = [(s1, "out 1"), (s2, "out 2")]
        results = await judge_all_results(items)

        assert len(results) == 2
        assert results[0]["scenario_id"] == 1
        assert results[0]["overall_pass"] is True
        assert results[1]["scenario_id"] == 2
        assert results[1]["overall_pass"] is False

    def test_format_judge_report(self) -> None:
        """Verify format_judge_report generates valid markdown table."""
        judge_entries = [
            {
                "scenario_id": 1,
                "scenario_name": "Simple factual query",
                "relevance": 5,
                "completeness": 5,
                "correctness": 5,
                "overall_pass": True,
                "explanation": "Correct",
            },
            {
                "scenario_id": 2,
                "scenario_name": "Weather lookup",
                "relevance": 3,
                "completeness": 3,
                "correctness": 3,
                "overall_pass": False,
                "explanation": "Incomplete",
            },
        ]
        md = format_judge_report(judge_entries)
        assert "# LLM Judge Evaluation" in md
        assert "Simple factual query" in md
        assert "Weather lookup" in md
        assert "AVERAGE" in md
        assert "50%" in md
