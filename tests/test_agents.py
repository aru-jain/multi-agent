"""Unit tests for agent nodes: planner, researcher, coder, db_analyst, and evaluator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import (
    CODER_SYSTEM_PROMPT,
    DB_ANALYST_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    create_coder_node,
    create_db_analyst_node,
    create_researcher_node,
    evaluator_node,
    planner_node,
)
from src.core.models import (
    AgentRole,
    BudgetUsage,
    SubTask,
    TaskStatus,
    WorkflowStatus,
)
from src.core.state import WorkflowState


def _make_base_state(**kwargs) -> WorkflowState:
    """Helper to construct a valid WorkflowState for tests."""
    base: WorkflowState = {
        "messages": [],
        "plan": [],
        "current_task_index": 0,
        "results": {},
        "step_count": 0,
        "budget": BudgetUsage(),
        "status": WorkflowStatus.RUNNING,
        "error": None,
        "trace_log": [],
        "goal": "Test goal",
        "run_id": "test-run-123",
    }
    base.update(kwargs)
    return base


class TestPlannerAgent:
    """Tests for the planner agent node."""

    @patch("src.agents.planner.ChatGroq")
    def test_planner_success(self, mock_llm_cls: MagicMock) -> None:
        """Test planner successfully decomposes a goal into subtasks."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        plan_json = json.dumps([
            {
                "id": "t1",
                "description": "Search documentation",
                "assigned_agent": "researcher",
                "requires_approval": False,
                "dependencies": [],
            },
            {
                "id": "t2",
                "description": "Write code",
                "assigned_agent": "coder",
                "requires_approval": False,
                "dependencies": ["t1"],
            },
        ])
        mock_llm.invoke.return_value = AIMessage(content=plan_json)

        state = _make_base_state(goal="Build a microservice")
        cmd = planner_node(state)

        assert cmd.goto == "router"
        update = cmd.update
        assert len(update["plan"]) == 2
        assert update["plan"][0].assigned_agent == AgentRole.RESEARCHER
        assert update["plan"][1].assigned_agent == AgentRole.CODER
        assert update["current_task_index"] == 0
        assert len(update["trace_log"]) == 1
        assert update["trace_log"][0].entry_type == "plan"

    @patch("src.agents.planner.ChatGroq")
    def test_planner_markdown_code_block(self, mock_llm_cls: MagicMock) -> None:
        """Test planner parsing JSON when wrapped in markdown code fence."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        wrapped = "```json\n[{\"id\": \"t1\", \"description\": \"Inspect db\", \"assigned_agent\": \"db_analyst\"}]\n```"
        mock_llm.invoke.return_value = AIMessage(content=wrapped)

        state = _make_base_state(goal="Inspect database schema")
        cmd = planner_node(state)

        assert cmd.goto == "router"
        assert len(cmd.update["plan"]) == 1
        assert cmd.update["plan"][0].assigned_agent == AgentRole.DB_ANALYST

    @patch("src.agents.planner.ChatGroq")
    def test_planner_failure_routing(self, mock_llm_cls: MagicMock) -> None:
        """Test planner handles invalid output by routing to __end__ with failure status."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Not a valid JSON array")

        state = _make_base_state(goal="Fail me")
        cmd = planner_node(state)

        assert cmd.goto == "__end__"
        assert cmd.update["status"] == WorkflowStatus.FAILED
        assert "Planning failed" in cmd.update["error"]
        assert cmd.update["trace_log"][0].entry_type == "error"


class TestResearcherAgent:
    """Tests for the researcher agent node."""

    @patch("src.agents.researcher.ChatGroq")
    def test_researcher_routes_to_tools(self, mock_llm_cls: MagicMock) -> None:
        """Test researcher routes to researcher_tools when tool calls are generated."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "LangGraph"}, "id": "call_1"}],
        )
        mock_bound.invoke.return_value = tool_call_msg

        node = create_researcher_node(tools=[])
        task = SubTask(id="r1", description="Research LangGraph", assigned_agent=AgentRole.RESEARCHER)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "researcher_tools"
        assert cmd.update["messages"][0] == tool_call_msg
        assert cmd.update["trace_log"][0].metadata["has_tool_calls"] is True

    @patch("src.agents.researcher.ChatGroq")
    def test_researcher_completes_to_evaluator(self, mock_llm_cls: MagicMock) -> None:
        """Test researcher saves result and routes to evaluator when no tool calls remain."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        final_msg = AIMessage(content="Found comprehensive LangGraph documentation.")
        mock_bound.invoke.return_value = final_msg

        node = create_researcher_node(tools=[])
        task = SubTask(id="r1", description="Research LangGraph", assigned_agent=AgentRole.RESEARCHER)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "evaluator"
        assert cmd.update["results"]["r1"] == "Found comprehensive LangGraph documentation."


class TestCoderAgent:
    """Tests for the coder agent node."""

    @patch("src.agents.coder.ChatGroq")
    def test_coder_routes_to_tools(self, mock_llm_cls: MagicMock) -> None:
        """Test coder routes to coder_tools when tool calls are generated."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "execute_python", "args": {"code": "print(1)"}, "id": "c1"}],
        )
        mock_bound.invoke.return_value = tool_call_msg

        node = create_coder_node(tools=[])
        task = SubTask(id="c1", description="Compute prime numbers", assigned_agent=AgentRole.CODER)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "coder_tools"
        assert cmd.update["trace_log"][0].agent == "coder"

    @patch("src.agents.coder.ChatGroq")
    def test_coder_completes_to_evaluator(self, mock_llm_cls: MagicMock) -> None:
        """Test coder saves result and routes to evaluator when code task is complete."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        final_msg = AIMessage(content="Prime calculation script passed all tests.")
        mock_bound.invoke.return_value = final_msg

        node = create_coder_node(tools=[])
        task = SubTask(id="c1", description="Compute prime numbers", assigned_agent=AgentRole.CODER)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "evaluator"
        assert cmd.update["results"]["c1"] == "Prime calculation script passed all tests."


class TestDbAnalystAgent:
    """Tests for the database analyst agent node."""

    @patch("src.agents.db_analyst.ChatGroq")
    def test_db_analyst_routes_to_tools_and_flags_write(self, mock_llm_cls: MagicMock) -> None:
        """Test db analyst routes to db_tools and identifies write_db approval metadata."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        tool_call_msg = AIMessage(
            content="Writing records to the audit log",
            tool_calls=[{"name": "write_db", "args": {"query": "INSERT INTO logs VALUES(1)"}, "id": "db1"}],
        )
        mock_bound.invoke.return_value = tool_call_msg

        node = create_db_analyst_node(tools=[])
        task = SubTask(id="db_task", description="Record logs", assigned_agent=AgentRole.DB_ANALYST)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "db_tools"
        assert cmd.update["trace_log"][0].metadata["requires_approval"] is True

    @patch("src.agents.db_analyst.ChatGroq")
    def test_db_analyst_completes_to_evaluator(self, mock_llm_cls: MagicMock) -> None:
        """Test db analyst routes to evaluator when queries are finished."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_llm_cls.return_value = mock_llm

        final_msg = AIMessage(content="Database inspection found 5 active tables.")
        mock_bound.invoke.return_value = final_msg

        node = create_db_analyst_node(tools=[])
        task = SubTask(id="db_task", description="Inspect schema", assigned_agent=AgentRole.DB_ANALYST)
        state = _make_base_state(plan=[task], current_task_index=0)

        cmd = node(state)
        assert cmd.goto == "evaluator"
        assert cmd.update["results"]["db_task"] == "Database inspection found 5 active tables."


class TestEvaluatorAgent:
    """Tests for the evaluator agent node."""

    @patch("src.agents.evaluator.ChatGroq")
    def test_evaluator_complete_intermediate_task(self, mock_llm_cls: MagicMock) -> None:
        """Test evaluator advances to router when an intermediate task completes."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content=json.dumps({"decision": "complete", "reason": "Output satisfies requirements"})
        )

        task1 = SubTask(id="t1", description="Task 1", assigned_agent=AgentRole.RESEARCHER)
        task2 = SubTask(id="t2", description="Task 2", assigned_agent=AgentRole.CODER)
        state = _make_base_state(
            plan=[task1, task2],
            current_task_index=0,
            results={"t1": "Research done"},
        )

        cmd = evaluator_node(state)
        assert cmd.goto == "router"
        assert cmd.update["current_task_index"] == 1
        assert task1.status == TaskStatus.COMPLETED

    @patch("src.agents.evaluator.ChatGroq")
    def test_evaluator_complete_last_task(self, mock_llm_cls: MagicMock) -> None:
        """Test evaluator routes to __end__ when the final task in the plan completes."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content=json.dumps({"decision": "complete", "reason": "All goals met"})
        )

        task = SubTask(id="t1", description="Only Task", assigned_agent=AgentRole.RESEARCHER)
        state = _make_base_state(
            plan=[task],
            current_task_index=0,
            results={"t1": "Final result"},
        )

        cmd = evaluator_node(state)
        assert cmd.goto == "__end__"
        assert cmd.update["status"] == WorkflowStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    @patch("src.agents.evaluator.ChatGroq")
    def test_evaluator_retry_decision(self, mock_llm_cls: MagicMock) -> None:
        """Test evaluator routes to router on retry decision."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content=json.dumps({"decision": "retry", "reason": "Incomplete data"})
        )

        task = SubTask(id="t1", description="Task", assigned_agent=AgentRole.RESEARCHER)
        state = _make_base_state(
            plan=[task],
            current_task_index=0,
            results={"t1": "Partial output"},
        )

        cmd = evaluator_node(state)
        assert cmd.goto == "router"
        assert task.status == TaskStatus.PENDING

    @patch("src.agents.evaluator.ChatGroq")
    def test_evaluator_replan_decision(self, mock_llm_cls: MagicMock) -> None:
        """Test evaluator routes to planner on replan decision."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content=json.dumps({"decision": "replan", "reason": "Goal cannot be met with current plan"})
        )

        task = SubTask(id="t1", description="Task", assigned_agent=AgentRole.RESEARCHER)
        state = _make_base_state(
            plan=[task],
            current_task_index=0,
            results={"t1": "Failure"},
        )

        cmd = evaluator_node(state)
        assert cmd.goto == "planner"
