"""Unit tests for src.graph modules: guards, callbacks, and workflow."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.checkpoint.memory import MemorySaver

from src.core.models import (
    AgentRole,
    BudgetUsage,
    SubTask,
    TaskStatus,
    TraceEntry,
    WorkflowConfig,
    WorkflowStatus,
)
from src.core.state import WorkflowState
from src.graph.callbacks import MODEL_PRICING, WorkflowCallbackHandler
from src.graph.guards import (
    GuardTripped,
    check_budget,
    check_loop_detection,
    check_step_limit,
    run_all_guards,
)
from src.graph.workflow import (
    _guard_node,
    _router_node,
    build_workflow,
    get_initial_state,
)


class TestGuards:
    """Tests for safety guardrails."""

    def test_guard_tripped_exception(self) -> None:
        """Test GuardTripped exception initialization."""
        exc = GuardTripped("test_guard", "Something went wrong")
        assert exc.guard_name == "test_guard"
        assert exc.message == "Something went wrong"
        assert "Guard 'test_guard' tripped" in str(exc)

    def test_check_step_limit_safe(self) -> None:
        """Test step limit check when steps are below limit."""
        state: WorkflowState = {"step_count": 10}  # type: ignore[typeddict-item]
        config = WorkflowConfig(max_steps=50)
        assert check_step_limit(state, config) is True

    def test_check_step_limit_exceeded(self) -> None:
        """Test step limit check when steps reach or exceed limit."""
        state: WorkflowState = {"step_count": 50}  # type: ignore[typeddict-item]
        config = WorkflowConfig(max_steps=50)
        assert check_step_limit(state, config) is False

        state_over: WorkflowState = {"step_count": 55}  # type: ignore[typeddict-item]
        assert check_step_limit(state_over, config) is False

    def test_check_budget_safe(self) -> None:
        """Test budget check within limits."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "budget": BudgetUsage(total_tokens=1000, total_cost_usd=0.05)
        }
        config = WorkflowConfig(max_tokens=100000, max_cost_usd=5.0)
        assert check_budget(state, config) is True

    def test_check_budget_tokens_exceeded(self) -> None:
        """Test budget check when total tokens exceed limit."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "budget": BudgetUsage(total_tokens=100001, total_cost_usd=1.0)
        }
        config = WorkflowConfig(max_tokens=100000, max_cost_usd=5.0)
        assert check_budget(state, config) is False

    def test_check_budget_cost_exceeded(self) -> None:
        """Test budget check when cost exceeds limit."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "budget": BudgetUsage(total_tokens=1000, total_cost_usd=5.5)
        }
        config = WorkflowConfig(max_tokens=100000, max_cost_usd=5.0)
        assert check_budget(state, config) is False

    def test_check_budget_dict_state(self) -> None:
        """Test budget check when budget is provided as a raw dictionary."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "budget": {"total_tokens": 500, "total_cost_usd": 0.02}  # type: ignore[typeddict-item]
        }
        config = WorkflowConfig(max_tokens=1000, max_cost_usd=1.0)
        assert check_budget(state, config) is True

    def test_check_loop_detection_short_history(self) -> None:
        """Test loop detection with fewer messages than the window."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "messages": [AIMessage(content="Hello"), HumanMessage(content="Hi")]
        }
        assert check_loop_detection(state, window=6) is True

    def test_check_loop_detection_diverse_messages(self) -> None:
        """Test loop detection when messages are diverse."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "messages": [
                AIMessage(content=f"Step {i} completed") for i in range(8)
            ]
        }
        assert check_loop_detection(state, window=6) is True

    def test_check_loop_detection_repetitive_loop(self) -> None:
        """Test loop detection when agent repeats identical messages."""
        state: WorkflowState = {  # type: ignore[typeddict-item]
            "messages": [
                AIMessage(content="I am doing X"),
                AIMessage(content="I am doing X"),
                AIMessage(content="I am doing X"),
                AIMessage(content="I am doing X"),
                AIMessage(content="I am doing X"),
                AIMessage(content="I am doing X"),
            ]
        }
        assert check_loop_detection(state, window=6) is False

    def test_run_all_guards_success(self) -> None:
        """Test run_all_guards passes when all conditions are normal."""
        state = get_initial_state("Build website")
        safe, err = run_all_guards(state)  # type: ignore[arg-type]
        assert safe is True
        assert err is None

    def test_run_all_guards_failure_step(self) -> None:
        """Test run_all_guards fails on step count."""
        state = get_initial_state("Build website")
        state["step_count"] = 100
        safe, err = run_all_guards(state)  # type: ignore[arg-type]
        assert safe is False
        assert err is not None
        assert "Step limit exceeded" in err


class TestCallbacks:
    """Tests for WorkflowCallbackHandler."""

    def test_initialization(self) -> None:
        """Verify callback handler starts with clean state."""
        handler = WorkflowCallbackHandler(model_name="llama-3.3-70b-versatile")
        assert handler.model_name == "llama-3.3-70b-versatile"
        assert handler.get_budget().total_tokens == 0
        assert len(handler.get_trace()) == 0

    def test_llm_tracking_with_token_usage(self) -> None:
        """Test LLM start and end tracking with standard token usage in llm_output."""
        handler = WorkflowCallbackHandler(model_name="llama-3.3-70b-versatile")
        run_id = uuid4()

        handler.on_llm_start({}, ["Write a query"], run_id=run_id)

        llm_result = LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )
        handler.on_llm_end(llm_result, run_id=run_id)

        budget = handler.get_budget()
        assert budget.prompt_tokens == 100
        assert budget.completion_tokens == 50
        assert budget.total_tokens == 150
        expected_cost = (100 * 0.59 + 50 * 0.79) / 1_000_000
        assert abs(budget.total_cost_usd - expected_cost) < 1e-6

        traces = handler.get_trace()
        assert len(traces) == 1
        assert traces[0].entry_type == "tool_result"
        assert traces[0].agent == "llm"
        assert traces[0].metadata["prompt_tokens"] == 100

    def test_tool_tracking(self) -> None:
        """Test tool start and end callback tracking."""
        handler = WorkflowCallbackHandler()
        run_id = uuid4()

        handler.on_tool_start({"name": "web_search"}, "AI agents", run_id=run_id)
        assert handler.get_budget().tool_calls == 1

        traces = handler.get_trace()
        assert len(traces) == 1
        assert traces[0].entry_type == "tool_call"
        assert traces[0].metadata["tool"] == "web_search"

        handler.on_tool_end("Found 3 articles", run_id=run_id)
        traces = handler.get_trace()
        assert len(traces) == 2
        assert traces[1].entry_type == "tool_result"
        assert traces[1].metadata["output_preview"] == "Found 3 articles"

    def test_reset(self) -> None:
        """Test reset restores handler to empty state."""
        handler = WorkflowCallbackHandler()
        run_id = uuid4()
        handler.on_tool_start({"name": "web_search"}, "AI", run_id=run_id)
        assert handler.get_budget().tool_calls == 1
        assert len(handler.get_trace()) == 1

        handler.reset()
        assert handler.get_budget().tool_calls == 0
        assert len(handler.get_trace()) == 0


class TestWorkflow:
    """Tests for workflow routing and initial state."""

    def test_get_initial_state(self) -> None:
        """Verify initial state construction."""
        state = get_initial_state("Analyze financial data", run_id="test-run-1")
        assert state["goal"] == "Analyze financial data"
        assert state["run_id"] == "test-run-1"
        assert state["status"] == WorkflowStatus.RUNNING
        assert state["step_count"] == 0
        assert state["current_task_index"] == 0
        assert state["plan"] == []
        assert state["results"] == {}
        assert state["error"] is None

    def test_guard_node_safe(self) -> None:
        """Verify guard node transitions to planner when safe."""
        state = get_initial_state("Test goal")
        cmd = _guard_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "planner"
        assert cmd.update == {"step_count": 1}

    def test_guard_node_tripped(self) -> None:
        """Verify guard node routes to __end__ and updates failure when guard trips."""
        state = get_initial_state("Test goal")
        state["step_count"] = 100  # Exceeds max_steps (50)
        cmd = _guard_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "__end__"
        assert cmd.update["status"] == WorkflowStatus.FAILED
        assert "Step limit exceeded" in cmd.update["error"]

    def test_router_node_completed_plan(self) -> None:
        """Verify router ends when task_index reaches end of plan."""
        state = get_initial_state("Test goal")
        state["plan"] = [
            SubTask(description="Task 1", assigned_agent=AgentRole.RESEARCHER)
        ]
        state["current_task_index"] = 1  # Index beyond plan length
        cmd = _router_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "__end__"
        assert cmd.update["status"] == WorkflowStatus.COMPLETED

    def test_router_node_route_to_researcher(self) -> None:
        """Verify router forwards to researcher node."""
        task = SubTask(
            id="t1",
            description="Find papers",
            assigned_agent=AgentRole.RESEARCHER,
        )
        state = get_initial_state("Test goal")
        state["plan"] = [task]
        state["current_task_index"] = 0

        cmd = _router_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "researcher"
        assert cmd.update["step_count"] == 1
        assert len(cmd.update["trace_log"]) == 1
        assert cmd.update["trace_log"][0].entry_type == "handoff"

    def test_router_node_route_to_coder(self) -> None:
        """Verify router forwards to coder node."""
        task = SubTask(
            id="t2",
            description="Write script",
            assigned_agent=AgentRole.CODER,
        )
        state = get_initial_state("Test goal")
        state["plan"] = [task]
        state["current_task_index"] = 0

        cmd = _router_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "coder"

    def test_router_node_route_to_db_analyst(self) -> None:
        """Verify router forwards to db_analyst node."""
        task = SubTask(
            id="t3",
            description="Query database",
            assigned_agent=AgentRole.DB_ANALYST,
        )
        state = get_initial_state("Test goal")
        state["plan"] = [task]
        state["current_task_index"] = 0

        cmd = _router_node(state)  # type: ignore[arg-type]
        assert cmd.goto == "db_analyst"

    @pytest.mark.asyncio
    async def test_build_workflow_compilation(self) -> None:
        """Verify graph builds and compiles without errors."""
        import os
        from unittest.mock import patch as _patch

        env_overrides = {
            "GROQ_API_KEY": "gsk-test-key-for-compilation-only",
        }
        with _patch.dict(os.environ, env_overrides):
            checkpointer = MemorySaver()
            graph = await build_workflow(checkpointer=checkpointer)
            assert graph is not None
            # Verify compiled graph has expected nodes
            assert "guard" in graph.nodes
            assert "planner" in graph.nodes
            assert "router" in graph.nodes
            assert "researcher" in graph.nodes
            assert "coder" in graph.nodes
            assert "db_analyst" in graph.nodes
            assert "evaluator" in graph.nodes
