"""Unit tests for src.core modules: config, models, and state."""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest

from src.core.config import Settings, get_settings
from src.core.models import (
    AgentRole,
    ApprovalRequest,
    BudgetUsage,
    SubTask,
    TaskStatus,
    ToolResult,
    TraceEntry,
    WorkflowConfig,
    WorkflowStatus,
)
from src.core.state import (
    WorkflowState,
    _increment,
    _merge_results,
    _merge_trace,
)


class TestConfig:
    """Tests for configuration settings."""

    def test_default_settings(self) -> None:
        """Test default values of Settings."""
        settings = Settings()
        assert bool(settings.groq_model)
        assert settings.max_steps == 50
        assert settings.max_tokens == 100_000
        assert settings.max_cost_usd == 5.0
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.database_url.startswith("postgresql")
        assert "redis://" in settings.redis_url

    def test_get_settings_factory(self) -> None:
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_env_var_override(self) -> None:
        """Test that environment variables override settings."""
        env_vars = {
            "GROQ_API_KEY": "test-key-123",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "MAX_STEPS": "25",
            "MAX_COST_USD": "2.5",
            "API_PORT": "9000",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            assert settings.groq_api_key == "test-key-123"
            assert settings.groq_model == "llama-3.1-8b-instant"
            assert settings.max_steps == 25
            assert settings.max_cost_usd == 2.5
            assert settings.api_port == 9000

    def test_settings_frozen(self) -> None:
        """Ensure Settings dataclass is immutable."""
        settings = Settings()
        with pytest.raises(AttributeError):
            settings.groq_model = "other-model"  # type: ignore[misc]


class TestModels:
    """Tests for Pydantic v2 models."""

    def test_enums(self) -> None:
        """Verify enum string representations."""
        assert TaskStatus.PENDING == "pending"
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.AWAITING_APPROVAL == "awaiting_approval"
        assert AgentRole.PLANNER == "planner"
        assert AgentRole.CODER == "coder"

    def test_subtask_creation(self) -> None:
        """Verify SubTask defaults and serialization."""
        task = SubTask(
            description="Write database schema",
            assigned_agent=AgentRole.DB_ANALYST,
        )
        assert len(task.id) == 8
        assert task.status == TaskStatus.PENDING
        assert task.requires_approval is False
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None

        # Verify model dump and reconstruction
        dumped = task.model_dump()
        reconstructed = SubTask.model_validate(dumped)
        assert reconstructed.id == task.id
        assert reconstructed.assigned_agent == AgentRole.DB_ANALYST

    def test_tool_result(self) -> None:
        """Verify ToolResult representation."""
        result = ToolResult(
            success=True,
            tool_name="web_search",
            data={"results": ["https://example.com"]},
            execution_time_ms=120.5,
        )
        assert result.success is True
        assert result.error is None
        assert result.data["results"] == ["https://example.com"]
        assert result.execution_time_ms == 120.5

    def test_trace_entry(self) -> None:
        """Verify TraceEntry timestamp and validation."""
        entry = TraceEntry(
            entry_type="agent_decision",
            agent="coder",
            content="Generated SQL migration script",
            metadata={"step": 1},
        )
        assert entry.timestamp is not None
        assert entry.entry_type == "agent_decision"
        assert entry.agent == "coder"
        assert entry.metadata["step"] == 1

    def test_budget_usage(self) -> None:
        """Verify BudgetUsage tracking."""
        budget = BudgetUsage(
            total_tokens=1500,
            prompt_tokens=1000,
            completion_tokens=500,
            total_cost_usd=0.03,
            total_steps=4,
            tool_calls=2,
        )
        assert budget.total_tokens == 1500
        assert budget.total_cost_usd == 0.03
        assert budget.tool_calls == 2

    def test_approval_request(self) -> None:
        """Verify ApprovalRequest generation."""
        req = ApprovalRequest(
            workflow_run_id="run-abc",
            tool_name="execute_sql",
            tool_args={"query": "DROP TABLE users;"},
            agent=AgentRole.DB_ANALYST,
            reason="Destructive DDL statement requested",
        )
        assert len(req.id) == 8
        assert req.workflow_run_id == "run-abc"
        assert req.agent == AgentRole.DB_ANALYST
        assert req.tool_name == "execute_sql"

    def test_workflow_config(self) -> None:
        """Verify WorkflowConfig default values."""
        cfg = WorkflowConfig()
        assert cfg.max_steps == 50
        assert cfg.max_tokens == 100_000
        assert cfg.max_cost_usd == 5.0
        assert cfg.max_retries_per_task == 3
        assert cfg.max_replans == 2
        assert cfg.tool_timeout_seconds == 30
        assert cfg.require_approval_for_writes is True


class TestStateAndReducers:
    """Tests for WorkflowState reducers and state construction."""

    def test_merge_trace(self) -> None:
        """Verify _merge_trace reducer."""
        e1 = TraceEntry(entry_type="plan", content="Initial plan formulated")
        e2 = TraceEntry(entry_type="agent_decision", content="Assigned task 1")
        merged = _merge_trace([e1], [e2])
        assert len(merged) == 2
        assert merged[0].content == "Initial plan formulated"
        assert merged[1].content == "Assigned task 1"

        # Edge cases with None
        assert len(_merge_trace(None, [e1])) == 1
        assert len(_merge_trace([e1], None)) == 1
        assert _merge_trace(None, None) == []

    def test_merge_results(self) -> None:
        """Verify _merge_results reducer."""
        r1 = {"task_1": "output_1"}
        r2 = {"task_2": "output_2"}
        merged = _merge_results(r1, r2)
        assert merged == {"task_1": "output_1", "task_2": "output_2"}

        # Overwrite behavior
        r3 = {"task_1": "updated_output"}
        merged_updated = _merge_results(merged, r3)
        assert merged_updated["task_1"] == "updated_output"

        # Edge cases with None
        assert _merge_results(None, r1) == {"task_1": "output_1"}
        assert _merge_results(r1, None) == {"task_1": "output_1"}
        assert _merge_results(None, None) == {}

    def test_increment(self) -> None:
        """Verify _increment reducer."""
        assert _increment(1, 2) == 3
        assert _increment(0, 1) == 1
        assert _increment(None, 5) == 5
        assert _increment(5, None) == 5
        assert _increment(None, None) == 0

    def test_workflow_state_structure(self) -> None:
        """Verify constructing a valid WorkflowState dictionary."""
        state: WorkflowState = {
            "messages": [],
            "plan": [
                SubTask(description="Task 1", assigned_agent=AgentRole.RESEARCHER)
            ],
            "current_task_index": 0,
            "results": {},
            "step_count": 0,
            "budget": BudgetUsage(),
            "status": WorkflowStatus.RUNNING,
            "error": None,
            "trace_log": [],
            "goal": "Build multi-agent architecture",
            "run_id": "run-001",
        }
        assert state["goal"] == "Build multi-agent architecture"
        assert state["status"] == WorkflowStatus.RUNNING
        assert len(state["plan"]) == 1
        assert state["plan"][0].assigned_agent == AgentRole.RESEARCHER

    def test_langgraph_stategraph_integration(self) -> None:
        """Verify WorkflowState works properly inside a LangGraph StateGraph."""
        from langgraph.graph import END, START, StateGraph

        def dummy_node(state: WorkflowState) -> dict:
            new_trace = TraceEntry(entry_type="agent_decision", content="Step executed")
            return {
                "step_count": 1,
                "results": {"step_1": "success"},
                "trace_log": [new_trace],
            }

        graph = StateGraph(WorkflowState)
        graph.add_node("dummy", dummy_node)
        graph.add_edge(START, "dummy")
        graph.add_edge("dummy", END)
        app = graph.compile()

        initial_state: WorkflowState = {
            "messages": [],
            "plan": [],
            "current_task_index": 0,
            "results": {"init": True},
            "step_count": 0,
            "budget": BudgetUsage(),
            "status": WorkflowStatus.RUNNING,
            "error": None,
            "trace_log": [],
            "goal": "Test LangGraph execution",
            "run_id": "test-run",
        }

        output = app.invoke(initial_state)
        assert output["step_count"] == 1
        assert output["results"]["init"] is True
        assert output["results"]["step_1"] == "success"
        assert len(output["trace_log"]) == 1
        assert output["trace_log"][0].content == "Step executed"
