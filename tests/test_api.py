"""Unit tests for FastAPI application and API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, lifespan
from src.api.routes import _active_workflows
from src.api.schemas import (
    ApprovalActionRequest,
    TraceResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowStatusResponse,
)
from src.core.models import BudgetUsage, SubTask, AgentRole, TaskStatus


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient instance without lifespan to avoid external connections."""
    return TestClient(app, raise_server_exceptions=False)


class TestSchemas:
    """Tests for API request and response schemas."""

    def test_workflow_run_request(self) -> None:
        """Verify WorkflowRunRequest defaults and validation."""
        req = WorkflowRunRequest(goal="Analyze market data")
        assert req.goal == "Analyze market data"
        assert req.max_steps == 50
        assert req.max_tokens == 100_000
        assert req.max_cost_usd == 5.0

    def test_workflow_status_response(self) -> None:
        """Verify WorkflowStatusResponse serialization."""
        resp = WorkflowStatusResponse(
            run_id="run-123",
            goal="Analyze data",
            status="running",
            current_task_index=1,
            total_tasks=3,
            step_count=5,
            budget={"total_tokens": 1200},
            plan=[{"id": "t1", "description": "Step 1"}],
            results={"t1": "Done"},
        )
        assert resp.run_id == "run-123"
        assert resp.current_task_index == 1
        assert resp.total_tasks == 3
        assert resp.results["t1"] == "Done"

    def test_workflow_run_response(self) -> None:
        """Verify WorkflowRunResponse."""
        resp = WorkflowRunResponse(
            run_id="run-123",
            status="running",
            message="Workflow started",
        )
        assert resp.run_id == "run-123"
        assert resp.status == "running"

    def test_approval_action_request(self) -> None:
        """Verify ApprovalActionRequest."""
        req = ApprovalActionRequest(feedback="Looks good")
        assert req.feedback == "Looks good"

        req_empty = ApprovalActionRequest()
        assert req_empty.feedback == ""

    def test_trace_response(self) -> None:
        """Verify TraceResponse."""
        resp = TraceResponse(
            run_id="run-123",
            entries=[{"entry_type": "plan", "content": "Initial plan"}],
            total_entries=1,
        )
        assert resp.run_id == "run-123"
        assert len(resp.entries) == 1
        assert resp.total_entries == 1


class TestApiRoutes:
    """Tests for API endpoints."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Verify root health check returns 200 and expected payload."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Multi-Agent AI Workflow API"
        assert data["status"] == "running"

    def test_health_endpoint(self, client: TestClient) -> None:
        """Verify /health endpoint returns 200 and healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_get_status_not_found(self, client: TestClient) -> None:
        """Verify 404 is returned for an unknown workflow run ID."""
        response = client.get("/workflow/unknown-id/status")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_trace_not_found(self, client: TestClient) -> None:
        """Verify 404 is returned for trace of an unknown run ID."""
        response = client.get("/workflow/unknown-id/trace")
        assert response.status_code == 404

    def test_approve_not_found(self, client: TestClient) -> None:
        """Verify 404 is returned when approving an unknown run ID."""
        response = client.post("/workflow/unknown-id/approve", json={"feedback": "ok"})
        assert response.status_code == 404

    def test_reject_not_found(self, client: TestClient) -> None:
        """Verify 404 is returned when rejecting an unknown run ID."""
        response = client.post("/workflow/unknown-id/reject", json={"feedback": "no"})
        assert response.status_code == 404

    def test_resume_not_found(self, client: TestClient) -> None:
        """Verify 404 is returned when resuming an unknown run ID."""
        response = client.post("/workflow/unknown-id/resume")
        assert response.status_code == 404

    def test_active_workflow_lifecycle(self, client: TestClient) -> None:
        """Verify status, trace, approve, reject, and resume with a mocked active workflow."""
        run_id = "test-run-lifecycle"

        # Mock graph state and methods
        mock_graph = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {
            "status": "running",
            "current_task_index": 0,
            "step_count": 2,
            "plan": [
                SubTask(
                    id="task-1",
                    description="Run DB query",
                    assigned_agent=AgentRole.DB_ANALYST,
                    status=TaskStatus.IN_PROGRESS,
                )
            ],
            "budget": BudgetUsage(total_tokens=500, total_cost_usd=0.01),
            "results": {"task-1": "in progress"},
            "error": None,
            "trace_log": [
                {"entry_type": "plan", "agent": "planner", "content": "Created plan"}
            ],
        }
        mock_graph.aget_state.return_value = mock_state
        mock_graph.ainvoke.return_value = {"status": "completed"}
        mock_graph.aupdate_state.return_value = None

        _active_workflows[run_id] = {
            "graph": mock_graph,
            "config": {"configurable": {"thread_id": run_id}},
            "goal": "Test lifecycle",
        }

        try:
            # 1. Test Status
            status_resp = client.get(f"/workflow/{run_id}/status")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            assert status_data["run_id"] == run_id
            assert status_data["goal"] == "Test lifecycle"
            assert status_data["total_tasks"] == 1
            assert status_data["step_count"] == 2

            # 2. Test Trace
            trace_resp = client.get(f"/workflow/{run_id}/trace")
            assert trace_resp.status_code == 200
            trace_data = trace_resp.json()
            assert trace_data["run_id"] == run_id
            assert len(trace_data["entries"]) == 1

            # 3. Test Approve
            approve_resp = client.post(f"/workflow/{run_id}/approve", json={"feedback": "approved"})
            assert approve_resp.status_code == 200
            assert approve_resp.json()["status"] == "approved"
            mock_graph.ainvoke.assert_awaited()

            # 4. Test Reject
            reject_resp = client.post(f"/workflow/{run_id}/reject", json={"feedback": "rejected query"})
            assert reject_resp.status_code == 200
            assert reject_resp.json()["status"] == "rejected"
            mock_graph.aupdate_state.assert_awaited()

            # 5. Test Resume
            resume_resp = client.post(f"/workflow/{run_id}/resume")
            assert resume_resp.status_code == 200
            assert resume_resp.json()["status"] == "resumed"

        finally:
            # Clean up
            _active_workflows.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_lifespan(self) -> None:
        """Verify lifespan handles startup and shutdown gracefully."""
        with (
            patch("src.api.main.create_checkpointer", AsyncMock()),
            patch("src.api.main.get_redis", AsyncMock()),
            patch("src.api.main.init_database", AsyncMock()),
            patch("src.api.main.cleanup_checkpointer", AsyncMock()),
            patch("src.api.main.close_redis", AsyncMock()),
        ):
            async with lifespan(app):
                pass
