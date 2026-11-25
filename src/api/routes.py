"""FastAPI router endpoints for workflow lifecycle management, human approval, and streaming."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from src.api.schemas import (
    ApprovalActionRequest,
    TraceResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowStatusResponse,
)
from src.core.models import BudgetUsage, WorkflowStatus
from src.persistence.checkpointer import get_checkpointer

try:
    from src.graph.workflow import build_workflow, get_initial_state
except ImportError:
    # Graceful fallback when graph module is not yet loaded
    async def build_workflow(checkpointer: Any = None) -> Any:  # type: ignore[misc]
        raise NotImplementedError("src.graph.workflow is not yet available")

    def get_initial_state(goal: str, run_id: str) -> dict[str, Any]:  # type: ignore[misc]
        return {
            "messages": [],
            "plan": [],
            "current_task_index": 0,
            "results": {},
            "step_count": 0,
            "budget": BudgetUsage(),
            "status": WorkflowStatus.RUNNING,
            "error": None,
            "trace_log": [],
            "goal": goal,
            "run_id": run_id,
        }

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflow", tags=["workflow"])

__all__ = ["router", "_active_workflows"]

# Store active workflow references (in production, use persistent storage)
_active_workflows: dict[str, dict[str, Any]] = {}


@router.post("/run", response_model=WorkflowRunResponse)
async def start_workflow(request: WorkflowRunRequest) -> WorkflowRunResponse:
    """Start a new multi-agent workflow.

    Args:
        request: Workflow run request specification.

    Returns:
        WorkflowRunResponse: Confirmation response with generated run ID.
    """
    run_id = str(uuid.uuid4())
    thread_id = run_id  # Use run_id as thread_id for checkpointer

    try:
        # Build the workflow graph
        if asyncio.iscoroutinefunction(build_workflow):
            graph = await build_workflow(checkpointer=get_checkpointer())
        else:
            graph = build_workflow(checkpointer=get_checkpointer())
            if asyncio.iscoroutine(graph):
                graph = await graph

        # Create initial state
        initial_state = get_initial_state(goal=request.goal, run_id=run_id)
        config = {"configurable": {"thread_id": thread_id}}

        # Store reference
        _active_workflows[run_id] = {
            "graph": graph,
            "config": config,
            "goal": request.goal,
        }

        # Run the workflow in background
        asyncio.create_task(_run_workflow(run_id, graph, initial_state, config))

        return WorkflowRunResponse(
            run_id=run_id,
            status="running",
            message=f"Workflow started for goal: {request.goal}",
        )
    except Exception as e:
        logger.error("Failed to start workflow: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_workflow(
    run_id: str, graph: Any, initial_state: dict[str, Any], config: dict[str, Any]
) -> None:
    """Run the workflow asynchronously in the background.

    Args:
        run_id: Unique workflow run identifier.
        graph: Compiled LangGraph workflow graph.
        initial_state: Initial state dictionary.
        config: Execution configuration containing thread_id.
    """
    try:
        result = await graph.ainvoke(initial_state, config=config)
        status_val = result.get("status") if isinstance(result, dict) else "done"
        logger.info("Workflow %s completed with status: %s", run_id, status_val)
    except Exception as e:
        logger.error("Workflow %s failed: %s", run_id, e)


@router.get("/{run_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(run_id: str) -> WorkflowStatusResponse:
    """Get the current status of a workflow.

    Args:
        run_id: Unique workflow run identifier.

    Returns:
        WorkflowStatusResponse: Detailed snapshot of the workflow state.
    """
    if run_id not in _active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {run_id} not found")

    workflow_info = _active_workflows[run_id]
    graph = workflow_info["graph"]
    config = workflow_info["config"]

    try:
        state = await graph.aget_state(config)
        values = state.values if state.values else {}

        plan = values.get("plan", [])
        plan_dicts: list[dict[str, Any]] = []
        for task in plan:
            if hasattr(task, "model_dump"):
                plan_dicts.append(task.model_dump())
            elif isinstance(task, dict):
                plan_dicts.append(task)

        budget = values.get("budget", BudgetUsage())
        if hasattr(budget, "model_dump"):
            budget_dict = budget.model_dump()
        elif isinstance(budget, dict):
            budget_dict = budget
        else:
            budget_dict = {}

        return WorkflowStatusResponse(
            run_id=run_id,
            goal=workflow_info["goal"],
            status=str(values.get("status", "unknown")),
            current_task_index=values.get("current_task_index", 0),
            total_tasks=len(plan),
            step_count=values.get("step_count", 0),
            budget=budget_dict,
            error=values.get("error"),
            plan=plan_dicts,
            results=values.get("results", {}),
        )
    except Exception as e:
        logger.error("Failed to get status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/trace", response_model=TraceResponse)
async def get_workflow_trace(run_id: str) -> TraceResponse:
    """Get the full trace log of a workflow.

    Args:
        run_id: Unique workflow run identifier.

    Returns:
        TraceResponse: List of trace events for the given run.
    """
    if run_id not in _active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {run_id} not found")

    workflow_info = _active_workflows[run_id]
    graph = workflow_info["graph"]
    config = workflow_info["config"]

    try:
        state = await graph.aget_state(config)
        values = state.values if state.values else {}
        trace_log = values.get("trace_log", [])

        entries: list[dict[str, Any]] = []
        for entry in trace_log:
            if hasattr(entry, "model_dump"):
                entries.append(entry.model_dump())
            elif isinstance(entry, dict):
                entries.append(entry)

        return TraceResponse(
            run_id=run_id,
            entries=entries,
            total_entries=len(entries),
        )
    except Exception as e:
        logger.error("Failed to get trace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/approve")
async def approve_workflow(
    run_id: str, request: ApprovalActionRequest
) -> dict[str, Any]:
    """Approve a pending human checkpoint and resume the workflow.

    Args:
        run_id: Unique workflow run identifier.
        request: Approval action payload.

    Returns:
        dict[str, Any]: Confirmation of approval and resumption.
    """
    if run_id not in _active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {run_id} not found")

    workflow_info = _active_workflows[run_id]
    graph = workflow_info["graph"]
    config = workflow_info["config"]

    try:
        # Resume the workflow from the interrupt point
        # Pass None to accept the pending tool call
        await graph.ainvoke(None, config=config)
        return {"status": "approved", "message": "Workflow resumed after approval"}
    except Exception as e:
        logger.error("Failed to approve workflow: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/reject")
async def reject_workflow(
    run_id: str, request: ApprovalActionRequest
) -> dict[str, Any]:
    """Reject a pending action and provide feedback.

    Args:
        run_id: Unique workflow run identifier.
        request: Rejection details and feedback.

    Returns:
        dict[str, Any]: Confirmation of rejection.
    """
    if run_id not in _active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {run_id} not found")

    workflow_info = _active_workflows[run_id]
    graph = workflow_info["graph"]
    config = workflow_info["config"]

    try:
        # Update state to skip the current tool call
        feedback_msg = (
            f"Action rejected by human. Feedback: {request.feedback}"
            if request.feedback
            else "Action rejected by human."
        )

        await graph.aupdate_state(
            config,
            {
                "messages": [HumanMessage(content=feedback_msg)],
                "trace_log": [
                    {"entry_type": "approval", "agent": "human", "content": feedback_msg}
                ],
            },
        )
        # Resume with the updated state
        await graph.ainvoke(None, config=config)
        return {"status": "rejected", "message": "Workflow updated with rejection feedback"}
    except Exception as e:
        logger.error("Failed to reject workflow: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/resume")
async def resume_workflow(run_id: str) -> dict[str, Any]:
    """Resume a paused workflow.

    Args:
        run_id: Unique workflow run identifier.

    Returns:
        dict[str, Any]: Confirmation of resumption.
    """
    if run_id not in _active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {run_id} not found")

    workflow_info = _active_workflows[run_id]
    graph = workflow_info["graph"]
    config = workflow_info["config"]

    try:
        await graph.ainvoke(None, config=config)
        return {"status": "resumed", "message": "Workflow resumed"}
    except Exception as e:
        logger.error("Failed to resume workflow: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/{run_id}/stream")
async def stream_workflow(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for real-time workflow updates.

    Args:
        websocket: WebSocket connection.
        run_id: Unique workflow run identifier.
    """
    await websocket.accept()

    if run_id not in _active_workflows:
        await websocket.send_json({"error": f"Workflow {run_id} not found"})
        await websocket.close()
        return

    try:
        while True:
            workflow_info = _active_workflows.get(run_id)
            if not workflow_info:
                await websocket.send_json({"status": "not_found"})
                break

            graph = workflow_info["graph"]
            config = workflow_info["config"]

            state = await graph.aget_state(config)
            values = state.values if state.values else {}
            status = str(values.get("status", "unknown"))

            await websocket.send_json(
                {
                    "run_id": run_id,
                    "status": status,
                    "step_count": values.get("step_count", 0),
                    "current_task_index": values.get("current_task_index", 0),
                }
            )

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(2)  # Poll every 2 seconds
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", run_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
