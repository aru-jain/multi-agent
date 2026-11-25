import type {
  TraceResponse,
  WorkflowRunRequest,
  WorkflowRunResponse,
  WorkflowStatusResponse,
} from "../types/workflow";

const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}

export async function startWorkflow(
  payload: WorkflowRunRequest
): Promise<WorkflowRunResponse> {
  return request<WorkflowRunResponse>("/workflow/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getWorkflowStatus(
  runId: string
): Promise<WorkflowStatusResponse> {
  return request<WorkflowStatusResponse>(`/workflow/${runId}/status`);
}

export async function getWorkflowTrace(runId: string): Promise<TraceResponse> {
  return request<TraceResponse>(`/workflow/${runId}/trace`);
}

export async function approveWorkflow(
  runId: string,
  feedback = ""
): Promise<{ status: string; message: string }> {
  return request(`/workflow/${runId}/approve`, {
    method: "POST",
    body: JSON.stringify({ feedback }),
  });
}

export async function rejectWorkflow(
  runId: string,
  feedback = ""
): Promise<{ status: string; message: string }> {
  return request(`/workflow/${runId}/reject`, {
    method: "POST",
    body: JSON.stringify({ feedback }),
  });
}

export async function resumeWorkflow(
  runId: string
): Promise<{ status: string; message: string }> {
  return request(`/workflow/${runId}/resume`, {
    method: "POST",
  });
}

export function getWebSocketUrl(runId: string): string {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return `${wsBase}/workflow/${runId}/stream`;
}
