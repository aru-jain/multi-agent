export type AgentRole = "planner" | "researcher" | "coder" | "db_analyst" | "evaluator";

export type TaskStatus = "pending" | "in_progress" | "completed" | "failed" | "awaiting_approval";

export type WorkflowStatus =
  | "running"
  | "paused"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "unknown";

export interface SubTask {
  id: string;
  description: string;
  assigned_agent: AgentRole;
  status: TaskStatus;
  requires_approval: boolean;
  dependencies: string[];
  result?: string | null;
  error?: string | null;
}

export interface BudgetUsage {
  total_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_cost_usd?: number;
  total_steps?: number;
  tool_calls?: number;
}

export interface TraceEntry {
  timestamp?: string;
  entry_type:
    | "plan"
    | "agent_decision"
    | "tool_call"
    | "tool_result"
    | "handoff"
    | "guard"
    | "approval"
    | "error"
    | string;
  agent?: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface WorkflowRunRequest {
  goal: string;
  max_steps: number;
  max_tokens: number;
  max_cost_usd: number;
}

export interface WorkflowRunResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface WorkflowStatusResponse {
  run_id: string;
  goal: string;
  status: string;
  current_task_index: number;
  total_tasks: number;
  step_count: number;
  budget: BudgetUsage;
  error?: string | null;
  plan: SubTask[];
  results: Record<string, unknown>;
}

export interface TraceResponse {
  run_id: string;
  entries: TraceEntry[];
  total_entries: number;
}

export interface RunHistoryItem {
  run_id: string;
  goal: string;
  status: WorkflowStatus;
  created_at: string;
  max_steps: number;
  max_tokens: number;
  max_cost_usd: number;
}

export function normalizeStatus(raw: string | undefined): WorkflowStatus {
  if (!raw) return "unknown";
  const cleaned = raw.toLowerCase().replace("workflowstatus.", "").trim();
  if (
    cleaned === "running" ||
    cleaned === "paused" ||
    cleaned === "awaiting_approval" ||
    cleaned === "completed" ||
    cleaned === "failed"
  ) {
    return cleaned;
  }
  return "unknown";
}
